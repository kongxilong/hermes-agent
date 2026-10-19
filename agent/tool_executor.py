"""Tool-call execution: sequential and concurrent dispatch, extracted from AIAgent.

Functions take the parent ``AIAgent`` first; ``run_agent`` keeps thin wrappers and is
reached lazily via ``_ra()`` so ``run_agent._set_interrupt`` patches still work. Every
call's identity travels as a ``_ToolCallRef``; both executors end in the same
observe → commit → project pipeline so the tool-result wire shape is produced once.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：agent/tool_executor.py（约 1752 行）
# 地位：★ 工具调用的执行器（并发 + 串行两条路径）。
#
# 背景：模型一次可能要求调用多个工具。
#   - 互不依赖的（比如同时读两个文件）→ 并发执行更快：execute_tool_calls_concurrent()
#   - 有依赖的 / 有副作用的（比如先写文件再运行）→ 必须按顺序：execute_tool_calls_sequential()
#
# 核心流程（两条路径最后汇合到同一段「观察 → 提交 → 投影」管线）：
#   解析参数 _parse_tool_call / _parse_tool_arguments
#     → 权限检查 _pre_tool_block / _blocked_tool_result
#     → 真正执行 _dispatch_authorized_once
#     → 记录与展示 _begin_tool_execution / _emit_tool_complete_and_risk
#     → 写回消息列表 _commit_tool_result
#
# 这个文件里很有学习价值的工程细节：
#   - 并发用 ThreadPoolExecutor（_max_workers_for_tool_batch 控制并发度）
#   - 超时处理：_ToolTimeoutResult / _resolve_concurrent_tool_timeout
#   - 顺序保证：_StartOrderGate / _WorkerStartOnce —— 并发执行但结果按顺序返回，
#     这样模型看到的 tool_result 顺序和它请求的工具顺序一致（很重要，否则模型会错乱）
#   - 心跳：_run_tool_activity_heartbeat —— 工具跑太久时定期输出「还在跑」，避免用户以为卡死
#
# 小知识：class _ToolTimeoutResult(str) 是「继承 str 的假结果」——用字符串冒充工具结果，
#   这样下游代码不用区分类型就能统一处理。这是一种常见的鸭子类型技巧。
# ==========================================================================
#

from __future__ import annotations

import concurrent.futures
import contextlib
import json
from pathlib import Path
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    build_tool_label as _build_tool_label,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.message_sanitization import coalesce_tool_call_id
from agent.inline_tool_executors import (
    INLINE_TOOL_EXECUTORS,
    InlineToolContext,
    emit_terminal_post_tool_call,
    tool_hook_ids,
)
from agent.tool_dispatch_helpers import (
    _NEVER_PARALLEL_TOOLS,
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    _plan_tool_batch_segments,
    make_tool_result_message,
)
from tools.terminal_tool_lifecycle import get_active_env
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
    extract_persisted_path,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)


_pairing_tool_call_id = coalesce_tool_call_id  # canonical id used by the persisted assistant message


# ┌── 函数 def _tc_name()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：tool_call: Any
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _tc_name(tool_call: Any) -> str:
    # ⤷ 出口 return：返回 getattr(getattr(tool_call, 'function', None), 'name', '') or…
    return getattr(getattr(tool_call, "function", None), "name", "") or "tool"


# ┌── 函数 def _record_persisted_path_for_stub()
# │  作用：记录溢出文件的路径，使得后续的结果引用桩不会悬空（尽最大努力）。
# │  参数：agent, tool_call_id: str, function_result
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _record_persisted_path_for_stub(agent, tool_call_id: str, function_result) -> None:
    """Record the spillover file path so a later result-reference stub can't dangle (best-effort)."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        path = extract_persisted_path(function_result) if isinstance(function_result, str) else None
        if path:
            agent._tool_guardrails.record_persisted_result(tool_call_id, path)
    except Exception as exc:
        logger.debug("persisted-path record for result stub failed: %s", exc)


# ┌── 函数 def _ensure_file_checkpoint()
# │  作用：对文件工具将要修改的同一工作区路径做检查点（用与……相同的方式解析）。
# │  参数：agent, function_name: str, function_args: dict, effective_task_id: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _ensure_file_checkpoint(agent, function_name: str, function_args: dict, effective_task_id: str) -> None:
    """Checkpoint the same workspace path that the file tool will mutate, resolved the way
    file tools do (against the task's live cwd, which differs from the process cwd in Docker)."""
    file_path = function_args.get("path", "")
    if not file_path:
        return
    from tools.file_tools_paths import _resolve_path_for_task

    resolved_path = _resolve_path_for_task(file_path, effective_task_id or "default")
    agent._checkpoint_mgr.ensure_checkpoint(
        agent._checkpoint_mgr.get_working_dir_for_path(str(resolved_path)), f"before {function_name}",
    )


# ┌── 函数 def _budget_for_agent()
# │  作用：按 agent 的上下文窗口缩放后的「工具结果预算配置」。未知长度时……
# │  参数：agent
# │  返回：BudgetConfig
# └──────────────────────────────────────────────────────────────
def _budget_for_agent(agent) -> BudgetConfig:
    """Tool-result BudgetConfig scaled to the agent's context window. Unknown length goes
    through ``budget_for_context_window(None)`` (not DEFAULT_BUDGET) so the MCP threshold
    override still applies.

    Large-context models keep the historical 100K/200K char defaults; small models (e.g. a 65K-token local
    model switched into mid-session) get a budget proportional to their window so a single large tool result
    can't push the request past the model's limit (#23767). Falls back to the default budget when the
    context length isn't resolvable.
    """
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        # ⤷ 出口 return：返回 budget_for_context_window(int(ctx) if ctx else None)
        return budget_for_context_window(int(ctx) if ctx else None)
    except Exception:
        # ⤷ 出口 return：返回 DEFAULT_BUDGET
        return DEFAULT_BUDGET

_MAX_TOOL_WORKERS = 8  # concurrent worker threads per batch
_DEFAULT_IMAGE_PARALLEL_REQUESTS = 4
# Generous: slow-but-valid tool work must never be preempted by the batch guard.
_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0
# Long enough for an approval round-trip, short enough that one wedged dispatch can't starve the batch.
_START_ORDER_GATE_TIMEOUT_S = 120.0
# Fallback only; the effective bound derives from approvals.timeout (_authorization_gate_lock_timeout).
_AUTHORIZATION_GATE_LOCK_TIMEOUT_S = 360.0


# ┌── 函数 def _authorization_gate_lock_timeout()
# │  作用：授权锁的上限 = ``tools.approval_human_wait.human_wait_ceiling``（审批超时 + ……）。
# │  参数：无
# │  返回：float
# └──────────────────────────────────────────────────────────────
def _authorization_gate_lock_timeout() -> float:
    """Authorization-lock bound = ``tools.approval_human_wait.human_wait_ceiling`` (approval timeout +
    margin, capped so it can't overflow Lock.acquire): never break serialization while a
    prompt is answerable, never let a wedged holder park workers forever. Deliberately NOT
    min()'d with the fallback so the gate never gives up early.

    Delegates to ``tools.approval_human_wait.human_wait_ceiling`` — the same bound that clamps a human-wait window's
    deadline contribution — so the two can't drift. Long enough that serialization is never broken while a
    legitimate approval prompt is still answerable; short enough that a wedged holder (hanging
    ``pre_tool_call`` plugin, dead approval client) cannot park other workers forever (#79719). Resolved
    once per gate (per batch), so a mid-process ``approvals.timeout`` change applies from the next batch.
    """
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.approval_human_wait import human_wait_ceiling

        # human_wait_ceiling is platform-safety-capped (agent/deadline.py MAX_SAFE_TIMEOUT_S): a huge
        # approvals.timeout can no longer overflow Lock.acquire's time_t on macOS (#83220). Deliberately NOT
        # min()'d with _AUTHORIZATION_GATE_LOCK_TIMEOUT_S — the gate must never give up while a legitimate
        # approval prompt is still answerable (#79719), so a configured approvals.timeout above 360s must
        # extend the gate.
        # ⤷ 出口 return：返回 human_wait_ceiling()
        return human_wait_ceiling()
    except Exception:
        # ⤷ 出口 return：返回 _AUTHORIZATION_GATE_LOCK_TIMEOUT_S
        return _AUTHORIZATION_GATE_LOCK_TIMEOUT_S


# ┌── 类 class _BatchAbandoned（继承自：BaseException）
# │  说明：在 worker 内部、且批次在派发前被放弃时抛出；这是一个 BaseException……
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _BatchAbandoned(BaseException):
    """Raised inside a worker when the batch was abandoned before dispatch; a BaseException
    so ``except Exception`` handlers in the middleware chain can't swallow it."""


# ┌── 函数 def _parse_tool_arguments()
# │  作用：解析模型给出的参数，不做修复、也不做强制类型转换。
# │  参数：raw_arguments: Any
# │  返回：tuple[dict, Optional[str]]
# └──────────────────────────────────────────────────────────────
def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """Parse model-emitted arguments without repairing or coercing them."""
    # ⤷ 关键分支 try/except：捕获 (json.JSONDecodeError, TypeError)，出错走兜底而不是崩溃。
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        # ⤷ 出口 return：返回 (arguments, None)
        return arguments, None
    # ⤷ 出口 return：返回 ({}, json.dumps({'error': 'Invalid tool arguments', 'message…
    return {}, json.dumps(
        {"error": "Invalid tool arguments", "message": "Tool arguments must be a valid JSON object; tool was not executed."},
        ensure_ascii=False,
    )


# ┌── 函数 def _resolve_concurrent_tool_timeout()
# │  作用：每个批次的并发截止时间：以 ``timeouts.tools.concurrent_batch`` 为准，……
# │  参数：无
# │  返回：float | None
# └──────────────────────────────────────────────────────────────
def _resolve_concurrent_tool_timeout() -> float | None:
    """Per-batch concurrent deadline: ``timeouts.tools.concurrent_batch`` wins,
    ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` is the legacy bridge, ``0``/negative disables."""
    from agent.deadline import resolve_timeout

    # ⤷ 出口 return：返回 resolve_timeout('tools.concurrent_batch', default=_DEFAULT_C…
    return resolve_timeout(
        "tools.concurrent_batch",
        default=_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
        env_var="HERMES_CONCURRENT_TOOL_TIMEOUT_S",
    )


# ┌── 函数 def _flush_session_db_after_tool_progress()
# │  作用：在把工具调用进度投射到任何 UI 之前，先冲刷到会话数据库：工具侧……
# │  参数：agent, messages: list, stage: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _flush_session_db_after_tool_progress(agent, messages: list, *, stage: str) -> bool:
    """Flush tool-call progress to the session DB before projecting it to any UI: tool side
    effects can kill/restart the process before turn-end persistence runs."""
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning

    # Persist exactly the checkpoint text the next model call will see, before stamping
    # this tool result as durable. Already-written rows must never be rewritten later.
    _maybe_inject_run_budget_wrapup(agent, messages)
    _maybe_inject_iteration_budget_warning(agent, messages)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        persisted = agent._flush_messages_to_session_db(messages) is not False
        if not persisted:
            agent._incremental_persistence_failed = True
            # The flush recorded any classified cause; default to 'unknown' only if nothing more specific exists.
            if getattr(agent, "_last_persistence_error_cause", None) is None:
                agent._last_persistence_error_cause = "unknown"
        # ⤷ 出口 return：返回 persisted
        return persisted
    except Exception as exc:
        agent._incremental_persistence_failed = True
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning("Incremental tool-call persistence failed after %s: %s", stage, exc)
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _image_generate_parallel_limit()
# │  作用：配置的图像生成并发上限（偏保守：后端突发请求会撞上速率限制）。
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _image_generate_parallel_limit() -> int:
    """Configured image-generation parallelism cap (conservative: backend bursts hit rate limits)."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        value = image_gen.get("max_parallel_requests") if isinstance(image_gen, dict) else None
    except Exception:
        value = None

    # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = _DEFAULT_IMAGE_PARALLEL_REQUESTS
    # ⤷ 出口 return：返回 max(1, min(limit, _MAX_TOOL_WORKERS))
    return max(1, min(limit, _MAX_TOOL_WORKERS))


# ┌── 函数 def _max_workers_for_tool_batch()
# │  作用：返回一个并发工具批次的 worker 上限。
# │  参数：runnable_calls
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _max_workers_for_tool_batch(runnable_calls) -> int:
    """Return the worker cap for a concurrent tool batch."""
    if not runnable_calls:
        # ⤷ 出口 return：返回 0
        return 0
    max_workers = _MAX_TOOL_WORKERS
    if any((call[2] if len(call) >= 3 else None) == "image_generate" for call in runnable_calls):
        max_workers = min(max_workers, _image_generate_parallel_limit())
    # ⤷ 出口 return：返回 min(len(runnable_calls), max_workers)
    return min(len(runnable_calls), max_workers)


# ┌── 函数 def _ra()
# │  作用：对 ``run_agent`` 的惰性引用，使得 ``run_agent._set_interrupt`` 这类补丁能生效。
# │  参数：无
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    # ⤷ 出口 return：返回 run_agent
    return run_agent


# ┌── 函数 def _is_interpreter_shutdown_submit_error()
# │  作用：关闭竞态的判定条件；``tools.interpreter_shutdown`` 同时认识 CPython 的两种退出消息变体。
# │  参数：exc: RuntimeError
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_interpreter_shutdown_submit_error(exc: RuntimeError) -> bool:
    """Shutdown-race predicate; ``tools.interpreter_shutdown`` knows both CPython message variants.

    Delegates so all sites (cron delivery, conversation-loop retry, tool submission) recognize both CPython
    shutdown-message variants instead of each matching its own substring (the bug class behind
    #55924/#58720).
    """
    from tools.interpreter_shutdown import interpreter_shutting_down

    # ⤷ 出口 return：返回 interpreter_shutting_down(exc)
    return interpreter_shutting_down(exc)


_emit_terminal_post_tool_call = emit_terminal_post_tool_call


# ┌── 类 class _ToolCallRef
# │  说明：一次工具调用在「每个钩子 / 结果消息」视角下的身份：即（可能被……的）……
# │  成员：共 4 个方法；对外可用 4 个
# └──────────────────────────────────────────────────────────────
@dataclass
class _ToolCallRef:
    """Identity of one tool call as every hook / result message sees it: the (possibly
    middleware-rewritten) name and args, the task, the pairing id and the request trace."""

    name: str
    args: dict
    task_id: str
    call_id: str
    trace: list

    # ┌── 函数 def middleware_kwargs()
    # │  作用：关键字形式的 ``_run_agent_tool_execution_middleware``（以及打在它上面的测试补丁）所期望的。
    # │  参数：self
    # │  返回：dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def middleware_kwargs(self) -> dict[str, Any]:
        """Keyword form ``_run_agent_tool_execution_middleware`` (and tests patching it) expect."""
        # ⤷ 出口 return：返回 {'function_name': self.name, 'function_args': self.args, 'ef…
        return {
            "function_name": self.name, "function_args": self.args, "effective_task_id": self.task_id,
            "tool_call_id": self.call_id, "middleware_trace": self.trace,
        }

    # ┌── 函数 def emit_post()
    # │  作用：为这次调用发出唯一的终局 ``post_tool_call``（``outcome`` = status / ……）。
    # │  参数：self, agent, result, trace = ..., **outcome
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def emit_post(self, agent, result, *, trace=None, **outcome) -> None:
        """Emit the one terminal ``post_tool_call`` for this call (``outcome`` = status /
        error_type / error_message / duration_ms). Resolved through the module attribute so
        tests patching ``_emit_terminal_post_tool_call`` still intercept."""
        _emit_terminal_post_tool_call(
            agent,
            function_name=self.name,
            function_args=self.args,
            result=result,
            effective_task_id=self.task_id,
            tool_call_id=self.call_id,
            middleware_trace=list(self.trace if trace is None else trace),
            **outcome,
        )

    # ┌── 函数 def emit_cancelled()
    # │  作用：为工具执行中途的 KeyboardInterrupt 合成 ``cancelled`` 结果，并发出它的钩子。
    # │  参数：self, agent, start_time: float
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def emit_cancelled(self, agent, start_time: float) -> str:
        """Synthesize the ``cancelled`` result for a KeyboardInterrupt mid-tool and emit its hook."""
        message = "Tool execution cancelled by user interrupt"
        result = json.dumps({"error": message, "status": "cancelled"}, ensure_ascii=False)
        self.emit_post(
            agent, result, duration_ms=int((time.time() - start_time) * 1000),
            status="cancelled", error_type="keyboard_interrupt", error_message=message,
        )
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def emit_invalid_arguments()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, agent, result: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def emit_invalid_arguments(self, agent, result: str) -> None:
        self.emit_post(
            agent, result, trace=[],
            status="error", error_type="invalid_tool_arguments", error_message="Tool arguments must be a valid JSON object",
        )


# ┌── 函数 def _append_skipped_tool_results()
# │  作用：为每个尚未开始的调用追加一条 ``tool`` 结果，使得助手的工具调用轮次永不……
# │  参数：agent, messages: list, tool_calls, effective_task_id: str, content: str, hook_error_type: Optional[str] = ..., hook_id: Optional[Callable[[Any], str]] = ..., flush_stage: Optional[str] = ...  …共 9 个
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _append_skipped_tool_results(
    agent,
    messages: list,
    tool_calls,
    effective_task_id: str,
    *,
    content: str,
    hook_error_type: Optional[str] = None,
    hook_id: Optional[Callable[[Any], str]] = None,
    flush_stage: Optional[str] = None,
    stop_on_flush_failure: bool = True,
) -> bool:
    """Append one ``tool`` result per unstarted call so the assistant tool-call turn never
    lacks matching results (role alternation). ``content`` is formatted with ``{name}``;
    ``hook_error_type`` also emits the terminal ``post_tool_call`` (status=cancelled) per
    call with ``hook_id`` overriding the hook's id; ``flush_stage`` flushes after each
    append and returns False on the first failed flush when ``stop_on_flush_failure``."""
    for tc in tool_calls:
        name = _tc_name(tc)
        result = content.format(name=name)
        messages.append(make_tool_result_message(name, result, _pairing_tool_call_id(tc), effect_disposition="none"))
        if hook_error_type is not None:
            _ToolCallRef(name, {}, effective_task_id, (hook_id or _pairing_tool_call_id)(tc), []).emit_post(
                agent, result,
                status="cancelled", error_type=hook_error_type, error_message="Tool execution skipped due to user interrupt",
            )
        if flush_stage is not None:
            flushed = _flush_session_db_after_tool_progress(agent, messages, stage=f"{flush_stage} {name}")
            if not flushed and stop_on_flush_failure:
                # ⤷ 出口 return：返回 False
                return False
    # ⤷ 出口 return：返回 True
    return True


# ┌── 函数 def _tool_search_scoped_names()
# │  作用：会话可以通过 ``tool_call`` 调用的、可延迟的工具名；unwrap 会绕过……
# │  参数：agent
# │  返回：frozenset
# └──────────────────────────────────────────────────────────────
def _tool_search_scoped_names(agent) -> frozenset:
    """Deferrable tool names the session may invoke via ``tool_call``; the unwrap bypasses
    the bridge's scope check in ``model_tools.handle_function_call``, so restricted sessions
    validate against this set. Cached on the agent, keyed by registry scope/generation."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
        # ⤷ 出口 return：返回 frozenset()
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        _registry.current_scope_key(),
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        # ⤷ 出口 return：返回 cached[1]
        return cached[1]
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        names = _ts.scoped_deferrable_names(model_tools.get_tool_definitions(
            enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True, skip_tool_search_assembly=True,
        ) or [])
    except Exception:
        names = frozenset()
    with contextlib.suppress(Exception):
        agent._tool_search_scope_cache = (cache_key, names)
    # ⤷ 出口 return：返回 names
    return names


# ┌── 函数 def _canonical_tool_name()
# │  作用：在 agent 主循环派发之前，先把旧式的工具名别名做映射。
# │  参数：function_name: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _canonical_tool_name(function_name: str) -> str:
    """Map legacy tool-name aliases BEFORE agent-loop dispatch."""
    from model_tools import _LEGACY_TOOL_ALIASES as _lta

    # ⤷ 出口 return：返回 _lta.get(function_name, function_name)
    return _lta.get(function_name, function_name)


# ┌── 函数 def _unwrap_tool_search_call()
# │  作用：剥开 ``tool_call`` 桥接层，使下游钩子（检查点、护栏、插件……）……
# │  参数：agent, function_name: str, function_args: dict, flatten_probe: bool = ...
# │  返回：tuple[str, dict, Optional[str]]
# └──────────────────────────────────────────────────────────────
def _unwrap_tool_search_call(
    agent, function_name: str, function_args: dict, *, flatten_probe: bool = False
) -> tuple[str, dict, Optional[str]]:
    """Peel the ``tool_call`` bridge so downstream hooks (checkpointing, guardrails, plugin
    hooks, activity feed) see the underlying tool; ``tool_call.function`` stays untouched for
    the transcript and tool_call_id pairing.

    The unwrap bypasses handle_function_call's scope check, so session toolset scope is
    enforced HERE. Returns ``(name, args, scope_block)``; ``scope_block`` is the block
    message when the underlying tool is out of scope or its args fail the deferred-schema
    probe (``flatten_probe`` collapses the probe's JSON payload to one plain string for
    callers that wrap the message in ``{"error": ...}``).
    """
    scope_block: Optional[str] = None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools import tool_search as _ts
        if function_name != _ts.TOOL_CALL_NAME:
            # ⤷ 出口 return：返回 (function_name, function_args, None)
            return function_name, function_args, None
        underlying, underlying_args, err = _ts.resolve_underlying_call(function_args)
        if err or not underlying:
            # ⤷ 出口 return：返回 (function_name, function_args, None)
            return function_name, function_args, None
        if underlying == _ts.CONNECTOR_BATCH_SENTINEL:
            # Both executors retain the wrapper: scope/probe/hooks run per entry
            # in the batch dispatcher, not against a synthetic registry name.
            # ⤷ 出口 return：返回 (function_name, function_args, None)
            return function_name, function_args, None
        if underlying not in _tool_search_scoped_names(agent):
            # ⤷ 出口 return：返回 (function_name, function_args, f"'{underlying}' is not avail…
            return function_name, function_args, (
                f"'{underlying}' is not available in this session. Use tool_search to find tools you can call."
            )
        # Validate before unwrapping: the generic bridge hides the concrete
        # parameter schema from provider-native tool-call validation.
        scope_block = _ts.validate_deferred_call_args(underlying, underlying_args)
        if scope_block is None:
            return underlying, underlying_args, None
        if flatten_probe:
            probe = json.loads(scope_block)
            scope_block = (
                f"{probe.get('error', '')} Parameters schema: "
                f"{json.dumps(probe.get('parameters', {}), ensure_ascii=False)}. "
                f"{probe.get('hint', '')}"
            ).strip()
    except Exception:
        pass
    # ⤷ 出口 return：返回 (function_name, function_args, scope_block)
    return function_name, function_args, scope_block


# ┌── 类 class _ParsedCall
# │  说明：经过别名归一化、参数解析、桥接 unwrap 之后的一条模型工具调用。
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class _ParsedCall:
    """One model tool call after alias canonicalization, arg parsing and bridge unwrap."""

    tool_call: Any
    name: str
    args: dict
    middleware_trace: list
    parse_error: Optional[str]
    scope_block: Optional[str]

    # ┌── 函数 def ref()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, task_id: str
    # │  返回：_ToolCallRef
    # └──────────────────────────────────────────────────────────────
    def ref(self, task_id: str) -> _ToolCallRef:
        # ⤷ 出口 return：返回 _ToolCallRef(self.name, self.args, task_id, _pairing_tool_ca…
        return _ToolCallRef(self.name, self.args, task_id, _pairing_tool_call_id(self.tool_call), self.middleware_trace)


# ┌── 函数 def _parse_tool_call()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：agent, tool_call, flatten_probe: bool = ...
# │  返回：_ParsedCall
# └──────────────────────────────────────────────────────────────
def _parse_tool_call(agent, tool_call, *, flatten_probe: bool = False) -> _ParsedCall:
    name = _canonical_tool_name(tool_call.function.name)
    args, parse_error = _parse_tool_arguments(tool_call.function.arguments)
    scope_block = None
    if parse_error is None:
        name, args, scope_block = _unwrap_tool_search_call(agent, name, args, flatten_probe=flatten_probe)
    # ⤷ 出口 return：返回 _ParsedCall(tool_call, name, args, [], parse_error, scope_bl…
    return _ParsedCall(tool_call, name, args, [], parse_error, scope_block)


# ┌── 类 class _ManagedToolResult
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class _ManagedToolResult:
    result: Any
    args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    blocked: bool
    dispatched: bool


# ┌── 类 class _ToolTimeoutResult（继承自：str）
# │  说明：为合成的「顺序工具超时」结果所做的标记。
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _ToolTimeoutResult(str):
    """Marker for a synthesized sequential-tool timeout result."""


# ┌── 类 class _ToolCancelledResult（继承自：str）
# │  说明：为合成的「顺序工具用户中断」结果所做的标记；其终局……
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _ToolCancelledResult(str):
    """Marker for a synthesized sequential-tool user-interrupt result; its terminal
    post_tool_call was already emitted, so a late-finishing abandoned worker must not report."""


# ┌── 类 class _ConcurrentToolAuthorizationGate
# │  说明：序列化策略提示词，并把「人工审批等待」排除在批次截止时间之外。
# │  成员：共 4 个方法；对外可用 2 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _ConcurrentToolAuthorizationGate:
    """Serialize policy prompts and exclude human approval waits from batch deadlines.

    The acquire is BOUNDED: on expiry the worker prompts unserialized rather than starving
    the batch behind a wedged plugin/approval client. Exclusion is measured at the SOURCE
    of the human wait (``tools.approval.human_wait_seconds``), NOT as gate residency —
    residency-based exclusion let a wedged plugin keep the deadline from ever firing.

    Serialization keeps concurrent approval prompts from interleaving on the user's screen. The acquire is
    BOUNDED: a worker wedged inside the gate (a hanging ``pre_tool_call`` plugin, or an approval round-trip
    to a client that went away) must not park every other worker forever. On expiry the worker runs its
    prompt unserialized — worst case is interleaved prompts, strictly better than permanent starvation (same
    tradeoff as the start-order gate, #79705).
    Gate residency is arbitrary code — using it as the exclusion signal let a wedged plugin grow the
    exclusion 1:1 with wall clock, keeping the batch deadline's ``remaining`` constant so it never fired and
    the turn hung forever (#79719). A wedged plugin now contributes nothing to the exclusion and the batch
    times out normally, while a genuine approval wait (which can legitimately exceed any fixed bound) is
    still excluded in full.
    """

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, lock_timeout: float | None = ..., session_key: str | None = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, *, lock_timeout: float | None = None, session_key: str | None = None) -> None:
        self._serialization_lock = threading.Lock()
        self._lock_timeout = _authorization_gate_lock_timeout() if lock_timeout is None else lock_timeout
        self._session_key = session_key
        if self._session_key is None:
            # Snapshot on the SUBMITTING thread: excluded_seconds() is polled from the
            # batch wait loop, whose context may differ from the workers'.
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                from tools.approval_context import get_current_session_key

                self._session_key = get_current_session_key()
            except Exception:
                logger.debug(
                    "authorization gate could not snapshot the session key; "
                    "human-wait exclusion will re-resolve it at poll time",
                    exc_info=True,
                )
        self._baseline_wait_seconds = self._human_wait_seconds()

    # ┌── 函数 def _human_wait_seconds()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：float
    # └──────────────────────────────────────────────────────────────
    def _human_wait_seconds(self) -> float:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.approval_human_wait import human_wait_seconds

            # ⤷ 出口 return：返回 human_wait_seconds(self._session_key)
            return human_wait_seconds(self._session_key)
        except Exception:
            # ⤷ 出口 return：返回 0.0
            return 0.0

    # ┌── 函数 def run()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, callback
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def run(self, callback):
        if not self._serialization_lock.acquire(timeout=self._lock_timeout):
            # Deterministic failure (bad command, non-MCP URL, 401/403): every retry hits the same wall.
            # Park immediately instead of burning the retry ladder and spamming N identical warnings
            # (#65673). Auth failures park here too rather than returning. Returning ends the run task, and
            # with it the only listener on ``_reconnect_event`` — so a 401 on the very first connect left
            # the server unrevivable for the life of the process, even after the user re-authenticated with
            # ``hermes mcp login``. Parking keeps the task alive so the 300s self-probe (and an explicit
            # /mcp refresh) can pick up fresh tokens.
            logger.warning(
                "authorization gate lock not acquired after %.1fs "
                "(holder wedged in a pre_tool_call plugin or approval "
                "round-trip?); running prompt unserialized",
                self._lock_timeout,
            )
            # ⤷ 出口 return：返回 callback()
            return callback()
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 callback()
            return callback()
        finally:
            self._serialization_lock.release()

    # ┌── 函数 def excluded_seconds()
    # │  作用：返回自批次开始以来累计的「人工审批等待」秒数。
    # │  参数：self
    # │  返回：float
    # └──────────────────────────────────────────────────────────────
    def excluded_seconds(self) -> float:
        """Return human-approval wait seconds accrued since the batch started."""
        # ⤷ 出口 return：返回 max(0.0, self._human_wait_seconds() - self._baseline_wait_se…
        return max(0.0, self._human_wait_seconds() - self._baseline_wait_seconds)


# ┌── 函数 def _registered_tool_worker()
# │  作用：记录这个 worker 的线程 id，用于中断的扇出（``AIAgent.interrupt()``）；在任何退出……
# │  参数：agent
# │  返回：（未标注类型）
# │  【语法】@contextmanager 把「有 yield 的函数」变成上下文管理器，调用方可以用 `with xxx() as v:` 自动做进入/清理。yield 之前是准备，yield 之后是收尾。
# └──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _registered_tool_worker(agent):
    """Track this worker tid for interrupt fan-out (``AIAgent.interrupt()``); on ANY exit
    (incl. BaseException) discard it and clear its interrupt bit so a recycled tid starts clean."""
    tid = threading.current_thread().ident
    with agent._tool_worker_threads_lock:
        agent._tool_worker_threads.add(tid)
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
        yield tid
    finally:
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.discard(tid)
        with contextlib.suppress(Exception):
            _ra()._set_interrupt(False, tid)


_NO_REASON = object()


# ┌── 函数 def _interrupt_worker_tids()
# │  作用：在每个 worker 线程 id 上置位中断标志（尽最大努力，经由 ``run_agent``）。
# │  参数：agent, tids, reason = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _interrupt_worker_tids(agent, tids, *, reason=_NO_REASON) -> None:
    """Raise the interrupt bit on each worker tid (best-effort, via ``run_agent``)."""
    kwargs = {} if reason is _NO_REASON else {"reason": reason}
    for tid in tids:
        with contextlib.suppress(Exception):
            _ra()._set_interrupt(True, tid, **kwargs)


# ┌── 函数 def _set_worker_activity_callback()
# │  作用：活跃度回调是线程局部的：必须在「当前线程」上绑定，工具层的活跃心跳才会触发。
# │  参数：agent
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _set_worker_activity_callback(agent) -> None:
    """The activity callback is thread-local: bind it on THIS thread so tool-layer heartbeats fire."""
    with contextlib.suppress(Exception):
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)


# Must stay far below the gateway turn-inactivity timeout (default 1800s) so a silent tool never looks idle.
_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S = 30.0


# ┌── 函数 def _run_tool_activity_heartbeat()
# │  作用：守护线程，每隔 ``interval`` 秒打一次 ``agent._touch_activity`` 时间戳，直到……
# │  参数：agent, stop_event: threading.Event, label: str, interval: float = _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _run_tool_activity_heartbeat(
    agent,
    stop_event: threading.Event,
    label: str,
    interval: float = _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S,
) -> None:
    """Daemon thread stamping ``agent._touch_activity`` every ``interval`` seconds until
    ``stop_event`` is set, so the gateway inactivity watchdog never abandons a turn whose
    tool runs silently. Wedged tools stay bounded by the tool layer's own timeouts."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        while not stop_event.wait(interval):
            agent._touch_activity(label)
    except Exception:
        pass  # a heartbeat must never break the agent loop


# ┌── 函数 def _run_with_activity_heartbeat()
# │  作用：在活跃心跳的保护下运行 ``fn()``；覆盖两条执行器路径。
# │  参数：agent, function_name: str, fn
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _run_with_activity_heartbeat(agent, function_name: str, fn):
    """Run ``fn()`` under the activity heartbeat; covers both executor paths."""
    stop = threading.Event()
    thread = threading.Thread(
        # Keep the gateway turn-inactivity watchdog from abandoning a turn whose tool call runs silently for
        # longer than the inactivity timeout (#84491): stamp activity periodically while the tool is in
        # flight, not just at start/completion. Both the sequential and the concurrent paths funnel through
        # here, so a single heartbeat covers every tool.
        target=_run_tool_activity_heartbeat,
        args=(agent, stop, f"tool running: {function_name}"),
        kwargs={"interval": _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S},
        daemon=True,
        name=f"tool-activity-hb-{function_name[:24]}",
    )
    thread.start()
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 fn()
        return fn()
    finally:
        stop.set()
        thread.join(timeout=2.0)


# ┌── 函数 def _blocked_tool_result()
# │  作用：为被 scope/插件（``block_message``）或……挡住的一次调用合成结果。
# │  参数：agent, ref: _ToolCallRef, block_message: Optional[str], block_error_type: str, guardrail_decision
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _blocked_tool_result(agent, ref: _ToolCallRef, *, block_message: Optional[str], block_error_type: str, guardrail_decision) -> str:
    """Synthesize the result for a call blocked by scope/plugin (``block_message``) or by
    guardrail policy (``guardrail_decision``) and emit its terminal post_tool_call."""
    if block_message is not None:
        result, error_type, error_message = json.dumps({"error": block_message}, ensure_ascii=False), block_error_type, block_message
    else:
        result = agent._guardrail_block_result(guardrail_decision)
        error_type = "guardrail_block"
        error_message = getattr(guardrail_decision, "message", None) or "Tool blocked by guardrail policy"
    ref.emit_post(agent, result, status="blocked", error_type=error_type, error_message=error_message)
    # ⤷ 出口 return：返回 result
    return result


# ┌── 函数 def _pre_tool_block()
# │  作用：运行 ``pre_tool_call`` 插件钩子；返回 ``(block_message, final_args)``，附带任何……
# │  参数：agent, ref: _ToolCallRef
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _pre_tool_block(agent, ref: _ToolCallRef):
    """Run ``pre_tool_call`` plugin hooks; returns ``(block_message, final_args)`` with any
    hook-modified args applied. Hook failures never block."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

        block_msg, modified_args = _dispatch_pre_tool_call_hooks(
            ref.name,
            ref.args,
            **tool_hook_ids(agent, ref.task_id, ref.call_id),
            middleware_trace=list(ref.trace),
        )
        # ⤷ 出口 return：返回 (block_msg, ref.args if modified_args is None else modified_…
        return block_msg, (ref.args if modified_args is None else modified_args)
    except Exception:
        # ⤷ 出口 return：返回 (None, ref.args)
        return None, ref.args


# ┌── 函数 def _dispatch_authorized_once()
# │  作用：Hermes 的策略链（scope → 插件 pre-hooks → 护栏），其后才是真正的一次派发。
# │  参数：agent, state: _ManagedToolResult, ref: _ToolCallRef, execute, scope_block: str | None, display_index: int | None, begin_execution, authorization_gate: _ConcurrentToolAuthorizationGate | None
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _dispatch_authorized_once(
    agent,
    state: _ManagedToolResult,
    ref: _ToolCallRef,
    *,
    execute,
    scope_block: str | None,
    display_index: int | None,
    begin_execution,
    authorization_gate: _ConcurrentToolAuthorizationGate | None,
) -> Any:
    """Hermes policy (scope → plugin pre-hooks → guardrails) then the one real dispatch.

    Plugin ``modify`` hooks may rewrite ``ref.args`` (mirrored into ``state.args``).
    ``begin_execution`` (concurrent start-order gate) is advanced exactly once on every
    path so later-ordered workers keep moving; blocked calls advance it without a callback.
    """
    def _advance_start_order(callback=None) -> None:
        if begin_execution is not None:
            begin_execution(callback)
        elif callback is not None:
            callback()

    block_message, block_error_type = scope_block, "tool_scope_block"
    if block_message is None:
        block_error_type = "plugin_block"
        resolve = lambda: _pre_tool_block(agent, ref)  # noqa: E731
        block_message, ref.args = resolve() if authorization_gate is None else authorization_gate.run(resolve)
        state.args = ref.args

    guardrail_decision = None
    if block_message is None:
        guardrail_decision = agent._tool_guardrails.before_call(ref.name, ref.args)
        if guardrail_decision.allows_execution:
            guardrail_decision = None

    if block_message is not None or guardrail_decision is not None:
        _advance_start_order()
        state.blocked = True
        # ⤷ 出口 return：返回 _blocked_tool_result(agent, ref, block_message=block_message…
        return _blocked_tool_result(
            agent, ref,
            block_message=block_message, block_error_type=block_error_type, guardrail_decision=guardrail_decision,
        )

    if ref.name == "memory":
        agent._turns_since_memory = 0
    elif ref.name == "skill_manage":
        agent._iters_since_skill = 0

    _advance_start_order(lambda: _begin_tool_execution(agent, ref, display_index))
    # ⤷ 出口 return：返回 _run_with_activity_heartbeat(agent, ref.name, lambda: execut…
    return _run_with_activity_heartbeat(agent, ref.name, lambda: execute(ref.args))


# ┌── 函数 def _run_agent_tool_execution_middleware()
# │  作用：在 Hermes 策略和派发之前，恰好运行一次 Relay 重写。
# │  参数：agent, function_name: str, function_args: dict, effective_task_id: str, tool_call_id: str, execute, scope_block: str | None = ..., display_index: int | None = ...  …共 11 个
# │  返回：_ManagedToolResult
# └──────────────────────────────────────────────────────────────
def _run_agent_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
    begin_execution=None,
    authorization_gate: _ConcurrentToolAuthorizationGate | None = None,
) -> _ManagedToolResult:
    """Run Relay rewrites before Hermes policy and dispatch exactly once."""
    from agent import relay_tools
    from hermes_cli.middleware import (
        apply_tool_request_middleware,
        run_tool_execution_middleware,
    )

    trace = middleware_trace if middleware_trace is not None else []
    state = _ManagedToolResult(result=None, args=function_args, middleware_trace=trace, blocked=False, dispatched=False)
    dispatch_lock = threading.Lock()

    def _authorized_dispatch(final_args: dict[str, Any]) -> Any:
        with dispatch_lock:
            if state.dispatched:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes tool execution callback invoked more than once")
            state.dispatched = True
            state.blocked = False
            state.args = final_args
        # ⤷ 出口 return：返回 _dispatch_authorized_once(agent, state, _ToolCallRef(functio…
        return _dispatch_authorized_once(
            agent,
            state,
            _ToolCallRef(function_name, final_args, effective_task_id, tool_call_id, trace),
            execute=execute,
            scope_block=scope_block,
            display_index=display_index,
            begin_execution=begin_execution,
            authorization_gate=authorization_gate,
        )

    def _hermes_pipeline(relay_args: dict[str, Any]) -> Any:
        request_result = apply_tool_request_middleware(
            function_name,
            relay_args,
            skip_relay=True,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )
        request_args = request_result.payload if isinstance(request_result.payload, dict) else relay_args
        trace.clear()
        trace.extend(request_result.trace)
        # ⤷ 出口 return：返回 run_tool_execution_middleware(function_name, request_args, l…
        return run_tool_execution_middleware(
            function_name,
            request_args,
            lambda next_args: _authorized_dispatch(next_args if isinstance(next_args, dict) else request_args),
            original_args=function_args,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )

    state.result, _relay_args = relay_tools.execute(
        function_name,
        function_args,
        _hermes_pipeline,
        session_id=str(getattr(agent, "session_id", "") or ""),
        tool_call_id=tool_call_id or None,
        metadata={
            "task_id": effective_task_id or "",
            "turn_id": getattr(agent, "_current_turn_id", "") or "",
            "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
            "tool_call_id": tool_call_id or "",
        },
    )
    # ⤷ 出口 return：返回 state
    return state


# Sequential wait-loop poll cadence: /stop lands within ~1s even if the tool never polls is_interrupted().
_SEQUENTIAL_INTERRUPT_POLL_SECONDS = 1.0


# ┌── 函数 def _resolve_sequential_tool_timeout()
# │  作用：单次顺序调用的截止时间：以 ``timeouts.tools.sequential_call`` 为准，否则取……
# │  参数：无
# │  返回：float | None
# └──────────────────────────────────────────────────────────────
def _resolve_sequential_tool_timeout() -> float | None:
    """Deadline for one sequential call: ``timeouts.tools.sequential_call``, else the
    concurrent batch deadline so the two paths can't drift; ``0``/negative disables.
    Deliberately NOT ``agent.deadline.run_bounded_sync``: both executors extend the
    deadline while an approval prompt is open, which a fixed deadline can't express."""
    from agent.deadline import resolve_timeout

    # ⤷ 出口 return：返回 resolve_timeout('tools.sequential_call', default=_resolve_co…
    return resolve_timeout("tools.sequential_call", default=_resolve_concurrent_tool_timeout())


# Tools whose call blocks on a long-running operation that supervises its own liveness: no generic
# sequential deadline. ``delegate_task`` in a nested orchestrator blocks for the whole batch by design
# (children carry heartbeats, the stale monitor, and ``delegation.child_timeout_seconds``); under the
# 420 s deadline every real batch "timed out" while its children ran on as orphans, and the orchestrator
# spent the following hours polling transcripts (measured: 332 timeouts, ~$4k of orchestrator turns in
# one run).
# ``manage_connections`` waits on the connection operation's own deadline; the generic deadline
# would return tool_timeout while its approval card is still open.
_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset({"delegate_task", "manage_connections"})


# ┌── 函数 def _abandoned_sequential_result()
# │  作用：为顺序执行器已经放弃的那个 worker 发出终局 post_tool_call……
# │  参数：agent, ref: _ToolCallRef, message: str, result_cls, **outcome
# │  返回：_ManagedToolResult
# └──────────────────────────────────────────────────────────────
def _abandoned_sequential_result(agent, ref: _ToolCallRef, message: str, result_cls, **outcome) -> _ManagedToolResult:
    """Emit the terminal post_tool_call for a worker the sequential runner gave up on
    (timeout / interrupt) and wrap ``message`` in its marker ``result_cls``."""
    ref.emit_post(agent, message, **outcome)
    # ⤷ 出口 return：返回 _ManagedToolResult(result=result_cls(message), args=ref.args…
    return _ManagedToolResult(result=result_cls(message), args=ref.args, middleware_trace=ref.trace, blocked=False, dispatched=True)


# ┌── 函数 def _poll_sequential_future()
# │  作用：以「中断轮询」的切片方式等待 worker，并按人工审批……来延长截止时间。
# │  参数：agent, future, function_name: str, deadline: float | None, started: float, authorization_gate
# │  返回：tuple[str, Any]
# └──────────────────────────────────────────────────────────────
def _poll_sequential_future(agent, future, function_name: str, deadline: float | None, started: float, authorization_gate) -> tuple[str, Any]:
    """Wait for the worker in interrupt-poll slices, extending the deadline by human approval
    wait; returns ``("done", result)``, ``("timeout", None)`` or ``("interrupted", None)``.
    A disabled deadline still polls: this loop is what makes a non-cooperative tool
    interruptible, so no deadline must not mean no interrupt checks."""
    _last_heartbeat = 0
    while True:
        wait_slice = _SEQUENTIAL_INTERRUPT_POLL_SECONDS
        if deadline is not None:
            remaining = deadline + authorization_gate.excluded_seconds() - time.monotonic()
            if remaining <= 0:
                # ⤷ 出口 return：返回 ('timeout', None)
                return "timeout", None
            wait_slice = min(wait_slice, remaining)
        # ⤷ 关键分支 try/except：捕获 concurrent.futures.TimeoutError，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 ('done', future.result(timeout=wait_slice))
            return "done", future.result(timeout=wait_slice)
        except concurrent.futures.TimeoutError:
            if agent._interrupt_requested:
                # ⤷ 出口 return：返回 ('interrupted', None)
                return "interrupted", None
            elapsed = int(time.monotonic() - started)
            if elapsed - _last_heartbeat >= 30:
                _last_heartbeat = elapsed
                agent._touch_activity(f"sequential tool running ({elapsed}s): {function_name}")


# ┌── 函数 def _run_sequential_tool_execution_middleware()
# │  作用：在并发执行器的截止时间约束下，于一个 worker 线程上运行一次顺序调用。
# │  参数：agent, function_name: str, function_args: dict, effective_task_id: str, tool_call_id: str, execute, scope_block: str | None = ..., display_index: int | None = ...  …共 9 个
# │  返回：_ManagedToolResult
# └──────────────────────────────────────────────────────────────
def _run_sequential_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
) -> _ManagedToolResult:
    """Run one sequential call on a worker thread under the concurrent executor's deadline.
    Interactive tools (``clarify``) own their wait via ``agent.clarify_timeout``; the
    generic deadline would report ``tool_timeout`` while the prompt is still live."""
    timeout_s = None if function_name in _SEQUENTIAL_DEADLINE_EXEMPT_TOOLS else _resolve_sequential_tool_timeout()
    ref = _ToolCallRef(function_name, function_args, effective_task_id, tool_call_id, middleware_trace)
    kwargs = dict(ref.middleware_kwargs(), execute=execute, scope_block=scope_block, display_index=display_index)
    if function_name in _NEVER_PARALLEL_TOOLS:
        # ⤷ 出口 return：返回 _run_agent_tool_execution_middleware(agent, **kwargs)
        return _run_agent_tool_execution_middleware(agent, **kwargs)

    from tools.daemon_pool import DaemonThreadPoolExecutor

    authorization_gate = _ConcurrentToolAuthorizationGate()
    worker_tid: list[int] = []

    def _run() -> _ManagedToolResult:
        with _registered_tool_worker(agent) as tid:
            worker_tid.append(tid)
            # ⤷ 出口 return：返回 _run_agent_tool_execution_middleware(agent, authorization_ga…
            return _run_agent_tool_execution_middleware(agent, authorization_gate=authorization_gate, **kwargs)

    if ref.trace is None:
        ref.trace = []
    executor = DaemonThreadPoolExecutor(max_workers=1)
    future = executor.submit(propagate_context_to_thread(_run))
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    started = time.monotonic()
    abandoned = False
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        state, result = _poll_sequential_future(agent, future, function_name, deadline, started, authorization_gate)
        if state == "done":
            # ⤷ 出口 return：返回 result
            return result
        if state == "interrupted":
            # interrupt() already fanned out to tracked tids, but this worker may have
            # registered after that ran; then 3s grace (mirrors the concurrent path).
            _interrupt_worker_tids(agent, worker_tid, reason=getattr(agent, "_tool_interrupt_reason", None))
            concurrent.futures.wait([future], timeout=3.0)
            if future.done() and not future.cancelled():
                # ⤷ 出口 return：返回 future.result()
                return future.result()
            interrupt_reason = getattr(agent, "_tool_interrupt_reason", None) or "interrupt requested"
            message = f"[Tool execution cancelled — {function_name} was abandoned: {interrupt_reason}]"
            logger.info(
                "sequential tool %s abandoned due to %s (%.1fs elapsed)",
                function_name, interrupt_reason, time.monotonic() - started,
            )
            result_cls, outcome = _ToolCancelledResult, dict(
                duration_ms=int((time.monotonic() - started) * 1000), status="cancelled",
                error_type="tool_interrupted", error_message=f"Tool execution cancelled: {interrupt_reason}",
            )
        else:
            assert timeout_s is not None  # only reachable when a deadline exists
            message = f"Error executing tool '{function_name}': timed out after {timeout_s:.1f}s"
            logger.warning("sequential tool %s timed out after %.1fs", function_name, timeout_s)
            result_cls, outcome = _ToolTimeoutResult, dict(
                duration_ms=int(timeout_s * 1000), status="timeout", error_type="tool_timeout", error_message=message,
            )
        abandoned = True
        future.cancel()
        if state == "timeout":
            _interrupt_worker_tids(agent, worker_tid)
        # ⤷ 出口 return：返回 _abandoned_sequential_result(agent, ref, message, result_cls…
        return _abandoned_sequential_result(agent, ref, message, result_cls, **outcome)
    finally:
        # Never join a wedged worker (daemon pool also keeps it out of the atexit join).
        executor.shutdown(wait=not abandoned, cancel_futures=abandoned)


# ┌── 函数 def _safe_callback()
# │  作用：若设置了 UI/桥接回调则调用它；回调失败只记日志，绝不会致命。
# │  参数：callback, label: str, *args, **kwargs
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _safe_callback(callback, label: str, *args, **kwargs) -> None:
    """Invoke a UI/bridge callback if set; a failing callback is logged, never fatal."""
    if not callback:
        return
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        callback(*args, **kwargs)
    except Exception as callback_error:
        logging.debug("%s callback error: %s", label, callback_error)


# ┌── 函数 def _begin_tool_execution()
# │  作用：对最终的工具参数，运行「面向用户」与检查点相关的预检。
# │  参数：agent, ref: _ToolCallRef, display_index: int | None
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _begin_tool_execution(agent, ref: _ToolCallRef, display_index: int | None) -> None:
    """Run user-visible and checkpoint preflight on final tool arguments."""
    function_name, function_args, effective_task_id, tool_call_id = ref.name, ref.args, ref.task_id, ref.call_id
    display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
    if _tool_progress_enabled(agent):
        prefix = f"Tool {display_index}" if display_index is not None else "Tool"
        if agent.verbose_logging:
            print(f"  📞 {prefix}: {function_name}({list(display_args.keys())})")
            print(agent._wrap_verbose("Args: ", json.dumps(display_args, indent=2, ensure_ascii=False)))
        else:
            print(f"  📞 {prefix}: {function_name}({list(function_args.keys())}) - {_preview(json.dumps(display_args, ensure_ascii=False), agent.log_prefix_chars)}")

    agent._current_tool = function_name
    agent._touch_activity(f"executing tool: {function_name}")
    _set_worker_activity_callback(agent)

    if agent.tool_progress_callback:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            preview = _build_tool_preview(function_name, display_args)
        except Exception as callback_error:
            logging.debug("Tool progress callback error: %s", callback_error)
        else:
            _safe_callback(agent.tool_progress_callback, "Tool progress", "tool.started", function_name, preview, display_args)
    _safe_callback(agent.tool_start_callback, "Tool start", tool_call_id, function_name, display_args)

    if not agent._checkpoint_mgr.enabled:
        return
    with contextlib.suppress(Exception):
        if function_name in {"write_file", "patch"}:
            _ensure_file_checkpoint(agent, function_name, function_args, effective_task_id)
        elif function_name == "terminal":
            command = function_args.get("command", "")
            if _is_destructive_command(command):
                from agent.runtime_cwd import scope_terminal_cwd
                cwd = function_args.get("workdir") or scope_terminal_cwd() or os.getcwd()
                agent._checkpoint_mgr.ensure_checkpoint(cwd, f"before terminal: {command[:60]}")


# ┌── 函数 def _emit_tool_complete_and_risk()
# │  作用：触发 ``tool_complete_callback``（除非被挡住），随后做 ``tool.output_risk`` 的风险投射。
# │  参数：agent, ref: _ToolCallRef, result, risk_metadata, blocked: bool
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _emit_tool_complete_and_risk(agent, ref: _ToolCallRef, result, risk_metadata, blocked: bool) -> None:
    """Fire ``tool_complete_callback`` (unless blocked) then the ``tool.output_risk`` projection."""
    if not blocked and agent.tool_complete_callback:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            display_args = _redact_tool_args_for_display(ref.name, ref.args) or ref.args
        except Exception as cb_err:
            logging.debug("Tool complete callback error: %s", cb_err)
        else:
            _safe_callback(agent.tool_complete_callback, "Tool complete", ref.call_id, ref.name, display_args, result)
    if risk_metadata is not None and risk_metadata.get("risk") != "low":
        _safe_callback(
            agent.tool_progress_callback, "Tool output risk",
            "tool.output_risk", ref.name, None, None, tool_call_id=ref.call_id, risk_metadata=risk_metadata,
        )


# ┌── 函数 def _commit_tool_result()
# │  作用：观察（仅 ``observed`` 结果）并记录结果；标记工具为已完成；持久化/……
# │  参数：agent, messages: list, ref: _ToolCallRef, function_result, budget: BudgetConfig, tool_duration: float, is_error: bool, blocked: bool  …共 13 个
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _commit_tool_result(
    agent,
    messages: list,
    ref: _ToolCallRef,
    function_result,
    *,
    budget: BudgetConfig,
    tool_duration: float,
    is_error: bool,
    blocked: bool,
    effect_disposition,
    observed: bool = False,
    error_preview: Callable[[Any], Any] = lambda result: result,
    success_log_chars: Optional[int] = None,
    verbose_text: Callable[[Any], Any] = lambda result: result,
):
    """Observe (``observed`` results only) and log the outcome; mark the tool done; persist/
    spill, hint, wrap and append the result; flush the session DB; project ``tool.completed``.

    Blocked calls never ran, so they are neither guardrail-observed nor fed to the file-
    mutation verifier; ``success_log_chars`` (sequential path) also logs the completion line.
    Returns ``(persisted_result, display_result, risk_metadata)`` (``display_result`` =
    pre-persist content for UI previews) or ``None`` when the flush failed (stop the batch).
    """
    function_name, function_args, tool_call_id, effective_task_id = ref.name, ref.args, ref.call_id, ref.task_id
    if observed:
        if not blocked:
            function_result = agent._append_guardrail_observation(
                function_name, function_args, function_result, failed=is_error, tool_call_id=tool_call_id,
            )
        if is_error:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, error_preview(function_result))
        elif success_log_chars is not None:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, success_log_chars)
        if not blocked:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                agent._record_file_mutation_result(function_name, function_args, function_result, is_error)
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)
        if agent.verbose_logging:
            logging.debug("Tool %s completed in %.2fs", function_name, tool_duration)
            _log_result = verbose_text(function_result)
            logging.debug("Tool result (%d chars): %s", len(_log_result), _log_result)

    agent._current_tool = None
    _status_suffix = " (error)" if is_error else ""
    agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s){_status_suffix}")

    persisted_result = function_result
    if not _is_multimodal_tool_result(persisted_result):
        persisted_result = maybe_persist_tool_result(
            content=persisted_result,
            tool_name=function_name,
            tool_use_id=tool_call_id,
            env=get_active_env(effective_task_id),
            config=budget,
        )
    _record_persisted_path_for_stub(agent, tool_call_id, persisted_result)

    subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
    if subdir_hints:
        if _is_multimodal_tool_result(persisted_result):
            # Hint goes on the text summary part so the model still sees it; image blocks untouched.
            _append_subdir_hint_to_multimodal(persisted_result, subdir_hints)
        else:
            persisted_result += subdir_hints

    # Multimodal dicts become an OpenAI-style content list; text-only servers get a
    # string-safe fallback so a rejected image result never poisons history.
    _tool_content = agent._tool_result_content_for_active_model(function_name, persisted_result)
    tool_message = make_tool_result_message(function_name, _tool_content, tool_call_id, effect_disposition=effect_disposition)
    messages.append(tool_message)
    if not _flush_session_db_after_tool_progress(agent, messages, stage=f"tool result {function_name}"):
        # ⤷ 出口 return：返回 None
        return None

    if not blocked:
        # ``tool.completed`` projects AFTER the canonical append + flush so resume can
        # reconstruct the result even if the UI bridge dies mid-projection.
        _safe_callback(
            agent.tool_progress_callback, "Tool progress",
            "tool.completed", function_name, None, None, duration=tool_duration, is_error=is_error, result=function_result,
        )
    # ⤷ 出口 return：返回 (persisted_result, function_result, tool_message.get('_tool_…
    return persisted_result, function_result, tool_message.get("_tool_output_risk")


# ┌── 函数 def _finalize_tool_batch()
# │  作用：先执行「每轮聚合预算」的强制，再做 /steer 注入——顺序固定，以便……
# │  参数：agent, messages: list, effective_task_id: str, num_tools: int, budget: BudgetConfig
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _finalize_tool_batch(agent, messages: list, effective_task_id: str, num_tools: int, budget: BudgetConfig) -> None:
    """Per-turn aggregate budget enforcement, then /steer injection — in that order, so the
    steer marker is never truncated/discarded when enforcement replaces a result."""
    if num_tools <= 0:
        return
    enforce_turn_budget(messages[-num_tools:], env=get_active_env(effective_task_id), config=budget)
    agent._apply_pending_steer_to_tool_results(messages, num_tools)


# ┌── 函数 def _tool_progress_enabled()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：agent
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _tool_progress_enabled(agent) -> bool:
    # ⤷ 出口 return：返回 not agent.quiet_mode and getattr(agent, 'tool_progress_mode'…
    return not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off"


# ┌── 函数 def _preview()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：text: str, limit: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _preview(text: str, limit: int) -> str:
    # ⤷ 出口 return：返回 text[:limit] + '...' if len(text) > limit else text
    return text[:limit] + "..." if len(text) > limit else text


# ┌── 函数 def _print_tool_completed()
# │  作用：非安静模式下打印的 ``✅ Tool N completed`` 行（详细日志下显示完整结果）。
# │  参数：agent, index: int, tool_duration: float, result
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _print_tool_completed(agent, index: int, tool_duration: float, result) -> None:
    """Non-quiet ``✅ Tool N completed`` line (full result under verbose logging)."""
    if agent.verbose_logging:
        print(f"  ✅ Tool {index} completed in {tool_duration:.2f}s")
        print(agent._wrap_verbose("Result: ", result))
    else:
        print(f"  ✅ Tool {index} completed in {tool_duration:.2f}s - {_preview(result if isinstance(result, str) else str(result), agent.log_prefix_chars)}")


# ── Concurrent batch machinery ──────────────────────────────────────────────


# ┌── 类 class _ToolOutcome
# │  说明：一个并发批次里已完成的 worker 槽位（``ref`` 持有最终的名字/参数/调用链）。
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class _ToolOutcome:
    """One finished worker slot of a concurrent batch (``ref`` holds the final name/args/trace)."""

    ref: _ToolCallRef
    result: Any
    duration: float
    is_error: bool
    blocked: bool


# ┌── 函数 def _start_order_gate_timeout()
# │  作用：门限值必须低于批次截止时间，否则 parked 的 worker 会被错误地……
# │  参数：batch_timeout: float | None
# │  返回：float
# └──────────────────────────────────────────────────────────────
def _start_order_gate_timeout(batch_timeout: float | None) -> float:
    """The gate bound must sit UNDER the batch deadline, else parked workers are falsely
    reported timed out without starting. A disabled deadline keeps the stock bound."""
    if batch_timeout is None:
        # ⤷ 出口 return：返回 _START_ORDER_GATE_TIMEOUT_S
        return _START_ORDER_GATE_TIMEOUT_S
    # ⤷ 出口 return：返回 min(_START_ORDER_GATE_TIMEOUT_S, batch_timeout / 2)
    return min(_START_ORDER_GATE_TIMEOUT_S, batch_timeout / 2)


# ┌── 类 class _StartOrderGate
# │  说明：按提交顺序串行化 worker 派发（提示按调用顺序出现）；``abandon()``……
# │  成员：共 3 个方法；对外可用 2 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _StartOrderGate:
    """Serialize worker dispatch by submit order (prompts appear in call order); ``abandon()``
    releases every parked worker so none dispatches a tool the turn already gave up on."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, timeout: float
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, timeout: float) -> None:
        self._condition = threading.Condition()
        self._next_order = 0
        self._timeout = timeout
        self.abandoned = threading.Event()

    # ┌── 函数 def abandon()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def abandon(self) -> None:
        self.abandoned.set()
        with self._condition:
            self._condition.notify_all()

    # ┌── 函数 def begin_in_order()
    # │  作用：等待 ``order``，运行 ``callback``，然后推进。若已被放弃则返回 False。
    # │  参数：self, order: int, callback = None, tool_name: str = ...
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def begin_in_order(self, order: int, callback=None, *, tool_name: str = "") -> bool:
        """Wait for ``order``, run ``callback``, advance. Returns False if abandoned."""
        with self._condition:
            # Bounded wait so one wedged dispatch can't starve later-ordered workers; on
            # expiry proceed out of order (interleaved prompts beat starvation). ``>=`` (not
            # ``==``) releases every skipped worker at once; abandoned short-circuits.
            in_order = self._condition.wait_for(
                lambda: self._next_order >= order or self.abandoned.is_set(), timeout=self._timeout,
            )
            if self.abandoned.is_set():
                # ⤷ 出口 return：返回 False
                return False  # the turn already synthesized this result; don't advance
            if not in_order:
                logger.warning(
                    "start-order gate timed out for %s (order=%d next=%d); proceeding out of order",
                    tool_name or "tool", order, self._next_order,
                )
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                if callback is not None:
                    callback()
            finally:
                self._next_order = max(self._next_order, order + 1)
                self._condition.notify_all()
        # ⤷ 出口 return：返回 True
        return True


# ┌── 类 class _WorkerStartOnce
# │  说明：一个 worker 对「启动顺序门」的句柄：最多推进一次，若……则抛出……
# │  成员：共 2 个方法；对外可用 1 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _WorkerStartOnce:
    """One worker's handle on the start-order gate: advances at most once, raising
    ``_BatchAbandoned`` (instead of dispatching late) when the batch was abandoned."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, gate: _StartOrderGate, order: int, tool_name: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, gate: _StartOrderGate, order: int, tool_name: str) -> None:
        self._gate, self._order, self._tool_name, self._advanced = gate, order, tool_name, False

    # ┌── 函数 def advance()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, callback = None
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def advance(self, callback=None) -> None:
        if self._advanced:
            return
        self._advanced = True
        if not self._gate.begin_in_order(self._order, callback, tool_name=self._tool_name):
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise _BatchAbandoned(self._tool_name)


# ┌── 类 class _ConcurrentBatch
# │  说明：一个并发工具批次的共享状态：各槽位结果、启动顺序以及……
# │  成员：共 7 个方法；对外可用 4 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _ConcurrentBatch:
    """Shared state of one concurrent tool batch: per-slot results, the start-order and
    authorization gates, and the deadline bookkeeping the wait loop needs."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, agent, messages: list, effective_task_id: str, parsed_calls: list[_ParsedCall], timeout_s: float | None
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, agent, messages: list, effective_task_id: str, parsed_calls: list[_ParsedCall], timeout_s: float | None) -> None:
        self.agent = agent
        self.messages = messages
        self.effective_task_id = effective_task_id
        self.parsed_calls = parsed_calls
        self.timeout_s = timeout_s
        self.results: list[Optional[_ToolOutcome]] = [None] * len(parsed_calls)
        for i, pc in enumerate(parsed_calls):
            if pc.parse_error is not None:
                self.results[i] = _ToolOutcome(pc.ref(effective_task_id), pc.parse_error, 0.0, True, True)
        self.gate = _StartOrderGate(_start_order_gate_timeout(timeout_s))
        self.authorization_gate = _ConcurrentToolAuthorizationGate()
        self.timed_out_indices: set[int] = set()

    # ┌── 函数 def _dispatch_worker()
    # │  作用：让一次调用经过中间件并合成它的槽位结果；当……时为 ``None``。
    # │  参数：self, index: int, ref: _ToolCallRef, scope_block, start_gate: _WorkerStartOnce
    # │  返回：Optional[_ToolOutcome]
    # └──────────────────────────────────────────────────────────────
    def _dispatch_worker(self, index: int, ref: _ToolCallRef, scope_block, start_gate: _WorkerStartOnce) -> Optional[_ToolOutcome]:
        """Run one call through the middleware and synthesize its slot outcome; ``None`` when
        abandoned at the gate (the main thread already wrote this slot; emitting would
        double-report the tool_call_id)."""
        agent = self.agent
        # Approval/sudo callbacks (thread-local) and the agent turn's ContextVars are propagated by
        # propagate_context_to_thread() at the submit site below (GHSA-qg5c-hvr5-hjgr, #13617).
        start = time.time()
        blocked = dispatched = False
        # ⤷ 关键分支 try/except：捕获 _BatchAbandoned、KeyboardInterrupt、Exception，出错走兜底而不是崩溃。
        try:
            managed = _run_agent_tool_execution_middleware(
                agent,
                **ref.middleware_kwargs(),
                execute=lambda next_args: agent._invoke_tool(
                    ref.name, next_args, ref.task_id, ref.call_id,
                    messages=self.messages,
                    pre_tool_block_checked=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                    tool_request_middleware_trace=list(ref.trace),
                ),
                scope_block=scope_block,
                display_index=index + 1,
                begin_execution=start_gate.advance,
                authorization_gate=self.authorization_gate,
            )
            result, ref.args, ref.trace = managed.result, managed.args, managed.middleware_trace
            blocked, dispatched = managed.blocked, managed.dispatched
        except _BatchAbandoned:
            logger.info("tool %s abandoned at start-order gate; skipping dispatch", ref.name)
            # ⤷ 出口 return：返回 None
            return None
        except KeyboardInterrupt:
            with contextlib.suppress(Exception):
                agent.interrupt("keyboard interrupt")
            result = ref.emit_cancelled(agent, start)
            duration = time.time() - start
            logger.info("tool %s cancelled (%.2fs)", ref.name, duration)
            # ⤷ 出口 return：返回 _ToolOutcome(ref, result, duration, True, False)
            return _ToolOutcome(ref, result, duration, True, False)
        except Exception as tool_error:
            result = f"Error executing tool '{ref.name}': {tool_error}"
            logger.error("_invoke_tool raised for %s: %s", ref.name, tool_error, exc_info=True)
        duration = time.time() - start
        if not blocked and not dispatched:
            ref.emit_post(agent, result, duration_ms=int(duration * 1000))
        is_error, _ = _detect_tool_failure(ref.name, result)
        if is_error:
            logger.info("tool %s failed (%.2fs): %s", ref.name, duration, result[:200])
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", ref.name, duration, len(result))
        # ⤷ 出口 return：返回 _ToolOutcome(ref, result, duration, is_error, blocked)
        return _ToolOutcome(ref, result, duration, is_error, blocked)

    # ┌── 函数 def run_worker()
    # │  作用：在线程中执行的 worker 函数。
    # │  参数：self, index: int, start_order: int
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def run_worker(self, index: int, start_order: int) -> None:
        """Worker function executed in a thread."""
        agent, pc = self.agent, self.parsed_calls[index]
        with _registered_tool_worker(agent) as _worker_tid:
            # An interrupt may have fanned out before our registration; apply it to our tid.
            if agent._interrupt_requested:
                _interrupt_worker_tids(agent, [_worker_tid], reason=getattr(agent, "_tool_interrupt_reason", None))
            _set_worker_activity_callback(agent)
            start_gate = _WorkerStartOnce(self.gate, start_order, pc.name)
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                outcome = self._dispatch_worker(index, pc.ref(self.effective_task_id), pc.scope_block, start_gate)
                if outcome is not None:
                    self.results[index] = outcome
            finally:
                with contextlib.suppress(_BatchAbandoned):
                    start_gate.advance()  # keep later-ordered workers moving

    # ┌── 函数 def submit_all()
    # │  作用：提交每一个可运行的槽位；在解释器关闭时，合成错误结果……
    # │  参数：self, executor, runnable: list[int]
    # │  返回：tuple[list, dict]
    # └──────────────────────────────────────────────────────────────
    def submit_all(self, executor, runnable: list[int]) -> tuple[list, dict]:
        """Submit every runnable slot; on interpreter shutdown, synthesize error results
        for the unsubmitted remainder instead of raising. ``propagate_context_to_thread``
        carries turn ContextVars and thread-local approval/sudo callbacks into the worker."""
        futures = []
        future_to_index = {}
        for submit_index, i in enumerate(runnable):
            # ⤷ 关键分支 try/except：捕获 RuntimeError，出错走兜底而不是崩溃。
            try:
                f = executor.submit(propagate_context_to_thread(self.run_worker), i, submit_index)
            except RuntimeError as submit_error:
                if not _is_interpreter_shutdown_submit_error(submit_error):
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise
                skipped = runnable[submit_index:]
                logger.warning(
                    "interpreter shutdown while scheduling concurrent tools; skipping %d unsubmitted tool(s)", len(skipped),
                )
                for skipped_i in skipped:
                    ref = self.parsed_calls[skipped_i].ref(self.effective_task_id)
                    if self.results[skipped_i] is None:
                        result = f"Error executing tool '{ref.name}': Python interpreter is shutting down; tool was not started"
                        self.results[skipped_i] = _ToolOutcome(ref, result, 0.0, True, False)
                break
            futures.append(f)
            future_to_index[f] = i
        # ⤷ 出口 return：返回 (futures, future_to_index)
        return futures, future_to_index

    # ┌── 函数 def _running_names()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, not_done, future_to_index
    # │  返回：list[str]
    # └──────────────────────────────────────────────────────────────
    def _running_names(self, not_done, future_to_index) -> list[str]:
        # ⤷ 出口 return：返回 [self.parsed_calls[future_to_index[f]].name for f in not_don…
        return [self.parsed_calls[future_to_index[f]].name for f in not_done if f in future_to_index]

    # ┌── 函数 def await_completion()
    # │  作用：以周期性心跳和中断检查来等待；当批次被……时返回 True。
    # │  参数：self, futures, future_to_index, deadline: float | None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def await_completion(self, futures, future_to_index, deadline: float | None) -> bool:
        """Wait with periodic heartbeats and interrupt checks; True when the batch was
        abandoned (deadline or interrupt) and the executor must not join its workers."""
        agent = self.agent
        _conc_start = time.time()
        while True:
            wait_timeout = 5.0
            if deadline is not None:
                remaining = deadline + self.authorization_gate.excluded_seconds() - time.monotonic()
                if remaining <= 0:
                    not_done = {f for f in futures if not f.done()}
                else:
                    wait_timeout = min(wait_timeout, remaining)
            if deadline is None or remaining > 0:
                _done, not_done = concurrent.futures.wait(futures, timeout=wait_timeout)
            if not not_done:
                # ⤷ 出口 return：返回 False
                return False

            timed_out = deadline is not None and time.monotonic() >= deadline + self.authorization_gate.excluded_seconds()
            if timed_out:
                self.timed_out_indices = {future_to_index[f] for f in not_done if f in future_to_index}
                logger.warning(
                    "concurrent tool batch timed out after %.1fs; %d tool(s) still running: %s",
                    self.timeout_s,
                    len(self.timed_out_indices),
                    ", ".join(self._running_names(not_done, future_to_index)[:5]),
                )
            elif agent._interrupt_requested:
                # Tools without interrupt checks (web_search, read_file) run to
                # completion; cancel unstarted futures so we don't block on them.
                agent._vprint(
                    f"{agent.log_prefix}⚡ Interrupt: cancelling {len(not_done)} pending concurrent tool(s)",
                    force=True,
                )
            else:
                _conc_elapsed = int(time.time() - _conc_start)
                # Heartbeat every ~30s (6 × 5s poll intervals)
                if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                    _still_running = self._running_names(not_done, future_to_index)
                    agent._touch_activity(
                        f"concurrent tools running ({_conc_elapsed}s, "
                        f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                    )
                continue
            for f in not_done:
                f.cancel()
            # Release gate-parked workers BEFORE interrupt fan-out so none later
            # dispatches a tool the turn already reported as timed out / interrupted.
            self.gate.abandon()
            if timed_out:
                with agent._tool_worker_threads_lock:
                    worker_tids = list(agent._tool_worker_threads)
                _interrupt_worker_tids(agent, worker_tids)
            else:
                # Give running tools a moment to notice the per-thread interrupt and exit gracefully.
                concurrent.futures.wait(not_done, timeout=3.0)
            # ⤷ 出口 return：返回 True
            return True

    # ┌── 函数 def run()
    # │  作用：在守护线程池上派发可运行的调用，并等待整个批次完成。
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Dispatch the runnable calls on a daemon pool and wait for the batch."""
        runnable = [i for i, pc in enumerate(self.parsed_calls) if pc.parse_error is None]
        if not runnable:
            return
        deadline = time.monotonic() + self.timeout_s if self.timeout_s is not None else None
        max_workers = _max_workers_for_tool_batch([(i, None, self.parsed_calls[i].name) for i in runnable])
        # Daemon workers: the stdlib pool's atexit join would let one wedged tool block exit.
        from tools.daemon_pool import DaemonThreadPoolExecutor
        executor = DaemonThreadPoolExecutor(max_workers=max_workers)
        abandon_executor = False
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            futures, future_to_index = self.submit_all(executor, runnable)
            abandon_executor = self.await_completion(futures, future_to_index, deadline)
        finally:
            # Every abandoning exit releases gate-parked workers and leaves wedged threads
            # detached rather than joining them; normal completion joins.
            if abandon_executor:
                self.gate.abandon()
            executor.shutdown(wait=not abandon_executor, cancel_futures=abandon_executor)


# ┌── 函数 def _unfinished_tool_result()
# │  作用：为没有任何 worker 填充的槽位合成结果（因截止时间、中断，或线程……）。
# │  参数：agent, ref: _ToolCallRef, timed_out: bool, timeout_s: float | None
# │  返回：tuple[str, float, Optional[str]]
# └──────────────────────────────────────────────────────────────
def _unfinished_tool_result(agent, ref: _ToolCallRef, *, timed_out: bool, timeout_s: float | None) -> tuple[str, float, Optional[str]]:
    """Synthesize the result for a slot no worker filled (deadline, interrupt, or a thread
    that never returned), emit its terminal post_tool_call, and return
    ``(function_result, tool_duration, effect_disposition)``."""
    if timed_out:
        suffix = f"{timeout_s:.1f}s" if timeout_s is not None else "the configured timeout"
        function_result = f"Error executing tool '{ref.name}': timed out after {suffix}"
        outcome = dict(duration_ms=int((timeout_s or 0.0) * 1000), status="timeout", error_type="tool_timeout", error_message=function_result)
        tool_duration, effect_disposition = float(timeout_s or 0.0), "unknown"
    elif agent._interrupt_requested:
        function_result = f"[Tool execution cancelled — {ref.name} was skipped due to user interrupt]"
        outcome = dict(status="cancelled", error_type="keyboard_interrupt", error_message="Tool execution cancelled by user interrupt")
        tool_duration, effect_disposition = 0.0, None
    else:
        function_result = f"Error executing tool '{ref.name}': thread did not return a result"
        outcome = dict(status="error", error_type="thread_missing_result", error_message=function_result)
        tool_duration, effect_disposition = 0.0, None
    ref.emit_post(agent, function_result, **outcome)
    # ⤷ 出口 return：返回 (function_result, tool_duration, effect_disposition)
    return function_result, tool_duration, effect_disposition


# ┌── 函数 def _append_batch_results()
# │  作用：按原始调用顺序追加每个槽位的结果；在遇到第一个……时返回 False。
# │  参数：agent, messages: list, effective_task_id: str, batch: _ConcurrentBatch, budget: BudgetConfig
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _append_batch_results(agent, messages: list, effective_task_id: str, batch: _ConcurrentBatch, budget: BudgetConfig) -> bool:
    """Append every slot's result in original call order; returns False at the first
    failed flush (the caller must stop the batch)."""
    for i, pc in enumerate(batch.parsed_calls):
        r = batch.results[i]
        # A worker may finish between the deadline snapshot and this loop;
        # prefer its real result over a fabricated timeout.
        if r is None:
            ref, is_error, blocked = pc.ref(effective_task_id), True, False
            function_result, tool_duration, effect_disposition = _unfinished_tool_result(
                agent, ref, timed_out=i in batch.timed_out_indices, timeout_s=batch.timeout_s,
            )
        else:
            ref, function_result, tool_duration, is_error, blocked = r.ref, r.result, r.duration, r.is_error, r.blocked
            effect_disposition = "none" if blocked else None
            if pc.parse_error is not None:
                ref.emit_invalid_arguments(agent, r.result)
        committed = _commit_tool_result(
            agent, messages, ref, function_result,
            budget=budget, tool_duration=tool_duration, is_error=is_error, blocked=blocked,
            effect_disposition=effect_disposition, observed=r is not None,
            error_preview=lambda res: _multimodal_text_summary(res)[:200],
        )
        if committed is None:
            # ⤷ 出口 return：返回 False
            return False
        _persisted, display_function_result, risk_metadata = committed

        if agent._should_emit_quiet_tool_messages():
            cute_msg = _get_cute_tool_message_impl(ref.name, ref.args, tool_duration, result=display_function_result)
            agent._safe_print(f"  {cute_msg}")
        elif _tool_progress_enabled(agent):
            _print_tool_completed(agent, i + 1, tool_duration, _multimodal_text_summary(display_function_result))

        _emit_tool_complete_and_risk(agent, ref, display_function_result, risk_metadata, blocked)
    # ⤷ 出口 return：返回 True
    return True


# ┌── 函数 def execute_tool_calls_concurrent()
# │  作用：并发执行工具调用；结果按原始调用顺序追加。
# │  参数：agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, finalize: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls concurrently; results are appended in original call order.
    ``finalize=False`` skips end-of-batch budget enforcement and /steer injection (the
    segmented dispatcher owns turn-end work)."""
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)
    _tool_budget = _budget_for_agent(agent)  # once per turn, not per result

    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        _append_skipped_tool_results(
            agent, messages, tool_calls, effective_task_id,
            content="[Tool execution cancelled — {name} was skipped due to user interrupt]",
            hook_error_type="user_interrupt",
            flush_stage="cancelled tool result",
            stop_on_flush_failure=False,
        )
        return

    parsed_calls = [_parse_tool_call(agent, tc) for tc in tool_calls]

    tool_names_str = ", ".join(pc.name for pc in parsed_calls)
    if _tool_progress_enabled(agent):
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")

    # Resolved before the batch is built so the start-order gate can clamp under the deadline.
    timeout_s = _resolve_concurrent_tool_timeout()
    batch = _ConcurrentBatch(agent, messages, effective_task_id, parsed_calls, timeout_s)
    agent._current_tool = tool_names_str
    agent._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

    spinner = _start_quiet_tool_spinner(agent, "", {}, label=f"⚡ running {num_tools} tools concurrently")
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        batch.run()
    finally:
        if spinner:
            finished = [r for r in batch.results if r is not None]
            spinner.stop(f"⚡ {len(finished)}/{num_tools} tools completed in {sum(r.duration for r in finished):.1f}s total")

    if not _append_batch_results(agent, messages, effective_task_id, batch, _tool_budget):
        return
    if finalize:
        _finalize_tool_batch(agent, messages, effective_task_id, len(parsed_calls), _tool_budget)


# ── Sequential dispatch ─────────────────────────────────────────────────────


# ┌── 函数 def _start_quiet_tool_spinner()
# │  作用：为一个工具调用启动安静模式下的可爱转圈动画，或返回 None；``gate=False``……
# │  参数：agent, function_name: str, function_args: dict, gate: bool = ..., label: Optional[str] = ...
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _start_quiet_tool_spinner(agent, function_name: str, function_args: dict, *, gate: bool = True, label: Optional[str] = None):
    """Start the quiet-mode kawaii spinner for one tool call, or return None; ``gate=False``
    skips ``_should_start_quiet_spinner`` (context-engine tools always spin)."""
    if not agent._should_emit_quiet_tool_messages() or (gate and not agent._should_start_quiet_spinner()):
        # ⤷ 出口 return：返回 None
        return None
    face = random.choice(KawaiiSpinner.get_waiting_faces())
    if label is None:
        display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
        label = f"{_get_tool_emoji(function_name)} {_build_tool_label(function_name, display_args) or function_name}"
    spinner = KawaiiSpinner(f"{face} {label}", spinner_type='dots', print_fn=agent._print_fn)
    spinner.start()
    # ⤷ 出口 return：返回 spinner
    return spinner


# ┌── 函数 def _finish_quiet_tool_spinner()
# │  作用：用可爱的完成行停下转圈动画，或在没有动画时直接打印它。
# │  参数：agent, spinner, function_name: str, function_args: dict, tool_duration: float, result
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _finish_quiet_tool_spinner(agent, spinner, function_name: str, function_args: dict, tool_duration: float, result) -> None:
    """Stop the spinner with the cute completion line, or print it when no spinner ran."""
    if spinner or agent._should_emit_quiet_tool_messages():
        cute = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=result)
        spinner.stop(cute) if spinner else agent._vprint(f"  {cute}")


# ┌── 函数 def _delegate_spinner_label()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：function_args: dict
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _delegate_spinner_label(function_args: dict) -> str:
    action = str(function_args.get("action") or "").strip().lower()
    tasks = function_args.get("tasks")
    if action in ("list", "steer", "stop"):
        # ⤷ 出口 return：返回 f'🔀 subagent {action}'
        return f"🔀 subagent {action}"
    if tasks and isinstance(tasks, list):
        # ⤷ 出口 return：返回 f'🔀 delegating {len(tasks)} tasks · (/agents to monitor)'
        return f"🔀 delegating {len(tasks)} tasks · (/agents to monitor)"
    goal_preview = (function_args.get("goal") or "")[:30]
    # ⤷ 出口 return：返回 f'🔀 {goal_preview} · (/agents to monitor)' if goal_preview e…
    return f"🔀 {goal_preview} · (/agents to monitor)" if goal_preview else "🔀 delegating · (/agents to monitor)"


# ┌── 类 class _SequentialDispatch
# │  说明：一次顺序调用如何执行：可调用对象，加上它的转圈动画/错误策略。
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class _SequentialDispatch:
    """How one sequential call executes: the callable plus its spinner/error policy."""

    execute: Callable[[dict], Any]
    spinner: Any = None
    middleware_trace_arg: Optional[list] = None  # forwarded to the middleware runner (registry closure reads it)
    error_result: Optional[Callable[[Exception], str]] = None  # None → exceptions propagate (inline/delegate own failures)
    error_log: str = ""
    handles_keyboard_interrupt: bool = False
    is_delegate: bool = False
    finish_spinner: bool = True
    finish_in_finally: bool = True  # inline tools print their completion line only on success


# ┌── 函数 def _resolve_sequential_dispatch()
# │  作用：为一次顺序调用挑选执行可调用对象并启动其转圈动画。优先级：……
# │  参数：agent, ref: _ToolCallRef, messages: list
# │  返回：_SequentialDispatch
# └──────────────────────────────────────────────────────────────
def _resolve_sequential_dispatch(agent, ref: _ToolCallRef, messages: list) -> _SequentialDispatch:
    """Pick the execute callable for one sequential call and start its spinner. Precedence:
    inline agent-level tools, delegate_task, context-engine tools, memory-provider tools,
    then the registry."""
    function_name, function_args, effective_task_id, tool_call_id, middleware_trace = (
        ref.name, ref.args, ref.task_id, ref.call_id, ref.trace,
    )
    if function_name != "delegate_task" and function_name in INLINE_TOOL_EXECUTORS:
        # Agent-level tools that need live AIAgent state; table shared with invoke_tool.
        inline_executor = INLINE_TOOL_EXECUTORS[function_name]
        inline_ctx = InlineToolContext(effective_task_id=effective_task_id, tool_call_id=tool_call_id, messages=messages)
        # ⤷ 出口 return：返回 _SequentialDispatch(lambda next_args: inline_executor(agent,…
        return _SequentialDispatch(lambda next_args: inline_executor(agent, next_args, inline_ctx), finish_in_finally=False)
    if function_name == "delegate_task":
        spinner = _start_quiet_tool_spinner(agent, function_name, function_args, label=_delegate_spinner_label(function_args))
        agent._delegate_spinner = spinner
        # ⤷ 出口 return：返回 _SequentialDispatch(agent._dispatch_delegate_task, spinner=s…
        return _SequentialDispatch(agent._dispatch_delegate_task, spinner=spinner, is_delegate=True)
    if agent._context_engine_tool_names and function_name in agent._context_engine_tool_names:
        # ⤷ 出口 return：返回 _SequentialDispatch(execute=lambda next_args: agent.context_…
        return _SequentialDispatch(
            execute=lambda next_args: agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages),
            spinner=_start_quiet_tool_spinner(agent, function_name, function_args, gate=False),
            error_result=lambda e: json.dumps({"error": f"Context engine tool '{function_name}' failed: {e}"}),
            error_log="context_engine.handle_tool_call raised for %s: %s",
        )
    if agent._memory_manager and agent._memory_manager.has_tool(function_name):
        # Memory-provider tools (hindsight_retain, honcho_search, ...) are not in the registry.
        # ⤷ 出口 return：返回 _SequentialDispatch(execute=lambda next_args: agent._memory_…
        return _SequentialDispatch(
            execute=lambda next_args: agent._memory_manager.handle_tool_call(function_name, next_args),
            spinner=_start_quiet_tool_spinner(agent, function_name, function_args),
            error_result=lambda e: json.dumps({"error": f"Memory tool '{function_name}' failed: {e}"}),
            error_log="memory_manager.handle_tool_call raised for %s: %s",
        )

    # Registry tools: post hook is owned by this executor (inner observer suppressed).
    def _execute(next_args: dict) -> Any:
        import model_tools

        with model_tools.suppress_post_tool_call_hook():
            # ⤷ 出口 return：返回 model_tools.handle_function_call(function_name, next_args, e…
            return model_tools.handle_function_call(
                function_name,
                next_args,
                effective_task_id,
                tool_call_id=tool_call_id,
                session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
                tool_request_middleware_trace=list(middleware_trace),
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            )

    # ⤷ 出口 return：返回 _SequentialDispatch(execute=_execute, spinner=_start_quiet_t…
    return _SequentialDispatch(
        execute=_execute,
        spinner=_start_quiet_tool_spinner(agent, function_name, function_args) if agent.quiet_mode else None,
        middleware_trace_arg=middleware_trace,
        error_result=lambda e: f"Error executing tool '{function_name}': {e}",
        error_log="handle_function_call raised for %s: %s",
        handles_keyboard_interrupt=True,
        finish_spinner=bool(agent.quiet_mode),
    )


# ┌── 函数 def _skip_remaining_sequential()
# │  作用：宣布一次中断，并为每个未开始的调用追加一条「已跳过」结果；当……时返回 False。
# │  参数：agent, messages: list, remaining, effective_task_id: str, notice: str, **skip_kwargs
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _skip_remaining_sequential(agent, messages: list, remaining, effective_task_id: str, *, notice: str, **skip_kwargs) -> bool:
    """Announce an interrupt and append one skipped result per unstarted call; False when
    a flush failed (the caller must stop the batch)."""
    agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining)} {notice}", force=True)
    # ⤷ 出口 return：返回 _append_skipped_tool_results(agent, messages, remaining, eff…
    return _append_skipped_tool_results(agent, messages, remaining, effective_task_id, **skip_kwargs)


# ┌── 函数 def _append_invalid_arguments_result()
# │  作用：为参数不是 JSON 对象的调用发出并追加「解析错误」结果。
# │  参数：agent, messages: list, ref: _ToolCallRef, parse_error: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _append_invalid_arguments_result(agent, messages: list, ref: _ToolCallRef, parse_error: str) -> bool:
    """Emit + append the parse-error result for a call whose arguments were not a JSON object."""
    ref.emit_invalid_arguments(agent, parse_error)
    messages.append(make_tool_result_message(ref.name, parse_error, ref.call_id))
    # ⤷ 出口 return：返回 _flush_session_db_after_tool_progress(agent, messages, stage…
    return _flush_session_db_after_tool_progress(agent, messages, stage=f"invalid tool arguments {ref.name}")


# ┌── 函数 def _run_sequential_call()
# │  作用：运行一次顺序调用（含其转圈动画/错误策略）；返回 ``(managed, duration)``。
# │  参数：agent, dispatch: _SequentialDispatch, ref: _ToolCallRef, scope_block: Optional[str], messages: list, remaining_calls, display_index: int, tool_start_time: float
# │  返回：tuple[_ManagedToolResult, float]
# └──────────────────────────────────────────────────────────────
def _run_sequential_call(
    agent,
    dispatch: _SequentialDispatch,
    ref: _ToolCallRef,
    *,
    scope_block: Optional[str],
    messages: list,
    remaining_calls,
    display_index: int,
    tool_start_time: float,
) -> tuple[_ManagedToolResult, float]:
    """Run one sequential call with its spinner/error policy; returns ``(managed, duration)``.
    KeyboardInterrupt (registry tools only) emits results for THIS and every remaining call
    before re-raising so the tool-call turn keeps matching results (alternation)."""
    _spinner_result = None
    # ⤷ 关键分支 try/except：捕获 KeyboardInterrupt、Exception，出错走兜底而不是崩溃。
    try:
        managed = _run_sequential_tool_execution_middleware(
            agent,
            **dict(ref.middleware_kwargs(), middleware_trace=dispatch.middleware_trace_arg),
            execute=dispatch.execute,
            scope_block=scope_block,
            display_index=display_index,
        )
        ref.args = managed.args
        _spinner_result = managed.result
    except KeyboardInterrupt:
        if not dispatch.handles_keyboard_interrupt:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        _spinner_result = ref.emit_cancelled(agent, tool_start_time)
        with contextlib.suppress(Exception):
            agent.interrupt("keyboard interrupt")
        _append_skipped_tool_results(
            agent, messages, remaining_calls, ref.task_id,
            content="[Tool execution cancelled — {name} was skipped due to keyboard interrupt]",
        )
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise
    except Exception as tool_error:
        if dispatch.error_result is None:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        function_result = dispatch.error_result(tool_error)
        logger.error(dispatch.error_log, ref.name, tool_error, exc_info=True)
        managed = _ManagedToolResult(result=function_result, args=ref.args, middleware_trace=ref.trace, blocked=False, dispatched=False)
    finally:
        if dispatch.is_delegate:
            agent._delegate_spinner = None
        tool_duration = time.time() - tool_start_time
        if dispatch.finish_spinner and dispatch.finish_in_finally:
            _finish_quiet_tool_spinner(agent, dispatch.spinner, ref.name, ref.args, tool_duration, _spinner_result)
    if dispatch.finish_spinner and not dispatch.finish_in_finally:
        _finish_quiet_tool_spinner(agent, dispatch.spinner, ref.name, ref.args, tool_duration, _spinner_result)
    # ⤷ 出口 return：返回 (managed, tool_duration)
    return managed, tool_duration


# ┌── 函数 def _publish_sequential_result()
# │  作用：对一次顺序调用依次执行：终局钩子 → 观察 → 提交 → 完成回调/打印……
# │  参数：agent, messages: list, ref: _ToolCallRef, managed: _ManagedToolResult, tool_duration: float, index: int, budget: BudgetConfig
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _publish_sequential_result(agent, messages: list, ref: _ToolCallRef, managed: _ManagedToolResult, *, tool_duration: float, index: int, budget: BudgetConfig) -> bool:
    """Terminal hook → observe → commit → completion callbacks/print for one sequential
    result; False when the incremental flush failed (the caller must stop the batch)."""
    ref.args, ref.trace, function_result = managed.args, managed.middleware_trace, managed.result
    _execution_timed_out = isinstance(function_result, (_ToolTimeoutResult, _ToolCancelledResult))
    # Multimodal dict results (_multimodal=True) are not sliceable as strings.
    _result_len = len(function_result) if isinstance(function_result, str) else len(str(function_result))
    _is_error_result, _ = _detect_tool_failure(ref.name, function_result)
    # Inline-dispatched runtime tools never reach handle_function_call, so the
    # executor owns the one terminal post_tool_call per tool_call_id (the inner
    # observer is suppressed); also stops an abandoned timeout worker reporting late.
    if not managed.blocked and not _execution_timed_out:
        ref.emit_post(agent, function_result, duration_ms=int(tool_duration * 1000))
    committed = _commit_tool_result(
        agent, messages, ref, function_result,
        budget=budget, tool_duration=tool_duration, is_error=_is_error_result, blocked=managed.blocked,
        effect_disposition="unknown" if _execution_timed_out else None, observed=True,
        error_preview=lambda res: res[:200] if isinstance(res, str) and not agent.verbose_logging else res,
        success_log_chars=_result_len,
        verbose_text=_multimodal_text_summary,
    )
    if committed is None:
        # ⤷ 出口 return：返回 False
        return False
    function_result, display_function_result, risk_metadata = committed

    _emit_tool_complete_and_risk(agent, ref, display_function_result, risk_metadata, managed.blocked)
    if _tool_progress_enabled(agent):
        _print_tool_completed(agent, index, tool_duration, function_result)
    # ⤷ 出口 return：返回 True
    return True


# ┌── 函数 def execute_tool_calls_sequential()
# │  作用：顺序执行工具调用（用于单次调用或交互式工具）。``finalize=False``……
# │  参数：agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, finalize: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls sequentially (single calls or interactive tools). ``finalize=False``
    skips end-of-batch budget enforcement and /steer injection (the segmented dispatcher
    owns turn-end work)."""
    _tool_budget = _budget_for_agent(agent)  # once per turn, not per result
    tool_calls = assistant_message.tool_calls

    for i, tool_call in enumerate(tool_calls, 1):
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        # Check interrupt BEFORE each tool so a "stop" during the previous one skips the rest.
        if agent._interrupt_requested:
            if not _skip_remaining_sequential(
                agent, messages, tool_calls[i - 1:], effective_task_id,
                notice="tool call(s)",
                content="[Tool execution cancelled — {name} was skipped due to user interrupt]",
                hook_error_type="user_interrupt",
                hook_id=lambda tc: getattr(tc, "id", "") or "",
                flush_stage="cancelled tool result",
            ):
                return
            break

        pc = _parse_tool_call(agent, tool_call, flatten_probe=True)
        ref = pc.ref(effective_task_id)
        if pc.parse_error is not None:
            if not _append_invalid_arguments_result(agent, messages, ref, pc.parse_error):
                return
            continue

        tool_start_time = time.time()
        dispatch = _resolve_sequential_dispatch(agent, ref, messages)
        managed, tool_duration = _run_sequential_call(
            agent, dispatch, ref,
            scope_block=pc.scope_block,
            messages=messages,
            remaining_calls=tool_calls[i - 1:],
            display_index=i,
            tool_start_time=tool_start_time,
        )
        if not _publish_sequential_result(agent, messages, ref, managed, tool_duration=tool_duration, index=i, budget=_tool_budget):
            return

        if agent._interrupt_requested and i < len(tool_calls):
            if not _skip_remaining_sequential(
                agent, messages, tool_calls[i:], effective_task_id,
                notice="remaining tool call(s)",
                content="[Tool execution skipped — {name} was not started. User sent a new message]",
                flush_stage="skipped tool result",
            ):
                return
            break

    if finalize:
        _finalize_tool_batch(agent, messages, effective_task_id, len(tool_calls), _tool_budget)


# ┌── 函数 def execute_tool_calls_segmented()
# │  作用：把一个混合批次按「有序的并行/顺序」分段来执行（即 ``(kind, calls)``……）。
# │  参数：agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments = None
# │  返回：None
# └──────────────────────────────────────────────────────────────
def execute_tool_calls_segmented(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments=None) -> None:
    """Execute a mixed batch as ordered parallel/sequential segments (the ``(kind, calls)``
    plan from ``_plan_tool_batch_segments``), preserving per-call result order and barrier
    boundaries exactly as fully-sequential execution. Turn-end work (budget + /steer) runs
    once here (segments run with ``finalize=False``); each segment executor checks the
    interrupt flag up front, so an interrupt drains later segments with one result per call."""
    from types import SimpleNamespace

    if segments is None:
        _active_env = get_active_env(effective_task_id)
        _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
        segments = _plan_tool_batch_segments(assistant_message.tool_calls, execution_cwd=_exec_cwd)

    for kind, calls in segments:
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        segment_message = SimpleNamespace(tool_calls=list(calls))
        run_segment = execute_tool_calls_concurrent if kind == "parallel" else execute_tool_calls_sequential
        run_segment(agent, segment_message, messages, effective_task_id, api_call_count, finalize=False)
        if getattr(agent, "_incremental_persistence_failed", False):
            return

    total_tools = len(assistant_message.tool_calls)
    if total_tools > 0:
        _finalize_tool_batch(agent, messages, effective_task_id, total_tools, _budget_for_agent(agent))


__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
    "execute_tool_calls_segmented",
]
