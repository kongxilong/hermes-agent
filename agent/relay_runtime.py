"""Profile-scoped NeMo Relay runtimes owned by the Hermes agent core."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：agent/relay_runtime.py（约 1352 行）
# 地位：NeMo Relay 运行时（按配置档隔离的会话/作用域管理）。
#
# 说明：Relay 是本项目里比较「框架化」的一层，负责把「一次会话」「一次工具调用」
#   包装成带作用域（scope）的可追踪单元，便于插桩、隔离和多租户。
#   初学阶段可以先跳过这个文件，等熟悉主流程再回来看。
#
# 主要类型：
#   class RelayRuntime            运行时本体（28 个方法）
#   class NoopRelayRuntime        空实现（Relay 没启用时用这个占位，避免到处写 if）
#                                 —— 这种「空对象模式」很值得学
#   class RelaySessionCoordinator 会话协调器：begin_turn / end_turn / acquire_conversation
#   class RelayOperationLease     操作租约（和 registration_lifecycle.py 的租约思路一致）
#   class RelayHostRegistry       按配置档（profile）隔离，for_profile() / shutdown_all()
#
# 常用入口：ensure_session() / run_in_session() / get_session_handle() / emit_mark()
# ==========================================================================
#

from __future__ import annotations

import atexit
import asyncio
import contextlib
import contextvars
import functools
import importlib
import inspect
import logging
import os
import threading
import tomllib
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from hermes_cli.relay_plugin_cutover import (RELAY_PLUGINS_CONFIG_ENV, configured_legacy_relay_env_vars)

logger = logging.getLogger(__name__)

SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
RELAY_PLUGINS_EXECUTION_CONSUMER = "hermes.nemo_relay.plugins"
_PROFILE_KEY_CACHE: dict[str, str] = {}

# Bound for native scope ops gating turn/session completion: a wedged pipeline costs one
# lost span, never a blocked agent.
_SCOPE_OP_TIMEOUT = 10.0



# ┌── 类 class _Lazy
# │  说明：Double-checked, thread-safe once-only factory; ``reset()`` re-arms it (tests).
# │  成员：共 3 个方法；对外可用 2 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _Lazy:
    """Double-checked, thread-safe once-only factory; ``reset()`` re-arms it (tests)."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, factory: Callable[[], Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory, self._value, self._lock = factory, None, threading.Lock()

    # ┌── 函数 def get()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def get(self) -> Any:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._factory()
        # ⤷ 出口 return：返回 self._value
        return self._value

    # ┌── 函数 def reset()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._value = None


# ┌── 函数 def _new_scope_op_executor()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _new_scope_op_executor() -> Any:
    # Daemon workers: a wedged call abandoned at timeout cannot block interpreter exit;
    # ``Future.result(timeout=...)`` still bounds callers when every worker is wedged.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    # ⤷ 出口 return：返回 DaemonThreadPoolExecutor(max_workers=8, thread_name_prefix='…
    return DaemonThreadPoolExecutor(max_workers=8, thread_name_prefix="relay-scope-op")


_SCOPE_OP_EXECUTOR = _Lazy(_new_scope_op_executor)
_scope_op_executor = _SCOPE_OP_EXECUTOR.get


# ┌── 函数 def runtime_metadata()
# │  作用：Return the scope metadata that stamps every Hermes-owned Relay scope.
# │  参数：runtime_id: str, **extra
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def runtime_metadata(runtime_id: str, **extra: Any) -> dict[str, Any]:
    """Return the scope metadata that stamps every Hermes-owned Relay scope."""
    # ⤷ 出口 return：返回 {RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION, RUNTIME_INSTANC…
    return {RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION, RUNTIME_INSTANCE_KEY: runtime_id, **extra}


# ┌── 函数 def _run_on_daemon_thread()
# │  作用：Run ``fn`` on a fresh daemon thread; re-raise its error or return its result.
# │  参数：fn: Callable[[], Any], name: str, timeout: float | None = ..., timeout_message: str = ...
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _run_on_daemon_thread(
    fn: Callable[[], Any], *, name: str, timeout: float | None = None, timeout_message: str = ""
) -> Any:
    """Run ``fn`` on a fresh daemon thread; re-raise its error or return its result.
    With ``timeout`` a still-running worker is abandoned with ``TimeoutError`` (daemon: cannot block exit)."""
    outcome: dict[str, Any] = {}

    def _target() -> None:
        # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagated below
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True, name=name)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise TimeoutError(timeout_message)
    if "error" in outcome:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise outcome["error"]
    # ⤷ 出口 return：返回 outcome.get('result')
    return outcome.get("result")


# ┌── 函数 def pop_relay_scope()
# │  作用：Pop a Relay scope, forwarding only the kwargs the live binding accepts.
# │  参数：relay: Any, handle: Any, output: Any = ..., metadata: Any = ..., timestamp: Any = ...
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def pop_relay_scope(relay: Any, handle: Any, *, output: Any = None, metadata: Any = None, timestamp: Any = None) -> Any:
    """Pop a Relay scope, forwarding only the kwargs the live binding accepts.
    ``scope.pop`` gained ``metadata`` in nemo-relay 0.4+; older wheels raise TypeError."""
    pop = relay.scope.pop
    kwargs = {k: v for k, v in (("output", output), ("metadata", metadata), ("timestamp", timestamp)) if v is not None}
    # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
    try:
        params = inspect.signature(pop).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    # ⤷ 出口 return：返回 pop(handle, **kwargs)
    return pop(handle, **kwargs)


# ┌── 函数 def _current_top()
# │  作用：Return the current top-of-stack scope handle, or None.
# │  参数：relay: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _current_top(relay: Any) -> Any:
    """Return the current top-of-stack scope handle, or None."""
    # Prefer scope.get_handle(): get_scope_stack() may return a native ScopeStack that scope.pop rejects.
    get_handle = getattr(getattr(relay, "scope", None), "get_handle", None)
    if callable(get_handle):
        with contextlib.suppress(Exception):
            # ⤷ 出口 return：返回 get_handle()
            return get_handle()
    top = relay.get_scope_stack()
    # Some builds return the live stack (list), others the top handle: only unwrap real lists.
    # ⤷ 出口 return：返回 (top[-1] if top else None) if isinstance(top, list) else top
    return (top[-1] if top else None) if isinstance(top, list) else top


# ┌── 函数 def _same_handle()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: Any, b: Any
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _same_handle(a: Any, b: Any) -> bool:
    # Native ScopeHandle has no value __eq__; compare by uuid when both expose one.
    if a is None or b is None:
        # ⤷ 出口 return：返回 a is b
        return a is b
    a_uuid = getattr(a, "uuid", None)
    # ⤷ 出口 return：返回 a is b or a == b or (a_uuid is not None and a_uuid == getatt…
    return a is b or a == b or (a_uuid is not None and a_uuid == getattr(b, "uuid", None))


# Process-wide plugin-configuration result shared by every currently hosted profile.
_RelayPluginConfigurationState = Enum("_RelayPluginConfigurationState", "UNINITIALIZED DISABLED ACTIVE FOREIGN FAILED")


# ┌── 类 class _RelayPluginConfigurationLoadError（继承自：RuntimeError）
# │  说明：An explicitly selected Relay plugin configuration could not be loaded.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _RelayPluginConfigurationLoadError(RuntimeError):
    """An explicitly selected Relay plugin configuration could not be loaded."""


# ┌── 类 class RelaySession
# │  说明：One isolated Relay scope stack owned by a Hermes session.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class RelaySession:
    """One isolated Relay scope stack owned by a Hermes session."""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None
    # Segmentation: rotation closes the session scope and pushes segment N+1 at a turn
    # boundary (the only LIFO-safe point). Flags are set by compaction, consumed at that boundary.
    segment: int = 0  # index of the CURRENT session scope (0 = first)
    segment_turns: int = 0  # turns completed within the current segment
    rotate_pending: bool = False  # consumed at next begin_turn
    close_pending: bool = False  # rotating compaction hit a live turn; end_turn consumes it


# ┌── 函数 def _load_segments_config()
# │  作用：gateway.telemetry.session_segments; both defaults OFF => rotation never fires.
# │  参数：无
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _load_segments_config() -> dict[str, Any]:
    """gateway.telemetry.session_segments; both defaults OFF => rotation never fires."""
    on_compaction = False
    max_turns = 0
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from gateway.run import _load_gateway_config  # late import
        telemetry = (_load_gateway_config().get("gateway") or {}).get("telemetry") or {}
        segments = telemetry.get("session_segments") or {}
        on_compaction = bool(segments.get("on_compaction", False))
        # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
        try:
            max_turns = max(0, int(segments.get("max_turns", 0) or 0))
        except (TypeError, ValueError):
            max_turns = 0
    except Exception:  # noqa: BLE001 - config absence (or a malformed section) must not crash
        pass
    # ⤷ 出口 return：返回 {'on_compaction': on_compaction, 'max_turns': max_turns}
    return {"on_compaction": on_compaction, "max_turns": max_turns}


_SEGMENTS_CONFIG = _Lazy(_load_segments_config)  # cached at first read
_segments_config = _SEGMENTS_CONFIG.get
_reset_segments_config_for_tests = _SEGMENTS_CONFIG.reset


# ┌── 类 class RelayOperationLease
# │  说明：Keep process-wide Relay plugins alive across a deferred operation.
# │  成员：共 3 个方法；对外可用 2 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class RelayOperationLease:
    """Keep process-wide Relay plugins alive across a deferred operation."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, runtime: 'RelayRuntime'
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, runtime: "RelayRuntime") -> None:
        self._lock, self._runtime = threading.Lock(), runtime

    # ┌── 函数 def run_in_session()
    # │  作用：Run cleanup while this lease still owns the runtime lifetime.
    # │  参数：self, session: RelaySession, callback: Callable[..., Any], *args, **kwargs
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def run_in_session(self, session: RelaySession, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run cleanup while this lease still owns the runtime lifetime."""
        with self._lock:
            if self._runtime is None:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes Relay operation lease is released")
            # ⤷ 出口 return：返回 self._runtime._run_in_session_untracked(session, callback, *…
            return self._runtime._run_in_session_untracked(session, callback, *args, **kwargs)

    # ┌── 函数 def release()
    # │  作用：Release this lease exactly once.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def release(self) -> None:
        """Release this lease exactly once."""
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime._end_operation()


# ┌── 类 class _ProcessRelayPluginConfiguration
# │  说明：Own one Relay plugin configuration across profile-scoped hosts.
# │  成员：共 9 个方法；对外可用 4 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _ProcessRelayPluginConfiguration:
    """Own one Relay plugin configuration across profile-scoped hosts."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: set[int] = set()
        self._state = _RelayPluginConfigurationState.UNINITIALIZED
        self._relay: Any = None  # set while a Hermes-owned configuration is active
        self._activation: Any = None

    # ┌── 函数 def acquire()
    # │  作用：Join the process configuration, initializing it for the first host.
    # │  参数：self, owner: Any, relay: Any
    # │  返回：_RelayPluginConfigurationState
    # └──────────────────────────────────────────────────────────────
    def acquire(self, owner: Any, relay: Any) -> _RelayPluginConfigurationState:
        """Join the process configuration, initializing it for the first host."""
        with self._lock:
            if not self._owners:
                # First host decides for the whole process; later hosts just join.
                self._state = self._preflight(relay) or self._activate(relay)
                if self._state is _RelayPluginConfigurationState.ACTIVE:
                    logger.info(
                        "Relay plugins are active process-wide and apply to all profiles hosted by this Hermes process."
                    )
            self._owners.add(id(owner))
            # ⤷ 出口 return：返回 self._state
            return self._state

    # ┌── 函数 def _activate()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, relay: Any
    # │  返回：_RelayPluginConfigurationState
    # └──────────────────────────────────────────────────────────────
    def _activate(self, relay: Any) -> _RelayPluginConfigurationState:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            if not self._initialize(relay):
                # ⤷ 出口 return：返回 _RelayPluginConfigurationState.DISABLED
                return _RelayPluginConfigurationState.DISABLED
        except Exception as exc:
            self._activation = None
            logger.warning("Hermes Relay plugin initialization failed: %s", exc, exc_info=True)
            # ⤷ 出口 return：返回 _RelayPluginConfigurationState.FAILED
            return _RelayPluginConfigurationState.FAILED
        self._relay = relay
        # ⤷ 出口 return：返回 _RelayPluginConfigurationState.ACTIVE
        return _RelayPluginConfigurationState.ACTIVE

    # ┌── 函数 def _preflight()
    # │  作用：Return a terminal state when the process cannot take ownership; None to proceed.
    # │  参数：self, relay: Any
    # │  返回：_RelayPluginConfigurationState | None
    # └──────────────────────────────────────────────────────────────
    def _preflight(self, relay: Any) -> _RelayPluginConfigurationState | None:
        """Return a terminal state when the process cannot take ownership; None to proceed."""
        if self._relay is not None and not self._clear_active():
            logger.warning(
                "Hermes Relay plugin cleanup is still pending; refusing to replace the process-global configuration"
            )
            # ⤷ 出口 return：返回 _RelayPluginConfigurationState.FAILED
            return _RelayPluginConfigurationState.FAILED
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            existing_report = relay.plugin.report()
        except Exception:
            logger.warning(
                "Hermes could not determine whether a process-global Relay plugin configuration is already "
                "active; refusing to replace it", exc_info=True,
            )
            # ⤷ 出口 return：返回 _RelayPluginConfigurationState.FAILED
            return _RelayPluginConfigurationState.FAILED
        if existing_report is not None:
            logger.warning(
                "A process-global Relay plugin configuration is already active outside Hermes native ownership; "
                "leaving it unchanged and disabling Hermes-managed Relay middleware for this process"
            )
            # ⤷ 出口 return：返回 _RelayPluginConfigurationState.FOREIGN
            return _RelayPluginConfigurationState.FOREIGN
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def _initialize()
    # │  作用：Initialize Relay from the selected plugins.toml; False when none is selected.
    # │  参数：self, relay: Any
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _initialize(self, relay: Any) -> bool:
        """Initialize Relay from the selected plugins.toml; False when none is selected."""
        configured_inputs = _configured_plugin_inputs(relay)
        if configured_inputs is None:
            # ⤷ 出口 return：返回 False
            return False
        plugin_config, dynamic_plugins = configured_inputs
        if dynamic_plugins:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                initialize = relay.plugin.initialize_with_dynamic_plugins
                activation = _resolve_plugin_awaitable(initialize(plugin_config, dynamic_plugins))
                if activation is None:
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise RuntimeError("NeMo Relay dynamic plugin initialization returned no activation handle")
                self._activation = activation
            except Exception as exc:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes Relay dynamic plugin activation failed") from exc
        if self._activation is None:
            # Reached only after explicit opt-in. Relay 0.8 no longer layers repository-local
            # configuration onto this explicitly selected payload.
            _resolve_plugin_awaitable(relay.plugin.initialize(plugin_config))
        # ⤷ 出口 return：返回 True
        return True

    # ┌── 函数 def release()
    # │  作用：Release one host and clear Relay after the final host exits.
    # │  参数：self, owner: Any
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def release(self, owner: Any) -> None:
        """Release one host and clear Relay after the final host exits."""
        with self._lock:
            if id(owner) in self._owners:
                self._owners.remove(id(owner))
                self.retry_pending_cleanup()

    # ┌── 函数 def reset_for_tests()
    # │  作用：Clear process-global state left by directly constructed test hosts.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def reset_for_tests(self) -> None:
        """Clear process-global state left by directly constructed test hosts."""
        with self._lock:
            self._owners.clear()
            self.retry_pending_cleanup()

    # ┌── 函数 def retry_pending_cleanup()
    # │  作用：Retry a failed final cleanup without disrupting live owners.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def retry_pending_cleanup(self) -> None:
        """Retry a failed final cleanup without disrupting live owners."""
        with self._lock:
            if not self._owners and self._clear_active():
                self._state = _RelayPluginConfigurationState.UNINITIALIZED

    # ┌── 函数 def _clear_active()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _clear_active(self) -> bool:
        relay, activation = self._relay, self._activation
        if relay is None:
            # ⤷ 出口 return：返回 True
            return True

        def close_configuration() -> Any:
            if activation is None:
                # ⤷ 出口 return：返回 _resolve_plugin_awaitable(relay.plugin.clear_async())
                return _resolve_plugin_awaitable(relay.plugin.clear_async())
            if callable(close := getattr(activation, "close", None)):
                # ⤷ 出口 return：返回 _resolve_plugin_awaitable(close())
                return _resolve_plugin_awaitable(close())
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise RuntimeError("NeMo Relay dynamic plugin activation has no close method")

        for what, step in (
            ("subscriber flush", lambda: _resolve_plugin_awaitable(relay.subscribers.flush_async())),
            ("configuration cleanup", close_configuration),
        ):
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                step()
            except Exception:
                logger.warning("Hermes Relay plugin %s failed", what, exc_info=True)
                return False
        self._relay = self._activation = None
        # ⤷ 出口 return：返回 True
        return True


_PLUGIN_CONFIGURATION = _ProcessRelayPluginConfiguration()
atexit.register(_PLUGIN_CONFIGURATION.retry_pending_cleanup)


# ┌── 类 class RelayRuntime
# │  说明：Own Relay session scopes and optional process plugin configuration.
# │  成员：共 28 个方法；对外可用 14 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class RelayRuntime:
    """Own Relay session scopes and optional process plugin configuration."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, relay: Any = None, profile_key: str | None = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock, self._execution_consumers_lock = threading.RLock(), threading.RLock()
        self._sessions: dict[str, RelaySession] = {}
        self._subagent_parents: dict[str, str] = {}
        self._subagent_parent_handles: dict[str, Any] = {}
        self._execution_consumers: set[str] = set()
        self._closing = self._shutdown_started = False
        self._shutdown_complete, self._operations_idle = threading.Event(), threading.Event()
        self._operations_idle.set()
        self._active_operations = 0
        self._plugin_configuration_state = _PLUGIN_CONFIGURATION.acquire(self, self.relay)
        # Cleared (with the atexit hook) by the first successful _finish_shutdown.
        self._plugin_configuration_registered = True
        if self._plugins_active():
            self.retain_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
        atexit.register(self.shutdown)

    # ┌── 函数 def _plugins_active()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _plugins_active(self) -> bool:
        # ⤷ 出口 return：返回 self._plugin_configuration_state is _RelayPluginConfiguratio…
        return self._plugin_configuration_state is _RelayPluginConfigurationState.ACTIVE

    # ┌── 函数 def retain_managed_execution()
    # │  作用：Keep managed LLM and tool execution active for one consumer.
    # │  参数：self, consumer: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def retain_managed_execution(self, consumer: str) -> None:
        """Keep managed LLM and tool execution active for one consumer."""
        if not consumer:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("Relay managed-execution consumer must not be empty")
        with self._execution_consumers_lock:
            self._execution_consumers.add(consumer)

    # ┌── 函数 def release_managed_execution()
    # │  作用：Release a consumer's managed-execution requirement.
    # │  参数：self, consumer: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def release_managed_execution(self, consumer: str) -> None:
        """Release a consumer's managed-execution requirement."""
        with self._execution_consumers_lock:
            self._execution_consumers.discard(consumer)

    # ┌── 函数 def managed_execution_enabled()
    # │  作用：Return whether a Hermes-managed consumer needs the Relay pipeline.
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def managed_execution_enabled(self) -> bool:
        """Return whether a Hermes-managed consumer needs the Relay pipeline."""
        with self._execution_consumers_lock:
            # ⤷ 出口 return：返回 bool(self._execution_consumers)
            return bool(self._execution_consumers)

    # ┌── 函数 def _open_session_scope()
    # │  作用：Push a fresh SESSION_SCOPE for ``session`` (bounded by ``_SCOPE_OP_TIMEOUT``); record handle + context.
    # │  参数：self, session: RelaySession, scope_metadata: dict[str, Any], resolve_parent: bool, exit_fallback: bool = ..., **push_kwargs
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _open_session_scope(
        self, session: RelaySession, scope_metadata: dict[str, Any], *, resolve_parent: bool,
        exit_fallback: bool = False, **push_kwargs: Any,
    ) -> None:
        """Push a fresh SESSION_SCOPE for ``session`` (bounded by ``_SCOPE_OP_TIMEOUT``); record handle + context.
        Subagents parent under their spawning turn/session handle (``resolve_parent`` creates the parent when
        unknown). ``exit_fallback``: at interpreter shutdown the executor refuses futures; push synchronously."""
        parent_handle = None
        if session.parent_session_id:
            with self._sessions_lock:
                parent_handle = self._subagent_parent_handles.get(session.session_id)
            if parent_handle is None and resolve_parent:
                parent = self.ensure_session({"session_id": session.parent_session_id})
                if parent is not None:
                    parent_handle = parent.handle
            scope_metadata["nemo_relay_scope_role"] = "subagent"
        context = contextvars.Context()
        args = (self.relay.scope.push, SESSION_SCOPE, self.relay.ScopeType.Agent)
        push_kwargs.update(handle=parent_handle, metadata=scope_metadata, input={})
        # ⤷ 关键分支 try/except：捕获 RuntimeError，出错走兜底而不是崩溃。
        try:
            future = _scope_op_executor().submit(context.run, *args, **push_kwargs)
            session.handle = future.result(timeout=_SCOPE_OP_TIMEOUT)
        except RuntimeError:
            if not exit_fallback:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise
            session.handle = context.run(*args, **push_kwargs)
        session.context = context

    # ┌── 函数 def ensure_session()
    # │  作用：Return the existing session scope or create it once.
    # │  参数：self, event: dict[str, Any], data: Any = ..., metadata: dict[str, Any] | None = ...
    # │  返回：RelaySession | None
    # └──────────────────────────────────────────────────────────────
    def ensure_session(
        self, event: dict[str, Any], *, data: Any = None, metadata: dict[str, Any] | None = None
    ) -> RelaySession | None:
        """Return the existing session scope or create it once."""
        session_id = _session_id(event)
        if not session_id:
            # ⤷ 出口 return：返回 None
            return None
        with self._sessions_lock:
            if self._closing:
                # ⤷ 出口 return：返回 None
                return None
            session = self._sessions.get(session_id)
            if session is None:
                session = self._sessions[session_id] = RelaySession(
                    session_id=session_id, parent_session_id=self._subagent_parents.get(session_id, "")
                )
        with session.lock:
            if session.closing:
                # ⤷ 出口 return：返回 None
                return None
            if session.handle is None:
                self._open_session_scope(
                    session, {**(metadata or {}), **runtime_metadata(self.runtime_id)},
                    resolve_parent=True, data=data, exit_fallback=True,
                )
        # ⤷ 出口 return：返回 session
        return session

    # ┌── 函数 def rotate_session_scope()
    # │  作用：Close the current session scope and open the next segment (turn boundary ONLY: LIFO).
    # │  参数：self, session: RelaySession, reason: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def rotate_session_scope(self, session: RelaySession, *, reason: str) -> None:
        """Close the current session scope and open the next segment (turn boundary ONLY: LIFO).
        Bookkeeping advances even when a native call fails so a degraded rotation cannot retry every turn."""
        with session.lock:
            if session.closing or session.handle is None:
                return
            old_handle = session.handle
            # Bookkeeping FIRST: a failed native call must not leave rotate_pending set.
            session.segment += 1
            session.segment_turns = 0
            session.rotate_pending = False
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                self.run_in_session(
                    session, self.relay.scope.pop, old_handle, output={"hermes.session.segment_reason": reason},
                    metadata=runtime_metadata(self.runtime_id), timeout=_SCOPE_OP_TIMEOUT,
                )
            except Exception:
                logger.warning(
                    "Hermes Relay segment close failed (session=%s segment=%d); abandoning the old segment span",
                    session.session_id, session.segment - 1, exc_info=True,
                )
            scope_metadata = runtime_metadata(
                self.runtime_id, **{"hermes.session.segment": session.segment, "hermes.session.segment_reason": reason},
            )
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                self._open_session_scope(session, scope_metadata, resolve_parent=False)
            except Exception:
                logger.warning(
                    "Hermes Relay segment open failed (session=%s segment=%d); keeping the prior scope handle",
                    session.session_id, session.segment, exc_info=True,
                )

    # ┌── 函数 def register_subagent()
    # │  作用：Open a child Agent scope under its spawning turn when available.
    # │  参数：self, event: dict[str, Any], metadata: dict[str, Any] | None = ...
    # │  返回：RelaySession | None
    # └──────────────────────────────────────────────────────────────
    def register_subagent(
        self, event: dict[str, Any], *, metadata: dict[str, Any] | None = None
    ) -> RelaySession | None:
        """Open a child Agent scope under its spawning turn when available."""
        parent_session_id = str(event.get("parent_session_id") or "")
        child_session_id = str(event.get("child_session_id") or "")
        if not parent_session_id or not child_session_id or parent_session_id == child_session_id:
            # ⤷ 出口 return：返回 None
            return None
        parent = self.ensure_session({"session_id": parent_session_id})
        parent_handle = None if parent is None else parent.handle
        turn = active_turn(parent_session_id)  # already proved liveness and (RelayRuntime host) an open session
        if turn is not None and turn.handle is not None and turn.lease.host is self:
            parent_handle = turn.handle
        with self._sessions_lock:
            if self._closing:
                # ⤷ 出口 return：返回 None
                return None
            self._subagent_parents[child_session_id] = parent_session_id
            if parent_handle is not None:
                self._subagent_parent_handles[child_session_id] = parent_handle
        # ⤷ 出口 return：返回 self.ensure_session({'session_id': child_session_id}, metada…
        return self.ensure_session({"session_id": child_session_id}, metadata=metadata)

    # ┌── 函数 def unregister_subagent()
    # │  作用：Close a delegated session and forget its parent relationship.
    # │  参数：self, event: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def unregister_subagent(self, event: dict[str, Any]) -> None:
        """Close a delegated session and forget its parent relationship."""
        child_session_id = str(event.get("child_session_id") or "")
        if child_session_id:
            self.close_session({"session_id": child_session_id})
            self._forget_subagent(child_session_id)

    # ┌── 函数 def _forget_subagent()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, session_id: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _forget_subagent(self, session_id: str) -> None:
        with self._sessions_lock:
            self._subagent_parents.pop(session_id, None)
            self._subagent_parent_handles.pop(session_id, None)

    # ┌── 函数 def _lookup()
    # │  作用：Registry lookup (closing sessions included) without creating one.
    # │  参数：self, session_id: str
    # │  返回：RelaySession | None
    # └──────────────────────────────────────────────────────────────
    def _lookup(self, session_id: str) -> RelaySession | None:
        """Registry lookup (closing sessions included) without creating one."""
        with self._sessions_lock:
            # ⤷ 出口 return：返回 self._sessions.get(session_id)
            return self._sessions.get(session_id)

    # ┌── 函数 def get_session()
    # │  作用：Return an active Hermes Relay session without creating one.
    # │  参数：self, session_id: str
    # │  返回：RelaySession | None
    # └──────────────────────────────────────────────────────────────
    def get_session(self, session_id: str) -> RelaySession | None:
        """Return an active Hermes Relay session without creating one."""
        with self._sessions_lock:
            session = None if self._closing else self._sessions.get(str(session_id or ""))
        if session is None:
            # ⤷ 出口 return：返回 None
            return None
        with session.lock:
            # ⤷ 出口 return：返回 None if session.closing else session
            return None if session.closing else session

    # ┌── 函数 def _session_context()
    # │  作用：Copy the current context and overlay the session's saved Relay vars (a copy: re-entrant from callbacks).
    # │  参数：self, session: RelaySession, allow_closing: bool
    # │  返回：contextvars.Context
    # └──────────────────────────────────────────────────────────────
    def _session_context(self, session: RelaySession, *, allow_closing: bool) -> contextvars.Context:
        """Copy the current context and overlay the session's saved Relay vars (a copy: re-entrant from callbacks)."""
        with session.lock:
            if session.closing and not allow_closing:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes Relay session is closing")
            if session.context is None or session.handle is None:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes Relay session context is unavailable")
            relay_context = session.context.copy()
        context = contextvars.copy_context()
        for variable, value in relay_context.items():
            context.run(variable.set, value)
        # ⤷ 出口 return：返回 context
        return context

    # ┌── 函数 def run_in_session()
    # │  作用：Run a Relay operation against a session's isolated scope stack.
    # │  参数：self, session: RelaySession, callback: Callable[..., Any], *args, allow_closing: bool = ..., timeout: float | None = ..., **kwargs
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def run_in_session(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, timeout: float | None = None, **kwargs: Any,
    ) -> Any:
        """Run a Relay operation against a session's isolated scope stack.
        ``timeout`` bounds the native call on the daemon executor (``TimeoutError`` on breach); ``None``
        runs synchronously. Lifecycle ops pass ``_SCOPE_OP_TIMEOUT``: a wedged pipeline costs one span."""
        with self._operation():
            # ⤷ 出口 return：返回 self._run_in_session_untracked(session, callback, *args, all…
            return self._run_in_session_untracked(
                session, callback, *args, allow_closing=allow_closing, timeout=timeout, **kwargs
            )

    # ┌── 函数 def _run_in_session_untracked()
    # │  作用：Run inside a session whose host-level lifetime is already held.
    # │  参数：self, session: RelaySession, callback: Callable[..., Any], *args, allow_closing: bool = ..., timeout: float | None = ..., **kwargs
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def _run_in_session_untracked(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, timeout: float | None = None, **kwargs: Any,
    ) -> Any:
        """Run inside a session whose host-level lifetime is already held."""
        context = self._session_context(session, allow_closing=allow_closing)

        def invoke() -> Any:
            self.relay.get_scope_stack()
            # ⤷ 出口 return：返回 callback(*args, **kwargs)
            return callback(*args, **kwargs)

        if timeout is None:
            # ⤷ 出口 return：返回 context.run(invoke)
            return context.run(invoke)
        exceeded = f"Relay scope operation exceeded {timeout}s"
        # ⤷ 关键分支 try/except：捕获 RuntimeError，出错走兜底而不是崩溃。
        try:
            future = _scope_op_executor().submit(context.run, invoke)
        except RuntimeError:
            # Interpreter shutdown: the executor refuses futures but the atexit close path must
            # still flush — still bounded so a wedged call cannot block exit.
            # ⤷ 出口 return：返回 _run_on_daemon_thread(lambda: context.run(invoke), name='rel…
            return _run_on_daemon_thread(
                lambda: context.run(invoke), name="relay-scope-op-exit", timeout=timeout,
                timeout_message=f"{exceeded} during interpreter shutdown; abandoning the native "
                "call so process exit can proceed",
            )
        # ⤷ 关键分支 try/except：捕获 FuturesTimeoutError，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 future.result(timeout=timeout)
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"{exceeded} (session={session.session_id}); abandoning the native call "
                "so the agent can continue — the span for this scope is lost"
            ) from exc

    # ┌── 异步函数（协程） def run_in_session_async()
    # │  作用：Create and await an operation inside the session's saved context.
    # │  参数：self, session: RelaySession, callback: Callable[..., Any], *args, allow_closing: bool = ..., **kwargs
    # │  返回：Any
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def run_in_session_async(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, **kwargs: Any,
    ) -> Any:
        """Create and await an operation inside the session's saved context."""
        with self._operation():
            context = self._session_context(session, allow_closing=allow_closing)

            async def invoke() -> Any:
                self.relay.get_scope_stack()
                result = callback(*args, **kwargs)
                # ⤷ 出口 return：返回 await result if inspect.isawaitable(result) else result
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                return await result if inspect.isawaitable(result) else result

            # ⤷ 出口 return：返回 await context.run(asyncio.create_task, invoke())
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            return await context.run(asyncio.create_task, invoke())

    # ┌── 函数 def _begin_operation()
    # │  作用：Admit one Relay call while keeping process plugins alive.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _begin_operation(self) -> None:
        """Admit one Relay call while keeping process plugins alive."""
        with self._sessions_lock:
            if self._closing:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise RuntimeError("Hermes Relay runtime is shutting down")
            self._active_operations += 1
            self._operations_idle.clear()

    # ┌── 函数 def _end_operation()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _end_operation(self) -> None:
        with self._sessions_lock:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._operations_idle.set()

    # ┌── 函数 def _operation()
    # │  作用：``_begin_operation`` / ``_end_operation`` around one tracked Relay call.
    # │  参数：self
    # │  返回：（未标注类型）
    # │  【语法】@contextmanager 把「有 yield 的函数」变成上下文管理器，调用方可以用 `with xxx() as v:` 自动做进入/清理。yield 之前是准备，yield 之后是收尾。
    # └──────────────────────────────────────────────────────────────
    @contextlib.contextmanager
    def _operation(self):
        """``_begin_operation`` / ``_end_operation`` around one tracked Relay call."""
        self._begin_operation()
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield
        finally:
            self._end_operation()

    # ┌── 函数 def acquire_operation_lease()
    # │  作用：Retain plugin lifetime for work that outlives one Relay await.
    # │  参数：self
    # │  返回：RelayOperationLease
    # └──────────────────────────────────────────────────────────────
    def acquire_operation_lease(self) -> RelayOperationLease:
        """Retain plugin lifetime for work that outlives one Relay await."""
        self._begin_operation()
        # ⤷ 出口 return：返回 RelayOperationLease(self)
        return RelayOperationLease(self)

    # ┌── 函数 def apply_tool_request_intercepts()
    # │  作用：Apply Relay request rewriting before Hermes authorizes a tool call.
    # │  参数：self, session_id: str, tool_name: str, args: dict[str, Any]
    # │  返回：dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def apply_tool_request_intercepts(self, *, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Apply Relay request rewriting before Hermes authorizes a tool call."""
        request_intercepts = getattr(getattr(self.relay, "tools", None), "request_intercepts", None)
        managed = self.managed_execution_enabled() and callable(request_intercepts)
        session = self.ensure_session({"session_id": session_id}) if managed else None
        if session is None:
            # ⤷ 出口 return：返回 args
            return args
        result = self.run_in_session(session, request_intercepts, tool_name, args)
        # ⤷ 出口 return：返回 result if isinstance(result, dict) else args
        return result if isinstance(result, dict) else args

    # ┌── 函数 def _pop_with_drain()
    # │  作用：Pop ``handle``; if that fails, drain orphans above it and retry once; return the retry's error.
    # │  参数：self, handle: Any, output: dict[str, Any], metadata: dict[str, Any], session_root: Any, drain_limit: int
    # │  返回：BaseException | None
    # └──────────────────────────────────────────────────────────────
    def _pop_with_drain(
        self, handle: Any, *, output: dict[str, Any], metadata: dict[str, Any], session_root: Any, drain_limit: int,
    ) -> BaseException | None:
        """Pop ``handle``; if that fails, drain orphans above it and retry once; return the retry's error.
        Must run inside ONE ``run_in_session`` callback so ContextVar stack views stay consistent."""
        with contextlib.suppress(Exception):
            pop_relay_scope(self.relay, handle, output=output, metadata=metadata)
            # ⤷ 出口 return：返回 None
            return None
        drained = 0
        for _ in range(drain_limit):
            top = _current_top(self.relay)
            if top is None or _same_handle(top, handle):
                break
            # Never pop the session root while draining for a nested handle.
            if session_root is not None and _same_handle(top, session_root) and handle is not session_root:
                break
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                orphan_output = {"outcome": "cancelled", "hermes.orphan_drain": True}
                pop_relay_scope(self.relay, top, output=orphan_output, metadata=metadata)
                drained += 1
            except Exception:
                logger.warning("Hermes Relay orphaned scope drain failed", exc_info=True)
                break
        if drained:
            logger.warning("Hermes Relay drained %d orphaned scope(s) before closing %s", drained, handle)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            pop_relay_scope(self.relay, handle, output=output, metadata=metadata)
            # ⤷ 出口 return：返回 None
            return None
        except Exception as retry_exc:
            # ⤷ 出口 return：返回 retry_exc
            return retry_exc

    # ┌── 函数 def _close_scope_handle()
    # │  作用：Pop ``handle``, draining orphaned children in the same session context; failure string or None.
    # │  参数：self, session: RelaySession, handle: Any, output: dict[str, Any] | None = ..., allow_closing: bool = ..., failure_label: str = ..., drain_limit: int = ..., operation_already_held: bool = ...
    # │  返回：str | None
    # └──────────────────────────────────────────────────────────────
    def _close_scope_handle(
        self, session: RelaySession, handle: Any, *, output: dict[str, Any] | None = None, allow_closing: bool = False,
        failure_label: str = "scope close failed", drain_limit: int = 32, operation_already_held: bool = False,
    ) -> str | None:
        """Pop ``handle``, draining orphaned children in the same session context; failure string or None.
        Relay scopes are strict LIFO; empty-stream retries + interrupt can abandon a physical LLM scope
        above TURN/SESSION. Drain+close is bounded so a wedged pipeline never blocks completion."""
        if handle is None:
            # ⤷ 出口 return：返回 None
            return None
        run_in_session = (self._run_in_session_untracked if operation_already_held else self.run_in_session)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            failure = run_in_session(
                session, self._pop_with_drain, handle, output=output or {},
                metadata=runtime_metadata(self.runtime_id), session_root=session.handle,
                drain_limit=drain_limit, allow_closing=allow_closing, timeout=_SCOPE_OP_TIMEOUT,
            )
        except Exception as exc:
            failure = exc
        # ⤷ 出口 return：返回 None if failure is None else f'{failure_label}: {failure}'
        return None if failure is None else f"{failure_label}: {failure}"

    # ┌── 函数 def close_session()
    # │  作用：Close one session scope and remove it from the core registry (no-op once shutting down).
    # │  参数：self, event: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def close_session(self, event: dict[str, Any]) -> None:
        """Close one session scope and remove it from the core registry (no-op once shutting down)."""
        with contextlib.suppress(RuntimeError), self._operation():  # _close_session itself never raises
            self._close_session(event)

    # ┌── 函数 def _close_session()
    # │  作用：Close one session already admitted by the host lifecycle gate.
    # │  参数：self, event: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _close_session(self, event: dict[str, Any]) -> None:
        """Close one session already admitted by the host lifecycle gate."""
        session_id = _session_id(event)
        session = self._lookup(session_id)
        if session is None:
            self._forget_subagent(session_id)
            return
        failure = None
        with session.lock:
            if session.closing:
                return
            session.closing = True
            if session.handle is not None:
                failure = self._close_scope_handle(
                    session, session.handle, output={}, allow_closing=True,
                    failure_label="session scope close failed", operation_already_held=True,
                )
        # No subscriber flush here: process-wide, may wait on other sessions and deadlock an asyncio
        # loop; final plugin teardown flushes once.
        with self._sessions_lock:
            if self._sessions.get(session_id) is session:
                del self._sessions[session_id]
            self._forget_subagent(session_id)
        if failure:
            logger.warning("Hermes Relay session %s closed with errors: %s", session_id, failure)

    # ┌── 函数 def shutdown()
    # │  作用：Close core scopes and release process plugin configuration.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        """Close core scopes and release process plugin configuration."""
        with self._sessions_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = self._closing = True
            has_active_operations = self._active_operations > 0
        if not has_active_operations:
            self._finish_shutdown()
            return
        thread = threading.Thread(
            target=lambda: (self._operations_idle.wait(), self._finish_shutdown()),
            name=f"hermes-nemo-relay-shutdown-{self.runtime_id[:8]}", daemon=True,
        )
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            thread.start()
        except Exception:
            with self._sessions_lock:
                self._shutdown_started = False
            logger.warning("Hermes Relay deferred shutdown could not start", exc_info=True)

    # ┌── 函数 def _finish_shutdown()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _finish_shutdown(self) -> None:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            with self._sessions_lock:
                session_ids = list(self._sessions)
            for session_id in session_ids:
                _warn_on_error("runtime operation", self._close_session, {"session_id": session_id})
            if self._plugin_configuration_registered:
                if self._plugins_active():
                    self.release_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
                _PLUGIN_CONFIGURATION.release(self)
                self._plugin_configuration_registered = False
                with contextlib.suppress(Exception):
                    atexit.unregister(self.shutdown)
        except Exception:
            with self._sessions_lock:
                self._shutdown_started = False
            logger.warning("Hermes Relay shutdown failed", exc_info=True)
            return
        with self._sessions_lock:
            self._shutdown_complete.set()


# ┌── 类 class NoopRelayRuntime
# │  说明：Explicit reduced-capability host for platforms without Relay wheels.
# │  成员：共 2 个方法；对外可用 2 个
# └──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NoopRelayRuntime:
    """Explicit reduced-capability host for platforms without Relay wheels."""

    profile_key: str
    reason: str

    # ┌── 函数 def apply_tool_request_intercepts()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, session_id: str, tool_name: str, args: dict[str, Any]
    # │  返回：dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def apply_tool_request_intercepts(self, *, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        # ⤷ 出口 return：返回 args
        return args

    # ┌── 函数 def retain_managed_execution()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：consumer: str
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def retain_managed_execution(consumer: str) -> None:
        pass

    release_managed_execution = retain_managed_execution
    managed_execution_enabled = staticmethod(lambda: False)
    shutdown = staticmethod(lambda: None)  # no resources are allocated on unsupported platforms


RelayHost = RelayRuntime | NoopRelayRuntime


# ┌── 类 class RelayHostRegistry
# │  说明：Own exactly one Relay host for each canonical Hermes profile.
# │  成员：共 3 个方法；对外可用 2 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class RelayHostRegistry:
    """Own exactly one Relay host for each canonical Hermes profile."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hosts: dict[str, RelayHost] = {}

    # ┌── 函数 def for_profile()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, profile_key: str | None = None, create: bool = ...
    # │  返回：RelayHost | None
    # └──────────────────────────────────────────────────────────────
    def for_profile(self, profile_key: str | None = None, *, create: bool = True) -> RelayHost | None:
        key = profile_key or current_profile_key()
        with self._lock:
            host = self._hosts.get(key)
            if host is not None or not create:
                # ⤷ 出口 return：返回 host
                return host
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                host = RelayRuntime(profile_key=key)
            except Exception as exc:
                logger.warning("Hermes Relay runtime initialization failed", exc_info=True)
                host = NoopRelayRuntime(profile_key=key, reason=str(exc))
            self._hosts[key] = host
            # ⤷ 出口 return：返回 host
            return host

    # ┌── 函数 def shutdown_all()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def shutdown_all(self) -> None:
        with self._lock:
            hosts, self._hosts = list(self._hosts.values()), {}
        for host in hosts:
            host.shutdown()


HOST_REGISTRY = RelayHostRegistry()


# ┌── 类 class ConversationLease
# │  说明：A resumable reference to one profile-scoped conversation scope.
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class ConversationLease:
    """A resumable reference to one profile-scoped conversation scope."""

    profile_key: str
    session_id: str
    platform: str
    host: RelayHost
    session: RelaySession | None
    parent_session_id: str = ""
    released: bool = False

    # ┌── 函数 def live_runtime()
    # │  作用：Return the real Relay host when this lease owns an open session.
    # │  参数：self
    # │  返回：RelayRuntime | None
    # └──────────────────────────────────────────────────────────────
    def live_runtime(self) -> RelayRuntime | None:
        """Return the real Relay host when this lease owns an open session."""
        # ⤷ 出口 return：返回 self.host if isinstance(self.host, RelayRuntime) and self.se…
        return self.host if isinstance(self.host, RelayRuntime) and self.session is not None else None


# ┌── 类 class RelayTurnContext
# │  说明：Runtime-only context for one Hermes turn or top-level task.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class RelayTurnContext:
    """Runtime-only context for one Hermes turn or top-level task."""

    lease: ConversationLease
    turn_id: str
    task_id: str
    handle: Any = None
    logical_llm_calls: dict[str, Any] = field(default_factory=dict, repr=False)
    logical_llm_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    finalize_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _previous_turn: RelayTurnContext | None = field(default=None, repr=False)
    _active_registered: bool = field(default=False, repr=False)
    relay_enabled: bool = True
    closed: bool = False


_CURRENT_TURN: contextvars.ContextVar[RelayTurnContext | None] = contextvars.ContextVar(
    "hermes_relay_turn", default=None
)

# >0 while the native pipeline is mid-dispatch of a Hermes tool/LLM callback. Nested managed
# execution there is structurally broken (the pipeline binds its Futures to the OUTER call's
# loop, blocked inside the synchronous callback), so resolve_execution_context() bypasses Relay.
# A ContextVar so the marker follows copy_context() into worker threads / per-thread loops.
_MANAGED_CALLBACK_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "hermes_relay_managed_callback_depth", default=0
)


# ┌── 函数 def managed_callback_guard()
# │  作用：Mark the current context as inside a managed Relay callback: everything the wrapped ``invoke()``
# │  参数：无
# │  返回：（未标注类型）
# │  【语法】@contextmanager 把「有 yield 的函数」变成上下文管理器，调用方可以用 `with xxx() as v:` 自动做进入/清理。yield 之前是准备，yield 之后是收尾。
# └──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def managed_callback_guard():
    """Mark the current context as inside a managed Relay callback: everything the wrapped ``invoke()``
    transitively calls (incl. work forwarded via copy_context()) runs unmanaged."""
    token = _MANAGED_CALLBACK_DEPTH.set(_MANAGED_CALLBACK_DEPTH.get() + 1)
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
        yield
    finally:
        _MANAGED_CALLBACK_DEPTH.reset(token)


# ┌── 函数 def _warn_on_error()
# │  作用：Run fail-open telemetry work: log ``Hermes Relay <what> failed`` and return None on error.
# │  参数：what: str, callback: Callable[..., Any], *args, **kwargs
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _warn_on_error(what: str, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run fail-open telemetry work: log ``Hermes Relay <what> failed`` and return None on error."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 callback(*args, **kwargs)
        return callback(*args, **kwargs)
    except Exception:
        logger.warning("Hermes Relay %s failed", what, exc_info=True)
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _fail_open()
# │  作用：Decorator form of ``_warn_on_error`` for telemetry hooks that must never block the caller.
# │  参数：what: str
# │  返回：Callable[[Callable[..., Any]], Callable[..., Any]]
# └──────────────────────────────────────────────────────────────
def _fail_open(what: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of ``_warn_on_error`` for telemetry hooks that must never block the caller."""
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            # ⤷ 出口 return：返回 _warn_on_error(what, fn, *args, **kwargs)
            return _warn_on_error(what, fn, *args, **kwargs)
        # ⤷ 出口 return：返回 guarded
        return guarded
    # ⤷ 出口 return：返回 wrap
    return wrap


# ┌── 函数 def _flag_open_session()
# │  作用：Set a pending-rotation/close flag unless the session is already closing.
# │  参数：session: RelaySession, flag: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _flag_open_session(session: RelaySession, flag: str) -> None:
    """Set a pending-rotation/close flag unless the session is already closing."""
    with session.lock:
        if not session.closing:
            setattr(session, flag, True)


# ┌── 类 class RelaySessionCoordinator
# │  说明：Own semantic conversation and turn lifetimes for Hermes core.
# │  成员：共 18 个方法；对外可用 9 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class RelaySessionCoordinator:
    """Own semantic conversation and turn lifetimes for Hermes core."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, registry: RelayHostRegistry = HOST_REGISTRY
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self, registry: RelayHostRegistry = HOST_REGISTRY) -> None:
        self.registry = registry
        self._initializer_lock = threading.RLock()
        self._session_initializers: dict[str, Callable[[RelayRuntime, dict[str, Any]], None]] = {}
        self._active_turns_lock = threading.RLock()
        self._active_turns: dict[tuple[str, str], set[int]] = {}

    # ┌── 函数 def register_session_initializer()
    # │  作用：Register idempotent profile/session preparation before scope creation.
    # │  参数：self, name: str, callback: Callable[[RelayRuntime, dict[str, Any]], None]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def register_session_initializer(self, name: str, callback: Callable[[RelayRuntime, dict[str, Any]], None]) -> None:
        """Register idempotent profile/session preparation before scope creation."""
        with self._initializer_lock:
            self._session_initializers[name] = callback

    # ┌── 函数 def _prepare_session()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, host: RelayRuntime, context: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _prepare_session(self, host: RelayRuntime, context: dict[str, Any]) -> None:
        with self._initializer_lock:
            initializers = list(self._session_initializers.items())
        for name, callback in initializers:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                callback(host, context)
            except Exception:
                logger.warning("Hermes Relay session initializer failed: %s", name, exc_info=True)

    # ┌── 函数 def acquire_conversation()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, profile_key: str, session_id: str, platform: str, parent_session_id: str = ..., model: str = ...
    # │  返回：ConversationLease
    # └──────────────────────────────────────────────────────────────
    def acquire_conversation(
        self, *, profile_key: str, session_id: str, platform: str, parent_session_id: str = "", model: str = "",
    ) -> ConversationLease:
        host = self.registry.for_profile(profile_key) or NoopRelayRuntime(profile_key, "Relay host creation was disabled")
        session = None
        if isinstance(host, RelayRuntime):
            context = {
                "profile_key": profile_key, "session_id": session_id, "platform": platform,
                "parent_session_id": parent_session_id, "model": model,
            }
            session = _warn_on_error("conversation initialization", self._open_conversation_session, host, context)
        # ⤷ 出口 return：返回 ConversationLease(profile_key=profile_key, session_id=sessio…
        return ConversationLease(
            profile_key=profile_key, session_id=session_id, platform=platform, host=host,
            session=session, parent_session_id=parent_session_id,
        )

    # ┌── 函数 def _open_conversation_session()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, host: RelayRuntime, context: dict[str, Any]
    # │  返回：RelaySession | None
    # └──────────────────────────────────────────────────────────────
    def _open_conversation_session(self, host: RelayRuntime, context: dict[str, Any]) -> RelaySession | None:
        self._prepare_session(host, context)
        session_id, parent_session_id = context["session_id"], context["parent_session_id"]
        metadata = {"hermes.execution_surface": context["platform"] or "unknown"}
        if parent_session_id and parent_session_id != session_id:
            event = {"parent_session_id": parent_session_id, "child_session_id": session_id}
            # ⤷ 出口 return：返回 host.register_subagent(event, metadata=metadata)
            return host.register_subagent(event, metadata=metadata)
        # ⤷ 出口 return：返回 host.ensure_session({'session_id': session_id}, metadata=met…
        return host.ensure_session({"session_id": session_id}, metadata=metadata)

    # ┌── 函数 def begin_turn()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, lease: ConversationLease, turn_id: str, task_id: str, metadata: dict[str, Any] | None = ...
    # │  返回：RelayTurnContext
    # └──────────────────────────────────────────────────────────────
    def begin_turn(
        self,
        lease: ConversationLease,
        *,
        turn_id: str,
        task_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> RelayTurnContext:
        if lease.released:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise RuntimeError("Hermes Relay conversation lease is released")
        turn = RelayTurnContext(lease=lease, turn_id=turn_id, task_id=task_id)
        key = (lease.profile_key, lease.session_id)
        with self._active_turns_lock:
            if self._active_turns.get(key):
                # One physical scope stack per session; concurrent turns' sibling scopes would not close LIFO.
                turn.relay_enabled = False
                logger.warning(
                    "Skipping Relay instrumentation for concurrent Hermes turn %s in session %s",
                    turn_id, lease.session_id,
                )
            else:
                self._active_turns[key] = {id(turn)}
                turn._active_registered = True
        host = lease.live_runtime() if turn.relay_enabled else None
        if host is not None:
            # Rotation happens HERE: no live turn scope on the stack, so the session scope can close/reopen LIFO.
            _warn_on_error("segment rotation", self._maybe_rotate_segment, host, lease.session)
            turn_metadata = dict(metadata or {})
            turn_metadata.update(
                runtime_metadata(
                    host.runtime_id,
                    **{"hermes.execution_surface": lease.platform or "unknown"},
                )
            )
            turn.handle = _warn_on_error(
                "turn initialization", host.run_in_session, lease.session, host.relay.scope.push,
                TURN_SCOPE, host.relay.ScopeType.Function, handle=lease.session.handle, input={},
                metadata=turn_metadata,
                timeout=_SCOPE_OP_TIMEOUT,
            )
        turn._previous_turn = _CURRENT_TURN.get()
        _CURRENT_TURN.set(turn)
        # ⤷ 出口 return：返回 turn
        return turn

    # ┌── 函数 def _maybe_rotate_segment()
    # │  作用：Rotate the session scope when compaction flagged it or the turn cap is hit.
    # │  参数：host: RelayRuntime, session: RelaySession
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _maybe_rotate_segment(host: RelayRuntime, session: RelaySession) -> None:
        """Rotate the session scope when compaction flagged it or the turn cap is hit."""
        config = _segments_config()
        cap = config["max_turns"]
        compaction = config["on_compaction"] and session.rotate_pending
        if compaction or (cap > 0 and session.segment_turns >= cap):
            host.rotate_session_scope(session, reason="compaction" if compaction else "max_turns")

    # ┌── 函数 def end_turn()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, turn: RelayTurnContext, outcome: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def end_turn(self, turn: RelayTurnContext, *, outcome: str) -> None:
        with turn.finalize_lock:
            if turn.closed:
                self._reset_turn_context(turn)
                return
            turn.closed = True
            lease = turn.lease
            host = lease.live_runtime()
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                if host is not None:
                    self._close_turn_scope(host, turn, outcome=outcome)
            finally:
                if turn._active_registered and host is not None:
                    with contextlib.suppress(Exception), lease.session.lock:  # accounting never blocks
                        lease.session.segment_turns += 1  # max_turns rotation trigger
                # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
                try:
                    # Delegated agents own one turn: close their conversation while the active-turn
                    # guard is held so a parent timeout fallback cannot race it.
                    if lease.parent_session_id and isinstance(lease.host, RelayRuntime):
                        _warn_on_error(
                            "child conversation finalization", lease.host.unregister_subagent,
                            {"child_session_id": lease.session_id},
                        )
                finally:
                    self._unregister_active_turn(turn)
                    self._reset_turn_context(turn)
                self._consume_deferred_close(lease)

    # ┌── 函数 def _close_turn_scope()
    # │  作用：Pop the turn's logical LLM children, then the turn scope itself (LIFO).
    # │  参数：self, host: RelayRuntime, turn: RelayTurnContext, outcome: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _close_turn_scope(self, host: RelayRuntime, turn: RelayTurnContext, *, outcome: str) -> None:
        """Pop the turn's logical LLM children, then the turn scope itself (LIFO)."""
        self._finish_logical_calls(turn, outcome=outcome)
        failure = host._close_scope_handle(
            turn.lease.session, turn.handle, output={"outcome": outcome}, failure_label="turn scope close failed",
        )
        if failure:
            logger.warning("Hermes Relay turn finalization failed: %s", failure)

    # ┌── 函数 def _consume_deferred_close()
    # │  作用：Close a session whose rotating-compaction close was deferred (``close_pending``).
    # │  参数：self, lease: ConversationLease
    # │  返回：None
    # │  【语法】带 @_fail_open 装饰器：调用本函数时，等价于 _consume_deferred_close = _fail_open(_consume_deferred_close)，即先经过一层包装。
    # └──────────────────────────────────────────────────────────────
    @_fail_open("deferred session close")
    def _consume_deferred_close(self, lease: ConversationLease) -> None:
        """Close a session whose rotating-compaction close was deferred (``close_pending``).
        The last live turn consumes it after its own scope popped and it left the active-turn table."""
        host = lease.live_runtime()
        if host is None:
            return
        with lease.session.lock:
            pending = lease.session.close_pending and not lease.session.closing
        if pending and not self.has_active_turn(profile_key=lease.profile_key, session_id=lease.session_id):
            host.close_session({"session_id": lease.session_id})

    # ┌── 函数 def notify_session_compacted()
    # │  作用：React to a completed compaction, per compaction mode; unknown sessions / disabled config are no-ops.
    # │  参数：self, profile_key: str, session_id: str, old_session_id: str = ...
    # │  返回：None
    # │  【语法】带 @_fail_open 装饰器：调用本函数时，等价于 notify_session_compacted = _fail_open(notify_session_compacted)，即先经过一层包装。
    # └──────────────────────────────────────────────────────────────
    @_fail_open("compaction notification")
    def notify_session_compacted(self, *, profile_key: str, session_id: str, old_session_id: str = "") -> None:
        """React to a completed compaction, per compaction mode; unknown sessions / disabled config are no-ops.
        In-place (``old_session_id`` empty/equal): flag rotation for the next turn boundary — never rotate
        immediately, a live turn under it breaks LIFO. Rotating (ids differ): the next turn gets a fresh
        session under the new id, so close the OLD session now or its scope stays an unexported orphan."""
        if not _segments_config()["on_compaction"]:
            return
        host = self.registry.for_profile(profile_key)
        if not isinstance(host, RelayRuntime):
            return
        if old_session_id and old_session_id != session_id:
            # A LIVE turn on the old session: closing now would pop under it (LIFO).
            old_session = host._lookup(old_session_id)
            if old_session is not None and self.has_active_turn(profile_key=profile_key, session_id=old_session_id):
                _flag_open_session(old_session, "close_pending")
            else:
                host.close_session({"session_id": old_session_id})
            return
        session = host._lookup(session_id)
        if session is not None:
            _flag_open_session(session, "rotate_pending")

    # ┌── 函数 def has_active_turn()
    # │  作用：Return whether a turn is still running for one profile/session.
    # │  参数：self, profile_key: str, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_active_turn(self, *, profile_key: str, session_id: str) -> bool:
        """Return whether a turn is still running for one profile/session."""
        with self._active_turns_lock:
            # ⤷ 出口 return：返回 bool(self._active_turns.get((profile_key, session_id)))
            return bool(self._active_turns.get((profile_key, session_id)))

    # ┌── 函数 def _unregister_active_turn()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, turn: RelayTurnContext
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _unregister_active_turn(self, turn: RelayTurnContext) -> None:
        if not turn._active_registered:
            return
        key = (turn.lease.profile_key, turn.lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active is not None:
                active.discard(id(turn))
                if not active:
                    del self._active_turns[key]
            turn._active_registered = False

    # ┌── 函数 def finish_logical_calls()
    # │  作用：Close logical LLM children before sibling task aggregation scopes.
    # │  参数：self, turn: RelayTurnContext, outcome: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def finish_logical_calls(self, turn: RelayTurnContext, *, outcome: str) -> None:
        """Close logical LLM children before sibling task aggregation scopes."""
        with turn.finalize_lock:
            if not turn.closed:
                self._finish_logical_calls(turn, outcome=outcome)

    # ┌── 函数 def _finish_logical_calls()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：turn: RelayTurnContext, outcome: str
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _finish_logical_calls(turn: RelayTurnContext, *, outcome: str) -> None:
        lease = turn.lease
        host = lease.live_runtime()
        if host is None:
            return
        with turn.logical_llm_lock:
            logical_calls = list(turn.logical_llm_calls.items())
            turn.logical_llm_calls.clear()
        while logical_calls:
            _request_id, logical_handle = logical_calls[-1]
            failure = host._close_scope_handle(
                lease.session, logical_handle, output={"outcome": outcome},
                failure_label="logical LLM scope close failed",
            )
            if failure is None:
                logical_calls.pop()
                continue
            with turn.logical_llm_lock:
                # Stack-owned: if the newest handle cannot close even after drain, older ones cannot either.
                for pending_request_id, pending_handle in logical_calls:
                    turn.logical_llm_calls.setdefault(pending_request_id, pending_handle)
            logger.warning("Hermes Relay logical LLM finalization failed: %s", failure)
            break

    # ┌── 函数 def _reset_turn_context()
    # │  作用：Unwind ``turn`` without disturbing a newer context-local turn.
    # │  参数：turn: RelayTurnContext
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _reset_turn_context(turn: RelayTurnContext) -> None:
        """Unwind ``turn`` without disturbing a newer context-local turn."""
        if _CURRENT_TURN.get() is not turn:
            return
        previous, seen = turn._previous_turn, {id(turn)}
        while previous is not None and previous.closed and id(previous) not in seen:
            seen.add(id(previous))
            previous = previous._previous_turn
        if previous is not None and previous.closed:  # cycle: no live ancestor
            previous = None
        _CURRENT_TURN.set(previous)

    # ┌── 函数 def release_conversation()
    # │  作用：Release a caller lease without closing a resumable conversation.
    # │  参数：lease: ConversationLease
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def release_conversation(lease: ConversationLease) -> None:
        """Release a caller lease without closing a resumable conversation."""
        lease.released = True

    # ┌── 函数 def finalize_conversation()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, profile_key: str, session_id: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def finalize_conversation(self, *, profile_key: str, session_id: str) -> None:
        host = self.registry.for_profile(profile_key, create=False)
        if isinstance(host, RelayRuntime):
            host.close_session({"session_id": session_id})


SESSION_COORDINATOR = RelaySessionCoordinator()


# ┌── 函数 def current_turn()
# │  作用：Return the turn context inherited by current async and thread work.
# │  参数：无
# │  返回：RelayTurnContext | None
# └──────────────────────────────────────────────────────────────
def current_turn() -> RelayTurnContext | None:
    """Return the turn context inherited by current async and thread work."""
    # ⤷ 出口 return：返回 _CURRENT_TURN.get()
    return _CURRENT_TURN.get()


# ┌── 函数 def relay_instrumentation_enabled()
# │  作用：Return whether this inherited turn may create Relay instrumentation.
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def relay_instrumentation_enabled() -> bool:
    """Return whether this inherited turn may create Relay instrumentation."""
    turn = current_turn()
    # ⤷ 出口 return：返回 turn is None or (turn.relay_enabled and (not turn.closed))
    return turn is None or (turn.relay_enabled and not turn.closed)


# ┌── 函数 def active_turn()
# │  作用：Return a live turn only when it belongs to the active profile/session.
# │  参数：session_id: str | None = None
# │  返回：RelayTurnContext | None
# └──────────────────────────────────────────────────────────────
def active_turn(session_id: str | None = None) -> RelayTurnContext | None:
    """Return a live turn only when it belongs to the active profile/session."""
    turn = current_turn()
    if turn is None or not turn.relay_enabled or turn.closed or turn.lease.released:
        # ⤷ 出口 return：返回 None
        return None
    lease = turn.lease
    if lease.profile_key != current_profile_key() or (session_id is not None and lease.session_id != session_id):
        # ⤷ 出口 return：返回 None
        return None
    if isinstance(lease.host, RelayRuntime) and (
        lease.session is None or lease.host.get_session(lease.session_id) is not lease.session
    ):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 turn
    return turn


# ┌── 函数 def resolve_execution_context()
# │  作用：Resolve one active turn/session parent for managed Relay execution.
# │  参数：session_id: str
# │  返回：tuple[RelayRuntime | None, RelaySession | None, Any]
# └──────────────────────────────────────────────────────────────
def resolve_execution_context(session_id: str) -> tuple[RelayRuntime | None, RelaySession | None, Any]:
    """Resolve one active turn/session parent for managed Relay execution."""
    # Nested managed execution is impossible (see _MANAGED_CALLBACK_DEPTH); the outer scope
    # still records the tool-level event.
    if _MANAGED_CALLBACK_DEPTH.get() > 0 or not relay_instrumentation_enabled():
        # A managed Relay callback is already executing on this logical call path (e.g. the native
        # ``tools.execute`` pipeline is mid-dispatch of a Hermes tool). Nested managed execution here is
        # structurally impossible: the native pipeline binds its Futures to the OUTER call's event loop,
        # which is blocked inside the synchronous tool callback until the tool returns. A nested managed LLM
        # call (the vision_analyze auxiliary path) therefore awaits a foreign-loop Future that can never
        # complete — "attached to a different loop" at best, deadlock at worst, and "Event loop is closed"
        # during shutdown when the orphaned Future is completed late (#77244).
        # ⤷ 出口 return：返回 (None, None, None)
        return None, None, None
    turn = active_turn(session_id)
    host = turn.lease.live_runtime() if turn is not None else None
    if host is not None:
        session = turn.lease.session
        # ⤷ 出口 return：返回 (host, session, turn.handle or session.handle)
        return host, session, turn.handle or session.handle
    # Consumers retain the profile host before reaching an out-of-turn adapter; never
    # initialize Relay for the default no-consumer path.
    runtime = get_runtime(create=False)
    if runtime is None or not runtime.managed_execution_enabled():
        # ⤷ 出口 return：返回 (None, None, None)
        return None, None, None
    session = runtime.get_session(session_id) or runtime.ensure_session({"session_id": session_id})
    # ⤷ 出口 return：返回 (runtime, session, None if session is None else session.hand…
    return runtime, session, None if session is None else session.handle


# ┌── 函数 def apply_tool_request_intercepts()
# │  作用：Return Relay-rewritten arguments at Hermes's authorization boundary.
# │  参数：session_id: str, tool_name: str, args: dict[str, Any]
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def apply_tool_request_intercepts(*, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return Relay-rewritten arguments at Hermes's authorization boundary."""
    runtime = get_runtime(create=False) if session_id else None
    if runtime is None:
        # ⤷ 出口 return：返回 args
        return args
    # ⤷ 出口 return：返回 runtime.apply_tool_request_intercepts(session_id=session_id,…
    return runtime.apply_tool_request_intercepts(session_id=session_id, tool_name=tool_name, args=args)


# ┌── 函数 def _is_relay_wrapped_callback_error()
# │  作用：Match Relay's native callback wrapper without masking policy errors.
# │  参数：relay_error: BaseException, callback_error: BaseException
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_relay_wrapped_callback_error(relay_error: BaseException, callback_error: BaseException) -> bool:
    """Match Relay's native callback wrapper without masking policy errors."""
    if relay_error is callback_error:
        # ⤷ 出口 return：返回 True
        return True
    if not isinstance(relay_error, RuntimeError):
        # ⤷ 出口 return：返回 False
        return False
    kind = type(callback_error)
    type_names = {kind.__name__, kind.__qualname__, f"{kind.__module__}.{kind.__qualname__}"}
    # ⤷ 出口 return：返回 any((str(relay_error).startswith(f'internal error: {name}: {…
    return any(str(relay_error).startswith(f"internal error: {name}: {callback_error}") for name in type_names)


# ┌── 函数 def get_runtime()
# │  作用：Return the Relay host for the active Hermes profile.
# │  参数：create: bool = ..., profile_key: str | None = ...
# │  返回：RelayRuntime | None
# └──────────────────────────────────────────────────────────────
def get_runtime(*, create: bool = True, profile_key: str | None = None) -> RelayRuntime | None:
    """Return the Relay host for the active Hermes profile."""
    host = HOST_REGISTRY.for_profile(profile_key, create=create)
    # ⤷ 出口 return：返回 host if isinstance(host, RelayRuntime) else None
    return host if isinstance(host, RelayRuntime) else None


# ┌── 函数 def current_profile_key()
# │  作用：Return the canonical profile identity used for runtime isolation.
# │  参数：无
# │  返回：str
# └──────────────────────────────────────────────────────────────
def current_profile_key() -> str:
    """Return the canonical profile identity used for runtime isolation."""
    home = get_hermes_home().expanduser()
    if not home.is_absolute():
        # ⤷ 出口 return：返回 str(home.resolve())
        return str(home.resolve())
    # ⤷ 出口 return：返回 _PROFILE_KEY_CACHE.get(str(home)) or _PROFILE_KEY_CACHE.setd…
    return _PROFILE_KEY_CACHE.get(str(home)) or _PROFILE_KEY_CACHE.setdefault(str(home), str(home.resolve()))


# ┌── 函数 def _load_nemo_relay()
# │  作用：Load the binding only when a producer or consumer needs Relay.
# │  参数：无
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _load_nemo_relay() -> Any:
    """Load the binding only when a producer or consumer needs Relay."""
    # ⤷ 出口 return：返回 importlib.import_module('nemo_relay')
    return importlib.import_module("nemo_relay")


# ┌── 函数 def _configured_plugin_inputs()
# │  作用：Load selected plugin inputs, or return ``None`` when none were selected.
# │  参数：relay: Any
# │  返回：tuple[dict[str, Any], list[Any]] | None
# └──────────────────────────────────────────────────────────────
def _configured_plugin_inputs(relay: Any) -> tuple[dict[str, Any], list[Any]] | None:
    """Load selected plugin inputs, or return ``None`` when none were selected."""
    configured = os.environ.get(RELAY_PLUGINS_CONFIG_ENV, "").strip()
    if not configured:
        if legacy_vars := configured_legacy_relay_env_vars(os.environ):
            logger.warning(
                "Legacy NeMo Relay exporter variables are set but no %s was provided. %s no longer activate "
                "Relay exporters; migrate the exporter configuration to a Relay plugins.toml file.",
                RELAY_PLUGINS_CONFIG_ENV, ", ".join(legacy_vars),
            )
        # ⤷ 出口 return：返回 None
        return None
    config_path = Path(configured).expanduser()
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        if "dynamic_plugins" in config:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("Hermes [[dynamic_plugins]] records are unsupported; use Relay [[plugins.dynamic]] records")
        dynamic_plugins = relay.plugin.load_dynamic_plugin_activation_specs(config_path) if "plugins" in config else []
        # ⤷ 出口 return：返回 ({k: v for k, v in config.items() if k != 'plugins'}, dynami…
        return {k: v for k, v in config.items() if k != "plugins"}, dynamic_plugins
    except Exception as exc:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise _RelayPluginConfigurationLoadError(
            f"Hermes Relay plugin configuration could not be loaded from {config_path}; continuing without Relay plugins"
        ) from exc


# ┌── 函数 def _resolve_plugin_awaitable()
# │  作用：Resolve Relay's async plugin API from synchronous host construction.
# │  参数：value: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _resolve_plugin_awaitable(value: Any) -> Any:
    """Resolve Relay's async plugin API from synchronous host construction."""
    if not inspect.isawaitable(value):
        # ⤷ 出口 return：返回 value
        return value
    # Only the "no running loop" probe is guarded: a RuntimeError raised by the awaitable itself
    # (re-raised from the daemon thread) must propagate, not fall through to a second asyncio.run.
    # ⤷ 关键分支 try/except：捕获 RuntimeError，出错走兜底而不是崩溃。
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # ⤷ 出口 return：返回 asyncio.run(value)
        return asyncio.run(value)
    # ⤷ 出口 return：返回 _run_on_daemon_thread(lambda: asyncio.run(value), name='herm…
    return _run_on_daemon_thread(lambda: asyncio.run(value), name="hermes-nemo-relay-plugin-lifecycle")


# ┌── 函数 def _session_id()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：event: dict[str, Any]
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _session_id(event: dict[str, Any]) -> str:
    # ⤷ 出口 return：返回 str(event.get('session_id') or '')
    return str(event.get("session_id") or "")


# ┌── 函数 def _reset_for_tests()
# │  作用：Reset all profile-scoped Relay hosts for isolated tests.
# │  参数：无
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    with SESSION_COORDINATOR._active_turns_lock:
        SESSION_COORDINATOR._active_turns.clear()
    HOST_REGISTRY.shutdown_all()
    _PLUGIN_CONFIGURATION.reset_for_tests()
    _PROFILE_KEY_CACHE.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from enum import auto  # noqa: F401,E402

# ┌── 函数 def emit_mark()
# │  作用：Emit a fail-open Relay mark under a Hermes session.
# │  参数：name: str, session_id: str, data: Any = ..., metadata: Any = ...
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def emit_mark(
    name: str,
    *,
    session_id: str,
    data: Any = None,
    metadata: Any = None,
) -> bool:
    """Emit a fail-open Relay mark under a Hermes session."""
    runtime = get_runtime(create=False)
    if runtime is None:
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 runtime.emit_mark(name, {'session_id': session_id}, data=dat…
        return runtime.emit_mark(
            name,
            {"session_id": session_id},
            data=data,
            metadata=metadata,
        )
    except Exception:
        logger.warning("Hermes Relay mark failed: %s", name, exc_info=True)
        # ⤷ 出口 return：返回 False
        return False

# ┌── 函数 def ensure_session()
# │  作用：Create or return the shared Relay session used by Hermes core.
# │  参数：session_id: str, **context
# │  返回：RelaySession | None
# └──────────────────────────────────────────────────────────────
def ensure_session(*, session_id: str, **context: Any) -> RelaySession | None:
    """Create or return the shared Relay session used by Hermes core."""
    runtime = get_runtime()
    if runtime is None:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 runtime.ensure_session({'session_id': session_id, **context}…
        return runtime.ensure_session({"session_id": session_id, **context})
    except Exception:
        logger.warning("Hermes Relay session initialization failed", exc_info=True)
        # ⤷ 出口 return：返回 None
        return None

# ┌── 函数 def get_host()
# │  作用：Return the explicit real or reduced-capability host for a profile.
# │  参数：create: bool = ..., profile_key: str | None = ...
# │  返回：RelayHost | None
# └──────────────────────────────────────────────────────────────
def get_host(
    *,
    create: bool = True,
    profile_key: str | None = None,
) -> RelayHost | None:
    """Return the explicit real or reduced-capability host for a profile."""
    # ⤷ 出口 return：返回 HOST_REGISTRY.for_profile(profile_key, create=create)
    return HOST_REGISTRY.for_profile(profile_key, create=create)

# ┌── 函数 def get_session_handle()
# │  作用：Return the shared Relay handle for direct core instrumentation.
# │  参数：session_id: str
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def get_session_handle(session_id: str) -> Any:
    """Return the shared Relay handle for direct core instrumentation."""
    runtime = get_runtime(create=False)
    # ⤷ 出口 return：返回 None if runtime is None else runtime.get_session_handle(sess…
    return None if runtime is None else runtime.get_session_handle(session_id)

# ┌── 函数 def run_in_session()
# │  作用：Run a scope, LLM, or tool API against a shared Hermes session.
# │  参数：session_id: str, callback: Callable[..., Any], *args, **kwargs
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def run_in_session(
    session_id: str,
    callback: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a scope, LLM, or tool API against a shared Hermes session."""
    runtime = get_runtime()
    if runtime is None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise RuntimeError("Hermes Relay runtime is unavailable")
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    if session is None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise RuntimeError("Hermes Relay session is unavailable")
    # ⤷ 出口 return：返回 runtime.run_in_session(session, callback, *args, **kwargs)
    return runtime.run_in_session(session, callback, *args, **kwargs)

# ┌── 异步函数（协程） def run_in_session_async()
# │  作用：Await a Relay operation inside a shared Hermes session context.
# │  参数：session_id: str, callback: Callable[..., Any], *args, **kwargs
# │  返回：Any
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def run_in_session_async(
    session_id: str,
    callback: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Await a Relay operation inside a shared Hermes session context."""
    runtime = get_runtime()
    if runtime is None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise RuntimeError("Hermes Relay runtime is unavailable")
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    if session is None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise RuntimeError("Hermes Relay session is unavailable")
    # ⤷ 出口 return：返回 await runtime.run_in_session_async(session, callback, *args,…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await runtime.run_in_session_async(session, callback, *args, **kwargs)
# ---- END PLUGIN-COMPAT ----
