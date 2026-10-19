"""Core NeMo Relay adapters for physical Hermes provider attempts."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：agent/relay_llm.py（约 890 行）
# 地位：把真实的模型调用包进 Relay 体系（做插桩、编解码、逻辑调用收尾）。
#
# 主要类型：
#   class _ManagedAttempt     一次受管的调用尝试：invoke / invoke_async / run_managed / resolve_failure
#   class ManagedLlmStream    流式响应的包装迭代器（17 个方法）
#   class AnthropicStreamAccumulator   把 Anthropic 的流式事件累积成一个完整响应
#                                      （observe → _on_message_start → _on_content_block_delta
#                                       → _on_message_delta → finalize）
#
# 主要函数：execute(request, callback) / execute_async / stream_current(request, stream_factory)
#
# 技术点：
#   - 编解码往返：_codec_round_trip_request_body / _provider_request_body
#     请求要在「我们的格式」和「服务商格式」之间转换，还要保证往返一致（_json_equal 校验）
#   - 逻辑调用收尾：complete_logical_call()
#   - 取消判定：_is_cancellation(error)
#
# 初学建议：这个文件偏框架层，可以先跳过。
# ==========================================================================
#

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import json
import logging
from collections.abc import Callable, Iterator
from functools import partial
from types import SimpleNamespace
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)


_PROVIDER_MESSAGE_EXTENSION_KEYS = frozenset({"reasoning_content", "reasoning_details"})
_RELAY_INTERNAL_PROVIDER_HEADERS = frozenset({"x-dynamo-parent-session-id", "x-dynamo-session-id"})
_LogicalCall = tuple[relay_runtime.RelayTurnContext, Any, str]


# api_mode -> (Relay operation name, codec class name on ``relay.codecs``)
_RELAY_PROTOCOL_BY_API_MODE = {
    "chat_completions": ("openai.chat_completions", "OpenAIChatCodec"),
    "codex_responses": ("openai.responses", "OpenAIResponsesCodec"),
    "anthropic_messages": ("anthropic.messages", "AnthropicMessagesCodec"),
}


# ┌── 函数 def _api_mode()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：metadata: dict[str, Any] | None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _api_mode(metadata: dict[str, Any] | None) -> str:
    # ⤷ 出口 return：返回 str((metadata or {}).get('api_mode') or '')
    return str((metadata or {}).get("api_mode") or "")


# ┌── 函数 def _relay_operation_name()
# │  作用：当 Hermes knows API 模式. 时返回 Relay's canonical operation 名称。
# │  参数：provider_name: str, metadata: dict[str, Any] | None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _relay_operation_name(provider_name: str, metadata: dict[str, Any] | None) -> str:
    """Return Relay's canonical operation name when Hermes knows the API mode."""
    protocol = _RELAY_PROTOCOL_BY_API_MODE.get(_api_mode(metadata))
    # ⤷ 出口 return：返回 protocol[0] if protocol is not None else provider_name
    return protocol[0] if protocol is not None else provider_name


# ┌── 函数 def _relay_metadata()
# │  作用：Preserve physical 服务商当 operation 名称是 canonicalized.
# │  参数：provider_name: str, metadata: dict[str, Any] | None
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _relay_metadata(provider_name: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve the physical provider when the operation name is canonicalized."""
    relay_metadata = _jsonable_dict(metadata or {})
    relay_metadata.setdefault("hermes.provider", provider_name)
    # ⤷ 出口 return：返回 relay_metadata
    return relay_metadata


# ┌── 类 class _ManagedAttempt
# │  说明：Relay 请求状态 shared 由同步, 异步, 且 streaming adapters.
# │  成员：共 12 个方法；对外可用 8 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class _ManagedAttempt:
    """Relay request state shared by the sync, async, and streaming adapters."""

    # ┌── 函数 def resolve()
    # │  作用：返回 managed 尝试用于 ``session_id`` (None: inherited 轮次's), 或 None 运行 unmanaged.。
    # │  参数：cls, session_id: str | None, request: dict[str, Any], metadata: dict[str, Any] | None, name: str, model_name: str
    # │  返回：'_ManagedAttempt | None'
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def resolve(
        cls, session_id: str | None, request: dict[str, Any], metadata: dict[str, Any] | None,
        *, name: str, model_name: str,
    ) -> "_ManagedAttempt | None":
        """Return the managed attempt for ``session_id`` (None: the inherited turn's), or None to run unmanaged."""
        if session_id is None:
            session_id = _current_session_id()
        if not session_id:
            # ⤷ 出口 return：返回 None
            return None
        runtime, session, parent = relay_runtime.resolve_execution_context(session_id)
        if runtime is None or session is None or not runtime.managed_execution_enabled():
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 cls(runtime, session, parent, request, metadata, name=name, …
        return cls(runtime, session, parent, request, metadata, name=name, model_name=model_name)

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, runtime: relay_runtime.RelayRuntime, session: Any, parent: Any, request: dict[str, Any], metadata: dict[str, Any] | None, name: str, model_name: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(
        self, runtime: relay_runtime.RelayRuntime, session: Any, parent: Any,
        request: dict[str, Any], metadata: dict[str, Any] | None, *, name: str, model_name: str,
    ) -> None:
        self.runtime, self.session, self.request, self.metadata = runtime, session, request, metadata
        self.logical = _logical_parent(runtime, session, parent, metadata)
        self.parent = self.logical[1] if self.logical is not None else parent
        self.body = _relay_request_body(request, metadata)
        self.relay_request = runtime.relay.LLMRequest({}, self.body)
        self.codec_baseline = _codec_round_trip_request_body(
            runtime.relay, self.relay_request, relay_request_body=self.body, metadata=metadata
        )
        self.operation = _relay_operation_name(name, metadata)
        self.relay_kwargs = {
            "handle": self.parent, "metadata": _relay_metadata(name, metadata), "model_name": model_name,
            "codec": _codec(runtime.relay, metadata), "response_codec": _codec(runtime.relay, metadata),
        }
        # Provider callback bookkeeping: "value"/"json" once it returned, "error" if it raised.
        self.raw_response: dict[str, Any] = {}
        self.context = contextvars.copy_context()

    # ┌── 函数 def provider_request()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, next_request: Any
    # │  返回：dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def provider_request(self, next_request: Any) -> dict[str, Any]:
        # ⤷ 出口 return：返回 _provider_request(self.request, next_request, relay_request_…
        return _provider_request(
            self.request, next_request, relay_request_body=self.body,
            codec_baseline_body=self.codec_baseline, metadata=self.metadata,
        )

    # ┌── 函数 def run_callback()
    # │  作用：运行 Hermes 回调在 fresh 复制的 captured 上下文.
    # │  参数：self, callback: Callable[..., Any], *args
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def run_callback(self, callback: Callable[..., Any], *args: Any) -> Any:
        """Run a Hermes callback in a fresh copy of the captured context.
        Relay can invoke callbacks while another still owns the captured Context (hence the
        copy); nested relay calls run unmanaged — see relay_runtime.managed_callback_guard."""
        def guarded() -> Any:
            # See #77244.
            # See #77244.
            # Hermes-side callbacks run while the native pipeline drives this stream; nested relay calls
            # they make must bypass managed execution (#77244).
            with relay_runtime.managed_callback_guard():
                # ⤷ 出口 return：返回 callback(*args)
                return callback(*args)

        # ⤷ 出口 return：返回 self.context.copy().run(guarded)
        return self.context.copy().run(guarded)

    # ┌── 函数 def _record()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, raw: Any
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def _record(self, raw: Any) -> Any:
        self.raw_response["value"] = raw
        self.raw_response["json"] = _jsonable(raw)
        # ⤷ 出口 return：返回 self.raw_response['json']
        return self.raw_response["json"]

    # ┌── 函数 def _recording_errors()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：Iterator[None]
    # │  【语法】@contextmanager 把「有 yield 的函数」变成上下文管理器，调用方可以用 `with xxx() as v:` 自动做进入/清理。yield 之前是准备，yield 之后是收尾。
    # └──────────────────────────────────────────────────────────────
    @contextlib.contextmanager
    def _recording_errors(self) -> Iterator[None]:
        # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
        try:
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield
        except BaseException as exc:
            self.raw_response["error"] = exc
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise

    # ┌── 函数 def invoke()
    # │  作用：服务商回调 handed Relay: 运行 ``回调`` 在…上 Relay's (possibly rewritten) 请求.
    # │  参数：self, callback: Callable[..., Any], next_request: Any
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def invoke(self, callback: Callable[..., Any], next_request: Any) -> Any:
        """Provider callback handed to Relay: run ``callback`` on Relay's (possibly rewritten) request."""
        with self._recording_errors():
            raw = self.run_callback(callback, self.provider_request(next_request))
        # ⤷ 出口 return：返回 self._record(raw)
        return self._record(raw)

    # ┌── 异步函数（协程） def invoke_async()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, callback: Callable[..., Any], next_request: Any
    # │  返回：Any
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def invoke_async(self, callback: Callable[..., Any], next_request: Any) -> Any:
        async def call_provider() -> Any:
            with relay_runtime.managed_callback_guard():  # nested relay calls run unmanaged
                # ⤷ 出口 return：返回 await callback(final_request)
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                return await callback(final_request)

        with self._recording_errors():
            final_request = self.provider_request(next_request)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            raw = await self.context.copy().run(asyncio.create_task, call_provider())
        # ⤷ 出口 return：返回 self._record(raw)
        return self._record(raw)

    # ┌── 函数 def run_managed()
    # │  作用：返回 awaitable 运行 ``relay_call`` inside 会话上下文.。
    # │  参数：self, relay_call: Callable[..., Any], *callbacks
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def run_managed(self, relay_call: Callable[..., Any], *callbacks: Any) -> Any:
        """Return the awaitable running ``relay_call`` inside the session context."""
        # ⤷ 出口 return：返回 self.runtime.run_in_session_async(self.session, relay_call, …
        return self.runtime.run_in_session_async(
            self.session, relay_call, self.operation, self.relay_request, *callbacks, **self.relay_kwargs,
        )

    # ┌── 函数 def resolve_failure()
    # │  作用：Re-抛出服务商's 持有错误, 或 recover completed 服务商结果.
    # │  参数：self, exc: BaseException, defer_logical_completion: bool
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def resolve_failure(self, exc: BaseException, defer_logical_completion: bool) -> Any:
        """Re-raise the provider's own error, or recover a completed provider result.
        Must be called from the ``except`` handling ``exc`` (bare ``raise``)."""
        callback_error = self.raw_response.get("error")
        if callback_error is not None and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error):
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise callback_error
        if not isinstance(exc, Exception) or callback_error is not None or "value" not in self.raw_response:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        logger.warning(
            "NeMo Relay LLM post-processing failed after provider success; returning the provider response",
            exc_info=True,
        )
        self._complete(defer_logical_completion)
        # ⤷ 出口 return：返回 self.raw_response['value']
        return self.raw_response["value"]

    # ┌── 函数 def result()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, managed: Any, defer_logical_completion: bool
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def result(self, managed: Any, defer_logical_completion: bool) -> Any:
        self._complete(defer_logical_completion)
        if "value" in self.raw_response and _json_equal(managed, self.raw_response["json"]):
            # ⤷ 出口 return：返回 self.raw_response['value']
            return self.raw_response["value"]
        # ⤷ 出口 return：返回 _namespace(managed)
        return _namespace(managed)

    # ┌── 函数 def _complete()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, defer_logical_completion: bool
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _complete(self, defer_logical_completion: bool) -> None:
        if not defer_logical_completion:
            _complete_logical(self.logical, outcome="success")


# ┌── 函数 def _current_session_id()
# │  作用：返回 inherited Hermes 轮次's 会话 id, 或 None outside 当前活跃的轮次.。
# │  参数：无
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _current_session_id() -> str | None:
    """Return the inherited Hermes turn's session id, or None outside a live turn."""
    turn = relay_runtime.active_turn()
    # ⤷ 出口 return：返回 None if turn is None else turn.lease.session_id
    return None if turn is None else turn.lease.session_id


# ┌── 函数 def execute()
# │  作用：运行 one non-streaming physical 服务商尝试 through Relay.
# │  参数：request: dict[str, Any], callback: Callable[[dict[str, Any]], Any], name: str, model_name: str, session_id: str | None = ..., metadata: dict[str, Any] | None = ..., defer_logical_completion: bool = ...
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def execute(
    request: dict[str, Any], callback: Callable[[dict[str, Any]], Any], *, name: str, model_name: str,
    session_id: str | None = None, metadata: dict[str, Any] | None = None, defer_logical_completion: bool = False,
) -> Any:
    """Run one non-streaming physical provider attempt through Relay.
    ``session_id`` defaults to the inherited Hermes turn's session (unmanaged when there is none)."""
    attempt = _ManagedAttempt.resolve(session_id, request, metadata, name=name, model_name=model_name)
    if attempt is None:
        # ⤷ 出口 return：返回 callback(request)
        return callback(request)
    # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
    try:
        managed = _run_awaitable(attempt.run_managed(
            attempt.runtime.relay.llm.execute, partial(attempt.invoke, callback)
        ))
    except BaseException as exc:
        # ⤷ 出口 return：返回 attempt.resolve_failure(exc, defer_logical_completion)
        return attempt.resolve_failure(exc, defer_logical_completion)
    # ⤷ 出口 return：返回 attempt.result(managed, defer_logical_completion)
    return attempt.result(managed, defer_logical_completion)


# ┌── 异步函数（协程） def execute_async()
# │  作用：异步 ``执行``.
# │  参数：request: dict[str, Any], callback: Callable[[dict[str, Any]], Any], name: str, model_name: str, session_id: str | None = ..., metadata: dict[str, Any] | None = ..., defer_logical_completion: bool = ...
# │  返回：Any
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def execute_async(
    request: dict[str, Any], callback: Callable[[dict[str, Any]], Any], *, name: str, model_name: str,
    session_id: str | None = None, metadata: dict[str, Any] | None = None, defer_logical_completion: bool = False,
) -> Any:
    """Async ``execute``."""
    attempt = _ManagedAttempt.resolve(session_id, request, metadata, name=name, model_name=model_name)
    if attempt is None:
        # ⤷ 出口 return：返回 await callback(request)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await callback(request)
    # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
    try:
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        managed = await attempt.run_managed(attempt.runtime.relay.llm.execute, partial(attempt.invoke_async, callback))
    except BaseException as exc:
        # ⤷ 出口 return：返回 attempt.resolve_failure(exc, defer_logical_completion)
        return attempt.resolve_failure(exc, defer_logical_completion)
    # ⤷ 出口 return：返回 attempt.result(managed, defer_logical_completion)
    return attempt.result(managed, defer_logical_completion)


# Run under the inherited Hermes turn when present (callers that do not know a session id).
execute_current = execute
execute_current_async = execute_async


# ┌── 函数 def _has_running_event_loop()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _has_running_event_loop() -> bool:
    with contextlib.suppress(RuntimeError):
        # ⤷ 出口 return：返回 asyncio.get_running_loop() is not None
        return asyncio.get_running_loop() is not None
    # ⤷ 出口 return：返回 False
    return False


# ┌── 函数 def stream_current()
# │  作用：运行服务商 stream under inherited Hermes 轮次当 present.
# │  参数：request: dict[str, Any], stream_factory: Callable[[dict[str, Any]], Any], name: str, model_name: str, finalizer: Callable[[], Any], metadata: dict[str, Any] | None = ..., defer_logical_completion: bool = ..., completed_response_predicate: Callable[[Any], bool] | None = ...
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def stream_current(
    request: dict[str, Any], stream_factory: Callable[[dict[str, Any]], Any], *, name: str, model_name: str,
    finalizer: Callable[[], Any], metadata: dict[str, Any] | None = None,
    defer_logical_completion: bool = False, completed_response_predicate: Callable[[Any], bool] | None = None,
) -> Any:
    """Run a provider stream under the inherited Hermes turn when present.
    With ``completed_response_predicate`` set, a factory that ignores ``stream=True`` and returns a
    complete response is unwrapped and returned directly (pre-Relay behavior). Detecting that primes
    the lazy pipeline: a genuine first chunk is buffered, but provider latency and pre-first-yield
    errors may surface before this returns.

    AnthropicAuxiliaryClient and other shims that ignore ``stream=True``), unwrap and return the completed
    response directly. This mirrors the pre-Relay behavior where ``call_llm(stream=True)`` returned the raw
    response and the consumer's own ``hasattr(stream, "choices")`` check handled it (#11732, #55933) —
    without the unwrap the response stays trapped as ``final_response`` on the inner ManagedLlmStream and
    the outer consumer sees an empty stream.
    """
    session_id = _current_session_id()
    # Inside a managed callback (on the Relay session's loop) a nested ManagedLlmStream would be
    # iterated synchronously on that loop, which asyncio forbids; the outer stream tracks this attempt.
    if session_id is None or _has_running_event_loop():
        # ⤷ 出口 return：返回 stream_factory(request)
        return stream_factory(request)
    managed = stream(
        request, stream_factory, session_id=session_id, name=name, model_name=model_name,
        finalizer=finalizer, metadata=metadata, defer_logical_completion=defer_logical_completion,
        completed_response_predicate=completed_response_predicate,
    )
    if completed_response_predicate is not None:
        # Relay may defer the provider callback until the first pull; prime once (a real first chunk is buffered).
        managed._prime_completed_response()
    # ⤷ 出口 return：返回 managed.final_response if managed.final_response is not None…
    return managed.final_response if managed.final_response is not None else managed


# ┌── 函数 def _aclose_on_loop()
# │  作用：Await ``stream.aclose()`` 在…上 ``循环`` 当 stream exposes one.
# │  参数：loop: asyncio.AbstractEventLoop, stream: Any
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _aclose_on_loop(loop: asyncio.AbstractEventLoop, stream: Any) -> None:
    """Await ``stream.aclose()`` on ``loop`` when the stream exposes one."""
    close = getattr(stream, "aclose", None)
    if not callable(close):
        return

    async def close_stream() -> None:  # create the coroutine on ``loop``, not the caller's thread
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await close()

    loop.run_until_complete(close_stream())


# ┌── 类 class ManagedLlmStream（继承自：Iterator[Any]）
# │  说明：Synchronous view 的 one Relay-managed 服务商 stream, driven 从工作线程.
# │  成员：共 17 个方法；对外可用 1 个；含魔术方法 __init__, __iter__, __next__, __del__
# └──────────────────────────────────────────────────────────────
class ManagedLlmStream(Iterator[Any]):
    """Synchronous view of one Relay-managed provider stream, driven from the worker thread."""

    final_response: Any = None
    output_modified = _closed = _provider_completed = False
    _loop: asyncio.AbstractEventLoop | None = None
    _stream = _raw_stream_resource = None
    _runtime_lease: relay_runtime.RelayOperationLease | None = None
    _close_error = _callback_error = None  # BaseException | None
    _logical: _LogicalCall | None = None
    _logical_response_model_name: str | None = None

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, request: dict[str, Any], stream_factory: Callable[[dict[str, Any]], Any], session_id: str, name: str, model_name: str, finalizer: Callable[[], Any], on_stream_created: Callable[[Any], None] | None = ...  …共 14 个
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(
        self, request: dict[str, Any], stream_factory: Callable[[dict[str, Any]], Any], *, session_id: str,
        name: str, model_name: str, finalizer: Callable[[], Any],
        on_stream_created: Callable[[Any], None] | None = None, on_chunk: Callable[[Any], None] | None = None,
        chunk_adapter: Callable[[Any], Any] | None = None, accept_chunk: Callable[[Any], bool] | None = None,
        completed_response_predicate: Callable[[Any], bool] | None = None,
        metadata: dict[str, Any] | None = None, defer_logical_completion: bool = False,
    ) -> None:
        self._defer_logical_completion = defer_logical_completion
        # Only auxiliary calls report model/provider on their logical scope.
        auxiliary = str((metadata or {}).get("call_role") or "").startswith("auxiliary:")
        self._logical_model_name, self._logical_provider_name = (model_name, name) if auxiliary else (None, None)
        self._on_chunk, self._chunk_adapter, self._accept_chunk = on_chunk, chunk_adapter or _namespace, accept_chunk
        self._stream_factory, self._on_stream_created, self._finalizer = stream_factory, on_stream_created, finalizer
        self._completed_response_predicate = completed_response_predicate
        self._raw_chunks: list[tuple[Any, Any]] = []
        self._prefetched_chunks: list[Any] = []
        attempt = _ManagedAttempt.resolve(session_id, request, metadata, name=name, model_name=model_name)
        if attempt is None:
            self._start_unmanaged(request)
            return
        self._logical = attempt.logical
        self._start_managed(attempt)

    # ┌── 函数 def _start_unmanaged()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, request: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _start_unmanaged(self, request: dict[str, Any]) -> None:
        raw_stream = self._stream_factory(request)
        predicate = self._completed_response_predicate
        if predicate is not None and predicate(raw_stream):
            self.final_response = raw_stream
            self._stream = iter(())
            return
        self._raw_stream_resource = raw_stream
        if self._on_stream_created is not None:
            self._on_stream_created(raw_stream)
        self._stream = iter(raw_stream)

    # ┌── 异步函数（协程） def _provider_stream()
    # │  作用：Relay's 服务商回调: 运行 factory 且 yield JSON-encoded chunks.
    # │  参数：self, attempt: _ManagedAttempt, next_request: Any
    # │  返回：（未标注类型）
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _provider_stream(self, attempt: _ManagedAttempt, next_request: Any):
        """Relay's provider callback: run the factory and yield JSON-encoded chunks."""
        run_callback = attempt.run_callback
        raw_stream = None
        # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
        try:
            raw_stream = run_callback(self._stream_factory, attempt.provider_request(next_request))
            predicate = self._completed_response_predicate
            if predicate is not None and run_callback(predicate, raw_stream):
                self.final_response = raw_stream
                self._provider_completed = True
                return
            if self._on_stream_created is not None:
                run_callback(self._on_stream_created, raw_stream)
            raw_iterator = run_callback(iter, raw_stream)
            while True:
                # ⤷ 关键分支 try/except：捕获 StopIteration，出错走兜底而不是崩溃。
                try:
                    chunk = run_callback(next, raw_iterator)
                except StopIteration:
                    break
                if self._accept_chunk is not None and not run_callback(self._accept_chunk, chunk):
                    break
                encoded_chunk = _jsonable(chunk)
                self._raw_chunks.append((encoded_chunk, chunk))
                # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
                yield encoded_chunk
            self._provider_completed = True
        except BaseException as exc:
            self._callback_error = exc
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        finally:
            close = getattr(raw_stream, "close", None)
            if callable(close):
                # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
                try:
                    run_callback(close)
                except BaseException as exc:
                    self._close_error = exc
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise

    # ┌── 函数 def _relay_finalizer()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, attempt: _ManagedAttempt
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def _relay_finalizer(self, attempt: _ManagedAttempt) -> Any:
        # Relay may call this while unwinding a provider-stream failure; keep the original
        # error instead of a secondary "missing terminal response".
        if self._callback_error is not None:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
        try:
            response = self.final_response
            if response is None:
                response = attempt.run_callback(self._finalizer)
            if self._logical_model_name is not None:
                self._logical_response_model_name = _response_model_name(response)
            # ⤷ 出口 return：返回 _jsonable(response)
            return _jsonable(response)
        except BaseException as exc:
            self._callback_error = exc
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise

    # ┌── 函数 def _start_managed()
    # │  作用：打开 Relay's stream 在…上 private 事件循环持有的由 这个 iterator.
    # │  参数：self, attempt: _ManagedAttempt
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _start_managed(self, attempt: _ManagedAttempt) -> None:
        """Open Relay's stream on a private event loop owned by this iterator."""

        def observe_chunk(chunk: Any) -> None:
            if self._on_chunk is not None:
                attempt.run_callback(self._on_chunk, _jsonable(chunk))

        self._runtime_lease = attempt.runtime.acquire_operation_lease()
        # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
        try:
            self._loop = loop = asyncio.new_event_loop()
            self._stream = loop.run_until_complete(
                attempt.run_managed(
                    attempt.runtime.relay.llm.stream_execute, partial(self._provider_stream, attempt),
                    observe_chunk, partial(self._relay_finalizer, attempt),
                )
            )
        except BaseException as exc:
            if self._loop is not None and self._recoverable_relay_failure(exc):
                self._preserve_pending_provider_chunks()
                return
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                if self._loop is not None:
                    self._finish_logical("cancelled" if _is_cancellation(exc) else "failed")
                    self._loop.close()
            finally:
                self._loop = None
                self._release_runtime_lease()
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise

    # ┌── 函数 def __iter__()
    # │  作用：魔术方法 __iter__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：'ManagedLlmStream'
    # └──────────────────────────────────────────────────────────────
    def __iter__(self) -> "ManagedLlmStream":
        # ⤷ 出口 return：返回 self
        return self

    # ┌── 函数 def _prime_completed_response()
    # │  作用：Advance 一旦当 preserving genuine first chunk.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _prime_completed_response(self) -> None:
        """Advance once while preserving a genuine first chunk."""
        if not self._closed and not self._prefetched_chunks:
            with contextlib.suppress(StopIteration):
                self._prefetched_chunks.append(next(self))

    # ┌── 函数 def _recoverable_relay_failure()
    # │  作用：Relay post-processing failed 在…之后服务商已 succeeded.
    # │  参数：self, exc: BaseException
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _recoverable_relay_failure(self, exc: BaseException) -> bool:
        """Relay post-processing failed after the provider already succeeded."""
        recoverable = isinstance(exc, Exception) and self._provider_completed and self._callback_error is None
        if recoverable:
            logger.warning(
                "NeMo Relay stream post-processing failed after provider success; preserving the provider result",
                exc_info=True,
            )
        # ⤷ 出口 return：返回 recoverable
        return recoverable

    # ┌── 函数 def _finish_logical()
    # │  作用：Complete logical LLM scope unless caller deferred 它.
    # │  参数：self, outcome: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _finish_logical(self, outcome: str) -> None:
        """Complete the logical LLM scope unless the caller deferred it."""
        if self._defer_logical_completion:
            return
        _complete_logical(
            self._logical, outcome=outcome, model_name=self._logical_model_name,
            provider_name=self._logical_provider_name, response_model_name=self._logical_response_model_name,
            operation_lease=self._runtime_lease,
        )
        self._logical = None

    # ┌── 函数 def __next__()
    # │  作用：魔术方法 __next__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def __next__(self) -> Any:
        if self._closed:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise StopIteration
        if self._prefetched_chunks:
            # ⤷ 出口 return：返回 self._prefetched_chunks.pop()
            return self._prefetched_chunks.pop()
        if self._loop is None:
            chunk = next(self._stream, self)  # self: exhausted sentinel
            if chunk is self or (self._accept_chunk is not None and not self._accept_chunk(chunk)):
                self._close(logical_outcome="cancelled")
                raise StopIteration
            # ⤷ 出口 return：返回 chunk
            return chunk

        async def next_chunk() -> Any:
            # ⤷ 出口 return：返回 await anext(self._stream)
            return await anext(self._stream)

        # ⤷ 关键分支 try/except：捕获 StopAsyncIteration、BaseException，出错走兜底而不是崩溃。
        try:
            chunk = self._loop.run_until_complete(next_chunk())
        except StopAsyncIteration:
            if self._raw_chunks:
                self.output_modified = True
            self._finish_logical("success")
            self._close(logical_outcome="cancelled")
            raise StopIteration from None
        except BaseException as exc:
            callback_error = self._callback_error
            if callback_error is not None and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error):
                self._close(logical_outcome="failed")
                raise callback_error
            if self._recoverable_relay_failure(exc):
                self._preserve_pending_provider_chunks()
                return next(self)
            self._close(logical_outcome="cancelled" if _is_cancellation(exc) else "failed")
            raise
        for index, (encoded, raw) in enumerate(self._raw_chunks):
            if _json_equal(chunk, encoded):
                if index > 0:
                    self.output_modified = True
                del self._raw_chunks[: index + 1]
                return raw
        self.output_modified = True
        # ⤷ 出口 return：返回 self._chunk_adapter(chunk)
        return self._chunk_adapter(chunk)

    # ┌── 函数 def close()
    # │  作用：关闭 explicitly abandoned stream 且 cancel 其 logical call.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Close an explicitly abandoned stream and cancel its logical call."""
        self._close(logical_outcome="cancelled")
        close_error, self._close_error = self._close_error, None
        if close_error is not None:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise close_error

    # ┌── 函数 def _preserve_pending_provider_chunks()
    # │  作用：切换 failed Relay stream 其 undelivered 服务商 chunks.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _preserve_pending_provider_chunks(self) -> None:
        """Switch a failed Relay stream to its undelivered provider chunks."""
        pending = [raw for _encoded, raw in self._raw_chunks]
        self._raw_chunks.clear()
        loop, relay_stream = self._loop, self._stream
        self._loop, self._stream, self._raw_stream_resource, self._accept_chunk = None, iter(pending), None, None
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            if loop is not None:
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    _aclose_on_loop(loop, relay_stream)
                except Exception:
                    logger.debug("Relay stream cleanup failed during provider fallback", exc_info=True)
                loop.close()
            self._finish_logical("success")
        finally:
            self._release_runtime_lease()

    # ┌── 函数 def _keep_first_close_error()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, exc: BaseException
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _keep_first_close_error(self, exc: BaseException) -> None:
        if self._close_error is None:
            self._close_error = exc

    # ┌── 函数 def _close_provider_resources()
    # │  作用：关闭 unmanaged 服务商 stream/resource 一旦每个 (它们可以是 same 对象).
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _close_provider_resources(self) -> None:
        """Close the unmanaged provider stream/resource once each (they may be the same object)."""
        resources = {id(r): r for r in (self._stream, self._raw_stream_resource) if r is not None}
        self._stream = None
        self._raw_stream_resource = None
        for resource in resources.values():
            close = getattr(resource, "close", None)
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                if callable(close):
                    close()
            except Exception as exc:
                self._keep_first_close_error(exc)
                logger.debug("Provider stream cleanup failed", exc_info=True)

    # ┌── 函数 def _close()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, logical_outcome: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _close(self, *, logical_outcome: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._prefetched_chunks.clear()
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            loop, self._loop = self._loop, None
            if loop is None:
                self._close_provider_resources()
            else:
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    _aclose_on_loop(loop, self._stream)
                except Exception as exc:
                    self._keep_first_close_error(exc)
            self._finish_logical(logical_outcome)
            if loop is not None:
                loop.close()
        finally:
            self._release_runtime_lease()

    # ┌── 函数 def _release_runtime_lease()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _release_runtime_lease(self) -> None:
        lease, self._runtime_lease = self._runtime_lease, None
        if lease is not None:
            lease.release()

    # ┌── 函数 def __del__()
    # │  作用：魔术方法 __del__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __del__(self) -> None:
        self._close(logical_outcome="cancelled")


stream = ManagedLlmStream


_ANTHROPIC_APPEND_DELTAS = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}


# ┌── 类 class AnthropicStreamAccumulator
# │  说明：Rebuild Anthropic 消息从 post-intercept SSE 事件.
# │  成员：共 8 个方法；对外可用 3 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class AnthropicStreamAccumulator:
    """Rebuild an Anthropic Message from post-intercept SSE events."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._message: dict[str, Any] = {}
        self._blocks: dict[int, dict[str, Any]] = {}

    # ┌── 函数 def observe()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, event: Any
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def observe(self, event: Any) -> None:
        payload = _jsonable(event)
        if isinstance(payload, dict):
            handler = self._EVENT_HANDLERS.get(payload.get("type"))
            if handler is not None:
                handler(self, payload)

    # ┌── 函数 def _on_message_start()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, payload: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _on_message_start(self, payload: dict[str, Any]) -> None:
        message = payload.get("message")
        if isinstance(message, dict):
            self._message.update({k: message[k] for k in ("id", "type", "role", "model", "usage") if k in message})

    # ┌── 函数 def _on_content_block_start()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, payload: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _on_content_block_start(self, payload: dict[str, Any]) -> None:
        index, block = payload.get("index"), payload.get("content_block")
        if isinstance(index, int) and isinstance(block, dict):
            self._blocks[index] = dict(block)

    # ┌── 函数 def _on_content_block_delta()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, payload: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _on_content_block_delta(self, payload: dict[str, Any]) -> None:
        index, delta = payload.get("index"), payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return
        block = self._blocks.setdefault(index, {})
        delta_type = delta.get("type")
        field = _ANTHROPIC_APPEND_DELTAS.get(delta_type)
        if field is not None:
            block[field] = str(block.get(field) or "") + str(delta.get(field) or "")
        elif delta_type == "input_json_delta":
            block["_partial_json"] = str(block.pop("_partial_json", "")) + str(delta.get("partial_json") or "")
        elif delta_type == "citations_delta" and "citation" in delta:
            block.setdefault("citations", []).append(delta["citation"])

    # ┌── 函数 def _on_message_delta()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, payload: dict[str, Any]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _on_message_delta(self, payload: dict[str, Any]) -> None:
        delta = payload.get("delta")
        if isinstance(delta, dict):
            self._message.update({k: delta[k] for k in ("stop_reason", "stop_sequence") if k in delta})
        if "usage" in payload:
            usage, current_usage = payload["usage"], self._message.get("usage")
            if isinstance(current_usage, dict) and isinstance(usage, dict):
                usage = {**current_usage, **usage}
            self._message["usage"] = usage

    _EVENT_HANDLERS = {
        "message_start": _on_message_start, "content_block_start": _on_content_block_start,
        "content_block_delta": _on_content_block_delta, "message_delta": _on_message_delta,
    }

    # ┌── 函数 def finalize()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def finalize(self) -> dict[str, Any]:
        blocks = [dict(self._blocks[index]) for index in sorted(self._blocks)]
        for block in blocks:
            partial = block.pop("_partial_json", None)
            if partial is not None:
                with contextlib.suppress(TypeError, ValueError):
                    partial = json.loads(partial)
                block["input"] = partial
        # ⤷ 出口 return：返回 {**self._message, 'content': blocks}
        return {**self._message, "content": blocks}

    # ┌── 函数 def response()
    # │  作用：返回 属性-shaped 响应已消费由 Hermes.。
    # │  参数：self, base: Any = None
    # │  返回：Any
    # └──────────────────────────────────────────────────────────────
    def response(self, base: Any = None) -> Any:
        """Return the attribute-shaped response consumed by Hermes."""
        assembled = self.finalize()
        content = assembled.pop("content", [])
        merged = {**_jsonable_dict(base), **assembled}
        if content or "content" not in merged:
            merged["content"] = content
        # ⤷ 出口 return：返回 _namespace(merged)
        return _namespace(merged)


# ┌── 函数 def _logical_parent()
# │  作用：返回 (轮次, handle, request_id) 用于轮次's logical LLM scope, pushing 它一旦.。
# │  参数：runtime: relay_runtime.RelayRuntime, session: Any, parent: Any, metadata: dict[str, Any] | None
# │  返回：_LogicalCall | None
# └──────────────────────────────────────────────────────────────
def _logical_parent(
    runtime: relay_runtime.RelayRuntime, session: Any, parent: Any, metadata: dict[str, Any] | None
) -> _LogicalCall | None:
    """Return (turn, handle, request_id) for the turn's logical LLM scope, pushing it once."""
    turn = relay_runtime.active_turn(session.session_id)
    request_id = str((metadata or {}).get("api_request_id") or "")
    if turn is None or not request_id or turn.lease.host is not runtime:
        # ⤷ 出口 return：返回 None
        return None
    with turn.finalize_lock:
        if turn.closed:
            # ⤷ 出口 return：返回 None
            return None
        with turn.logical_llm_lock:
            handle = turn.logical_llm_calls.get(request_id)
            if handle is None:
                call_role = str((metadata or {}).get("call_role") or "primary")
                handle = turn.logical_llm_calls[request_id] = runtime.run_in_session(
                    session, runtime.relay.scope.push, relay_runtime.LOGICAL_LLM_SCOPE,
                    runtime.relay.ScopeType.Function, handle=parent, input={},
                    metadata=relay_runtime.runtime_metadata(runtime.runtime_id, **{"hermes.call_role": call_role}),
                )
    # ⤷ 出口 return：返回 (turn, handle, request_id)
    return turn, handle, request_id


# ┌── 函数 def _complete_logical()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：logical: _LogicalCall | None, outcome: str, model_name: str | None = ..., provider_name: str | None = ..., response_model_name: str | None = ..., operation_lease: relay_runtime.RelayOperationLease | None = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _complete_logical(
    logical: _LogicalCall | None, *, outcome: str, model_name: str | None = None, provider_name: str | None = None,
    response_model_name: str | None = None, operation_lease: relay_runtime.RelayOperationLease | None = None,
) -> None:
    if logical is None:
        return
    turn, handle, request_id = logical
    lease = turn.lease
    if not isinstance(lease.host, relay_runtime.RelayRuntime):
        return
    output = {"outcome": outcome}
    if model_name is not None and provider_name is not None:
        output.update({"model": model_name, "provider": provider_name})
        if response_model_name is not None:
            output["response_model"] = response_model_name
    with turn.finalize_lock:
        with turn.logical_llm_lock:
            if turn.logical_llm_calls.get(request_id) is not handle:
                return
        if lease.session is None:
            return
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            (operation_lease or lease.host).run_in_session(
                lease.session, relay_runtime.pop_relay_scope, lease.host.relay, handle,
                output=output, metadata=relay_runtime.runtime_metadata(lease.host.runtime_id),
            )
        except Exception:
            # Provider result is authoritative; retain the handle so turn finalization can retry.
            logger.warning("Hermes Relay logical LLM finalization failed", exc_info=True)
            return
        with turn.logical_llm_lock:
            if turn.logical_llm_calls.get(request_id) is handle:
                del turn.logical_llm_calls[request_id]


# ┌── 函数 def _is_cancellation()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：error: BaseException
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_cancellation(error: BaseException) -> bool:
    # ⤷ 出口 return：返回 isinstance(error, (asyncio.CancelledError, InterruptedError,…
    return isinstance(error, (asyncio.CancelledError, InterruptedError, KeyboardInterrupt))


# ┌── 函数 def complete_logical_call()
# │  作用：Complete 活跃轮次's logical LLM call 在…之后 caller validation.
# │  参数：api_request_id: str, outcome: str, model_name: str | None = ..., provider_name: str | None = ..., response_model_name: str | None = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def complete_logical_call(
    api_request_id: str, *, outcome: str, model_name: str | None = None,
    provider_name: str | None = None, response_model_name: str | None = None,
) -> None:
    """Complete the active turn's logical LLM call after caller validation."""
    turn = relay_runtime.active_turn()
    if turn is None or not api_request_id:
        return
    with turn.logical_llm_lock:
        handle = turn.logical_llm_calls.get(api_request_id)
    if handle is not None:
        _complete_logical(
            (turn, handle, api_request_id), outcome=outcome, model_name=model_name,
            provider_name=provider_name, response_model_name=response_model_name,
        )


# ┌── 函数 def _response_model_name()
# │  作用：当 one 是 available. 时返回 服务商-reported 模型名。
# │  参数：response: Any
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _response_model_name(response: Any) -> str | None:
    """Return a provider-reported model name when one is available."""
    value = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
    # ⤷ 出口 return：返回 value if isinstance(value, str) and value.strip() else None
    return value if isinstance(value, str) and value.strip() else None


# ┌── 函数 def _provider_request()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：original: dict[str, Any], request: Any, relay_request_body: dict[str, Any], codec_baseline_body: dict[str, Any] | None, metadata: dict[str, Any] | None
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _provider_request(
    original: dict[str, Any], request: Any, *, relay_request_body: dict[str, Any],
    codec_baseline_body: dict[str, Any] | None, metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    content = getattr(request, "content", request)
    if not isinstance(content, dict):
        content = relay_request_body
    final = dict(original)
    if codec_baseline_body is not None and not _json_equal(content, relay_request_body):
        baseline = codec_baseline_body
        intercepted = _provider_request_body(content, metadata)
        # Typed codecs may not represent provider-specific fields: overlay only values that changed
        # from the codec-facing baseline so unrelated intercepts cannot delete/normalize unknown arguments.
        for key in baseline.keys() | intercepted.keys():
            if key not in intercepted:
                final.pop(key, None)
            elif key not in baseline or not _json_equal(intercepted[key], baseline[key]):
                final[key] = intercepted[key]
        _restore_provider_message_extensions(original, final, baseline=baseline, intercepted=intercepted)
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict):
        headers = {k: v for k, v in headers.items() if str(k).lower() not in _RELAY_INTERNAL_PROVIDER_HEADERS}
    # Relay's managed-call trace header maps to ``extra_headers`` for known SDK adapters and custom
    # requests that already use that container; other native transports take protocol kwargs directly
    # and may reject an SDK-only argument. Non-trace middleware headers are preserved as before.
    supports_extra_headers = _RELAY_PROTOCOL_BY_API_MODE.get(_api_mode(metadata)) is not None or "extra_headers" in original
    if headers and not supports_extra_headers:
        headers = {k: v for k, v in headers.items() if str(k).lower() != "traceparent"}
    if headers:
        final["extra_headers"] = {**dict(final.get("extra_headers") or {}), **headers}
    # ⤷ 出口 return：返回 final
    return final


# ┌── 函数 def _rewrite_tools()
# │  作用：Rewrite 每个字典工具那个 ``match``es (在 place 在…上 ``body["工具"]`` 当它 是列表).
# │  参数：body: dict[str, Any], match: Callable[[dict], bool], rewrite: Callable[[dict], dict]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _rewrite_tools(body: dict[str, Any], match: Callable[[dict], bool], rewrite: Callable[[dict], dict]) -> None:
    """Rewrite each dict tool that ``match``es (in place on ``body["tools"]`` when it is a list)."""
    tools = body.get("tools")
    if isinstance(tools, list):
        body["tools"] = [rewrite(t) if isinstance(t, dict) and match(t) else t for t in tools]


# ┌── 函数 def _codex_codec_tools()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：body: dict[str, Any]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _codex_codec_tools(body: dict[str, Any]) -> None:
    # The Responses SDK accepts ``tools=None`` as "no tools" while Relay's typed codec
    # wants an array or an absent field; only the codec-facing copy is normalized.
    if body.get("tools") is None:
        body.pop("tools", None)
    _rewrite_tools(
        body, lambda t: t.get("type") == "function" and "function" not in t,
        lambda t: {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}},
    )


# ┌── 函数 def _chat_codec_tools()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：body: dict[str, Any]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _chat_codec_tools(body: dict[str, Any]) -> None:
    _rewrite_tools(body, lambda t: "function" in t and "type" not in t, lambda t: {"type": "function", **t})


# api_mode -> in-place normalizer producing the codec-facing ``tools`` shape.
_CODEC_TOOL_NORMALIZERS = {"codex_responses": _codex_codec_tools, "chat_completions": _chat_codec_tools}


# ┌── 函数 def _relay_request_body()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：request: dict[str, Any], metadata: dict[str, Any] | None
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _relay_request_body(request: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    body = _jsonable_dict(request)
    # ``timeout`` configures the SDK client, not the wire: never expose it to intercepts.
    body.pop("timeout", None)
    normalize = _CODEC_TOOL_NORMALIZERS.get(_api_mode(metadata))
    if normalize is not None:
        normalize(body)
    # ⤷ 出口 return：返回 body
    return body


# ┌── 函数 def _restore_provider_message_extensions()
# │  作用：恢复服务商 wire 字段那个 Relay's typed codec 无法 represent.
# │  参数：original: dict[str, Any], final: dict[str, Any], baseline: dict[str, Any], intercepted: dict[str, Any]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _restore_provider_message_extensions(
    original: dict[str, Any], final: dict[str, Any], *, baseline: dict[str, Any], intercepted: dict[str, Any],
) -> None:
    """Restore provider wire fields that Relay's typed codec cannot represent."""
    message_lists = tuple(body.get("messages") for body in (original, final, baseline, intercepted))
    if not all(isinstance(m, list) for m in message_lists) or len({len(m) for m in message_lists}) != 1:
        return
    for messages in zip(*message_lists, strict=True):
        if not all(isinstance(message, dict) for message in messages):
            continue
        original_message, final_message, baseline_message, intercepted_message = messages
        for key in _PROVIDER_MESSAGE_EXTENSION_KEYS:
            if key in original_message and not any(
                key in m for m in (baseline_message, intercepted_message, final_message)
            ):
                final_message[key] = original_message[key]


# ┌── 函数 def _codec_round_trip_request_body()
# │  作用：返回 codec-仅请求形态 used identify real rewrites.。
# │  参数：relay: Any, relay_request: Any, relay_request_body: dict[str, Any], metadata: dict[str, Any] | None
# │  返回：dict[str, Any] | None
# └──────────────────────────────────────────────────────────────
def _codec_round_trip_request_body(
    relay: Any, relay_request: Any, *, relay_request_body: dict[str, Any], metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the codec-only request shape used to identify real rewrites."""
    codec = _codec(relay, metadata)
    if codec is None:
        # ⤷ 出口 return：返回 _provider_request_body(relay_request_body, metadata)
        return _provider_request_body(relay_request_body, metadata)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        encoded = codec.encode(codec.decode(relay_request), relay_request)
        content = getattr(encoded, "content", encoded)
    except Exception:
        logger.warning("NeMo Relay request codec baseline failed; ignoring request rewrites", exc_info=True)
        # ⤷ 出口 return：返回 None
        return None
    if isinstance(content, dict):
        # ⤷ 出口 return：返回 _provider_request_body(content, metadata)
        return _provider_request_body(content, metadata)
    logger.warning("NeMo Relay request codec returned an unsupported baseline; ignoring request rewrites")
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _provider_request_body()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：content: dict[str, Any], metadata: dict[str, Any] | None
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _provider_request_body(content: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    body = dict(content)
    if _api_mode(metadata) == "codex_responses":
        _rewrite_tools(
            body, lambda t: t.get("type") == "function" and isinstance(t.get("function"), dict),
            lambda t: {"type": "function", **dict(t["function"])},
        )
    # ⤷ 出口 return：返回 body
    return body


# ┌── 函数 def _codec()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：relay: Any, metadata: dict[str, Any] | None
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _codec(relay: Any, metadata: dict[str, Any] | None) -> Any:
    protocol = _RELAY_PROTOCOL_BY_API_MODE.get(_api_mode(metadata))
    codec = getattr(getattr(relay, "codecs", None), protocol[1], None) if protocol is not None else None
    # ⤷ 出口 return：返回 codec() if callable(codec) else None
    return codec() if callable(codec) else None


# ┌── 函数 def _jsonable()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        # ⤷ 出口 return：返回 value
        return value
    if isinstance(value, dict):
        # ⤷ 出口 return：返回 {str(key): _jsonable(item) for key, item in value.items()}
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        # ⤷ 出口 return：返回 [_jsonable(item) for item in value]
        return [_jsonable(item) for item in value]
    model_dump = getattr(type(value), "model_dump", None)
    if callable(model_dump):
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # warnings=False: pydantic's generic-union warning would leak to the terminal
            # mid-response; TypeError = duck-typed model_dump without pydantic's signature.
            try:
                return _jsonable(value.model_dump(mode="json", warnings=False))
            except TypeError:
                return _jsonable(value.model_dump())
        except Exception:
            pass
    # ⤷ 关键分支 try/except：捕获 (TypeError, AttributeError)，出错走兜底而不是崩溃。
    try:
        attributes = {str(key): item for key, item in vars(value).items() if not str(key).startswith("_")}
    except (TypeError, AttributeError):
        return str(value)
    # ⤷ 出口 return：返回 _jsonable(attributes) if attributes else str(value)
    return _jsonable(attributes) if attributes else str(value)


# ┌── 函数 def _jsonable_dict()
# │  作用：``_jsonable`` 用于值 那个必须是 JSON 对象; anything 否则 becomes ``{}``.
# │  参数：value: Any
# │  返回：dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _jsonable_dict(value: Any) -> dict[str, Any]:
    """``_jsonable`` for values that must be a JSON object; anything else becomes ``{}``."""
    payload = _jsonable(value)
    # ⤷ 出口 return：返回 payload if isinstance(payload, dict) else {}
    return payload if isinstance(payload, dict) else {}


# ┌── 函数 def _namespace()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        # ⤷ 出口 return：返回 SimpleNamespace(**{str(key): _namespace(item) for key, item …
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        # ⤷ 出口 return：返回 [_namespace(item) for item in value]
        return [_namespace(item) for item in value]
    # ⤷ 出口 return：返回 value
    return value


# ┌── 函数 def _canonical_json()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any, encode: Callable[[Any], Any] = _jsonable
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _canonical_json(value: Any, encode: Callable[[Any], Any] = _jsonable) -> str:
    # ⤷ 出口 return：返回 json.dumps(encode(value), sort_keys=True, separators=(',', '…
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"))


# ┌── 函数 def _json_equal()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：left: Any, right: Any
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _json_equal(left: Any, right: Any) -> bool:
    # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 _canonical_json(left) == _canonical_json(right)
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _run_awaitable()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any, loop_error: str = ...
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _run_awaitable(
    value: Any, *, loop_error: str = "Synchronous Relay LLM execution cannot run on an event-loop thread",
) -> Any:
    if not inspect.isawaitable(value):
        # ⤷ 出口 return：返回 value
        return value
    if _has_running_event_loop():
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise RuntimeError(loop_error)
    # ⤷ 出口 return：返回 asyncio.run(value)
    return asyncio.run(value)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import dataclass  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
