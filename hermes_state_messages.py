"""Transcript persistence for SessionDB: message append/replace/rewind, reactions, resume assembly,
replayed-user dedupe. Mixin bound via the MRO, built on SessionDB's _read_ctx/_execute_write/_read_* primitives."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：hermes_state_messages.py（约 1313 行）
# 地位：对话转录（消息）的持久化 mixin。
#
# 职责：消息的追加 / 替换 / 回退（rewind）、表情回应（reactions）、
#   恢复会话时组装消息列表、去重「重放的用户消息」。
#
# 核心类：class SessionMessagesMixin（66 个方法）
#   重点：
#     append_message                  追加一条消息
#     append_messages_batch           批量追加（性能）
#     _encode_content / _decode_content   消息内容的序列化（多模态内容是复杂结构）
#     _bump_conversation_generation   会话「代」计数器（用于并发控制/缓存失效）
#     _check_transcript_write_guards  写入前的守卫检查
#
# 实用细节：
#   _scrub_surrogates()   清理无法编码的代理字符对（surrogate pairs），SQLite 存不了会报错
#   _coerce_timestamp()   时间戳规范化
#   _parse_tool_calls()   tool_calls 的解析
#
# 配套：mixin 通过 MRO 绑定到 SessionDB，依赖 SessionDB 提供的
#   _read_ctx / _execute_write / _read_* 这些底层原语。
# ==========================================================================
#

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY, split_user_originated_turn
from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from hermes_cli.timefmt import coerce_epoch
from hermes_state_common import (
    _COMPRESSION_LOCK_ROW_SQL, _ENDED_ROW_SQL, _RESET_END_REASONS, _RESET_END_REASONS_SQL, _ended_by_compression,
    _legacy_reset_child_sql, _placeholders, _sql_json_extract)

logger = logging.getLogger("hermes_state")  # caplog tests pin the origin module's name

# One INSERT shape for every message writer (append, batch, replace, compact, import).
_INSERT_MESSAGE_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind,
                   display_metadata, display_identity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
_BUMP_GENERATION_SQL = """
            INSERT INTO conversation_generations (source, session_key, generation)
            VALUES (?, ?, 1)
            ON CONFLICT(source, session_key) DO UPDATE
                SET generation = conversation_generations.generation + 1
            """

_TURN_LEASE_ROW_SQL = "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?"
_DELETE_COMPRESSION_LOCK_SQL = "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?"
_DISPLAY_ACTIVE_CLAUSE = " AND (active = 1 OR compacted = 1)"
_DISPLAY_META_ROW_SQL = "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?"
_ACTIVE_IDS_SQL = "SELECT id FROM messages WHERE session_id = ? AND active = 1 ORDER BY id"
_SET_COUNTERS_SQL = "UPDATE sessions SET message_count = ?, tool_call_count = ?"
_RESET_COUNTERS_SQL = "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?"
_SET_DISPLAY_META_SQL = "UPDATE messages SET display_metadata = ? WHERE id = ?"
_ARCHIVE_ACTIVE_SQL = "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ? AND active = 1"
_INVALID = object()  # _json_or sentinel where the fallback must be distinguishable from JSON null


# ┌── 函数 def _json_or()
# │  作用：``JSON.加载(raw)``; 在…上 failure 日志 *警告* 且返回 *兜底*.
# │  参数：raw: Any, fallback: Any, warning: str
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _json_or(raw: Any, fallback: Any, warning: str) -> Any:
    """``json.loads(raw)``; on failure log *warning* and return *fallback*."""
    # ⤷ 关键分支 try/except：捕获 (json.JSONDecodeError, TypeError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 json.loads(raw)
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(warning)
        # ⤷ 出口 return：返回 fallback
        return fallback


# ┌── 函数 def _coerce_timestamp()
# │  作用：Explicit 消息时间戳 (datetime 或 number) 或 *默认值* 当 invalid 或 outside sane
# │  参数：value: Any, default: float
# │  返回：float
# └──────────────────────────────────────────────────────────────
def _coerce_timestamp(value: Any, default: float) -> float:
    """Explicit message timestamp (datetime or number) or *default* when invalid or outside the sane
    epoch window — the write-side twin of the readers' ``coerce_epoch``: a bad row is never persisted."""
    result = coerce_epoch(value, field="message timestamp")
    if result is None:
        # ⤷ 出口 return：返回 default
        return default
    # SQLite reads both signed zero spellings back as 0.0. Hash the value
    # in that stored form so -0.0 and 0.0 retain the historical SQL/Python
    # identity equality.
    # ⤷ 出口 return：返回 0.0 if result == 0.0 else result
    return 0.0 if result == 0.0 else result


# ┌── 函数 def _parse_tool_calls()
# │  作用：tool_calls 是列表 (当前活跃的 Agent) 或 JSON 字符串 (导入/export); 解析从而 JSON.dumps never double-encodes.
# │  参数：tool_calls: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _parse_tool_calls(tool_calls: Any) -> Any:
    """tool_calls is a list (live agent) or JSON string (import/export); parse so json.dumps never double-encodes."""
    if not isinstance(tool_calls, str):
        # ⤷ 出口 return：返回 tool_calls
        return tool_calls
    # ⤷ 关键分支 try/except：捕获 (json.JSONDecodeError, TypeError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 json.loads(tool_calls)
        return json.loads(tool_calls)
    except (json.JSONDecodeError, TypeError):
        # ⤷ 出口 return：返回 []
        return []


# ┌── 函数 def _tool_calls_count()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：tool_calls: Any
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _tool_calls_count(tool_calls: Any) -> int:
    # ⤷ 出口 return：返回 0 if tool_calls is None else len(tool_calls) if isinstance(t…
    return 0 if tool_calls is None else (len(tool_calls) if isinstance(tool_calls, list) else 1)


# ┌── 函数 def _tool_calls_len()
# │  作用：Count 的 stored ``tool_calls`` column: 列表 length, *scalar* 用于 truthy non-列表, 否则 0.
# │  参数：raw: Any, scalar: int = 0
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _tool_calls_len(raw: Any, scalar: int = 0) -> int:
    """Count of a stored ``tool_calls`` column: list length, *scalar* for a truthy non-list, else 0."""
    parsed = _parse_tool_calls(raw)
    # ⤷ 出口 return：返回 len(parsed) if isinstance(parsed, list) else scalar if parse…
    return len(parsed) if isinstance(parsed, list) else (scalar if parsed else 0)


# ┌── 函数 def _scrub_surrogates()
# │  作用：Lone surrogates make sqlite3 抛出 UnicodeEncodeError 且 abort whole 写入.
# │  参数：value: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _scrub_surrogates(value: Any) -> Any:
    """Lone surrogates make sqlite3 raise UnicodeEncodeError and abort the whole write."""
    # ⤷ 出口 return：返回 _sanitize_surrogates(value) if isinstance(value, str) else v…
    return _sanitize_surrogates(value) if isinstance(value, str) else value


# ┌── 函数 def _stale_holder()
# │  作用：锁/lease row whose holder 是 expired 或 provably dead local 进程.
# │  参数：row, now: float
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _stale_holder(row, now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process."""
    from hermes_state import _compression_lock_holder_process_is_dead
    # ⤷ 出口 return：返回 float(row['expires_at']) <= now or _compression_lock_holder_…
    return float(row["expires_at"]) <= now or _compression_lock_holder_process_is_dead(row["holder"])


# ┌── 类 class SessionMessagesMixin
# │  说明：消息追加/replace/rewind, reactions, resume conversations, 重放 dedupe.
# │  成员：共 66 个方法；对外可用 35 个
# └──────────────────────────────────────────────────────────────
class SessionMessagesMixin:
    """Message append/replace/rewind, reactions, resume conversations, replay dedupe."""

    # ┌── 函数 def _bump_conversation_generation()
    # │  作用：Advance peer's conversation generation past boundary, 在 txn 那个写入它. 仅
    # │  参数：self, conn, session_id: str, end_reason: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _bump_conversation_generation(self, conn, session_id: str, end_reason: str) -> None:
        """Advance the peer's conversation generation past a boundary, in the txn that writes it. Only
        ``_RESET_END_REASONS`` count (compression continues one conversation). Never derived from session
        rows (deletes/prunes could re-emit a retired affinity identity); it only ever increments."""
        if end_reason not in _RESET_END_REASONS:
            return
        row = conn.execute("SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return
        source, session_key = (str(row[k] or "").strip() for k in ("source", "session_key"))
        if source and session_key:
            conn.execute(_BUMP_GENERATION_SQL, (source, session_key))

    # ┌── 函数 def _encode_content()
    # │  作用：Serialize 列表/字典内容 (multimodal parts) 作为 sentinel-prefixed JSON 字符串 (sqlite3 binds
    # │  参数：cls, content: Any
    # │  返回：Any
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize list/dict content (multimodal parts) as a sentinel-prefixed JSON string (sqlite3 binds
        only scalars). Lone UTF-16 surrogates (unsanitized web-scraped tool results) are scrubbed here: left
        raw, sqlite3 raises UnicodeEncodeError and the session silently stops persisting. Pairs with
        :meth:`_decode_content`."""
        if isinstance(content, str):
            # ⤷ 出口 return：返回 _sanitize_surrogates(content)
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            # ⤷ 出口 return：返回 content
            return content
        # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 cls._CONTENT_JSON_PREFIX + json.dumps(content)
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)  # ensure_ascii escapes surrogates: bindable
        except (TypeError, ValueError):
            # ⤷ 出口 return：返回 _sanitize_surrogates(str(content))
            return _sanitize_surrogates(str(content))

    # ┌── 函数 def _decode_content()
    # │  作用：Reverse :meth:`_encode_content`; 返回 scalars unchanged.
    # │  参数：cls, content: Any
    # │  返回：Any
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            # ⤷ 出口 return：返回 _json_or(content[len(cls._CONTENT_JSON_PREFIX):], content, '…
            return _json_or(content[len(cls._CONTENT_JSON_PREFIX):], content,
                "Failed to decode JSON-encoded message content; returning raw string")
        # ⤷ 出口 return：返回 content
        return content

    # ┌── 函数 def _encode_display_metadata()
    # │  作用：Serialize ``display_metadata`` 用于其 文本 column; 已-serialized JSON 字符串
    # │  参数：display_metadata: Any
    # │  返回：Optional[str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column; an already-serialized JSON string
        (import/replace paths) is not double-encoded."""
        if not display_metadata:
            # ⤷ 出口 return：返回 None
            return None
        if isinstance(display_metadata, str):
            display_metadata = _json_or(display_metadata, _INVALID, "Ignoring non-JSON display metadata on write")
            if display_metadata is _INVALID:
                # ⤷ 出口 return：返回 None
                return None
            if not isinstance(display_metadata, dict):
                logger.warning("Ignoring non-object display metadata on write")
                # ⤷ 出口 return：返回 None
                return None
        elif not isinstance(display_metadata, dict):
            logger.warning("Ignoring unexpected display metadata type on write: %s", type(display_metadata).__name__)
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 json.dumps(display_metadata)
        return json.dumps(display_metadata)

    # ┌── 函数 def _decode_display_metadata()
    # │  作用：Decode ``display_metadata`` column 字典 (never 原始文本: desktop ``'task_count'
    # │  参数：raw: Any
    # │  返回：Optional[Dict[str, Any]]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column to a dict (never raw TEXT: the desktop does ``'task_count'
        in meta``). Pre-guard rows are double-encoded, so a second string layer is unwrapped."""
        if raw is None:
            # ⤷ 出口 return：返回 None
            return None
        meta = raw
        for _ in range(2):  # pre-guard rows carry a second string layer
            if isinstance(meta, str):
                meta = _json_or(meta, _INVALID, "Ignoring invalid display metadata on message row")
        if meta is _INVALID:
            # ⤷ 出口 return：返回 None
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 meta
        return meta

    # ┌── 函数 def _reasoning_json_text()
    # │  作用：Serialize 结构化的 reasoning 字段用于其 文本 column. Strings 是 stored 作为-是: 轮-trips
    # │  参数：value: Any
    # │  返回：Optional[str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _reasoning_json_text(value: Any) -> Optional[str]:
        """Serialize a structured reasoning field for its TEXT column. Strings are stored as-is: round-trips
        (get_messages -> replace_messages) hand back raw TEXT; re-dumping would double-encode it and
        reasoning-replay consumers (``isinstance(..., list)``) would drop it."""
        # ⤷ 出口 return：返回 None if not value else value if isinstance(value, str) else …
        return None if not value else (value if isinstance(value, str) else json.dumps(value))

    # ┌── 函数 def _check_transcript_write_guards()
    # │  作用：Transcript-写入 admission 检查, 运行 INSIDE 写入 txn 由每个 writer. Ordinary 追加
    # │  参数：self, conn, session_id: str, compression_lock_holder: Optional[str], turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0, reject_active_turn_lease: bool = False, reject_active_compression_lock: bool = False  …共 9 个
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _check_transcript_write_guards(self, conn, session_id: str, compression_lock_holder: Optional[str],
        turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False, reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False) -> None:
        """Transcript-write admission checks, run INSIDE the write txn by every writer. Ordinary appends do
        NOT check compression_locks: the lock only stops two COMPRESSIONS colliding and archive_and_compact()
        commits against a watermark, so concurrent appends are safe (blocking them killed turns during slow
        summaries). Destructive user mutations opt in via ``reject_active_*`` so a compressor that captured
        its watermark cannot resurrect the removed turn.

        Shared by :meth:`append_message` and :meth:`append_messages_batch` so the two writers can never
        diverge on these correctness invariants (this guard has already needed targeted fixes — see the
        #74478 patience note below). User-initiated transcript mutations may opt in to rejecting an active
        unowned turn lease in that same transaction.
        """
        from hermes_state import SessionCompressionInProgressError
        from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError
        # NOTE (#75316 redesign): appends do NOT check compression_locks. The lock's job is to stop two
        # COMPRESSIONS colliding, not to fence ordinary transcript writes. Concurrent appends during a
        # compression are safe by construction: archive_and_compact() commits against a watermark captured
        # at compression start and clones every row that arrived after it back into the live transcript, in
        # the same write transaction. Blocking appends here was the root cause of a whole symptom family —
        # turns dying as session_persistence_failed while a slow provider summary held the lease (#74568,
        # #77386), including stale locks from dead PIDs blocking writes for the full TTL. Keep that narrow
        # fence opt-in so ordinary appends retain the watermark behavior.
        if reject_active_compression_lock:
            active_lock = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
            if active_lock is not None:
                if _stale_holder(active_lock, time.time()):
                    conn.execute(_DELETE_COMPRESSION_LOCK_SQL, (session_id, active_lock["holder"]))
                elif active_lock["holder"] != compression_lock_holder:
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer")
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            lease = conn.execute(_TURN_LEASE_ROW_SQL, (conversation_id,)).fetchone()
            now = time.time()
            if turn_lease_holder:
                if lease is None or lease["holder"] != turn_lease_holder:
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write for {session_id!r}")
                if float(lease["expires_at"]) <= now:
                    # Expiry makes the row reclaimable, not taken over; BEGIN IMMEDIATE serializes this
                    # renewal with acquisition, so a still-matching owner recovers from a starved refresher.
                    conn.execute("UPDATE session_turn_leases SET expires_at = ? "
                        "WHERE conversation_id = ? AND holder = ?",
                        (now + max(0.1, float(turn_lease_ttl_seconds)), conversation_id, turn_lease_holder))
            elif lease is not None:
                if not _stale_holder(lease, now):
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript mutation for {session_id!r}")
                # Same reclaim rule as acquisition; deleting also fences a stale late flush after the mutation.
                conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                    (conversation_id, lease["holder"]))
        if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()) and not allow_closed_compression_parent:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise CompressionSessionClosedError(session_id)

    # ┌── 函数 def _message_row_params()
    # │  作用：Bind 值用于 ``_INSERT_MESSAGE_SQL`` 从 one 消息字典 (*tool_calls* 已已解析;
    # │  参数：self, session_id: str, role: str, msg: Dict[str, Any], tool_calls: Any, message_timestamp: float, keep_reasoning: bool
    # │  返回：tuple
    # └──────────────────────────────────────────────────────────────
    def _message_row_params(self, session_id: str, role: str, msg: Dict[str, Any], tool_calls: Any,
        message_timestamp: float, *, keep_reasoning: bool) -> tuple:
        """Bind values for ``_INSERT_MESSAGE_SQL`` from one message dict (*tool_calls* already parsed;
        *keep_reasoning* False NULLs every reasoning column). ``platform_message_id`` falls back to
        ``message_id`` (yuanbao's message-dict convention)."""
        _str_or_none = lambda v: _scrub_surrogates(v) if isinstance(v, str) else None  # noqa: E731
        _reasoning = lambda key: msg.get(key) if keep_reasoning else None  # noqa: E731
        encoded_content = self._encode_content(msg.get("content"))
        encoded_tool_calls = json.dumps(tool_calls) if tool_calls else None
        encoded_tool_name = _scrub_surrogates(msg.get("tool_name"))
        display_metadata = self._encode_display_metadata(msg.get("display_metadata"))
        identity_row = {
            "role": role, "content": encoded_content, "timestamp": message_timestamp,
            "tool_call_id": msg.get("tool_call_id"), "tool_calls": encoded_tool_calls,
            "tool_name": encoded_tool_name, "display_kind": msg.get("display_kind"),
            "display_metadata": display_metadata,
        }
        # ⤷ 出口 return：返回 (session_id, role, encoded_content, msg.get('tool_call_id'),…
        return (session_id, role, encoded_content, msg.get("tool_call_id"),
            encoded_tool_calls, encoded_tool_name,
            msg.get("effect_disposition"), message_timestamp, msg.get("token_count"), msg.get("finish_reason"),
            _scrub_surrogates(_reasoning("reasoning")), _scrub_surrogates(_reasoning("reasoning_content")),
            *(self._reasoning_json_text(_reasoning(k))
              for k in ("reasoning_details", "codex_reasoning_items", "codex_message_items")),
            msg.get("platform_message_id") or msg.get("message_id"),
            1 if msg.get("observed") else 0, 1 if msg.get("_compressed_summary") else 0, 1,
            _str_or_none(msg.get("api_content")), _str_or_none(msg.get("display_kind")),
            display_metadata, self._display_identity(self._display_dedupe_key(identity_row)))

    # ┌── 函数 def _bump_session_counters()
    # │  作用：递增会话.* counters 在…之后 insert; *unit* bakes ``+ 1`` literal 进入 SQL.
    # │  参数：conn, session_id: str, inserted: int, tool_calls: int, unit: bool
    # │  返回：None
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _bump_session_counters(conn, session_id: str, inserted: int, tool_calls: int, *, unit: bool) -> None:
        """Bump sessions.* counters after an insert; *unit* bakes the ``+ 1`` literal into the SQL."""
        inc, params = ("1", ()) if unit else ("?", (inserted,))
        if tool_calls > 0:
            conn.execute(
                f"""UPDATE sessions SET message_count = message_count + {inc},
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                (*params, tool_calls, session_id))
        elif inserted > 0:
            conn.execute(
                f"UPDATE sessions SET message_count = message_count + {inc} WHERE id = ?", (*params, session_id))

    # ┌── 函数 def append_message()
    # │  作用：追加 one 消息; 返回 row id 且递增会话 counters. ``platform_message_id``:
    # │  参数：self, session_id: str, role: str, content: str = None, tool_name: str = None, tool_calls: Any = None, tool_call_id: str = None, token_count: int = None  …共 25 个
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def append_message(
        self, session_id: str, role: str, content: str = None, tool_name: str = None, tool_calls: Any = None,
        tool_call_id: str = None, token_count: int = None, finish_reason: str = None, reasoning: str = None,
        reasoning_content: str = None, reasoning_details: Any = None, codex_reasoning_items: Any = None,
        codex_message_items: Any = None, platform_message_id: str = None, observed: bool = False,
        effect_disposition: Optional[str] = None, _compressed_summary: bool = False, timestamp: Any = None,
        api_content: Optional[str] = None, display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None, compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> int:
        """Append one message; returns the row id and bumps the session counters. ``platform_message_id``:
        the platform's own id. ``api_content``: byte-fidelity sidecar, the exact string sent to the API when
        it differed from ``content``, stored as sent except lone surrogates."""
        msg = dict(locals())  # every keyword above is a message-dict field of the same name
        # Encode outside the write txn (display metadata first: log-order parity).
        msg["display_metadata"] = self._encode_display_metadata(display_metadata)
        tool_calls = _parse_tool_calls(tool_calls)
        message_timestamp = _coerce_timestamp(timestamp, time.time())
        params = self._message_row_params(
            session_id, role, msg, tool_calls, message_timestamp, keep_reasoning=True)
        def _do(conn):
            self._check_transcript_write_guards(conn, session_id, compression_lock_holder,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            self._bump_session_counters(conn, session_id, 1, _tool_calls_count(tool_calls), unit=True)
            # ⤷ 出口 return：返回 msg_id
            return msg_id
        # THE critical write (failure aborts the turn): long patience so a sibling legitimately
        # holding the lock for seconds (VACUUM, checkpoint) can't kill it.
        # ⤷ 出口 return：返回 self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_P…
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    # ┌── 函数 def append_delegation_delivery()
    # │  作用：Record detached API 结果一旦, between 客户端轮次, including 重放在…之后 rotation.
    # │  参数：self, session_id: str, content: str, metadata: Dict[str, Any]
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def append_delegation_delivery(self, session_id: str, content: str, metadata: Dict[str, Any]) -> int:
        """Record a detached API result once, between client turns, including replay after rotation.

        The event's unit id, not its text or active flag, is the identity. Check and insert
        share the writer transaction, so independent gateway processes cannot duplicate it.
        """
        delegation_id = metadata.get("delegation_id")
        if not delegation_id:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("Delegation delivery requires a stable delegation_id")
        msg = {"content": content, "display_kind": "async_delegation_complete", "display_metadata": metadata}
        params = self._message_row_params(session_id, "user", msg, None, time.time(), keep_reasoning=True)

        def _do(conn):
            existing = conn.execute(
                """WITH RECURSIVE lineage(id) AS (
                    SELECT ? UNION
                    SELECT s.parent_session_id FROM sessions s JOIN lineage l ON s.id = l.id
                    JOIN sessions p ON p.id = s.parent_session_id WHERE p.end_reason = 'compression'
                ) SELECT m.id FROM messages m JOIN lineage l ON m.session_id = l.id
                WHERE m.display_kind = 'async_delegation_complete'
                AND json_extract(m.display_metadata, '$.delegation_id') = ?
                AND coalesce(json_extract(m.display_metadata, '$.delivery_notice'), '') = ? LIMIT 1""",
                (session_id, delegation_id, metadata.get("delivery_notice", ""))).fetchone()
            if existing is not None:
                # ⤷ 出口 return：返回 existing[0]
                return existing[0]
            self._check_transcript_write_guards(conn, session_id, None, reject_active_turn_lease=True)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            self._bump_session_counters(conn, session_id, 1, 0, unit=True)
            # ⤷ 出口 return：返回 msg_id
            return msg_id

        # ⤷ 出口 return：返回 self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_P…
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    # ┌── 函数 def append_messages_batch()
    # │  作用：追加 *消息* 在 ONE 写入 txn (全部 rows land 或 None, 护栏运行一旦); 返回 inserted
    # │  参数：self, session_id: str, messages: List[Dict[str, Any]], compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None, turn_lease_ttl_seconds: float = 300.0
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def append_messages_batch(
        self, session_id: str, messages: List[Dict[str, Any]], compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None,
        turn_lease_ttl_seconds: float = 300.0) -> int:
        """Append *messages* in ONE write txn (all rows land or none, guards run once); returns the inserted
        count. ``chunk_rows`` bounds txn size for LARGE copies (branch seeds; FTS triggers run per row)."""
        if not messages:
            # ⤷ 出口 return：返回 0
            return 0
        if chunk_rows is not None and len(messages) > chunk_rows:
            # ⤷ 出口 return：返回 sum((self.append_messages_batch(session_id, messages[start:s…
            return sum(self.append_messages_batch(session_id, messages[start:start + chunk_rows],
                    compression_lock_holder=compression_lock_holder, turn_lease_holder=turn_lease_holder,
                    turn_lease_ttl_seconds=turn_lease_ttl_seconds)
                for start in range(0, len(messages), chunk_rows))
        def _do(conn):
            self._check_transcript_write_guards(conn, session_id, compression_lock_holder,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            from agent.transcript_repair import resolve_and_repair_transcript_batch
            inserted_rows = resolve_and_repair_transcript_batch(conn, session_id, messages,
                encode_content_fn=self._encode_content, decode_content_fn=self._decode_content)
            inserted, tool_calls_total = self._insert_message_rows(conn, session_id, inserted_rows)
            self._bump_session_counters(conn, session_id, inserted, tool_calls_total, unit=False)
            # ⤷ 出口 return：返回 inserted
            return inserted
        # ⤷ 出口 return：返回 self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_P…
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    # ┌── 函数 def set_latest_matching_message_display_kind()
    # │  作用：Stamp presentation metadata 在…上这个轮次's freshly 已持久化的 row (newest active row 由内容,
    # │  参数：self, session_id: str, role: str, content: str, display_kind: str, display_metadata: Optional[Dict[str, Any]] = ...
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def set_latest_matching_message_display_kind(self, session_id: str, *, role: str, content: str,
                                                 display_kind: str,
                                                 display_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Stamp presentation metadata on this turn's freshly persisted row (newest active row by content,
        right after the serial turn flushed); the model still sees ``role``/``content`` unchanged, so
        producer provenance survives without classifying by content at render time."""
        if not session_id or not content or not display_kind:
            # ⤷ 出口 return：返回 False
            return False
        def _do(conn):
            row = conn.execute("SELECT id FROM messages WHERE session_id = ? AND role = ? "
                "AND content = ? AND active = 1 ORDER BY id DESC LIMIT 1",
                (session_id, role, self._encode_content(content))).fetchone()
            if row is None:
                # ⤷ 出口 return：返回 False
                return False
            conn.execute("UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?",
                (_scrub_surrogates(display_kind), self._encode_display_metadata(display_metadata), row[0]))
            # ⤷ 出口 return：返回 True
            return True
        # ⤷ 出口 return：返回 self._execute_write(_do)
        return self._execute_write(_do)

    # ┌── 函数 def _reaction_list()
    # │  作用：Well-formed (字典) reactions stored under ``REACTIONS_METADATA_KEY``.
    # │  参数：self, meta: Optional[Dict[str, Any]]
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def _reaction_list(self, meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Well-formed (dict) reactions stored under ``REACTIONS_METADATA_KEY``."""
        reactions = (meta or {}).get(self.REACTIONS_METADATA_KEY)
        # ⤷ 出口 return：返回 [r for r in reactions if isinstance(r, dict)] if isinstance(…
        return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []

    # ┌── 函数 def set_message_reaction()
    # │  作用：设置 (``emoji=None``: 清除) *author*'s reaction. Tapback semantics: one per author per 消息;
    # │  参数：self, session_id: str, message_row_id: int, emoji: Optional[str], author: str = ...
    # │  返回：Optional[List[Dict[str, Any]]]
    # └──────────────────────────────────────────────────────────────
    def set_message_reaction(self, session_id: str, message_row_id: int, emoji: Optional[str], *,
                             author: str = "user") -> Optional[List[Dict[str, Any]]]:
        """Set (``emoji=None``: clear) *author*'s reaction. Tapback semantics: one per author per message;
        the same emoji again clears, a different one replaces. Returns the list after the write, or
        ``None`` for a foreign row."""
        if not session_id or message_row_id is None:
            # ⤷ 出口 return：返回 None
            return None
        def _do(conn):
            row = conn.execute(_DISPLAY_META_ROW_SQL, (message_row_id, session_id)).fetchone()
            if row is None:
                # ⤷ 出口 return：返回 None
                return None
            meta = self._decode_display_metadata(row[0]) or {}
            existing = self._reaction_list(meta)
            reactions = [r for r in existing if r.get("author") != author]
            previous = next((r for r in existing if r.get("author") == author), None)
            if emoji and (previous is None or previous.get("emoji") != emoji):
                reactions.append({"emoji": _scrub_surrogates(emoji), "author": author, "at": time.time()})
            if reactions:
                meta[self.REACTIONS_METADATA_KEY] = reactions
            else:
                meta.pop(self.REACTIONS_METADATA_KEY, None)
            conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta) if meta else None, message_row_id))
            # ⤷ 出口 return：返回 reactions
            return reactions
        # ⤷ 出口 return：返回 self._execute_write(_do)
        return self._execute_write(_do)

    # ┌── 函数 def get_message_reactions()
    # │  作用：Reaction 列表已持久化的在…上 one 消息 row (never ``None``).
    # │  参数：self, session_id: str, message_row_id: int
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def get_message_reactions(self, session_id: str, message_row_id: int) -> List[Dict[str, Any]]:
        """Reaction list persisted on one message row (never ``None``)."""
        if not session_id or message_row_id is None:
            # ⤷ 出口 return：返回 []
            return []
        row = self._read_one(_DISPLAY_META_ROW_SQL, (message_row_id, session_id))
        # ⤷ 出口 return：返回 self._reaction_list(self._decode_display_metadata(row[0])) i…
        return self._reaction_list(self._decode_display_metadata(row[0])) if row is not None else []

    # ┌── 函数 def take_unseen_reactions()
    # │  作用：返回 *author*'s 不-yet-已上报 reactions 且 mark them seen. Reactions 是 announced 在…上。
    # │  参数：self, session_id: str, author: str = ...
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def take_unseen_reactions(self, session_id: str, *, author: str = "user") -> List[Dict[str, Any]]:
        """Return *author*'s not-yet-surfaced reactions and mark them seen. Reactions are announced on the
        NEXT user turn (never by rewriting the reacted message: cache-safe); ``seen`` makes it exactly once."""
        if not session_id:
            # ⤷ 出口 return：返回 []
            return []
        def _do(conn):
            pending = []
            for row in conn.execute("SELECT id, role, content, display_metadata FROM messages "
                    "WHERE session_id = ? AND active = 1 AND display_metadata IS NOT NULL ORDER BY id",
                    (session_id,)).fetchall():
                meta = self._decode_display_metadata(row["display_metadata"])
                reactions = meta.get(self.REACTIONS_METADATA_KEY) if meta else None
                if not isinstance(reactions, list):
                    continue
                changed = False
                for reaction in reactions:
                    if not isinstance(reaction, dict) or reaction.get("author") != author or reaction.get("seen"):
                        continue
                    reaction["seen"] = True
                    changed = True
                    content = self._decode_content(row["content"])
                    pending.append({
                        "row_id": row["id"], "role": row["role"], "emoji": reaction.get("emoji") or "",
                        "text": content if isinstance(content, str) else ""})
                if changed:
                    conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta), row["id"]))
            # ⤷ 出口 return：返回 pending
            return pending
        # ⤷ 出口 return：返回 self._execute_write(_do)
        return self._execute_write(_do)

    # ┌── 函数 def latest_message_row_id()
    # │  作用：Row id 的 most recent active *角色* 消息, 或 ``None``. ``offset`` 步骤 back; ``require_text``
    # │  参数：self, session_id: str, role: str = ..., offset: int = ..., require_text: bool = ...
    # │  返回：Optional[int]
    # └──────────────────────────────────────────────────────────────
    def latest_message_row_id(self, session_id: str, *, role: str = "user", offset: int = 0,
                              require_text: bool = True) -> Optional[int]:
        """Row id of the most recent active *role* message, or ``None``. ``offset`` steps back; ``require_text``
        skips rows without plain-text content so "the latest message" never resolves to an invisible bubble."""
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            # ⤷ 出口 return：返回 None
            return None
        text_filter = "AND content IS NOT NULL AND TRIM(content) != '' " if require_text else ""
        row = self._read_one("SELECT id FROM messages WHERE session_id = ? AND role = ? "
            f"AND active = 1 {text_filter}ORDER BY id DESC LIMIT 1 OFFSET ?",
            (session_id, role, int(offset)))
        # ⤷ 出口 return：返回 row[0] if row else None
        return row[0] if row else None

    # ┌── 函数 def latest_conversation_role()
    # │  作用：角色的 newest active user/assistant/工具 row, 或 ``None``. ``session_meta`` /
    # │  参数：self, session_id: str
    # │  返回：Optional[str]
    # └──────────────────────────────────────────────────────────────
    def latest_conversation_role(self, session_id: str) -> Optional[str]:
        """Role of the newest active user/assistant/tool row, or ``None``. ``session_meta`` /
        ``system`` rows are transcript bookkeeping stripped before the model sees history, so
        they must not hide an open user tail from the failed-turn boundary check."""
        row = self._read_one(
            "SELECT role FROM messages WHERE session_id = ? AND active = 1 "
            "AND role NOT IN ('session_meta', 'system') ORDER BY id DESC LIMIT 1", (session_id,))
        # ⤷ 出口 return：返回 row[0] if row else None
        return row[0] if row else None

    # ┌── 函数 def get_message_role()
    # │  作用：角色的 active 消息在 *row_id* 在 *session_id*, 或 ``None``.
    # │  参数：self, session_id: str, row_id: int
    # │  返回：Optional[str]
    # └──────────────────────────────────────────────────────────────
    def get_message_role(self, session_id: str, row_id: int) -> Optional[str]:
        """Role of the active message at *row_id* in *session_id*, or ``None``."""
        if not session_id:
            # ⤷ 出口 return：返回 None
            return None
        row = self._read_one("SELECT role FROM messages WHERE id = ? AND session_id = ? AND active = 1", (int(row_id), session_id))
        # ⤷ 出口 return：返回 row[0] if row else None
        return row[0] if row else None

    # ┌── 函数 def _insert_message_rows()
    # │  作用：Insert *消息* 作为 fresh active rows 在 caller's txn -> ``(inserted, tool_call_count)``.
    # │  参数：self, conn, session_id: str, messages: List[Dict[str, Any]]
    # │  返回：tuple[int, int]
    # └──────────────────────────────────────────────────────────────
    def _insert_message_rows(self, conn, session_id: str, messages: List[Dict[str, Any]]) -> tuple[int, int]:
        """Insert *messages* as fresh active rows in the caller's txn -> ``(inserted, tool_call_count)``.
        Never touches sessions.* counters (callers reconcile differently); reasoning kept for assistant rows."""
        now_ts = time.time()
        inserted = tool_calls_total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            message_timestamp = _coerce_timestamp(msg.get("timestamp"), now_ts)
            cur = conn.execute(_INSERT_MESSAGE_SQL, self._message_row_params(
                session_id, role, msg, tool_calls, message_timestamp, keep_reasoning=role == "assistant"))
            if cur.lastrowid is not None:
                msg["_row_id"] = cur.lastrowid
            inserted += 1
            tool_calls_total += _tool_calls_count(tool_calls)
            now_ts = max(now_ts, message_timestamp) + 1e-6
        # ⤷ 出口 return：返回 (inserted, tool_calls_total)
        return inserted, tool_calls_total

    # ┌── 函数 def replace_messages()
    # │  作用：Atomically replace 会话's 消息 (/重试, /undo, /compress). DESTRUCTIVE 由默认值 (rows
    # │  参数：self, session_id: str, messages: List[Dict[str, Any]], active_only: bool = False, archive_dropped: bool = False, reject_active_turn_lease: bool = False
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]], active_only: bool = False,
        archive_dropped: bool = False, reject_active_turn_lease: bool = False) -> None:
        """Atomically replace a session's messages (/retry, /undo, /compress). DESTRUCTIVE by default (rows
        DELETEd, leave FTS). ``active_only`` spares soft-archived rows (needed with in-place compaction).
        ``archive_dropped`` SOFT-archives live rows rewind-style: what rewind/edit/regenerate must use, since
        DELETE leaves nothing to recover. ``reject_active_turn_lease``: in-txn lease check for user rewrites.

        Pass ``archive_dropped=True`` to SOFT-archive the live rows instead of DELETEing them: the replaced
        turns stay on disk with ``active = 0``, ``compacted = 0`` — the same "the user took it back" marking
        :meth:`rewind_to_message` applies — and stay readable via :meth:`get_messages` with
        ``include_inactive=True``. This is the mode a rewind/edit/regenerate must use: those flows overwrite
        a transcript the user may not have meant to drop, and a plain DELETE also evicts the rows from the
        FTS index, leaving nothing to recover from (#82756). It implies active-only handling —
        already-archived rows are never touched — so ``active_only`` is redundant with it. The rewritten set
        is inserted as fresh active rows exactly as in the destructive path, so the live view is identical
        either way; only the durability of the dropped turns differs.
        """
        from hermes_state_errors import CompressionSessionClosedError
        def _do(conn):
            if reject_active_turn_lease:
                self._check_transcript_write_guards(
                    conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
            elif _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise CompressionSessionClosedError(session_id)
            if archive_dropped:
                # FTS triggers don't fire on `active`: replaced turns stay searchable (include_inactive=True).
                conn.execute("UPDATE messages SET active = 0 WHERE session_id = ? AND active = 1", (session_id,))
            else:
                conn.execute(f"DELETE FROM messages WHERE session_id = ?{' AND active = 1' if active_only else ''}", (session_id,))
            conn.execute(_RESET_COUNTERS_SQL, (session_id,))
            total_messages, total_tool_calls = self._insert_message_rows(conn, session_id, messages)
            conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?", (total_messages, total_tool_calls, session_id))
        self._execute_write(_do)

    # ┌── 函数 def has_archived_messages()
    # │  作用：True 若会话有 任意 soft-archived (``active = 0``) rows (测试/diagnostics).
    # │  参数：self, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_archived_messages(self, session_id: str) -> bool:
        """True if the session has any soft-archived (``active = 0``) rows (tests/diagnostics).

        Cheap existence probe — does not load rows. NOTE: production rewrite paths no longer branch on this
        (they pass ``active_only=True`` unconditionally — a probe can fail open or race a concurrent
        ``archive_and_compact``, #80216); kept for tests and diagnostics.
        """
        # ⤷ 出口 return：返回 self._read_one('SELECT 1 FROM messages WHERE session_id = ? …
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1", (session_id,)) is not None

    # ┌── 函数 def get_active_message_watermark()
    # │  作用：MAX(id) 的 active rows (0 若 None), captured 在 compression START: 每个 active row above 它
    # │  参数：self, session_id: str
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def get_active_message_watermark(self, session_id: str) -> int:
        """MAX(id) of the active rows (0 if none), captured at compression START: every active row above it
        arrived concurrently and must survive compaction verbatim."""
        if not session_id:
            # ⤷ 出口 return：返回 0
            return 0
        # ⤷ 出口 return：返回 int(self._read_one('SELECT COALESCE(MAX(id), 0) FROM message…
        return int(self._read_one(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1", (session_id,))[0])

    # ┌── 函数 def _tail_rows_after_watermark()
    # │  作用：``(ids, tool_call_count)`` 的 concurrent-tail rows selected 由 *sql* (``SELECT id, tool_calls``).
    # │  参数：self, conn, sql: str, params
    # │  返回：Tuple[List[int], int]
    # └──────────────────────────────────────────────────────────────
    def _tail_rows_after_watermark(self, conn, sql: str, params) -> Tuple[List[int], int]:
        """``(ids, tool_call_count)`` of the concurrent-tail rows selected by *sql* (``SELECT id, tool_calls``)."""
        rows = conn.execute(sql, params).fetchall()
        # ⤷ 出口 return：返回 ([int(r['id']) for r in rows], sum((_tool_calls_len(r['tool_…
        return [int(r["id"]) for r in rows], sum(_tool_calls_len(r["tool_calls"]) for r in rows)

    # ┌── 函数 def _clone_message_rows()
    # │  作用：Pure-SQL clone 的 *tail_ids* 作为 fresh 当前活跃的 rows (new id/展示 order, active=1, compacted=0;
    # │  参数：self, conn, tail_ids: List[int], session_id: Optional[str] = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _clone_message_rows(self, conn, tail_ids: List[int], *, session_id: Optional[str] = None) -> None:
        """Pure-SQL clone of *tail_ids* as fresh live rows (new id/display order, active=1, compacted=0;
        message payload columns stay byte-exact and FTS triggers index the clones), into *session_id* when given."""
        retarget = session_id is not None
        # A clone is a newly positioned display generation. Copy its indexed
        # identity, but let the insert trigger assign order from rows that are
        # still display-visible (the source may just have become rewind-only).
        skip = ("id", "active", "compacted", "display_order") + (("session_id",) if retarget else ())
        col_list = ", ".join(c for c in self._message_column_names(conn) if c not in skip)
        conn.execute(
            f"INSERT INTO messages ({col_list}, {'session_id, ' if retarget else ''}active, compacted) "
            f"SELECT {col_list}, {'?, ' if retarget else ''}1, 0 FROM messages "
            f"WHERE id IN ({_placeholders(tail_ids)}) ORDER BY id",
            [session_id, *tail_ids] if retarget else tail_ids)

    # ┌── 函数 def archive_and_compact()
    # │  作用：Non-destructive 在-place 压缩 under ONE 会话 id: soft-archive active rows (``active=0,
    # │  参数：self, session_id: str, compacted_messages: List[Dict[str, Any]], model_config_patch: Optional[Dict[str, Any]] = None, watermark: Optional[int] = None, lock_holder: Optional[str] = None, tail_count: int = 0
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def archive_and_compact(self, session_id: str, compacted_messages: List[Dict[str, Any]],
        model_config_patch: Optional[Dict[str, Any]] = None, watermark: Optional[int] = None,
        lock_holder: Optional[str] = None, tail_count: int = 0) -> int:
        """Non-destructive in-place compaction under ONE session id: soft-archive the active rows (``active=0,
        compacted=1``: summarized away, still searchable) and insert *compacted_messages* as fresh active
        rows, atomically; returns the new ACTIVE count (= ``message_count``). *watermark* (compression
        START): rows ``id > watermark`` arrived during the slow summary and are re-sequenced after the
        compacted set by a pure-SQL clone (fresh ids); ``None`` archives everything. *lock_holder*: verified
        in-txn so a reclaimed lease fails instead of clobbering the winner. *tail_count*: the LAST N compacted
        rows are the verbatim carried tail; their originals and the clones' originals are superseded
        duplicates and get rewind flags (``active=0, compacted=0``) so search doesn't return each carried
        message once per compaction. ``model_config_patch`` merges in the same txn (``None`` removes a key).

        Concurrent-append safety (#75316): when *watermark* is provided (the value of
        :meth:`get_active_message_watermark` captured at compression START), rows that arrived during the
        slow provider summary call (``id > watermark``) are NOT summarized away. They are re-sequenced after
        the compacted set by a pure-SQL column clone (every column except ``id`` — content, api_content,
        platform_message_id, token counts, reasoning sidecars all survive byte-exact, and the FTS triggers
        index the clones naturally), and the originals are archived. NOTE: re-sequencing assigns the tail
        rows fresh ids; consumers that reference durable row ids re-resolve by content (see 3e8ab0610).
        """
        from hermes_state import SessionCompressionInProgressError
        def _do(conn):
            if lock_holder is not None:
                lock_row = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
                if lock_row is None or lock_row["holder"] != lock_holder or float(lock_row["expires_at"]) <= time.time():
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise SessionCompressionInProgressError(
                        f"Compression lease for {session_id!r} lost before commit; refusing to publish a stale compaction")
            patch = model_config_patch is not None
            # on_missing="raise": never commit against a vanished session row (caller keeps the original).
            patched_model_config = self._merge_model_config_json(
                conn, session_id, model_config_patch, on_missing="raise") if patch else None
            tail_ids, tail_tool_calls = ([], 0) if watermark is None else self._tail_rows_after_watermark(
                conn, "SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 AND id > ? ORDER BY id",
                (session_id, int(watermark)))
            # Rewind targets sit AT/BELOW the watermark (all the compressor saw); unbounded, a
            # concurrent append would steal a LIMIT slot.
            rewind_ids: list[int] = []
            if tail_count > 0:
                bound = watermark is not None
                rewind_ids = [int(row["id"]) for row in conn.execute(
                    f"SELECT id FROM messages WHERE session_id = ? AND active = 1{' AND id <= ?' if bound else ''} "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, *((int(watermark),) if bound else ()), int(tail_count))).fetchall()]
            rewind_ids += tail_ids
            if rewind_ids:
                placeholders = _placeholders(rewind_ids)
                conn.execute("UPDATE messages SET active = 0, compacted = 0 "
                    f"WHERE session_id = ? AND id IN ({placeholders})", [session_id, *rewind_ids])
                conn.execute(f"{_ARCHIVE_ACTIVE_SQL} AND id NOT IN ({placeholders})", [session_id, *rewind_ids])
            else:
                conn.execute(_ARCHIVE_ACTIVE_SQL, (session_id,))
            inserted, tool_calls_total = self._insert_message_rows(conn, session_id, compacted_messages)
            if tail_ids:
                self._clone_message_rows(conn, tail_ids)
                inserted += len(tail_ids)
                tool_calls_total += tail_tool_calls
            conn.execute(f"{_SET_COUNTERS_SQL}{', model_config = ?' if patch else ''} WHERE id = ?",
                (inserted, tool_calls_total, *((patched_model_config,) if patch else ()), session_id))
            # ⤷ 出口 return：返回 inserted
            return inserted
        # ⤷ 出口 return：返回 self._execute_write(_do)
        return self._execute_write(_do)

    # ┌── 函数 def _message_column_names()
    # │  作用：Column 名称的 消息 table, cached per-连接 era.
    # │  参数：self, conn
    # │  返回：List[str]
    # └──────────────────────────────────────────────────────────────
    def _message_column_names(self, conn) -> List[str]:
        """Column names of the messages table, cached per-connection era."""
        if not getattr(self, "_message_columns_cache", None):
            self._message_columns_cache = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        # ⤷ 出口 return：返回 self._message_columns_cache
        return self._message_columns_cache

    # ┌── 函数 def set_latest_user_api_content()
    # │  作用：Backfill ``api_content`` sidecar onto newest ACTIVE user row (0/1 rows). Preflight 压缩
    # │  参数：self, session_id: str, content: Any, api_content: str
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row (0/1 rows). Preflight compaction
        inserts that row BEFORE the sidecar exists and the later persist identity-skips compacted dicts;
        without this a reload reopens the prompt-cache divergence. ``content`` match guards a racing rewrite.

        POSITIONAL, and only safe when the caller already knows the newest
        active user row IS the message it stamped. The content match is NOT
        sufficient on its own: repeated identical user turns ("ok", "y",
        "continue") make an OLDER row compare equal, so calling this before
        the current turn's row exists overwrites the previous turn's sidecar
        with this turn's bytes — durable wrong-bytes replay, a worse cache
        break than the missing sidecar. When the caller holds the durable row
        id (``_row_id``, synced onto the live dict by
        ``sync_flushed_message_markers`` and stamped by
        :meth:`_insert_message_rows`), use :meth:`set_message_api_content`
        instead — it addresses the exact row and cannot land on a neighbour.
        """
        # ⤷ 出口 return：返回 self._write_rowcount("UPDATE messages SET api_content = ? WH…
        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = (SELECT id FROM messages "
            "WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1"
            ") AND content IS ?",
            (_scrub_surrogates(api_content), session_id, self._encode_content(content)))

    # ┌── 函数 def set_message_api_content()
    # │  作用：把 ``api_content`` 附属数据回填到「一条」已知的持久化行上。
    # │  参数：self, session_id: str, row_id: int, content: Any, api_content: str
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def set_message_api_content(
        self, session_id: str, row_id: int, content: Any, api_content: str
    ) -> int:
        """Backfill the ``api_content`` sidecar onto ONE known durable row.

        Row-addressed counterpart to :meth:`set_latest_user_api_content`: the
        caller passes the ``_row_id`` the write path stamped on the live
        message dict, so the update cannot drift onto a neighbouring row that
        merely carries the same text.

        Used by the turn prologue whenever the current turn's user row was
        already materialized before the sidecar could be composed (in-place
        preflight compaction, a close/early flush that raced the prologue).
        The crash persist then marker-skips that message, so this is the only
        way the stamped bytes reach the store.

        ``active = 1`` and the ``content`` match stay as defensive guards: a
        row the compaction archived, or one a racing rewrite changed, is left
        untouched.
        """
        if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
            # ⤷ 出口 return：返回 0
            return 0
        # ⤷ 出口 return：返回 self._write_rowcount("UPDATE messages SET api_content = ? WH…
        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = ? AND session_id = ? "
            "AND role = 'user' AND active = 1 AND content IS ?",
            (_scrub_surrogates(api_content), row_id, session_id, self._encode_content(content)))

    # ┌── 函数 def _display_dedupe_key()
    # │  作用：Historical 展示身份, including 已归一化当前活跃的内容从 user 交接 carriers.
    # │  参数：self, row
    # │  返回：Tuple[Any, ...]
    # └──────────────────────────────────────────────────────────────
    def _display_dedupe_key(self, row) -> Tuple[Any, ...]:
        """Historical display identity, including normalized live content from user handoff carriers."""
        dedupe_content = row["content"]
        if row["role"] == "user":
            handoff, live_view = split_user_originated_turn({
                "role": "user", "content": self._decode_content(row["content"]),
                "display_kind": row["display_kind"],
                "display_metadata": self._decode_display_metadata(row["display_metadata"])})
            if handoff is not None and live_view is not None:
                dedupe_content = self._encode_content(live_view.get("content"))
        # ⤷ 出口 return：返回 (row['role'], dedupe_content, row['timestamp'], row['tool_ca…
        return (row["role"], dedupe_content, row["timestamp"],
                row["tool_call_id"], row["tool_calls"], row["tool_name"])

    # ┌── 函数 def _display_identity()
    # │  作用：Fixed-width durable 身份用于 indexed 展示-generation lookup.
    # │  参数：key: Tuple[Any, ...]
    # │  返回：bytes
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _display_identity(key: Tuple[Any, ...]) -> bytes:
        """Fixed-width durable identity for indexed display-generation lookup."""
        # ⤷ 出口 return：返回 hashlib.sha256(repr(key).encode('utf-8', 'surrogatepass')).d…
        return hashlib.sha256(repr(key).encode("utf-8", "surrogatepass")).digest()

    # ┌── 函数 def _dedupe_display_generations()
    # │  作用：Collapse 压缩 generations 从而每个 logical 消息 appears 一旦 ( protected tail 是已复制
    # │  参数：self, rows
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _dedupe_display_generations(self, rows):
        """Collapse compaction generations so each logical message appears once (the protected tail is copied
        into each generation: same role/content/timestamp, different ``active``/id); prefer the live row, then
        the newest. The ONE definition every display projection shares. *rows* must be ordered by ``id``."""
        seen: Dict[Tuple[Any, ...], Any] = {}
        first_id: Dict[Tuple[Any, ...], int] = {}
        for row in rows:
            key = self._display_dedupe_key(row)
            cur = seen.get(key)
            if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
                seen[key] = row
            first_id[key] = min(first_id.get(key, row["id"]), row["id"])
        # Order by the logical message's FIRST row, not the chosen representative's: a protected-tail
        # copy in a newer generation has a higher id than messages emitted after the original.
        # ⤷ 出口 return：返回 [seen[key] for key in sorted(seen, key=first_id.__getitem__)…
        return [seen[key] for key in sorted(seen, key=first_id.__getitem__)]

    # ┌── 函数 def _ensure_display_order()
    # │  作用：Backfill one legacy 会话一旦, preserving pre-index 展示身份 exactly.
    # │  参数：self, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _ensure_display_order(self, session_id: str) -> bool:
        """Backfill one legacy session once, preserving the pre-index display identity exactly."""
        with self._read_ctx() as conn:
            columns = set(self._message_column_names(conn))
        if not {"display_order", "display_identity"} <= columns:
            # ⤷ 出口 return：返回 False
            return False
        missing_sql = (
            "SELECT 1 FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) "
            "AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1")
        if self._read_one(missing_sql, (session_id,)) is None:
            # ⤷ 出口 return：返回 True
            return True
        if getattr(self, "read_only", False):
            # ⤷ 出口 return：返回 False
            return False

        def _do(conn):
            missing = conn.execute(missing_sql, (session_id,)).fetchone()
            if missing is None:
                # ⤷ 出口 return：返回 True
                return True
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) ORDER BY id",
                (session_id,)).fetchall()
            first_id: Dict[Tuple[Any, ...], int] = {}
            keyed_rows = []
            for row in rows:
                key = self._display_dedupe_key(row)
                first_id[key] = min(first_id.get(key, row["id"]), row["id"])
                keyed_rows.append((row["id"], key))
            conn.executemany(
                "UPDATE messages SET display_order = ?, display_identity = ? WHERE id = ?",
                [(first_id[key], self._display_identity(key), row_id) for row_id, key in keyed_rows])
            # ⤷ 出口 return：返回 True
            return True

        # ⤷ 出口 return：返回 bool(self._execute_write(_do))
        return bool(self._execute_write(_do))

    # ┌── 函数 def _row_to_message_dict()
    # │  作用：``字典(row)`` 带内容/tool_calls/display_metadata decoded; *summary_flag* 保留
    # │  参数：self, row, warn_context: str, summary_flag: bool
    # │  返回：Dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def _row_to_message_dict(self, row, *, warn_context: str, summary_flag: bool) -> Dict[str, Any]:
        """``dict(row)`` with content/tool_calls/display_metadata decoded; *summary_flag* keeps
        ``_compressed_summary`` only as ``True``."""
        msg = dict(row)
        msg.pop("display_identity", None)
        msg.pop("display_order", None)
        if summary_flag and msg.pop("_compressed_summary", 0):
            msg["_compressed_summary"] = True
        msg["content"] = self._decode_content(msg["content"])
        if msg.get("tool_calls"):
            msg["tool_calls"] = _json_or(
                msg["tool_calls"], [], f"Failed to deserialize tool_calls in {warn_context}, falling back to []")
        if msg.get("display_metadata") is not None:
            msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
        # ⤷ 出口 return：返回 msg
        return msg

    # ┌── 函数 def _active_clause()
    # │  作用：Audit: 每个 row; 展示: active plus 压缩-archived (never Undo/Rewind rows); 默认值: 当前活跃的.
    # │  参数：include_inactive: bool, include_compacted: bool
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _active_clause(include_inactive: bool, include_compacted: bool) -> str:
        """Audit: every row; display: active plus compaction-archived (never Undo/Rewind rows); default: live."""
        # ⤷ 出口 return：返回 '' if include_inactive else _DISPLAY_ACTIVE_CLAUSE if includ…
        return "" if include_inactive else (_DISPLAY_ACTIVE_CLAUSE if include_compacted else " AND active = 1")

    # ┌── 函数 def get_messages()
    # │  作用：加载消息在 insertion order (id, never 时间戳: clocks regress). ``include_inactive``:
    # │  参数：self, session_id: str, include_inactive: bool = False, include_compacted: bool = False, limit: Optional[int] = None, offset: int = 0, latest: bool = False, after_id: Optional[int] = None
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def get_messages(self, session_id: str, include_inactive: bool = False, include_compacted: bool = False,
                     limit: Optional[int] = None, offset: int = 0, latest: bool = False,
                     after_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load messages in insertion order (id, never timestamp: clocks regress). ``include_inactive``:
        rewind rows too; ``include_compacted``: compaction-archived display history (not rewind rows).
        ``latest`` pages back from the newest but returns chronological order; ``after_id``: keyset paging."""
        if after_id is not None and (latest or offset):
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("after_id is incompatible with latest/offset paging")
        if after_id is not None and include_compacted:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        active_clause = self._active_clause(include_inactive, include_compacted)
        if include_compacted and not include_inactive and self._ensure_display_order(session_id):
            direction = "DESC" if latest else "ASC"
            sql = f"""WITH page AS (
                    SELECT display_order FROM messages
                    WHERE session_id = ? AND (active = 1 OR compacted = 1)
                    GROUP BY display_order ORDER BY display_order {direction}
                    LIMIT ? OFFSET ?
                )
                SELECT chosen.* FROM page
                JOIN messages AS chosen ON chosen.id = (
                    SELECT candidate.id FROM messages AS candidate
                    WHERE candidate.session_id = ?
                      AND candidate.display_order = page.display_order
                      AND (candidate.active = 1 OR candidate.compacted = 1)
                    ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1
                )
                ORDER BY page.display_order ASC"""
            rows = self._read_all(sql, [session_id, -1 if limit is None else limit, offset, session_id])
        elif include_compacted:
            # Read-only legacy stores cannot persist display identities; retain the exact old projection.
            rows = self._dedupe_display_generations(self._read_all(
                "SELECT * FROM messages WHERE session_id = ?" + active_clause + " ORDER BY id ASC", [session_id]))
            rows = rows[::-1][offset:][:limit][::-1] if latest else rows[offset:][:limit]
        else:
            sql = (f"SELECT * FROM messages WHERE session_id = ?{active_clause}"
                f"{' AND id > ?' if after_id is not None else ''} ORDER BY id {'DESC' if latest else 'ASC'}")
            params: list = [session_id] if after_id is None else [session_id, after_id]
            if limit is not None or offset:
                # SQLite's OFFSET requires LIMIT; -1 means "no limit".
                sql += " LIMIT ? OFFSET ?"
                params.extend([-1 if limit is None else limit, offset])
            rows = self._read_all(sql, params)
            if latest:
                rows.reverse()
        # ⤷ 出口 return：返回 [self._row_to_message_dict(row, warn_context='get_messages',…
        return [self._row_to_message_dict(row, warn_context="get_messages", summary_flag=True) for row in rows]

    # ┌── 函数 def find_pr_url_messages()
    # │  作用：工具结果 containing ``/pull/``: deliberately loose scan, oldest-first 从而 caller takes last.
    # │  参数：self, session_ids: List[str]
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def find_pr_url_messages(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """Tool results containing ``/pull/``: a deliberately loose scan, oldest-first so the caller takes the last."""
        ids = [s for s in session_ids if s]
        chunks = (ids[start : start + 900] for start in range(0, len(ids), 900))  # SQLite's bound-variable ceiling.
        # ⤷ 出口 return：返回 [{'session_id': row[0], 'content': row[1]} for chunk in chun…
        return [{"session_id": row[0], "content": row[1]} for chunk in chunks for row in self._read_all(
                f"""SELECT session_id, content FROM messages
                    WHERE session_id IN ({_placeholders(chunk)})
                      AND role = 'tool' AND content LIKE '%/pull/%'
                    ORDER BY id ASC""",
                chunk)]

    # ┌── 函数 def get_messages_around()
    # │  作用：Up *window* 消息 either side 的 anchor id (ascending). ``messages_before``/``_after`` count
    # │  参数：self, session_id: str, around_message_id: int, window: int = 5
    # │  返回：Dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def get_messages_around(self, session_id: str, around_message_id: int, window: int = 5) -> Dict[str, Any]:
        """Up to *window* messages either side of an anchor id (ascending). ``messages_before``/``_after`` count
        strictly around the anchor (fewer than *window* = session boundary). Empty for a foreign anchor."""
        window = max(window, 0)
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                                (around_message_id, session_id)).fetchone():
                # ⤷ 出口 return：返回 {'window': [], 'messages_before': 0, 'messages_after': 0}
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1)).fetchall()
            after_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window)).fetchall()
        window_msgs = [self._row_to_message_dict(r, warn_context="get_messages_around", summary_flag=False)
                       for r in (*reversed(before_rows), *after_rows)]
        # before_rows includes the anchor itself.
        # ⤷ 出口 return：返回 {'window': window_msgs, 'messages_before': max(0, len(before…
        return {"window": window_msgs, "messages_before": max(0, len(before_rows) - 1), "messages_after": len(after_rows)}

    # ┌── 函数 def resolve_resume_session_id()
    # │  作用：Redirect resume target descendant holding 消息: follow compression chain
    # │  参数：self, session_id: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant holding the messages: follow the compression chain to
        the live tip (lineage-aware, so delegate/branch children never hijack it), then walk
        ``parent_session_id`` forward to the DEEPEST node with messages (a continuation may hold newer
        turns), skipping branch/delegate/reset/tool children. Unchanged when nothing has messages; depth cap 32.

        Context compression ends the current session and forks a new child session (linked via
        ``parent_session_id``). The flush cursor is reset, so the child is where new messages actually land
        — the parent ends up with ``message_count = 0`` rows unless messages had already been flushed to it
        before compression. See #15000.
        """
        if not session_id:
            # ⤷ 出口 return：返回 session_id
            return session_id
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            session_id = self.get_compression_tip(session_id) or session_id
        except Exception:
            pass
        with self._read_ctx() as conn:
            current = session_id
            seen = {current}
            best = None  # deepest node with messages
            for _ in range(32):
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    if conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (current,)).fetchone() is not None:
                        best = current
                    child_row = conn.execute(
                        "SELECT id FROM sessions AS child WHERE child.parent_session_id = ? "
                        f"  AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL "
                        f"  AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL "
                        f"  AND {_sql_json_extract('child.model_config', '$._reset_from')} IS NULL "
                        f"  AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
                        "  AND COALESCE(child.source, '') != 'tool' "
                        "ORDER BY child.started_at DESC, child.id DESC LIMIT 1", (current,)).fetchone()
                except Exception:
                    # ⤷ 出口 return：返回 session_id
                    return session_id
                if child_row is None or not child_row["id"] or child_row["id"] in seen:
                    break
                current = child_row["id"]
                seen.add(current)
            # ⤷ 出口 return：返回 best if best is not None else session_id
            return best if best is not None else session_id

    # ┌── 函数 def _fetch_conversation_rows()
    # │  作用：``_CONVERSATION_ROW_COLUMNS`` rows 用于 *session_ids* ORDER 由 id (timestamps 是不 monotonic
    # │  参数：self, session_ids: List[str], active_clause: str, with_session_id: bool
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _fetch_conversation_rows(self, session_ids: List[str], active_clause: str, *, with_session_id: bool):
        """``_CONVERSATION_ROW_COLUMNS`` rows for *session_ids* ORDER BY id (timestamps are not monotonic
        and would break tool-call adjacency)."""
        # ⤷ 出口 return：返回 self._read_all(f"SELECT {('session_id, ' if with_session_id …
        return self._read_all(
            f"SELECT {'session_id, ' if with_session_id else ''}{self._CONVERSATION_ROW_COLUMNS} "
            f"FROM messages WHERE session_id IN ({_placeholders(session_ids)})"
            f"{active_clause} ORDER BY id", tuple(session_ids))

    # ┌── 函数 def get_messages_as_conversation()
    # │  作用：加载消息在 OpenAI format. ``include_compacted`` (deduped 展示 history) 是用于展示读取
    # │  参数：self, session_id: str, include_ancestors: bool = False, include_inactive: bool = False, repair_alternation: bool = False, include_row_ids: bool = False, include_compacted: bool = False
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def get_messages_as_conversation(self, session_id: str, include_ancestors: bool = False,
                                     include_inactive: bool = False, repair_alternation: bool = False,
                                     include_row_ids: bool = False,
                                     include_compacted: bool = False) -> List[Dict[str, Any]]:
        """Load messages in OpenAI format. ``include_compacted`` (deduped display history) is for DISPLAY reads
        only: the model-fed restore must not regrow what compaction summarized away. ``repair_alternation``
        repairs the loaded list for LIVE REPLAY callers (a durable ``user;user`` pair would re-trigger the
        per-request repair forever), preserving summary markers before repair so derivative context
        cannot merge with an original user turn; the stored transcript is never mutated."""
        rows = self._fetch_conversation_rows(
            self._resume_lineage_ids(session_id) if include_ancestors else [session_id],
            self._active_clause(include_inactive, include_compacted), with_session_id=False)
        if include_compacted:
            rows = self._dedupe_display_generations(rows)
        # ⤷ 出口 return：返回 self._rows_to_conversation(rows, session_id=session_id, incl…
        return self._rows_to_conversation(rows, session_id=session_id, include_ancestors=include_ancestors,
            repair_alternation=repair_alternation, include_row_ids=include_row_ids,
            include_summary_markers=repair_alternation)

    # ┌── 函数 def _dedupe_replayed_user()
    # │  作用：Ancestor-lineage dedupe 的 one decoded user *msg* -> ``(skip, exact_clone_key)``. Rotation
    # │  参数：self, messages, msg, exact_user_clones
    # │  返回：Tuple[bool, Any]
    # └──────────────────────────────────────────────────────────────
    def _dedupe_replayed_user(self, messages, msg, exact_user_clones) -> Tuple[bool, Any]:
        """Ancestor-lineage dedupe of one decoded user *msg* -> ``(skip, exact_clone_key)``. Rotation
        column-clones the concurrent tail into the child, so copies need not be adjacent: the exact
        ``(timestamp, canonical content)`` clone index is checked first, then the adjacent heuristic. A
        rotated child carrier wins over the ancestor copy."""
        canonical_content = self._canonical_replayed_user_content(msg)[0]
        exact_clone_key = self._exact_replayed_user_clone_key(msg.get("timestamp"), canonical_content)
        previous_exact = exact_user_clones.get(exact_clone_key) if exact_clone_key is not None else None
        duplicate = None
        if previous_exact is not None:
            previous_index = next((i for i, candidate in enumerate(messages) if candidate is previous_exact), None)
            if previous_index is not None:
                duplicate = (previous_index, True)
        if duplicate is None:
            duplicate = self._find_duplicate_replayed_user_message(messages, msg)
        if duplicate is None:
            # ⤷ 出口 return：返回 (False, exact_clone_key)
            return False, exact_clone_key
        duplicate_index, prefer_current = duplicate
        if prefer_current:
            messages.pop(duplicate_index)
        # ⤷ 出口 return：返回 (not prefer_current, exact_clone_key)
        return not prefer_current, exact_clone_key

    # ┌── 函数 def _rows_to_conversation()
    # │  作用：Decode fetched rows (ordered 由 id, pre-filtered) 进入 OpenAI format, stable 键 order. 每个字典是
    # │  参数：self, rows, session_id: str, include_ancestors: bool, repair_alternation: bool, include_row_ids: bool = ..., include_summary_markers: bool = ...
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def _rows_to_conversation(self, rows, *, session_id: str, include_ancestors: bool, repair_alternation: bool,
                              include_row_ids: bool = False,
                              include_summary_markers: bool = False) -> List[Dict[str, Any]]:
        """Decode fetched rows (ordered by id, pre-filtered) into OpenAI format, stable key order. Every dict is
        stamped ``_DB_PERSISTED_MARKER_KEY`` (born durable) so an identity-losing handoff never re-appends the
        transcript on flush. ``_row_id`` is opt-in (gateway reactions); reasoning restored on assistant rows
        only; ``api_content`` VERBATIM (no sanitize/strip) so replay keeps the provider prompt cache byte-stable."""
        from hermes_state import _strip_background_review_harness, _strip_stale_tool_call_markers
        messages = []
        exact_user_clones: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            # Underscore-prefixed like ``_row_id``: transports strip it before the wire; compression's
            # assembly copies strip it so rotated child handoffs still flush (_fresh_compaction_message_copy).
            msg = {"role": row["role"], "content": content, _DB_PERSISTED_MARKER_KEY: True}
            # Born durable (#92231): this dict is materialized FROM a durable row, so stamp the persistence
            # marker at the source instead of relying on every restore caller to thread the loaded list back
            # through a flush as ``conversation_history=`` — any identity-losing handoff (compression's
            # durable-snapshot adoption, incremental persists with no history arg) would otherwise re-append
            # the ENTIRE transcript on flush.
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
            msg.update((col, row[col]) for col in ("api_content", "display_kind") if row[col])
            if row["display_metadata"] and (decoded := self._decode_display_metadata(row["display_metadata"])) is not None:
                msg["display_metadata"] = decoded
            if include_summary_markers and row["_compressed_summary"]:
                msg["_compressed_summary"] = True
            msg.update(
                (col, row[col]) for col in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if row[col])
            if row["tool_calls"]:
                msg["tool_calls"] = _json_or(
                    row["tool_calls"], [], "Failed to deserialize tool_calls in conversation replay, falling back to []")
            if row["platform_message_id"]:  # platform-side id exposed as ``message_id`` (JSONL transcript compat)
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            if row["role"] == "assistant":
                msg.update((col, row[col]) for col in ("finish_reason", "reasoning") if row[col])
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                msg.update(
                    (col, _json_or(row[col], None, f"Failed to deserialize {col}, falling back to None"))
                    for col in ("reasoning_details", "codex_reasoning_items", "codex_message_items") if row[col])
            if include_ancestors:
                skip, exact_clone_key = self._dedupe_replayed_user(messages, msg, exact_user_clones)
                if skip:
                    continue
                if exact_clone_key is not None:
                    exact_user_clones[exact_clone_key] = msg
            messages.append(msg)
        # Defense-in-depth: strip a background-review harness turn (older builds shared the parent's
        # session_id) plus its curator reply, and bare tool-call marker content ("[memory]") persisted as an answer.
        messages = _strip_stale_tool_call_markers(_strip_background_review_harness(messages))
        if repair_alternation and messages:
            from agent.agent_runtime_helpers import repair_message_sequence
            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info("Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence", repaired, session_id)
        # ⤷ 出口 return：返回 messages
        return messages

    # ┌── 函数 def get_resume_conversations()
    # │  作用：``(model_history, display_history)`` 用于 resume 从 ONE SELECT; byte-identical separate
    # │  参数：self, session_id: str
    # │  返回：Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    # └──────────────────────────────────────────────────────────────
    def get_resume_conversations(self, session_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """``(model_history, display_history)`` for a resume from ONE SELECT; byte-identical to the separate
        reads. model: the tip's active rows, alternation-repaired, summary marker kept for pre-compress
        checkpointing. display: the full lineage (``/branch`` stands alone), compaction-archived rows deduped.

        The display projection also includes rows preserved by IN-PLACE compaction (``active=0,
        compacted=1``), deduped by :meth:`_dedupe_display_generations`. Without them a compacted
        conversation resumes showing only its summary plus the carried-forward tail — the user's own turns
        read as deleted even though every row is still on disk, and the REST transcript read (which has
        always included them) disagreed with this one about the same session (#92080).
        """
        rows = self._fetch_conversation_rows(
            self._resume_lineage_ids(session_id), _DISPLAY_ACTIVE_CLAUSE, with_session_id=True)
        # The model projection stays active-only: it is the compressed working context.
        model_history = self._rows_to_conversation(
            [r for r in rows if r["session_id"] == session_id and r["active"]], session_id=session_id,
            include_ancestors=False, repair_alternation=True, include_row_ids=True, include_summary_markers=True)
        display_history = self._rows_to_conversation(
            self._dedupe_display_generations(rows), session_id=session_id,
            include_ancestors=True, repair_alternation=False, include_row_ids=True)
        # ⤷ 出口 return：返回 (model_history, display_history)
        return model_history, display_history

    # ┌── 函数 def _resume_lineage_ids()
    # │  作用：会话 ids 展示 resume materializes: compression lineage, 或会话 alone 用于
    # │  参数：self, session_id: str
    # │  返回：List[str]
    # └──────────────────────────────────────────────────────────────
    def _resume_lineage_ids(self, session_id: str) -> List[str]:
        """Session ids a display resume materializes: the compression lineage, or the session alone for an
        explicit ``/branch`` copy. Shared with the resume guard so it counts exactly what a resume loads."""
        # ⤷ 出口 return：返回 [session_id] if self._is_explicit_branch_session(session_id)…
        return [session_id] if self._is_explicit_branch_session(session_id) else self._session_lineage_root_to_tip(session_id)

    # ┌── 函数 def _resume_count_scope()
    # │  作用：``tip_only``: tip's ACTIVE rows (模型恢复); 否则 full-lineage 展示设置.
    # │  参数：self, session_id: str, tip_only: bool
    # │  返回：Tuple[List[str], str]
    # └──────────────────────────────────────────────────────────────
    def _resume_count_scope(self, session_id: str, tip_only: bool) -> Tuple[List[str], str]:
        """``tip_only``: the tip's ACTIVE rows (model restore); else the full-lineage DISPLAY set."""
        if tip_only:
            # ⤷ 出口 return：返回 ([session_id], 'active = 1')
            return [session_id], "active = 1"
        # ⤷ 出口 return：返回 (self._resume_lineage_ids(session_id), '(active = 1 OR compa…
        return self._resume_lineage_ids(session_id), "(active = 1 OR compacted = 1)"

    # ┌── 函数 def get_resume_message_count()
    # │  作用：Count rows resume 会 materialize (see ``_resume_count_scope``).
    # │  参数：self, session_id: str, tip_only: bool = ...
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def get_resume_message_count(self, session_id: str, *, tip_only: bool = False) -> int:
        """Count the rows a resume would materialize (see ``_resume_count_scope``)."""
        session_ids, active_clause = self._resume_count_scope(session_id, tip_only)
        # ⤷ 出口 return：返回 int(self._read_one(f'SELECT COUNT(*) FROM messages WHERE ses…
        return int(self._read_one(
            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({_placeholders(session_ids)}) AND {active_clause}",
            tuple(session_ids))[0])

    # ┌── 函数 def assert_resume_safe()
    # │  作用：Resume row count, 或抛出 ``SessionResumeTooLargeError``. ``max_messages=None`` 读取配置; 0
    # │  参数：self, session_id: str, max_messages: Optional[int] = None, tip_only: bool = ...
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def assert_resume_safe(self, session_id: str, max_messages: Optional[int] = None, *, tip_only: bool = False) -> int:
        """Resume row count, or raise ``SessionResumeTooLargeError``. ``max_messages=None`` reads config; 0
        disables the guard without counting. ``tip_only`` bounds only the tip's active rows for callers that
        never materialize the lineage: a heavily compressed conversation is a success, not a rejection."""
        from hermes_state import SessionResumeTooLargeError, resolved_max_resume_messages
        if max_messages is None:
            max_messages = resolved_max_resume_messages()
        if max_messages < 0:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            # ⤷ 出口 return：返回 0
            return 0
        session_ids, active_clause = self._resume_count_scope(session_id, tip_only)
        message_count = int(self._read_one("SELECT COUNT(*) FROM ("
            f"SELECT 1 FROM messages WHERE session_id IN ({_placeholders(session_ids)}) "
            f"AND {active_clause} LIMIT ?)", (*session_ids, max_messages + 1))[0])
        if message_count > max_messages:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise SessionResumeTooLargeError(
                message_count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
        # ⤷ 出口 return：返回 message_count
        return message_count

    # ┌── 函数 def get_ancestor_display_prefix()
    # │  作用：Ancestor-仅展示消息的 lineage (row ``session_id != tip``) 那个 ``会话.resume``
    # │  参数：self, session_id: str
    # │  返回：List[Dict[str, Any]]
    # └──────────────────────────────────────────────────────────────
    def get_ancestor_display_prefix(self, session_id: str) -> List[Dict[str, Any]]:
        """Ancestor-only display messages of a lineage (row ``session_id != tip``) that ``session.resume``
        prepends. Identified by row origin, not ``display[:len(display) - len(model)]``, so alternation
        repair cannot overcount."""
        session_ids = self._resume_lineage_ids(session_id)
        if len(session_ids) <= 1:
            # ⤷ 出口 return：返回 []
            return []
        rows = self._dedupe_display_generations(
            self._fetch_conversation_rows(session_ids, _DISPLAY_ACTIVE_CLAUSE, with_session_id=True))
        ancestor_ids = {int(row["id"]) for row in rows if row["session_id"] != session_id and row["id"] is not None}
        if not ancestor_ids:
            # ⤷ 出口 return：返回 []
            return []
        lineage = self._rows_to_conversation(
            rows, session_id=session_id, include_ancestors=True, repair_alternation=False, include_row_ids=True)
        # ⤷ 出口 return：返回 [{k: v for k, v in message.items() if k != '_row_id'} for me…
        return [{k: v for k, v in message.items() if k != "_row_id"}
            for message in lineage if message.get("_row_id") in ancestor_ids]

    # ┌── 函数 def get_conversation_root()
    # │  作用：ROOT id 的 lineage: stable conversation id across compression segments 且委派给
    # │  参数：self, session_id: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def get_conversation_root(self, session_id: str) -> str:
        """ROOT id of the lineage: the stable conversation id across compression segments and delegate
        subagents (Nous Portal usage tagging). Unchanged when there is no recorded parent."""
        chain = self._session_lineage_root_to_tip(session_id)
        # ⤷ 出口 return：返回 chain[0] if chain and chain[0] else session_id
        return chain[0] if chain and chain[0] else session_id

    # ┌── 函数 def _canonical_replayed_user_content()
    # │  作用：返回 canonical 当前活跃的内容且 whether *msg* 是 composite.。
    # │  参数：msg: Dict[str, Any]
    # │  返回：Tuple[Any, bool]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _canonical_replayed_user_content(msg: Dict[str, Any]) -> Tuple[Any, bool]:
        """Return canonical live content and whether *msg* is composite."""
        if msg.get("role") != "user":
            # ⤷ 出口 return：返回 (None, False)
            return None, False
        handoff, live_view = split_user_originated_turn(msg)
        is_composite = handoff is not None and live_view is not None
        # ⤷ 出口 return：返回 (live_view.get('content') if is_composite else msg.get('cont…
        return live_view.get("content") if is_composite else msg.get("content"), is_composite

    # ┌── 函数 def _exact_replayed_user_clone_key()
    # │  作用：返回 hashable 键用于 column-exact rotation clone.。
    # │  参数：timestamp: Any, content: Any
    # │  返回：Optional[Tuple[Any, str]]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _exact_replayed_user_clone_key(timestamp: Any, content: Any) -> Optional[Tuple[Any, str]]:
        """Return a hashable key for a column-exact rotation clone."""
        if timestamp is None or content in (None, "", []):
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 (timestamp, json.dumps(content, ensure_ascii=False, sort_key…
            return timestamp, json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            # ⤷ 出口 return：返回 None
            return None

    # ┌── 函数 def _find_duplicate_replayed_user_message()
    # │  作用：Adjacent 重放 duplicate ``(index, prefer_current)`` 或 None. Rotation 可以持久化 current ask
    # │  参数：messages: List[Dict[str, Any]], msg: Dict[str, Any]
    # │  返回：Optional[Tuple[int, bool]]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _find_duplicate_replayed_user_message(messages: List[Dict[str, Any]],
                                              msg: Dict[str, Any]) -> Optional[Tuple[int, bool]]:
        """Adjacent replay duplicate ``(index, prefer_current)`` or None. Rotation may persist the current ask
        in the parent and again inside a composite child carrier: carriers compare by canonical live payload,
        ordinary users by exact string. The child carrier wins (it owns the durable row id and scaffold)."""
        if msg.get("role") != "user":
            # ⤷ 出口 return：返回 None
            return None
        canonical = SessionMessagesMixin._canonical_replayed_user_content
        content, prefer_current = canonical(msg)
        if content in (None, "", []):
            # ⤷ 出口 return：返回 None
            return None
        for index in range(len(messages) - 1, -1, -1):
            prev = messages[index]
            if prev.get("role") == "user":
                prev_content, prev_is_composite = canonical(prev)
                if prev_content == content and (prefer_current or prev_is_composite or isinstance(content, str)):
                    # ⤷ 出口 return：返回 (index, prefer_current)
                    return index, prefer_current
            elif prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                # ⤷ 出口 return：返回 None
                return None
        # ⤷ 出口 return：返回 None
        return None

    # ========================================================================= Rewind (soft-delete) — see
    # /rewind slash command + issue #21910
    # =========================================================================
    # ┌── 函数 def get_active_message_ids()
    # │  作用：Ordered physical active ids 用于 rewind CAS 检查 (includes legacy 测试框架 rows projections omit).
    # │  参数：self, session_id: str
    # │  返回：List[int]
    # └──────────────────────────────────────────────────────────────
    def get_active_message_ids(self, session_id: str) -> List[int]:
        """Ordered physical active ids for rewind CAS checks (includes legacy harness rows projections omit)."""
        # ⤷ 出口 return：返回 [int(row[0]) for row in self._read_all(_ACTIVE_IDS_SQL, (ses…
        return [int(row[0]) for row in self._read_all(_ACTIVE_IDS_SQL, (session_id,))]

    # ┌── 函数 def _active_transcript_counts()
    # │  作用：返回 active 消息/工具调用 counts inside caller's txn.。
    # │  参数：conn, session_id: str
    # │  返回：tuple[int, int]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _active_transcript_counts(conn, session_id: str) -> tuple[int, int]:
        """Return active message/tool-call counts inside the caller's txn."""
        rows = conn.execute("SELECT tool_calls FROM messages WHERE session_id = ? AND active = 1", (session_id,)).fetchall()
        # ⤷ 出口 return：返回 (len(rows), sum((_tool_calls_len(row[0], scalar=1) for row i…
        return len(rows), sum(_tool_calls_len(row[0], scalar=1) for row in rows)

    # ┌── 函数 def _split_rewind_target()
    # │  作用：校验 active rewind target; 返回其 交接 scaffold (或 None). ``ValueError``: inactive /
    # │  参数：self, target_row: Dict[str, Any], expected_target_content: Any, preserve_compaction_handoff: bool
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _split_rewind_target(self, target_row: Dict[str, Any], expected_target_content: Any, preserve_compaction_handoff: bool):
        """Validate an active rewind target; return its handoff scaffold (or None). ``ValueError``: inactive /
        non-user-originated / missing composite carrier; ``RuntimeError``: live payload changed."""
        if not target_row.get("active"):
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("rewind target is not active")
        handoff, live_view = split_user_originated_turn({
            **target_row, "content": self._decode_content(target_row.get("content")),
            "display_metadata": self._decode_display_metadata(target_row.get("display_metadata"))})
        if live_view is None:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("rewind target is not a user-originated turn")
        live_content = live_view.get("content")
        if isinstance(live_content, str):
            live_content = sanitize_context(live_content).strip()
        if expected_target_content is not None and live_content != expected_target_content:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise RuntimeError("rewind target changed before it could be persisted")
        if preserve_compaction_handoff and handoff is None:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError("preserve_compaction_handoff requires an active composite carrier")
        # ⤷ 出口 return：返回 handoff if preserve_compaction_handoff else None
        return handoff if preserve_compaction_handoff else None

    # ┌── 函数 def rewind_to_message()
    # │  作用：Soft-delete (``active=0``) 每个消息带 id >= *target_message_id*, target included ( caller
    # │  参数：self, session_id: str, target_message_id: int, preserve_compaction_handoff: bool = ..., expected_active_ids: Optional[List[int]] = ..., expected_target_content: Any = ...
    # │  返回：Dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def rewind_to_message(self, session_id: str, target_message_id: int, *, preserve_compaction_handoff: bool = False,
                          expected_active_ids: Optional[List[int]] = None,
                          expected_target_content: Any = None) -> Dict[str, Any]:
        """Soft-delete (``active=0``) every message with id >= *target_message_id*, target included (the caller
        pre-fills it as the next prompt). Returns ``{"rewound_count", "target_message", "new_head_id"}``, plus
        ``replacement_message_id`` with ``preserve_compaction_handoff`` (archives a composite summary carrier,
        inserts its hidden handoff scaffold as the new head). ``ValueError``: target missing or not ``user``.
        ``expected_active_ids`` / ``expected_target_content`` pin the active set and canonical live payload
        in-txn before any mutation (presentation-only metadata changes don't invalidate a rewind). A live turn
        lease refuses; expired/dead holders are reclaimed. ``rewind_count`` always increments."""
        def _do(conn):
            self._check_transcript_write_guards(
                conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
            if expected_active_ids is not None:
                active_rows = conn.execute(_ACTIVE_IDS_SQL, (session_id,)).fetchall()
                if [int(r[0]) for r in active_rows] != expected_active_ids:
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise RuntimeError("active transcript changed before the rewind could be persisted")
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?", (target_message_id, session_id)).fetchone()
            if row is None:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise ValueError(f"message {target_message_id} not found in session {session_id}")
            target_row = dict(row)
            if target_row.get("role") != "user":
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise ValueError(
                    f"rewind target must be a 'user' message (got role={target_row.get('role')!r}, id={target_message_id})")
            replacement_message_id = replacement = None
            if preserve_compaction_handoff or expected_target_content is not None:
                replacement = self._split_rewind_target(target_row, expected_target_content, preserve_compaction_handoff)
            ids = [r[0] for r in conn.execute("SELECT id FROM messages WHERE session_id = ? AND id >= ? AND active = 1",
                                             (session_id, target_message_id)).fetchall()]
            if ids:
                conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({_placeholders(ids)})", ids)
            if replacement is not None:
                self._insert_message_rows(conn, session_id, [replacement])
                replacement_message_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = ?", (session_id,))
            message_count, tool_call_count = self._active_transcript_counts(conn, session_id)
            conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?", (message_count, tool_call_count, session_id))
            head_id = conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1", (session_id,)).fetchone()[0]
            # ⤷ 出口 return：返回 (target_row, ids, head_id, replacement_message_id)
            return target_row, ids, head_id, replacement_message_id
        target_row, rewound, new_head_id, replacement_message_id = self._execute_write(_do)
        # Decode for the prompt-buffer prefill without a second fallible DB operation.
        target_row["content"] = self._decode_content(target_row.get("content"))
        # ⤷ 出口 return：返回 {'rewound_count': len(rewound), 'target_message': target_row…
        return {"rewound_count": len(rewound), "target_message": target_row, "new_head_id": new_head_id,
                **({"replacement_message_id": replacement_message_id} if preserve_compaction_handoff else {})}

    # ┌── 函数 def message_count()
    # │  作用：Count 消息, optionally 用于 specific 会话.
    # │  参数：self, session_id: str = None
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        sql = "SELECT COUNT(*) FROM messages" + (" WHERE session_id = ?" if session_id else "")
        # ⤷ 出口 return：返回 self._read_one(sql, (session_id,) if session_id else ())[0]
        return self._read_one(sql, (session_id,) if session_id else ())[0]

    # ┌── 函数 def has_gateway_input_owner()
    # │  作用：Probe 已接受-输入 marker 不 allocating 消息 bodies 或 archives.
    # │  参数：self, session_id: str, owner: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_gateway_input_owner(self, session_id: str, owner: str) -> bool:
        """Probe the accepted-input marker without allocating message bodies or archives."""
        # ⤷ 出口 return：返回 self._read_one("SELECT 1 FROM messages WHERE session_id = ? …
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND role = 'user' "
            "AND observed = 0 AND (active = 1 OR compacted = 1) "
            "AND CASE WHEN json_valid(display_metadata) "
            "THEN json_extract(display_metadata, '$.gateway_input_owner') END = ? LIMIT 1",
            (session_id, owner)) is not None

    # ┌── 函数 def has_platform_message_id()
    # │  作用：当 *platform_message_id* exists (partial-index probe; gateway's transient-failure dedupe). 时返回 True。
    # │  参数：self, session_id: str, platform_message_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        """True when *platform_message_id* exists (partial-index probe; the gateway's transient-failure dedupe).

        Uses the idx_messages_platform_msg_id partial index for efficient lookup. Used by the gateway's
        transient-failure dedupe guard (#47237) to skip re-persisting a user message that was already saved
        on a prior retry of the same inbound platform message.
        """
        # ⤷ 出口 return：返回 self._read_one('SELECT 1 FROM messages WHERE session_id = ? …
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
            (session_id, platform_message_id)) is not None

    # ┌── 函数 def _is_explicit_fork_child_row()
    # │  作用：当 *会话* 是 branch, 委派给, 或工具 child 的其 parent. Markers 仅 count 当它们 时返回 True。
    # │  参数：self, session: Dict[str, Any]
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _is_explicit_fork_child_row(self, session: Dict[str, Any]) -> bool:
        """True when *session* is a branch, delegate, or tool child of its parent. Markers only count when they
        point at ``parent_session_id``: compression copies ``model_config`` onto the continuation, so
        presence-only matching would misclassify it (same binding as ``_NON_CONTINUATION_CHILD_FILTER_SQL``)."""
        if session.get("source") == "tool":
            # ⤷ 出口 return：返回 True
            return True
        cfg = session.get("model_config")
        if isinstance(cfg, str):
            # ⤷ 关键分支 try/except：捕获 json.JSONDecodeError，出错走兜底而不是崩溃。
            try:
                cfg = json.loads(cfg)
            except json.JSONDecodeError:
                # ⤷ 出口 return：返回 False
                return False
        if not isinstance(cfg, dict):
            # ⤷ 出口 return：返回 False
            return False
        markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"))
        parent_id = session.get("parent_session_id")
        # ⤷ 出口 return：返回 parent_id in markers if parent_id else any((m is not None fo…
        return parent_id in markers if parent_id else any(m is not None for m in markers)

    # ┌── 函数 def is_explicit_fork_child()
    # │  作用：读取-仅 :meth:`_is_explicit_fork_child_row`; missing row 是不 fork (prompt_cache_scope 保留
    # │  参数：self, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def is_explicit_fork_child(self, session_id: str) -> bool:
        """Read-only :meth:`_is_explicit_fork_child_row`; a missing row is not a fork (prompt_cache_scope keeps
        a declared conversation key from crossing the fork boundary)."""
        session = self.get_session(session_id)
        # ⤷ 出口 return：返回 bool(session and self._is_explicit_fork_child_row(session))
        return bool(session and self._is_explicit_fork_child_row(session))

    # ┌── 函数 def latest_conversation_boundary()
    # │  作用：Conversation boundaries (``_RESET_END_REASONS`` ends) 这个 peer 有 crossed, 或 ``None`` 若 never
    # │  参数：self, session_key: str, source: str
    # │  返回：Optional[int]
    # └──────────────────────────────────────────────────────────────
    def latest_conversation_boundary(self, session_key: str, source: str) -> Optional[int]:
        """Conversation boundaries (``_RESET_END_REASONS`` ends) this peer has crossed, or ``None`` if never
        reset. The peer is ``(session_key, source)``, never the key alone (an API caller may legally reuse a
        Telegram row's key). Read from ``conversation_generations`` (advanced in each boundary's txn), not an
        aggregate over session rows: deletes/prunes would re-emit a retired pair. Rows are never GC'd
        (dropping one re-issues generation 1: the ABA this prevents). Wall-clock-free, so a backwards NTP
        correction cannot reorder it. DBs upgraded mid-conversation take their first generation from the
        next boundary written (a pre-upgrade reset shares its predecessor's scope once: costs a warm
        prompt-cache bucket, never crosses an identity)."""
        if not session_key or not source:
            # ⤷ 出口 return：返回 None
            return None
        row = self._read_one(
            "SELECT generation FROM conversation_generations WHERE source = ? AND session_key = ?",
            (source, session_key))
        generation = int(row["generation"]) if row is not None and row["generation"] is not None else 0
        # ⤷ 出口 return：返回 generation if generation > 0 else None
        return generation if generation > 0 else None

    # ┌── 函数 def clear_messages()
    # │  作用：Delete 全部消息用于会话且 reset 其 counters.
    # │  参数：self, session_id: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(_RESET_COUNTERS_SQL, (session_id,))
        self._execute_write(_do)

    # ┌── 函数 def purge_stale_tool_call_markers()
    # │  作用：Permanently 清除 bare 工具调用 marker 内容 ("[内存]") left 由 pre-fix 会话
    # │  参数：self, dry_run: bool = ..., backup: bool = ...
    # │  返回：Dict[str, Any]
    # └──────────────────────────────────────────────────────────────
    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
        """Permanently clear bare tool-call marker content ("[memory]") left by pre-fix sessions
        (``_rows_to_conversation`` repairs it in memory; this stops the re-scan). Only ``content`` is touched.
        ``backup``: ``VACUUM INTO`` snapshot first (none when nothing changes)."""
        from hermes_state import _STALE_TOOL_CALL_MARKER_RE
        def _find_affected(conn) -> List[int]:
            cursor = conn.execute("SELECT id, content FROM messages "
                "WHERE role = 'assistant' AND tool_calls IS NOT NULL AND tool_calls != ''")
            # ⤷ 出口 return：返回 [row['id'] for row in cursor.fetchall() if isinstance(row['c…
            return [row["id"] for row in cursor.fetchall()
                if isinstance(row["content"], str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(row["content"].strip())]
        def _result(affected, backup_path=None):
            # ⤷ 出口 return：返回 {'dry_run': dry_run, 'rows_affected': len(affected), 'row_id…
            return {"dry_run": dry_run, "rows_affected": len(affected), "row_ids": affected, "backup_path": backup_path}
        with self._read_ctx() as conn:
            affected_ids = _find_affected(conn)
        if dry_run or not affected_ids:
            # ⤷ 出口 return：返回 _result(affected_ids)
            return _result(affected_ids)
        backup_path: Optional[str] = None
        if backup:
            import datetime
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.db_path.with_name(f"{self.db_path.name}.pre-clean-markers-backup-{stamp}"))
            with self._lock:
                self._conn.execute("VACUUM INTO ?", (backup_path,))
            logger.info("Backed up state.db to %s before clean-markers write", backup_path)
        def _do(conn):
            ids = _find_affected(conn)
            if ids:
                conn.execute(f"UPDATE messages SET content = '' WHERE id IN ({_placeholders(ids)})", ids)
            # ⤷ 出口 return：返回 ids
            return ids
        affected_ids = self._execute_write(_do)
        if affected_ids:
            logger.info("Permanently cleared %d stale tool-call marker row(s) in state.db (#78148)", len(affected_ids))
        # ⤷ 出口 return：返回 _result(affected_ids, backup_path)
        return _result(affected_ids, backup_path)
