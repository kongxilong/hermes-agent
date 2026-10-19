"""Base platform adapter interface; every platform adapter inherits from BasePlatformAdapter."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：gateway/platforms/base.py（约 4395 行）
# 地位：★ 所有平台适配器的抽象基类（BasePlatformAdapter）。
#
# 它是什么：
#   定义「一个聊天平台要接入 Hermes 需要实现哪些能力」。
#   具体平台（Telegram / Discord / Slack ...）的实现都在 plugins/platforms/<name>/adapter.py，
#   各自继承这个基类。想接一个新平台，照着这个基类实现一遍就对了。
#
# 核心类：class BasePlatformAdapter(ABC) —— 172 个方法！
#   主要分组：
#     消息收发：send_draft（草稿流式）/ render_message_event / format_tool_event / format_tool_preview
#     长度限制：max_message_length_for_chat / message_len_fn（不同平台字数上限不同，要切片）
#     媒体处理：cache_image_from_bytes / cache_audio_from_url / cache_video_from_bytes /
#               cache_document_from_bytes / validate_media_delivery_path
#     错误分类：classify_send_error / is_chat_level_not_found
#
# 几个非常实用的设计点：
#   - 媒体指令：_strip_media_tag_directives() —— 模型回复里可能夹带 MEDIA:/path 这样的指令，
#     发出去前要剥掉并转成真正的附件
#   - 路径安全：validate_media_delivery_path() —— 只许发白名单目录内、且是最近生成的文件，
#     防止 Agent 把 /etc/passwd 之类的文件发出去
#   - 代理：resolve_proxy_url / proxy_kwargs_for_aiohttp / should_bypass_proxy
#   - TTS：transcode_to_ogg_opus / StreamingTTSHandle（语音消息）
#
# 数据结构：SendResult / CachedMedia / ExecApprovalPrompt / AudioFormat 都是 @dataclass。
# ==========================================================================
#

import asyncio
import contextlib
import inspect
import ipaddress
import logging
import os
import random
import re
import socket as _socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from utils import normalize_proxy_url
from agent.proxy_bypass import first_proxy_env_value, should_bypass_proxy as _should_bypass_proxy

logger = logging.getLogger(__name__)


# ┌── 函数 def _consume_detached_handler_exception()
# │  作用：完成回调用于 detached 致命错误的 handler task (载体被取消在
# │  参数：task: 'asyncio.Task'
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _consume_detached_handler_exception(task: "asyncio.Task") -> None:
    """Done-callback for a detached fatal-error handler task (carrier cancelled in
    ``_notify_fatal_error``): retrieve its exception so asyncio never logs "never retrieved"."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Detached fatal-error handler task failed: %s", exc, exc_info=exc)


# Audio exts for native audio delivery; Telegram's narrower sets stay separate (.m2a is audio to
# Hermes but not to sendAudio).
_AUDIO_MIME_TYPES = {
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".mp3": "audio/mpeg", ".m2a": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/m4a", ".flac": "audio/flac"}
_AUDIO_EXTS = frozenset(_AUDIO_MIME_TYPES)
# Outbound dispatch partition for MEDIA/local files (image batch vs send_video).
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
# Telegram sendAudio accepts only MP3 / M4A; others go via sendVoice (Opus/OGG) or as a document.
_TELEGRAM_AUDIO_ATTACHMENT_EXTS = frozenset({'.mp3', '.m4a'})
_TELEGRAM_VOICE_EXTS = frozenset({'.ogg', '.opus'})


# ┌── 函数 def transcode_to_ogg_opus()
# │  作用：尽最大努力 ffmpeg 转码 Ogg/Opus (voip-tuned) 用于原生语音 bubbles: 已写入
# │  参数：path: str, bitrate: str = ..., timeout: int = ..., output_path: 'str | None' = ...
# │  返回：'str | None'
# └──────────────────────────────────────────────────────────────
def transcode_to_ogg_opus(path: str, *, bitrate: str = "32k", timeout: int = 60,
                          output_path: "str | None" = None) -> "str | None":
    """Best-effort ffmpeg transcode to Ogg/Opus (voip-tuned) for native voice bubbles: the written
    ``.ogg`` path (a NEW temp file unless ``output_path`` is given; caller cleans up), or None when
    ffmpeg is missing/fails. ``output_path`` may equal ``path`` (in-place container repair) — the
    encode goes through a sidecar so a failed run never truncates the source. Blocking (to_thread)."""
    import shutil as _shutil
    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        # ⤷ 出口 return：返回 None
        return None
    if output_path is None:
        fd, ogg_path = tempfile.mkstemp(prefix="voice_transcode_", suffix=".ogg")
        os.close(fd)
    else:
        ogg_path = output_path
    in_place = os.path.abspath(str(path)) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path),
             "-acodec", "libopus", "-ac", "1", "-b:a", bitrate, "-vbr", "on",
             "-application", "voip", "-compression_level", "10", "-f", "ogg", work_path],
            capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, ogg_path)
            # ⤷ 出口 return：返回 ogg_path
            return ogg_path
        logger.warning("ffmpeg Ogg/Opus transcode of %s failed (returncode=%s): %s", path, result.returncode,
                       (result.stderr or b"").decode("utf-8", errors="replace")[:500])
    except Exception:
        logger.warning("voice transcode to Ogg/Opus failed for %s", path, exc_info=True)
    with contextlib.suppress(OSError):
        os.unlink(work_path)
    # ⤷ 出口 return：返回 None
    return None
_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS = 30.0
# History dedup is best-effort: stay well below the Discord heartbeat watchdog and fail open.
_HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS = 5.0
# Timed-out reads can't be cancelled mid-SQLite: cap the isolated threads so wedged lookups
# can't exhaust the shared executor.
_HISTORY_MEDIA_LOOKUP_MAX_WORKERS = 2
_HISTORY_MEDIA_LOOKUP_ADMISSION = threading.BoundedSemaphore(_HISTORY_MEDIA_LOOKUP_MAX_WORKERS)


# ┌── 函数 def _platform_name()
# │  作用：归一化平台枚举 / raw 字符串进入 lowercase 名称.
# │  参数：platform
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _platform_name(platform) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    # ⤷ 出口 return：返回 str(value or '').lower()
    return str(value or "").lower()


# ┌── 函数 def _or_default()
# │  作用：``thunk()``, 或 ``默认值`` 当它 抛出 one 的 ``exc`` (numeric 配置/env coercion).
# │  参数：thunk, default, exc = (TypeError, ValueError)
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _or_default(thunk, default, exc=(TypeError, ValueError)):
    """``thunk()``, or ``default`` when it raises one of ``exc`` (numeric config/env coercion)."""
    # ⤷ 关键分支 try/except：捕获 exc，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 thunk()
        return thunk()
    except exc:
        # ⤷ 出口 return：返回 default
        return default


# ┌── 函数 def _float_env()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：name: str, default: float
# │  返回：float
# └──────────────────────────────────────────────────────────────
def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    # ⤷ 出口 return：返回 _or_default(lambda: float(raw) if raw else default, default)
    return _or_default(lambda: float(raw) if raw else default, default)


# ┌── 函数 def _thread_metadata_for_source()
# │  作用：平台感知的线程元数据用于适配器发送. Telegram 私信 topics 路由带
# │  参数：source, reply_to_message_id: str | None = None
# │  返回：dict | None
# └──────────────────────────────────────────────────────────────
def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Platform-aware thread metadata for adapter sends. Telegram DM topics route with
    ``message_thread_id`` + a reply anchor; anchorless synthetic/resumed sends fall back to
    ``direct_messages_topic_id`` when supported."""
    thread_id = getattr(source, "thread_id", None)
    platform = _platform_name(getattr(source, "platform", None))
    metadata = {"thread_id": thread_id} if thread_id is not None else {}
    # Slack workspace identity is routing state: carry it so a multi-workspace Socket Mode
    # gateway never falls back to its primary WebClient.
    scope_id = getattr(source, "scope_id", None) if platform == "slack" else None
    if scope_id:
        metadata["slack_team_id"] = str(scope_id)
    if not metadata:
        # ⤷ 出口 return：返回 None
        return None
    if platform == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        if str(thread_id) not in {"", "1"}:
            metadata["direct_messages_topic_id"] = str(thread_id)
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    # Routed profile (multiplex / profile_routes): outbound prune paths must not assume the
    # adapter's static profile stamp.
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile:
        metadata["hermes_profile"] = profile
    # ⤷ 出口 return：返回 metadata
    return metadata


# ┌── 函数 def _thread_metadata_for_event()
# │  作用：``_thread_metadata_for_source`` 用于事件, anchored 在…上其 回复 ID.
# │  参数：event
# │  返回：dict | None
# └──────────────────────────────────────────────────────────────
def _thread_metadata_for_event(event) -> dict | None:
    """``_thread_metadata_for_source`` for an event, anchored on its reply id."""
    # ⤷ 出口 return：返回 _thread_metadata_for_source(event.source, _reply_anchor_for_…
    return _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))


# ┌── 函数 def _mark_notify_metadata()
# │  作用：Clone metadata 且 mark user-visible reply 作为 notify-worthy.
# │  参数：metadata: dict | None
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    # ⤷ 出口 return：返回 notify_metadata
    return notify_metadata


# ┌── 函数 def _reply_anchor_for_event()
# │  作用：返回 reply_to id 用于平台那个 need 回复语义.。
# │  参数：event
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics."""
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None))
    thread_id = getattr(source, "thread_id", None)
    raw_message = getattr(event, "raw_message", None)
    if (platform == "slack" and isinstance(raw_message, dict)
            and raw_message.get("_hermes_no_thread_response")):
        # Slack reaction handoff = new top-level message; a message_id anchor would make
        # _resolve_thread_ts() reply in a nonexistent thread.
        # ⤷ 出口 return：返回 None
        return None
    if platform == "telegram" and thread_id:
        # Forum topics route by topic metadata (no reply); DM-topic lanes reply to the triggering
        # message — replying to the topic seed/anchor can render outside the active lane.
        if getattr(source, "chat_type", None) != "dm":
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 getattr(event, 'message_id', None) or getattr(event, 'reply_…
        return getattr(event, "message_id", None) or getattr(event, "reply_to_message_id", None)
    if platform == "feishu" and thread_id and getattr(event, "reply_to_message_id", None):
        # ⤷ 出口 return：返回 getattr(event, 'reply_to_message_id', None)
        return getattr(event, "reply_to_message_id", None)
    # ⤷ 出口 return：返回 getattr(event, 'message_id', None)
    return getattr(event, "message_id", None)


# ┌── 函数 def _media_failure_text()
# │  作用：面向用户的 "couldn't deliver" 提示; ``file_name`` 是仅 名称 ever shown.
# │  参数：kind: str, file_name: 'str | None' = None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _media_failure_text(kind: str, file_name: "str | None" = None) -> str:
    """User-facing "couldn't deliver" notice; ``file_name`` is the only name ever shown."""
    suffix = f" ({file_name})" if file_name else ""
    # ⤷ 出口 return：返回 f"⚠️ Couldn't deliver the {kind} attachment{suffix}."
    return f"⚠️ Couldn't deliver the {kind} attachment{suffix}."


# ┌── 函数 def should_send_media_as_audio()
# │  作用：当 媒体文件应当 use 平台's 音频发送器. Telegram: explicit 时返回 True。
# │  参数：platform, ext: str, is_voice: bool = False
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def should_send_media_as_audio(platform, ext: str, is_voice: bool = False) -> bool:
    """True when a media file should use the platform's audio sender. Telegram: explicit
    ``is_voice`` ([[audio_as_voice]]) routes ANY format to the voice sender (adapter transcodes
    non-Opus); otherwise only sendAudio's MP3/M4A qualify — a plain Opus/OGG attachment is never
    turned into a voice bubble, everything else → document. Other platforms: any audio ext."""
    normalized_ext = (ext or "").lower()
    if normalized_ext not in _AUDIO_EXTS:
        # ⤷ 出口 return：返回 False
        return False
    if _platform_name(platform) != "telegram":
        # ⤷ 出口 return：返回 True
        return True
    # ⤷ 出口 return：返回 is_voice or normalized_ext in _TELEGRAM_AUDIO_ATTACHMENT_EXT…
    return is_voice or normalized_ext in _TELEGRAM_AUDIO_ATTACHMENT_EXTS


# ┌── 函数 def build_auto_tts_output_path()
# │  作用：Unique 临时输出路径用于 gateway auto-TTS: ``.ogg`` 用于 ``OPUS_VOICE_PLATFORMS``
# │  参数：platform
# │  返回：str
# └──────────────────────────────────────────────────────────────
def build_auto_tts_output_path(platform) -> str:
    """Unique temp output path for gateway auto-TTS: ``.ogg`` for ``OPUS_VOICE_PLATFORMS``
    (the tool's ``_repair_ogg_container`` then guarantees real Opus bytes), else ``.mp3``.
    Platform-awareness lives HERE because ``_clear_session_env`` wipes the TTS tool's
    ``HERMES_SESSION_PLATFORM`` contextvar before the post-handler auto-TTS block runs.

    Platforms whose native voice bubbles require Ogg/Opus (``tools.tts_tool.OPUS_VOICE_PLATFORMS`` — the
    single source of truth) get an explicit ``.ogg`` path; the tool's central container repair
    (``_repair_ogg_container``) then guarantees real Ogg/Opus bytes for every provider, including MP3-only
    backends like Edge TTS. Everything else keeps the MP3 default. See #36685, #57049.
    """
    from tools.tts_tool import OPUS_VOICE_PLATFORMS
    ext = "ogg" if _platform_name(platform) in OPUS_VOICE_PLATFORMS else "mp3"
    audio_path = os.path.join(
        tempfile.gettempdir(), "hermes_voice", f"tts_reply_{uuid.uuid4().hex[:12]}.{ext}")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    # ⤷ 出口 return：返回 audio_path
    return audio_path


# ┌── 函数 def utf16_len()
# │  作用：UTF-16 码元在 *s* — Telegram's 4 096 限制 counts 那些, 从而扩展平面字符
# │  参数：s: str
# │  返回：int
# └──────────────────────────────────────────────────────────────
def utf16_len(s: str) -> int:
    """UTF-16 code units in *s* — Telegram's 4 096 limit counts those, so astral chars
    (emoji, CJK Ext B) cost **two** units although Python's ``len()`` counts one.

    Ported from nearai/ironclaw#2304 which discovered the same discrepancy in Rust's ``chars().count()``.
    """
    # ⤷ 出口 return：返回 len(s.encode('utf-16-le')) // 2
    return len(s.encode("utf-16-le")) // 2


# ┌── 函数 def _custom_unit_to_cp()
# │  作用：Largest codepoint offset *n* 带 ``len_fn(s[:n]) <= 预算`` (binary search).
# │  参数：s: str, budget: int, len_fn
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _custom_unit_to_cp(s: str, budget: int, len_fn) -> int:
    """Largest codepoint offset *n* with ``len_fn(s[:n]) <= budget`` (binary search)."""
    if len_fn(s) <= budget:
        # ⤷ 出口 return：返回 len(s)
        return len(s)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    # ⤷ 出口 return：返回 lo
    return lo


# ┌── 函数 def _prefix_within_utf16_limit()
# │  作用：Longest prefix 的 *s* 带 UTF-16 length ≤ *限制*; never splits surrogate pair.
# │  参数：s: str, limit: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _prefix_within_utf16_limit(s: str, limit: int) -> str:
    """Longest prefix of *s* with UTF-16 length ≤ *limit*; never splits a surrogate pair."""
    # ⤷ 出口 return：返回 s[:_custom_unit_to_cp(s, limit, utf16_len)]
    return s[:_custom_unit_to_cp(s, limit, utf16_len)]


# ┌── 函数 def is_network_accessible()
# │  作用：True 若 *宿主* 会 expose 服务端 beyond loopback (incl. IPv4-mapped
# │  参数：host: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_network_accessible(host: str) -> bool:
    """True if *host* would expose the server beyond loopback (incl. IPv4-mapped
    ::ffff:127.0.0.1); hostnames are resolved and DNS failure fails closed (True)."""
    with contextlib.suppress(ValueError):  # ValueError: hostname — resolve below
        addr = ipaddress.ip_address(host)
        # ::ffff:127.0.0.1 reports is_loopback=False; check the mapped IPv4 explicitly.
        mapped = getattr(addr, "ipv4_mapped", None)
        # ⤷ 出口 return：返回 not (addr.is_loopback or (mapped and mapped.is_loopback))
        return not (addr.is_loopback or (mapped and mapped.is_loopback))
    # ⤷ 关键分支 try/except：捕获 (_socket.gaierror, OSError)，出错走兜底而不是崩溃。
    try:
        resolved = _socket.getaddrinfo(host, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
        # Network-accessible if any resolved address is non-loopback.
        # ⤷ 出口 return：返回 any((not ipaddress.ip_address(sockaddr[0]).is_loopback for *…
        return any(not ipaddress.ip_address(sockaddr[0]).is_loopback for *_, sockaddr in resolved)
    except (_socket.gaierror, OSError):
        # ⤷ 出口 return：返回 True
        return True


# ┌── 函数 def _detect_macos_system_proxy()
# │  作用：读取 macOS system HTTP(S) proxy via ``scutil --proxy``: ``http://宿主:port``
# │  参数：无
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _detect_macos_system_proxy() -> str | None:
    """Read the macOS system HTTP(S) proxy via ``scutil --proxy``: ``http://host:port``
    when an HTTP(S) proxy is enabled, else None (non-macOS or any subprocess error)."""
    if sys.platform != "darwin":
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        out = subprocess.check_output(["scutil", "--proxy"], timeout=3, text=True, encoding='utf-8',
                                      errors='replace', stderr=subprocess.DEVNULL)
    except Exception:
        # ⤷ 出口 return：返回 None
        return None
    props = {
        key.strip(): val.strip()
        for key, sep, val in (line.strip().partition(" : ") for line in out.splitlines()) if sep}
    # Prefer HTTPS, fall back to HTTP
    for enable_key, host_key, port_key in (
        ("HTTPSEnable", "HTTPSProxy", "HTTPSPort"), ("HTTPEnable", "HTTPProxy", "HTTPPort")):
        if props.get(enable_key) == "1" and props.get(host_key) and props.get(port_key):
            # ⤷ 出口 return：返回 f'http://{props[host_key]}:{props[port_key]}'
            return f"http://{props[host_key]}:{props[port_key]}"
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def should_bypass_proxy()
# │  作用：当 NO_PROXY/no_proxy matches 在 least one target 宿主 (exact 宿主, domain / 时返回 True。
# │  参数：target_hosts: str | list[str] | tuple[str, ...] | set[str] | None
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def should_bypass_proxy(target_hosts: str | list[str] | tuple[str, ...] | set[str] | None) -> bool:
    """True when NO_PROXY/no_proxy matches at least one target host (exact hosts, domain /
    wildcard suffixes, IP literals, CIDR ranges, optional host:port entries, ``*``)."""
    # ⤷ 出口 return：返回 _should_bypass_proxy(target_hosts)
    return _should_bypass_proxy(target_hosts)


# ┌── 函数 def resolve_proxy_url()
# │  作用：Proxy URL: *platform_env_var* (e.g. ``DISCORD_PROXY``) first, 然后适配器's 持有 YAML
# │  参数：platform_env_var: str | None = None, target_hosts: str | list[str] | tuple[str, ...] | set[str] | None = ..., configured: str | None = ...
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def resolve_proxy_url(
    platform_env_var: str | None = None, *,
    target_hosts: str | list[str] | tuple[str, ...] | set[str] | None = None,
    configured: str | None = None) -> str | None:
    """Proxy URL: *platform_env_var* (e.g. ``DISCORD_PROXY``) first, then the adapter's own YAML
    value *configured* (``telegram.proxy_url``), then HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (any
    case), then the macOS system proxy — the latter two only when ``gateway.trust_env`` is true.
    None when nothing is found or NO_PROXY matches a target.

    *platform_env_var* is a per-adapter, per-profile-configurable setting (each proxy URL can
    embed credentials, e.g. ``http://user:pass@host``) so it is read scope-aware: under a
    secondary multiplex profile it comes from that profile's own ``.env``, not the shared
    process env another profile's ``TELEGRAM_PROXY``/``DISCORD_PROXY``/etc. may hold; the YAML
    value is the same profile's, so a secondary keeps its configured route without any env
    bridge (#108440). The generic ``HTTPS_PROXY``/``HTTP_PROXY``/``ALL_PROXY`` fallback stays a raw
    process-env read — those are OS/system-level network settings, not a per-profile Hermes concept."""
    from gateway.platforms._shared import get_scoped_secret as _get_scoped_proxy_var
    value = (_get_scoped_proxy_var(platform_env_var, "") or "").strip() if platform_env_var else ""
    if not value:
        value = str(configured or "").strip()
    if not value:
        if not gateway_trust_env():  # only the explicit per-platform var is honored
            # ⤷ 出口 return：返回 None
            return None
        value = first_proxy_env_value()
    proxy = normalize_proxy_url(value or _detect_macos_system_proxy())
    # ⤷ 出口 return：返回 None if proxy and should_bypass_proxy(target_hosts) else pro…
    return None if proxy and should_bypass_proxy(target_hosts) else proxy


# ┌── 函数 def _aiohttp_socks_connector()
# │  作用：``aiohttp_socks.ProxyConnector`` 用于 ``proxy_url``, 或 None 当 aiohttp_socks 是 missing
# │  参数：proxy_url: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _aiohttp_socks_connector(proxy_url: str):
    """``aiohttp_socks.ProxyConnector`` for ``proxy_url``, or None when aiohttp_socks is missing
    (SOCKS logs a warning; HTTP callers fall back to ``proxy=``). ``rdns=True`` forces remote DNS
    through the proxy — required by Shadowrocket/Clash-style SOCKS and against GFW DNS pollution."""
    # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
    try:
        from aiohttp_socks import ProxyConnector
        # ⤷ 出口 return：返回 ProxyConnector.from_url(proxy_url, rdns=True)
        return ProxyConnector.from_url(proxy_url, rdns=True)
    except ImportError:
        if proxy_url.lower().startswith("socks"):
            logger.warning("aiohttp_socks not installed — SOCKS proxy %s ignored. "
                           "Run: pip install aiohttp-socks", proxy_url)
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def proxy_kwargs_for_bot()
# │  作用：Kwargs 用于 ``命令.机器人()`` / ``discord.客户端()``: SOCKS → ``{"connector"}``,
# │  参数：proxy_url: str | None
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def proxy_kwargs_for_bot(proxy_url: str | None) -> dict:
    """Kwargs for ``commands.Bot()`` / ``discord.Client()``: SOCKS → ``{"connector"}``,
    HTTP → ``{"proxy": url}``, None → ``{}``."""
    if not proxy_url:
        # ⤷ 出口 return：返回 {}
        return {}
    if proxy_url.lower().startswith("socks"):
        connector = _aiohttp_socks_connector(proxy_url)
        # ⤷ 出口 return：返回 {'connector': connector} if connector is not None else {}
        return {"connector": connector} if connector is not None else {}
    # ⤷ 出口 return：返回 {'proxy': proxy_url}
    return {"proxy": proxy_url}


# ┌── 函数 def _config_section()
# │  作用：读取-仅 ``配置.YAML`` section ``名称``; ``{}`` 当 unreadable/missing/不字典.
# │  参数：name: str
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _config_section(name: str) -> dict:
    """Read-only ``config.yaml`` section ``name``; ``{}`` when unreadable/missing/not a dict."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.config import load_config_readonly as _load_config
        cfg = _load_config()  # read-only: .get() only, never mutated
    except Exception:
        # ⤷ 出口 return：返回 {}
        return {}
    section = cfg.get(name) if isinstance(cfg, dict) else None
    # ⤷ 出口 return：返回 section if isinstance(section, dict) else {}
    return section if isinstance(section, dict) else {}


# ┌── 函数 def gateway_trust_env()
# │  作用：``gateway.trust_env`` 从配置.YAML (默认值 True): whether gateway
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def gateway_trust_env() -> bool:
    """``gateway.trust_env`` from config.yaml (default True): whether gateway
    ``aiohttp.ClientSession``s honor HTTP(S)_PROXY / NO_PROXY / SSL_CERT_FILE. Set false
    when the gateway inherits a proxy env it must not use. Fail-open to default."""
    value = _config_section("gateway").get("trust_env", True)
    if isinstance(value, str):
        # ⤷ 出口 return：返回 value.strip().lower() not in {'0', 'false', 'no', 'off'}
        return value.strip().lower() not in {"0", "false", "no", "off"}
    # ⤷ 出口 return：返回 bool(value) if value is not None else True
    return bool(value) if value is not None else True


# ┌── 函数 def proxy_kwargs_for_aiohttp()
# │  作用：``(session_kwargs, request_kwargs)`` 用于 standalone ``aiohttp.ClientSession``. 带
# │  参数：proxy_url: str | None
# │  返回：tuple[dict, dict]
# └──────────────────────────────────────────────────────────────
def proxy_kwargs_for_aiohttp(proxy_url: str | None) -> tuple[dict, dict]:
    """``(session_kwargs, request_kwargs)`` for a standalone ``aiohttp.ClientSession``. With
    aiohttp-socks every scheme uses a connector (mautrix-style libs never forward per-request
    ``proxy=``); without it HTTP falls back to ``({}, {"proxy": url})`` and SOCKS is ignored."""
    if not proxy_url:
        # ⤷ 出口 return：返回 ({}, {})
        return {}, {}
    connector = _aiohttp_socks_connector(proxy_url)
    if connector is not None:
        # ⤷ 出口 return：返回 ({'connector': connector}, {})
        return {"connector": connector}, {}
    # ⤷ 出口 return：返回 ({}, {}) if proxy_url.lower().startswith('socks') else ({}, …
    return ({}, {}) if proxy_url.lower().startswith("socks") else ({}, {"proxy": proxy_url})


# ┌── 函数 def is_host_excluded_by_no_proxy()
# │  作用：当 ``hostname`` matches ``NO_PROXY`` entry (``no_proxy_value`` 覆盖值 时返回 True。
# │  参数：hostname: str, no_proxy_value: str | None = None
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_host_excluded_by_no_proxy(hostname: str, no_proxy_value: str | None = None) -> bool:
    """Return True when ``hostname`` matches a ``NO_PROXY`` entry (``no_proxy_value`` overrides the
    environment); same matcher as :func:`should_bypass_proxy`."""
    # ⤷ 出口 return：返回 _should_bypass_proxy(hostname, no_proxy_value=no_proxy_value…
    return _should_bypass_proxy(hostname, no_proxy_value=no_proxy_value)


import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Callable, Awaitable, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import fence_state_after
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key
from gateway.session_transcript import TranscriptReadError
from hermes_constants import get_default_hermes_root, get_hermes_dir, get_hermes_home

if TYPE_CHECKING:
    from agent.display import ToolPreview

# ┌── 类 class AudioFormat
# │  说明：Declared PCM format 用于 streaming-TTS 会话: 每个 ``write_streaming_tts``
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
# --------------------------------------------------------------------------- Streaming TTS format
# descriptor and handle (#60671) ---------------------------------------------------------------------------
class AudioFormat:
    """Declared PCM format for a streaming-TTS session: every ``write_streaming_tts``
    chunk must be raw little-endian PCM at this rate / channels / sample width."""
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (int16 = 2)


# ┌── 类 class StreamingTTSHandle
# │  说明：Opaque handle returned 由 ``begin_streaming_tts``; adapters 可以 extend 它带
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class StreamingTTSHandle:
    """Opaque handle returned by ``begin_streaming_tts``; adapters may extend it with
    platform state. The base fields are consumer bookkeeping / cancellation."""
    chat_id: str = ""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    # True once the first PCM chunk is written: a later failure then ends cleanly instead of
    # falling back to whole-file TTS (don't replay already-audible output).
    audible: bool = False
    aborted: bool = False  # set by abort_streaming_tts; late chunks are dropped


# ┌── 函数 def streaming_tts_turn_key()
# │  作用：Per-轮次 streaming-TTS suppression 键 — 轮次-scoped (不 chat-scoped) 从而
# │  参数：session_key: str | None, turn_marker: Any = None, event: Any = ...
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def streaming_tts_turn_key(session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> str | None:
    """Per-turn streaming-TTS suppression key — turn-scoped (not chat-scoped) so
    overlapping turns in one chat can't suppress each other's fallback paths.
    ``turn_marker`` is normally the run generation, else the event's message/update id."""
    if not session_key:
        # ⤷ 出口 return：返回 None
        return None
    if turn_marker is None and event is not None:
        turn_marker = getattr(event, "message_id", None) or getattr(event, "platform_update_id", None)
    # ⤷ 出口 return：返回 None if turn_marker is None else f'{session_key}:{turn_marke…
    return None if turn_marker is None else f"{session_key}:{turn_marker}"


# ┌── 函数 def streaming_tts_should_skip_whole_file()
# │  作用：Pure, 轮次-scoped auto-TTS suppression decision (testable 不适配器 stack).
# │  参数：completed_turns: set[str], session_key: str | None, turn_marker: Any = None, event: Any = ...
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def streaming_tts_should_skip_whole_file(completed_turns: set[str], session_key: str | None,
                                         turn_marker: Any = None, *, event: Any = None) -> bool:
    """Pure, turn-scoped auto-TTS suppression decision (testable without the adapter stack)."""
    turn_key = streaming_tts_turn_key(session_key, turn_marker, event=event)
    # ⤷ 出口 return：返回 bool(turn_key and turn_key in completed_turns)
    return bool(turn_key and turn_key in completed_turns)


GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE = (
    "Secure secret entry is not supported over messaging. "
    "Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually.")


# ┌── 函数 def safe_url_for_log()
# │  作用：返回 URL 字符串 safe 用于日志 (无 query/fragment/userinfo).。
# │  参数：url: str, max_len: int = 80
# │  返回：str
# └──────────────────────────────────────────────────────────────
def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo)."""
    raw = str(url) if max_len > 0 and url is not None else ""
    if not raw:
        # ⤷ 出口 return：返回 ''
        return ""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        parsed = urlsplit(raw)
    except Exception:
        # ⤷ 出口 return：返回 raw[:max_len]
        return raw[:max_len]
    safe = raw
    if parsed.scheme and parsed.netloc:
        # Strip potential embedded credentials (user:pass@host).
        path = parsed.path or ""
        basename = path.rsplit("/", 1)[-1]
        tail = "" if path in ("", "/") else f"/.../{basename}" if basename else "/..."
        safe = f"{parsed.scheme}://{parsed.netloc.rsplit('@', 1)[-1]}{tail}"
    if len(safe) <= max_len:
        # ⤷ 出口 return：返回 safe
        return safe
    # ⤷ 出口 return：返回 '.' * max_len if max_len <= 3 else f'{safe[:max_len - 3]}...…
    return "." * max_len if max_len <= 3 else f"{safe[:max_len - 3]}..."


# ┌── 异步函数（协程） def _ssrf_redirect_guard()
# │  作用：Re-校验每个 redirect target ( public URL 302-ing http://169.254.169.254/ 会
# │  参数：response
# │  返回：（未标注类型）
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _ssrf_redirect_guard(response):
    """Re-validate each redirect target (a public URL 302-ing to http://169.254.169.254/ would
    bypass the pre-flight is_safe_url()). Async because httpx awaits response event hooks."""
    from tools.url_safety import is_safe_url, redirect_target_from_response
    redirect_url = redirect_target_from_response(response)
    if redirect_url and not is_safe_url(redirect_url):
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Blocked redirect to private/internal address: {safe_url_for_log(redirect_url)}")


# Inbound images are cached locally for the vision tool (platform URLs are ephemeral).
# Import-time default; tests monkeypatch it, getters re-resolve per call.
IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")


# Inbound media cap (``gateway.max_inbound_media_bytes``): payloads are buffered fully in memory,
# so an uncapped upload (Discord Nitro: 500 MB) could OOM-kill the gateway.
# Inbound image / audio / video payloads are buffered fully into process memory before being written to the
# cache directory. With no cap, a single large upload (Discord Nitro allows 500 MB) — or a remote URL in an
# inbound message payload pointing at an arbitrarily large file — can spike RAM and OOM-kill the gateway.
# The ``cache_*_from_bytes`` helpers (the shared funnel every platform reaches eventually) and the
# ``cache_*_from_url`` downloaders enforce this cap, so the protection holds regardless of which platform
# adapter or code path produced the bytes. Configurable via ``gateway.max_inbound_media_bytes`` in
# config.yaml. ``0`` disables the cap. Default 128 MiB — generous enough for ordinary photos/voice
# notes/short clips while still bounding a hostile upload.
# --------------------------------------------------------------------------- See #13145.
DEFAULT_INBOUND_MEDIA_MAX_BYTES = 128 * 1024 * 1024


# ┌── 函数 def get_inbound_media_max_bytes()
# │  作用：Max inbound media bytes held 在内存 (``gateway.max_inbound_media_bytes``);
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def get_inbound_media_max_bytes() -> int:
    """Max inbound media bytes held in memory (``gateway.max_inbound_media_bytes``);
    ``0`` / negative / unparseable disables the cap; unreadable config → default."""
    # ⤷ 出口 return：返回 _or_default(lambda: int(_config_section('gateway')['max_inbo…
    return _or_default(lambda: int(_config_section("gateway")["max_inbound_media_bytes"]),
                       DEFAULT_INBOUND_MEDIA_MAX_BYTES, (KeyError, TypeError, ValueError))


# ┌── 函数 def validate_inbound_media_size()
# │  作用：抛出 ``ValueError`` 若 inbound payload exceeds cap (``max_bytes`` 的 ``0``
# │  参数：size: int, media_type: str = ..., max_bytes: Optional[int] = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def validate_inbound_media_size(
    size: int, *, media_type: str = "media", max_bytes: Optional[int] = None) -> None:
    """Raise ``ValueError`` if an inbound payload exceeds the cap (``max_bytes`` of ``0``
    disables it; pass it explicitly to resolve the limit once across an incremental read)."""
    limit = get_inbound_media_max_bytes() if max_bytes is None else max_bytes
    if limit and size > limit:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Inbound {media_type} payload is too large ({size} bytes > {limit} bytes)")


# ┌── 异步函数（协程） def _read_httpx_body_with_limit()
# │  作用：读取 httpx streaming body under media cap: 拒绝 oversized ``内容-Length``
# │  参数：response, media_type: str
# │  返回：bytes
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _read_httpx_body_with_limit(response, *, media_type: str) -> bytes:
    """Read an httpx streaming body under the media cap: reject an oversized ``Content-Length``
    early, then re-check the running total per chunk (a lying/absent header can't smuggle more)."""
    max_bytes = get_inbound_media_max_bytes()
    content_length = response.headers.get("content-length")
    if content_length:
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            declared_size = int(content_length)
        except ValueError:
            logger.debug("Ignoring invalid Content-Length for inbound %s: %r", media_type, content_length)
        else:
            validate_inbound_media_size(declared_size, media_type=media_type, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        validate_inbound_media_size(total, media_type=media_type, max_bytes=max_bytes)
        chunks.append(chunk)
    # ⤷ 出口 return：返回 b''.join(chunks)
    return b"".join(chunks)


# ┌── 函数 def _cache_dir_accessors()
# │  作用：``(get_<种类>_cache_dir, cleanup_<种类>_cache)`` pair. getter resolves fresh via
# │  参数：kind: str, constant_name: str, new_subpath: str, old_name: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _cache_dir_accessors(kind: str, constant_name: str, new_subpath: str, old_name: str):
    """``(get_<kind>_cache_dir, cleanup_<kind>_cache)`` pair. The getter resolves fresh via
    get_hermes_dir (active profile) unless a test monkeypatched the module constant away from
    its import-time default, and creates the directory; ``cleanup(max_age_hours=24)`` deletes
    older files and returns the count."""
    def get_dir() -> Path:
        d = get_hermes_dir(new_subpath, old_name)
        current = globals().get(constant_name)
        default = _CACHE_DIR_IMPORT_DEFAULTS.get(constant_name)
        if current is not None and default is not None and current != default:
            d = Path(current)
        d.mkdir(parents=True, exist_ok=True)
        # ⤷ 出口 return：返回 d
        return d

    def cleanup(max_age_hours: int = 24) -> int:
        # ⤷ 出口 return：返回 _cleanup_cache_dir(get_dir(), max_age_hours)
        return _cleanup_cache_dir(get_dir(), max_age_hours)
    get_dir.__name__ = get_dir.__qualname__ = f"get_{kind}_cache_dir"
    cleanup.__name__ = cleanup.__qualname__ = f"cleanup_{kind}_cache"
    # ⤷ 出口 return：返回 (get_dir, cleanup)
    return get_dir, cleanup


get_image_cache_dir, cleanup_image_cache = _cache_dir_accessors(
    "image", "IMAGE_CACHE_DIR", "cache/images", "image_cache")


# ┌── 函数 def _looks_like_image()
# │  作用：返回 True 若 *data* starts 带 known image magic-byte sequence.。
# │  参数：data: bytes
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _looks_like_image(data: bytes) -> bool:
    """Return True if *data* starts with a known image magic-byte sequence."""
    # ⤷ 出口 return：返回 len(data) >= 4 and (data[:8] == b'\x89PNG\r\n\x1a\n' or data…
    return len(data) >= 4 and (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
               or data[:6] in {b"GIF87a", b"GIF89a"} or data[:2] == b"BM"
               or (data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP"))


# ┌── 函数 def _write_cache_file()
# │  作用：写入 ``data`` ``<cache_dir>/<prefix>_<uuid12><ext>``; 返回路径字符串.
# │  参数：cache_dir: Path, prefix: str, ext: str, data: bytes
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _write_cache_file(cache_dir: Path, prefix: str, ext: str, data: bytes) -> str:
    """Write ``data`` to ``<cache_dir>/<prefix>_<uuid12><ext>``; return the path string."""
    filepath = cache_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    filepath.write_bytes(data)
    # ⤷ 出口 return：返回 str(filepath)
    return str(filepath)


# ┌── 函数 def cache_image_from_bytes()
# │  作用：Save raw image bytes 缓存且 返回 absolute 路径; 抛出
# │  参数：data: bytes, ext: str = '.jpg'
# │  返回：str
# └──────────────────────────────────────────────────────────────
def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """Save raw image bytes to the cache and return the absolute path; raises
    ValueError when *data* isn't an image (e.g. an upstream HTML error page)."""
    validate_inbound_media_size(len(data), media_type="image")
    if not _looks_like_image(data):
        snippet = data[:80].decode("utf-8", errors="replace")
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Refusing to cache non-image data as {ext} (starts with: {snippet!r})")
    # ⤷ 出口 return：返回 _write_cache_file(get_image_cache_dir(), 'img', ext, data)
    return _write_cache_file(get_image_cache_dir(), "img", ext, data)


# ┌── 异步函数（协程） def cache_image_from_bytes_async()
# │  作用：缓存 image bytes 不 blocking caller's 事件循环.
# │  参数：data: bytes, ext: str = '.jpg'
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_image_from_bytes_async(data: bytes, ext: str = ".jpg") -> str:
    """Cache image bytes without blocking the caller's event loop."""
    # ⤷ 出口 return：返回 await asyncio.to_thread(cache_image_from_bytes, data, ext)
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(cache_image_from_bytes, data, ext)


# ┌── 异步函数（协程） def _cache_media_from_url()
# │  作用：Shared downloader behind ``cache_*_from_url``: SSRF-已检查 (pre-flight + per-redirect;
# │  参数：url: str, ext: str, retries: int, media_type: str, accept: str, cache_fn, log_label: str
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _cache_media_from_url(url: str, ext: str, retries: int, *, media_type: str, accept: str,
                                cache_fn, log_label: str) -> str:
    """Shared downloader behind ``cache_*_from_url``: SSRF-checked (pre-flight + per-redirect;
    raises ValueError), size-capped, linear-backoff retries on timeouts / 429 / 5xx."""
    from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
    import httpx
    if not is_safe_url(url):
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Blocked unsafe URL (SSRF protection): {safe_url_for_log(url)}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)", "Accept": accept}
    async with create_ssrf_safe_async_client(
        timeout=30.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
    ) as client:
        for attempt in range(retries + 1):
            # ⤷ 关键分支 try/except：捕获 (httpx.TimeoutException, httpx.HTTPStatusError)，出错走兜底而不是崩溃。
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    content = await _read_httpx_body_with_limit(response, media_type=media_type)
                # ⤷ 出口 return：返回 await asyncio.to_thread(cache_fn, content, ext)
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                return await asyncio.to_thread(cache_fn, content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    logger.debug("%s cache retry %d/%d for %s (%.1fs): %s", log_label, attempt + 1,
                                 retries, safe_url_for_log(url), wait, exc)
                    await asyncio.sleep(wait)
                    continue
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise


# ┌── 异步函数（协程） def cache_image_from_url()
# │  作用：Download image URL 进入 image 缓存; 返回 absolute 路径.
# │  参数：url: str, ext: str = '.jpg', retries: int = 2
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_image_from_url(url: str, ext: str = ".jpg", retries: int = 2) -> str:
    """Download an image URL into the image cache; return the absolute path."""
    # ⤷ 出口 return：返回 await _cache_media_from_url(url, ext, retries, media_type='i…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _cache_media_from_url(
        url, ext, retries, media_type="image", accept="image/*,*/*;q=0.8",
        cache_fn=cache_image_from_bytes, log_label="Media")


# ┌── 函数 def _cleanup_cache_dir()
# │  作用：Delete 文件在 *cache_dir* older 比 *max_age_hours*; 返回 count 已移除.
# │  参数：cache_dir: Path, max_age_hours: int
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _cleanup_cache_dir(cache_dir: Path, max_age_hours: int) -> int:
    """Delete files in *cache_dir* older than *max_age_hours*; return the count removed."""
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            with contextlib.suppress(OSError):
                f.unlink()
                removed += 1
    # ⤷ 出口 return：返回 removed
    return removed


# Audio cache utilities (same pattern as images; feeds the STT tool).
AUDIO_CACHE_DIR = get_hermes_dir("cache/audio", "audio_cache")
get_audio_cache_dir, cleanup_audio_cache = _cache_dir_accessors(
    "audio", "AUDIO_CACHE_DIR", "cache/audio", "audio_cache")


# ┌── 函数 def cache_audio_from_bytes()
# │  作用：Save raw audio bytes 缓存 (container-sniffed ext); 返回路径.
# │  参数：data: bytes, ext: str = '.ogg'
# │  返回：str
# └──────────────────────────────────────────────────────────────
def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """Save raw audio bytes to the cache (container-sniffed ext); return the path."""
    # tools.audio_container is the ONE owner of container detection (outbound TTS repair + here).
    from tools.audio_container import sniff_audio_ext
    validate_inbound_media_size(len(data), media_type="audio")
    # ⤷ 出口 return：返回 _write_cache_file(get_audio_cache_dir(), 'audio', sniff_audi…
    return _write_cache_file(get_audio_cache_dir(), "audio", sniff_audio_ext(data, ext), data)


# ┌── 异步函数（协程） def cache_audio_from_bytes_async()
# │  作用：缓存 audio bytes 不 blocking caller's 事件循环.
# │  参数：data: bytes, ext: str = '.ogg'
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_audio_from_bytes_async(data: bytes, ext: str = ".ogg") -> str:
    """Cache audio bytes without blocking the caller's event loop."""
    # ⤷ 出口 return：返回 await asyncio.to_thread(cache_audio_from_bytes, data, ext)
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(cache_audio_from_bytes, data, ext)


# ┌── 异步函数（协程） def cache_audio_from_url()
# │  作用：Download audio URL 进入 audio 缓存; 返回 absolute 路径.
# │  参数：url: str, ext: str = '.ogg', retries: int = 2
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_audio_from_url(url: str, ext: str = ".ogg", retries: int = 2) -> str:
    """Download an audio URL into the audio cache; return the absolute path."""
    # ⤷ 出口 return：返回 await _cache_media_from_url(url, ext, retries, media_type='a…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _cache_media_from_url(
        url, ext, retries, media_type="audio", accept="audio/*,*/*;q=0.8",
        cache_fn=cache_audio_from_bytes, log_label="Audio")


# Video cache utilities (same pattern; referenced by local path).
VIDEO_CACHE_DIR = get_hermes_dir("cache/videos", "video_cache")
get_video_cache_dir, cleanup_video_cache = _cache_dir_accessors(
    "video", "VIDEO_CACHE_DIR", "cache/videos", "video_cache")

SUPPORTED_VIDEO_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}


# ┌── 函数 def cache_video_from_bytes()
# │  作用：Save raw video bytes 缓存且 返回 absolute 文件路径.
# │  参数：data: bytes, ext: str = '.mp4'
# │  返回：str
# └──────────────────────────────────────────────────────────────
def cache_video_from_bytes(data: bytes, ext: str = ".mp4") -> str:
    """Save raw video bytes to the cache and return the absolute file path."""
    validate_inbound_media_size(len(data), media_type="video")
    # ⤷ 出口 return：返回 _write_cache_file(get_video_cache_dir(), 'video', ext, data)
    return _write_cache_file(get_video_cache_dir(), "video", ext, data)


# ┌── 异步函数（协程） def cache_video_from_bytes_async()
# │  作用：缓存 video bytes 不 blocking caller's 事件循环.
# │  参数：data: bytes, ext: str = '.mp4'
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_video_from_bytes_async(data: bytes, ext: str = ".mp4") -> str:
    """Cache video bytes without blocking the caller's event loop."""
    # ⤷ 出口 return：返回 await asyncio.to_thread(cache_video_from_bytes, data, ext)
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(cache_video_from_bytes, data, ext)


# Document / screenshot cache utilities (same pattern; referenced by local path).
DOCUMENT_CACHE_DIR = get_hermes_dir("cache/documents", "document_cache")
SCREENSHOT_CACHE_DIR = get_hermes_dir("cache/screenshots", "browser_screenshots")
get_document_cache_dir, cleanup_document_cache = _cache_dir_accessors(
    "document", "DOCUMENT_CACHE_DIR", "cache/documents", "document_cache")
get_screenshot_cache_dir, cleanup_screenshot_cache = _cache_dir_accessors(
    "screenshot", "SCREENSHOT_CACHE_DIR", "cache/screenshots", "browser_screenshots")

# Import-time defaults; _resolve_cache_dir compares against these to detect a test monkeypatch.
_CACHE_DIR_IMPORT_DEFAULTS = {
    "IMAGE_CACHE_DIR": IMAGE_CACHE_DIR, "AUDIO_CACHE_DIR": AUDIO_CACHE_DIR,
    "VIDEO_CACHE_DIR": VIDEO_CACHE_DIR, "DOCUMENT_CACHE_DIR": DOCUMENT_CACHE_DIR,
    "SCREENSHOT_CACHE_DIR": SCREENSHOT_CACHE_DIR}

# Launch-time homes: fine for the static ALLOW roots below (per-profile cache roots are
# enumerated at check time), never for the credential DENY side — see _credential_home_roots.
_HERMES_HOME = get_hermes_home()
_HERMES_ROOT = get_default_hermes_root()
MEDIA_DELIVERY_ALLOW_DIRS_ENV = "HERMES_MEDIA_ALLOW_DIRS"
MEDIA_DELIVERY_TRUST_RECENT_ENV = "HERMES_MEDIA_TRUST_RECENT_FILES"
MEDIA_DELIVERY_TRUST_RECENT_SECONDS_ENV = "HERMES_MEDIA_TRUST_RECENT_SECONDS"
# Strict mode = allowlist+recency validation; off by default (the denylist still blocks
# credential / system paths). Set true on public-facing gateways.
MEDIA_DELIVERY_STRICT_ENV = "HERMES_MEDIA_DELIVERY_STRICT"
# Canonical cache subdirs of deliverable artifacts; also enumerates per-profile cache roots.
_MEDIA_DELIVERY_CACHE_SUBDIRS = ("images", "audio", "videos", "documents", "screenshots")
MEDIA_DELIVERY_SAFE_ROOTS = (
    IMAGE_CACHE_DIR, AUDIO_CACHE_DIR, VIDEO_CACHE_DIR, DOCUMENT_CACHE_DIR, SCREENSHOT_CACHE_DIR,
    *(_HERMES_HOME / d for d in (
        "image_cache", "audio_cache", "video_cache", "document_cache", "browser_screenshots")),
    # Canonical cache layout, alongside the legacy *_cache dirs (installs may have both).
    *(_HERMES_HOME / "cache" / d for d in _MEDIA_DELIVERY_CACHE_SUBDIRS))

# Recency window (s) for trusting fresh files: artifacts land seconds before delivery,
# pre-existing host files (/etc/passwd, ~/.ssh/id_rsa) are days/months old.
_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS = 600

# Hard denylist even for "recent" files (credentials, system state, /proc); the cache-dir
# allowlist still beats it.
_MEDIA_DELIVERY_DENIED_PREFIXES = (
    "/etc", "/proc", "/sys", "/dev", "/root", "/boot", "/var/log", "/var/lib", "/var/run")

# Credential / config dirs denied under $HOME (Library/Keychains = macOS), resolved at check time.
_MEDIA_DELIVERY_DENIED_HOME_SUBPATHS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config", ".azure", ".gcloud",
    "Library/Keychains")

# ┌── 函数 def _sqlite_files()
# │  作用：SQLite store plus 其 WAL/SHM/rollback-journal sidecars.
# │  参数：name: str
# │  返回：tuple[str, ...]
# └──────────────────────────────────────────────────────────────
def _sqlite_files(name: str) -> tuple[str, ...]:
    """A SQLite store plus its WAL/SHM/rollback-journal sidecars."""
    # ⤷ 出口 return：返回 (name, f'{name}-wal', f'{name}-shm', f'{name}-journal')
    return (name, f"{name}-wal", f"{name}-shm", f"{name}-journal")


# Credential stores at the HERMES_HOME root, denied per-file so skills/, logs/ and agent-written
# files stay deliverable (cache subdirs are allowlisted BEFORE this). A superset of the
# agent/file_safety.py read+write denies so exfil never trails the read guard. google_token.json's mtime bumps every turn (defeats the
# recency window); pairing/ and mcp-tokens/ (live OAuth tokens) are denied as whole trees.
_ROOT_CREDENTIAL_PATHS = (
    ".env", "auth.json", "auth.lock", "credentials", "config.yaml", ".anthropic_oauth.json",
    "google_token.json", "google_oauth_pending.json", os.path.join("auth", "google_oauth.json"),
    "webhook_subscriptions.json", os.path.join("cache", "bws_cache.json"),
    os.path.join("cache", "bws_cache.enc.json"), "pairing", "mcp-tokens",
    # Whole conversation history (every secret ever pasted into a chat) and the copied browser
    # cookie/login store; sessions/ is the legacy transcript dir. SQLite sidecars are listed
    # too: WAL mode touches state.db-wal on every write, so recency trust alone would leak them.
    "sessions", "browser-profile", *_sqlite_files("state.db"), *_sqlite_files("kanban.db"))


# ┌── 函数 def _profile_cache_roots()
# │  作用：按配置档的缓存 roots ``<root>/配置档/<名称>/缓存/{images,...}`` ( static safe
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _profile_cache_roots() -> List[Path]:
    """Per-profile cache roots ``<root>/profiles/<name>/cache/{images,...}`` (the static safe
    roots cover only the active HERMES_HOME). Enumerated at check time so profiles created after
    startup count and are allowlisted BEFORE the ``/root`` denylist (HERMES_HOME symlinked).

    ``HERMES_HOME=/opt/data``) while the model emits a profile-scoped path silently fails delivery.
    Enumerated dynamically at check time so profiles created after startup are covered, and so the resolved
    profile path is allowlisted *before* the ``/root`` system denylist is consulted (which otherwise wins
    when HERMES_HOME is symlinked under a denied prefix and $HOME is not that prefix). See issue #31733.
    """
    # ⤷ 出口 return：返回 [p / 'cache' / subdir for p in _profile_dirs() for subdir in…
    return [p / "cache" / subdir for p in _profile_dirs() for subdir in _MEDIA_DELIVERY_CACHE_SUBDIRS]


# ┌── 函数 def _profile_dirs()
# │  作用：每个 ``<root>/配置档/<名称>`` directory, 读取在 检查 time.
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _profile_dirs() -> List[Path]:
    """Every ``<root>/profiles/<name>`` directory, read at check time."""
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 [p for p in (_HERMES_ROOT / 'profiles').iterdir() if p.is_di…
        return [p for p in (_HERMES_ROOT / "profiles").iterdir() if p.is_dir()]
    except OSError:
        # ⤷ 出口 return：返回 []
        return []


# ┌── 函数 def _credential_home_roots()
# │  作用：每个 Hermes 主目录 whose credential stores denylist 必须 cover: ACTIVE 主目录
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _credential_home_roots() -> List[Path]:
    """Every Hermes home whose credential stores the denylist must cover: the ACTIVE home
    (the per-turn HERMES_HOME override under ``gateway.multiplex_profiles``), the shared root
    and every ``<root>/profiles/*``. Enumerated at check time like ``_profile_cache_roots`` on
    the allow side — a denylist frozen at import covers only the launch profile, so a
    ``MEDIA:<root>/profiles/<other>/.env`` emitted in any profile's turn would upload it."""
    # ⤷ 出口 return：返回 list(dict.fromkeys((get_hermes_home(), _HERMES_ROOT, *_profi…
    return list(dict.fromkeys((get_hermes_home(), _HERMES_ROOT, *_profile_dirs())))


# ┌── 函数 def _kanban_root()
# │  作用：Kanban 是 root-shared across 配置档由 design (``kanban_db.kanban_home``).
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _kanban_root() -> Path:
    """Kanban is root-shared across profiles by design (``kanban_db.kanban_home``)."""
    # ⤷ 出口 return：返回 Path(os.environ.get('HERMES_KANBAN_HOME', '').strip() or _HE…
    return Path(os.environ.get("HERMES_KANBAN_HOME", "").strip() or _HERMES_ROOT).expanduser()


# ┌── 函数 def _kanban_board_dirs()
# │  作用：每个 directory under ``<root>/kanban/boards`` (lax 在…上 purpose: DENY side 必须 catch
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _kanban_board_dirs() -> List[Path]:
    """Every directory under ``<root>/kanban/boards`` (lax on purpose: the DENY side must catch a
    board whatever its name; the allow side filters further)."""
    with contextlib.suppress(OSError):
        # ⤷ 出口 return：返回 [p for p in (_kanban_root() / 'kanban' / 'boards').iterdir()…
        return [p for p in (_kanban_root() / "kanban" / "boards").iterdir() if p.is_dir()]
    # ⤷ 出口 return：返回 []
    return []


# ┌── 函数 def _kanban_attachment_roots()
# │  作用：返回 durable Kanban attachment roots 不导入 kanban_db.。
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _kanban_attachment_roots() -> List[Path]:
    """Return durable Kanban attachment roots without importing kanban_db."""
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        # ⤷ 出口 return：返回 [Path(override).expanduser()]
        return [Path(override).expanduser()]
    roots = [_kanban_root() / "kanban" / "attachments"]
    roots.extend(path / "attachments" for path in _kanban_board_dirs()
                 if not path.is_symlink() and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", path.name)
                 and (path / "kanban.db").is_file())
    # ⤷ 出口 return：返回 roots
    return roots


# ┌── 函数 def _media_delivery_allowed_roots()
# │  作用：返回 roots 从其 模型-emitted local media 可以是 delivered.。
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _media_delivery_allowed_roots() -> List[Path]:
    """Return roots from which model-emitted local media may be delivered."""
    from gateway.media_policy import media_delivery_allow_dirs
    operator_roots = (
        root for chunk in media_delivery_allow_dirs().split(os.pathsep)
        for raw_root in chunk.split(",")
        if (root := Path(os.path.expanduser(raw_root.strip()))).is_absolute())
    # ⤷ 出口 return：返回 [*map(Path, MEDIA_DELIVERY_SAFE_ROOTS), *_profile_cache_root…
    return [*map(Path, MEDIA_DELIVERY_SAFE_ROOTS), *_profile_cache_roots(),
            *_kanban_attachment_roots(), *operator_roots]


# ┌── 函数 def _media_delivery_recency_seconds()
# │  作用：Recency window (seconds) 用于 trusting fresh 文件; 0 = pure-白名单模式.
# │  参数：无
# │  返回：float
# └──────────────────────────────────────────────────────────────
def _media_delivery_recency_seconds() -> float:
    """Recency window (seconds) for trusting fresh files; 0 = pure-allowlist mode."""
    from gateway.media_policy import media_delivery_trust_recent, media_delivery_trust_recent_seconds
    if not media_delivery_trust_recent():
        # ⤷ 出口 return：返回 0.0
        return 0.0
    custom = media_delivery_trust_recent_seconds().strip()
    default = float(_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS)
    # ⤷ 出口 return：返回 _or_default(lambda: max(0.0, float(custom)) if custom else d…
    return _or_default(lambda: max(0.0, float(custom)) if custom else default, default)


# ┌── 函数 def _kanban_board_db_paths()
# │  作用：Named-board ``kanban.数据库`` stores (+ sidecars): 它们 sit beside ATTACHMENTS dir
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _kanban_board_db_paths() -> List[Path]:
    """Named-board ``kanban.db`` stores (+ sidecars): they sit beside the ATTACHMENTS dir
    ``_kanban_attachment_roots`` allowlists and hold every task, comment and run transcript."""
    # ⤷ 出口 return：返回 [board / name for board in _kanban_board_dirs() for name in …
    return [board / name for board in _kanban_board_dirs() for name in _sqlite_files("kanban.db")]


# ┌── 函数 def _media_delivery_denied_paths()
# │  作用：返回 absolute denylist 路径 under 其 delivery 是 never allowed.。
# │  参数：无
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _media_delivery_denied_paths() -> List[Path]:
    """Return absolute denylist paths under which delivery is never allowed."""
    home = Path(os.path.expanduser("~"))
    # ⤷ 出口 return：返回 [*map(Path, _MEDIA_DELIVERY_DENIED_PREFIXES), *(home / sub f…
    return [*map(Path, _MEDIA_DELIVERY_DENIED_PREFIXES),
            *(home / sub for sub in _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS),
            *(r / rel for r in _credential_home_roots() for rel in _ROOT_CREDENTIAL_PATHS),
            *_kanban_board_db_paths()]


# ┌── 函数 def _resolve_path()
# │  作用：``路径[.expanduser()].resolve(strict)`` 或 None 当它 fails (OSError / RuntimeError /
# │  参数：path: Path, strict: bool = ..., expand: bool = ...
# │  返回：Optional[Path]
# └──────────────────────────────────────────────────────────────
def _resolve_path(path: Path, *, strict: bool = False, expand: bool = False) -> Optional[Path]:
    """``path[.expanduser()].resolve(strict)`` or None when it fails (OSError / RuntimeError /
    ValueError — embedded NUL, symlink loop, undeterminable home, missing file under ``strict``)."""
    # ⤷ 关键分支 try/except：捕获 (OSError, RuntimeError, ValueError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 (path.expanduser() if expand else path).resolve(strict=stric…
        return (path.expanduser() if expand else path).resolve(strict=strict)
    except (OSError, RuntimeError, ValueError):
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _path_under_denied_prefix()
# │  作用：True 若 ``resolved`` lives under deny-listed system 路径 — except denied prefix 那个
# │  参数：resolved: Path
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _path_under_denied_prefix(resolved: Path) -> bool:
    """True if ``resolved`` lives under a deny-listed system path — except a denied prefix that
    IS the running user's own home: ``/root`` is listed so a non-root gateway can't deliver
    another user's home, but a root-run gateway's own deliverables live under ``$HOME=/root``.
    Credential sub-dirs (``~/.ssh``, ``~/.hermes/.env``) stay blocked (more-specific entries)."""
    home = _resolve_path(Path(os.path.expanduser("~")))
    for denied in _media_delivery_denied_paths():
        resolved_denied = _resolve_path(denied, expand=True)
        if resolved_denied is None:
            continue
        hit = resolved == resolved_denied or _path_is_within(resolved, resolved_denied)
        if hit and resolved_denied != home:
            # ⤷ 出口 return：返回 True
            return True
    # ⤷ 出口 return：返回 False
    return False


# ┌── 函数 def _file_is_recently_produced()
# │  作用：True 若 mtime 是 within ``window_seconds`` — 会话-scoped trust 信号: Agent
# │  参数：resolved: Path, window_seconds: float
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _file_is_recently_produced(resolved: Path, window_seconds: float) -> bool:
    """True if mtime is within ``window_seconds`` — a session-scoped trust signal: agents
    produce artifacts seconds before sending; pre-existing host files are days/months old."""
    if window_seconds <= 0:
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 time.time() - resolved.stat().st_mtime <= window_seconds
        return (time.time() - resolved.stat().st_mtime) <= window_seconds
    except OSError:
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _path_is_within()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：path: Path, root: Path
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _path_is_within(path: Path, root: Path) -> bool:
    with contextlib.suppress(ValueError):
        path.relative_to(root)
        # ⤷ 出口 return：返回 True
        return True
    # ⤷ 出口 return：返回 False
    return False


# ┌── 函数 def _tenv()
# │  作用：Scope-aware TERMINAL_* 读取: per-轮次 scope carries ACTIVE 配置档's settings 当
# │  参数：name: str, default: str = ''
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _tenv(name: str, default: str = "") -> str:
    """Scope-aware TERMINAL_* read: the per-turn scope carries the ACTIVE profile's settings while
    os.getenv reads whatever a prior turn pinned into the process env. Only ImportError falls
    back — a refusal scope must raise rather than rebuild another profile's policy from the env."""
    # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        # ⤷ 出口 return：返回 os.getenv(name, default)
        return os.getenv(name, default)
    # ⤷ 出口 return：返回 terminal_env(name, default)
    return terminal_env(name, default)


# ┌── 函数 def _parse_docker_volume_mounts()
# │  作用：解析 ``TERMINAL_DOCKER_VOLUMES`` (JSON 列表的 ``宿主:container[:模式]``) 进入
# │  参数：无
# │  返回：List[Tuple[Path, Path]]
# └──────────────────────────────────────────────────────────────
def _parse_docker_volume_mounts() -> List[Tuple[Path, Path]]:
    """Parse ``TERMINAL_DOCKER_VOLUMES`` (JSON list of ``host:container[:mode]``) into
    ``(host_path, container_path)``; named volumes / non-absolute hosts can't resolve here."""
    raw = _tenv("TERMINAL_DOCKER_VOLUMES", "").strip()
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        import json as _json
        parsed = _json.loads(raw) if raw else []
    except Exception:
        # ⤷ 出口 return：返回 []
        return []
    mounts: List[Tuple[Path, Path]] = []
    for entry in parsed if isinstance(parsed, list) else ():
        spec = entry.strip() if isinstance(entry, str) else ""
        # Prefer the first ':/' so absolute container paths are unambiguous.
        sep = spec.find(":/")
        if sep <= 0:
            continue
        container_raw = spec[sep + 1:].split(":", 1)[0]  # starts with /
        # Skip named volumes (no absolute/drive host path).
        host_expanded = os.path.expanduser(spec[:sep])
        if not (host_expanded.startswith("/") or (len(host_expanded) > 1 and host_expanded[1] == ":")):
            continue
        host_path, container_path = _resolve_path(Path(host_expanded)), Path(container_raw)
        if host_path is not None and container_path.is_absolute():
            mounts.append((host_path, container_path))
    # ⤷ 出口 return：返回 mounts
    return mounts


# ┌── 函数 def _docker_sandbox_dir_candidates()
# │  作用：Candidate 宿主 sandbox dir 名称用于 delivering 会话, best first. 镜像
# │  参数：session_key: str = ''
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def _docker_sandbox_dir_candidates(session_key: str = "") -> List[str]:
    """Candidate host sandbox dir names for the delivering session, best first. Mirrors
    ``_resolve_container_task_id`` (tools/terminal_tool.py): containers are PROFILE-scoped
    (``default``, else ``profile:<name>``); legacy ``session:<key>`` sandboxes stay as a fallback.
    The key is passed explicitly because delivery runs after the turn's contextvars were cleared.

    Takes the key explicitly because the delivery pipeline runs after ``_handle_message_with_agent`` cleared
    the turn's session contextvars (#93950) — an ambient lookup here would silently collapse onto
    ``default`` and miss the session's real sandbox.
    """
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.environments.path_utils import sanitize_task_id_for_path
    except Exception:
        # ⤷ 出口 return：返回 ['default']
        return ["default"]
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name() or "default"
    except Exception:
        profile = "default"
    candidates: List[str] = []
    # Explicit trusted-profiles opt-in: one shared container identity.
    if shared := _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip():
        candidates.append(sanitize_task_id_for_path(f"shared:{shared}"))
    if profile != "default":
        candidates.append(sanitize_task_id_for_path(f"profile:{profile}"))
    candidates.append("default")
    if session_key:  # bug-window legacy layout: per-session sandboxes
        candidates.append(sanitize_task_id_for_path(f"session:{session_key}"))
    # ⤷ 出口 return：返回 candidates
    return candidates


_TRUTHY = {"1", "true", "yes", "on"}


# ┌── 函数 def _docker_env_active()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _docker_env_active() -> bool:
    # ⤷ 出口 return：返回 _tenv('TERMINAL_ENV', '').strip().lower() == 'docker'
    return _tenv("TERMINAL_ENV", "").strip().lower() == "docker"


# ┌── 函数 def _docker_persistent_active()
# │  作用：Docker backend 带 persistent containers ( 默认值) enabled.
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _docker_persistent_active() -> bool:
    """Docker backend with persistent containers (the default) enabled."""
    # ⤷ 出口 return：返回 _docker_env_active() and _tenv('TERMINAL_CONTAINER_PERSISTEN…
    return _docker_env_active() and _tenv("TERMINAL_CONTAINER_PERSISTENT", "true").strip().lower() in _TRUTHY


# ┌── 函数 def _docker_persistent_sandbox_roots()
# │  作用：Existing ``<sandbox>/Docker/<candidate>/<leaf>`` 宿主 dirs 在 candidate order;
# │  参数：session_key: str, leaf: str
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _docker_persistent_sandbox_roots(session_key: str, leaf: str) -> List[Path]:
    """Existing ``<sandbox>/docker/<candidate>/<leaf>`` host dirs in candidate order;
    the translator tries each until the file resolves. Empty unless Docker + persistent."""
    if not _docker_persistent_active():
        # ⤷ 出口 return：返回 []
        return []
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.environments.base import get_sandbox_dir
        base = get_sandbox_dir() / "docker"
        # ⤷ 出口 return：返回 [cand for name in _docker_sandbox_dir_candidates(session_key…
        return [cand for name in _docker_sandbox_dir_candidates(session_key)
                if (cand := (base / name / leaf).resolve(strict=False)).is_dir()]
    except Exception:
        # ⤷ 出口 return：返回 []
        return []


# ┌── 函数 def _default_docker_workspace_host_roots()
# │  作用：Existing 宿主 candidates 用于 ``/workspace``: explicit cwd mount
# │  参数：session_key: str = ''
# │  返回：List[Path]
# └──────────────────────────────────────────────────────────────
def _default_docker_workspace_host_roots(session_key: str = "") -> List[Path]:
    """Existing host candidates for ``/workspace``: the explicit cwd mount
    (``TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE``) if set, else the persistent sandbox layouts."""
    if not _docker_persistent_active():
        # ⤷ 出口 return：返回 []
        return []
    if _tenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").strip().lower() in _TRUTHY:
        cwd = _tenv("TERMINAL_CWD") or os.getcwd()
        # ⤷ 关键分支 try/except：捕获 (OSError, RuntimeError, ValueError)，出错走兜底而不是崩溃。
        try:
            host = Path(os.path.expanduser(cwd)).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            # ⤷ 出口 return：返回 []
            return []
        # ⤷ 出口 return：返回 [host] if host.is_dir() else []
        return [host] if host.is_dir() else []
    # ⤷ 出口 return：返回 _docker_persistent_sandbox_roots(session_key, 'workspace')
    return _docker_persistent_sandbox_roots(session_key, "workspace")


# ┌── 函数 def _cache_dir_container_mounts()
# │  作用：(宿主, container) pairs 用于 auto-mounted Hermes 缓存 dirs (``/root/.hermes/...`` 在
# │  参数：无
# │  返回：List[Tuple[Path, Path]]
# └──────────────────────────────────────────────────────────────
def _cache_dir_container_mounts() -> List[Tuple[Path, Path]]:
    """(host, container) pairs for the auto-mounted Hermes cache dirs (``/root/.hermes/...`` in
    MEDIA tags); longer prefixes than the ``/root`` home mount, so longest-prefix match wins."""
    if not _docker_env_active():
        # ⤷ 出口 return：返回 []
        return []
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.credential_files import get_cache_directory_mounts
        # ⤷ 出口 return：返回 [(Path(m['host_path']), Path(m['container_path'])) for m in …
        return [(Path(m["host_path"]), Path(m["container_path"])) for m in get_cache_directory_mounts()]
    except Exception:
        # ⤷ 出口 return：返回 []
        return []


# ┌── 函数 def _warn_unresolved_docker_media()
# │  作用：名称为何 container-absolute MEDIA 路径 failed translation (otherwise 仅信号是
# │  参数：candidate: Path, session_key: str, reason: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _warn_unresolved_docker_media(candidate: Path, session_key: str, reason: str) -> None:
    """Name WHY a container-absolute MEDIA path failed translation (otherwise the only signal is
    the generic "Skipping unsafe MEDIA directive path" line). Docker-only; host rejections quiet.

    See #93950.
    """
    if not _docker_env_active():
        return
    logger.warning("Docker MEDIA path %s did not resolve to a host sandbox file (%s%s); "
                   "the producing container's sandbox directory may not exist yet or "
                   "was pruned", _log_safe_path(str(candidate)), reason,
                   f", session_key={session_key}" if session_key else "")


# ┌── 函数 def _translate_docker_container_media_path()
# │  作用：Container-absolute 路径 -> 宿主路径 via longest-prefix match over ``docker_volumes``,
# │  参数：candidate: Path, session_key: str = ''
# │  返回：Optional[Path]
# └──────────────────────────────────────────────────────────────
def _translate_docker_container_media_path(candidate: Path, session_key: str = "") -> Optional[Path]:
    """Container-absolute path -> host path via longest-prefix match over ``docker_volumes``, the
    auto-mounted cache dirs (``/root/.hermes/...``), persistent ``/workspace`` and ``/root``."""
    if not candidate.is_absolute():
        # ⤷ 出口 return：返回 None
        return None
    # In-process gateways (Desktop, `hermes serve`) may not have bridged terminal.* config into
    # TERMINAL_* env yet; the bridge is idempotent.
    with contextlib.suppress(Exception):
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
    mounts = [*_parse_docker_volume_mounts(), *_cache_dir_container_mounts()]
    mounted = {c.as_posix() for _, c in mounts}
    # Synthetic /workspace mounts: profile-scoped layout first, then legacy per-session.
    if "/workspace" not in mounted:
        mounts.extend((root, Path("/workspace")) for root in _default_docker_workspace_host_roots(session_key))
    # Synthetic /root mounts catch stray home writes (/root/out.png; cache mounts are longer
    # prefixes). /root/.hermes/* that missed a cache mount is the container's credential surface —
    # translating it via the home mount would dodge the host denylist.
    if "/root" not in mounted and not candidate.as_posix().startswith("/root/.hermes"):
        mounts.extend(
            (root, Path("/root")) for root in _docker_persistent_sandbox_roots(session_key, "home"))
    if not mounts:
        _warn_unresolved_docker_media(candidate, session_key, "no sandbox mounts resolved")
        # ⤷ 出口 return：返回 None
        return None
    # Longest container-prefix match; equal-length prefixes are tried in insertion order.
    candidate_posix = candidate.as_posix()
    matched = [(host_root, container_root, len(prefix)) for host_root, container_root in mounts
               for prefix in (container_root.as_posix().rstrip("/") or "/",)
               if candidate_posix == prefix or candidate_posix.startswith(prefix + "/")]
    if not matched:
        _warn_unresolved_docker_media(candidate, session_key, "no mounted prefix matches")
        # ⤷ 出口 return：返回 None
        return None
    for host_root, container_root, _score in sorted(matched, key=lambda m: -m[2]):
        translated = _resolve_path(host_root / candidate.relative_to(container_root), strict=True)
        if translated is not None and (
                translated == host_root or _path_is_within(translated, host_root)):
            # ⤷ 出口 return：返回 translated
            return translated
    _warn_unresolved_docker_media(candidate, session_key, "host file missing from sandbox")
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def validate_media_delivery_path()
# │  作用：Safe absolute 文件路径用于 native media delivery, 否则 None. 默认值: 任意 existing
# │  参数：path: str, session_key: str = ''
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def validate_media_delivery_path(path: str, session_key: str = "") -> Optional[str]:
    """Safe absolute file path for native media delivery, else None. Default: any existing
    regular file outside the credential / system denylist (symmetric with inbound). Strict
    (``HERMES_MEDIA_DELIVERY_STRICT=1``, public bots where prompt injection must not exfiltrate
    host secrets): MUST be under a Hermes cache, an operator root (``HERMES_MEDIA_ALLOW_DIRS``),
    or freshly produced within the recency window. Symlinks are resolved before any check."""
    candidate = _normalize_media_tag_path(path)
    if not candidate:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 (OSError, RuntimeError, ValueError)，出错走兜底而不是崩溃。
    try:
        expanded = Path(os.path.expanduser(candidate))
    except (OSError, RuntimeError, ValueError):
        # expanduser raises ValueError("embedded null byte") for a ~\x00 path.
        return None
    if not expanded.is_absolute():
        # ⤷ 出口 return：返回 None
        return None
    # Docker agents emit MEDIA:/workspace/... — map container paths to host paths first.
    resolved = _translate_docker_container_media_path(expanded, session_key=session_key)
    if resolved is None:
        resolved = _resolve_path(expanded, strict=True)
    if resolved is None or not resolved.is_file():
        # ⤷ 出口 return：返回 None
        return None
    # Cache / operator allowlist is trusted unconditionally, regardless of mode.
    for root in _media_delivery_allowed_roots():
        resolved_root = _resolve_path(root, expand=True)
        if resolved_root is not None and _path_is_within(resolved, resolved_root):
            return str(resolved)
    # Non-strict (default): anything not denylisted (/etc, /proc, ~/.ssh, Hermes-root secrets).
    from gateway.media_policy import media_delivery_strict
    if not media_delivery_strict():
        # ⤷ 出口 return：返回 None if _path_under_denied_prefix(resolved) else str(resolve…
        return None if _path_under_denied_prefix(resolved) else str(resolved)
    # Strict: recency trust for fresh files (pandoc -o /tmp/x.pdf); denylist still applies.
    window = _media_delivery_recency_seconds()
    if (window > 0 and not _path_under_denied_prefix(resolved)
            and _file_is_recently_produced(resolved, window)):
        return str(resolved)
    # ⤷ 出口 return：返回 None
    return None


# Control chars + Unicode line separators (NEL, LS, PS) that log aggregators treat as breaks: a
# model-emitted path must not forge a log line.
_LOG_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")


# ┌── 函数 def _log_safe_path()
# │  作用：返回 single-line, length-bounded 路径用于日志输出.。
# │  参数：path: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _log_safe_path(path: str) -> str:
    """Return a single-line, length-bounded path for log output."""
    # ⤷ 出口 return：返回 _LOG_UNSAFE_CHARS.sub('?', str(path))[:200]
    return _LOG_UNSAFE_CHARS.sub("?", str(path))[:200]


# ┌── 函数 def _validated_delivery_path()
# │  作用：``validate_media_delivery_path`` plus shared "Skipping unsafe ..." 警告. 路径
# │  参数：raw_path, session_key: str, label: str
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _validated_delivery_path(raw_path, session_key: str, label: str) -> Optional[str]:
    """``validate_media_delivery_path`` plus the shared "Skipping unsafe ..." warning. A path the
    host cannot see is retried against the active remote sandbox (ssh/modal/...; #466)."""
    raw = str(raw_path)
    safe_path = validate_media_delivery_path(raw, session_key=session_key)
    if not safe_path:
        from gateway.media_fetch import fetch_remote_media
        safe_path = fetch_remote_media(raw)
    if not safe_path:
        # Say WHY: a path that does not exist on the host is the common case (a model hallucinated or
        # a sandbox path failed to translate) and is not a security rejection.
        reason = "not found on this host" if not _existing_regular_file(raw) else "denied by the delivery policy"
        logger.warning("Skipping %s (%s): %s", label, reason, _log_safe_path(raw))
    # ⤷ 出口 return：返回 safe_path
    return safe_path


# ┌── 函数 def _existing_regular_file()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：raw: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _existing_regular_file(raw: str) -> bool:
    # ⤷ 关键分支 try/except：捕获 (OSError, RuntimeError, ValueError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 Path(os.path.expanduser(raw)).is_file()
        return Path(os.path.expanduser(raw)).is_file()
    except (OSError, RuntimeError, ValueError):
        # ⤷ 出口 return：返回 False
        return False


SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv",
    ".log": "text/plain", ".json": "application/json", ".xml": "application/xml",
    ".yaml": "application/yaml", ".yml": "application/yaml", ".toml": "application/toml",
    ".ini": "text/plain", ".cfg": "text/plain", ".zip": "application/zip",
    ".doc": "application/msword", ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ts": "text/plain", ".py": "text/plain", ".sh": "text/plain"}

# Files safe to inline into the prompt when small. An extension gate, NOT a blind UTF-8 decode
# (PDF/zip/docx can start with decodable ASCII). Non-members are still cached by path.
_TEXT_INJECT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".jsonl", ".ndjson", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs",
    ".java", ".kt", ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".swift", ".m",
    ".scala", ".clj", ".ex", ".exs", ".erl", ".sql", ".graphql", ".proto", ".tf", ".hcl",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".rst", ".tex", ".srt", ".vtt", ".diff",
    ".patch"}

# Image exts platforms may deliver as "documents" (file-picker uploads); routed to the image cache.
SUPPORTED_IMAGE_DOCUMENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif"}

# Media-delivery ext allowlist — SINGLE SOURCE OF TRUTH for both extractors and the cleanup
# regexes: a tag is stripped only when deliverable, unknown-ext paths survive in the body.
# Both extractors that turn response text into native attachments derive their extension set from this
# tuple: * ``extract_media()``       — explicit ``MEDIA:<path>`` tags * ``extract_local_files()`` — bare
# absolute/home paths the agent mentions Historically these two carried independently-maintained extension
# lists. ``extract_media`` had a narrow list (no .md/.json/.yaml/.xml/.html/...) while
# ``extract_local_files`` had a broad one. Combined with the unconditional ``MEDIA:\\s*\\S+`` cleanup at the
# dispatch sites, that mismatch created a silent black hole: a ``MEDIA:/report.md`` tag failed the narrow
# extract_media match, got stripped from the body by the loose cleanup regex, and was then invisible to
# extract_local_files — the file was never delivered (issue #34517). Keeping one list eliminates the drift;
# building the cleanup regexes from the same set means a tag is only stripped when its extension is one we
# can actually deliver, so an unknown-extension path survives in the body instead of vanishing. Covers
# images (inline), video (inline where supported), audio (voice/audio), documents/spreadsheets/presentations
# (send_document), archives, and rendered web output. The dispatch partition (image vs video vs document)
# lives in ``gateway/run.py``. ---------------------------------------------------------------------------
MEDIA_DELIVERY_EXTS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",  # images (embed inline)
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp",  # video (embed inline where supported)
    ".mp3", ".m2a", ".wav", ".ogg", ".opus", ".m4a", ".flac",  # audio (voice/audio where supported)
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".epub",  # documents (file attachments)
    ".xlsx", ".xls", ".ods", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # spreadsheets/data
    ".kmz", ".kml", ".geojson", ".gpx",  # geospatial / GIS
    ".pptx", ".ppt", ".odp", ".key",  # presentations
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".apk", ".ipa",  # archives
    ".html", ".htm")  # web / rendered output

# Bare extensions (no dot) longest-first so a shorter ext never matches as a prefix of a longer one.
_MEDIA_EXT_ALTERNATION = "|".join(sorted((e.lstrip(".") for e in MEDIA_DELIVERY_EXTS), key=len, reverse=True))

# Anchored ``MEDIA:<path>`` cleanup pattern (dispatch path + streaming consumer). Strips only tags
# whose path ends in a deliverable extension (optionally quoted/backticked); unknown-ext tags stay
# for the bare-path detector. Rules: anchors ``~/``, ``/``, ``X:\`` / ``X:/``; up to 3 quote/
# emphasis markers each side (code/blockquote contexts are masked earlier); non-greedy with
# ``MEDIA:`` as a boundary so glued tags (``MEDIA:/a.pngMEDIA:/b.png``) never merge; sentence-final
# ``.`` is a boundary only before whitespace/EOL (``data.csv.`` -> ``data.csv``, ``archive.tar.gz``
# extends past ``.tar``); CJK full-width punctuation terminates paths (``早报.pdf（782.6 KB）``).
# A ``MEDIA:`` tag with an unknown extension is left in the text so it can still be picked up by the
# bare-path detector (extract_local_files) downstream rather than silently deleted. Path anchors: ``~/``
# (Unix home-relative), ``/`` (Unix absolute), ``X:\\`` or ``X:/`` (Windows drive-letter absolute — #34632).
# Emphasis tolerance: models routinely wrap the tag in Markdown emphasis (``**MEDIA:/x.pdf**``,
# ``*MEDIA:/x.pdf*``, ``_MEDIA:/x.pdf_``) when they present a file to the user. The old single-quote anchor
# (``[`"']?``) and the closing lookahead (which lacked ``*``/``_``) failed to match such tags, so the file
# was silently never delivered and the literal ``MEDIA:`` text leaked into the chat. Allow a short run of
# emphasis/quote markers on both sides so the tag is recognised regardless of cosmetic Markdown. Code-block
# / inline-code / blockquote contexts are still neutralised earlier by ``_mask_protected_spans`` (#35695),
# so example tags remain non-deliverable. The trailing lookahead also accepts ``MEDIA:`` as a boundary, so
# the next tag stops the current match cleanly (#68773). The whitespace guard keeps multi-part extensions
# intact — for ``archive.tar.gz`` the ``.`` after ``tar`` is followed by ``g``, so the match must extend to
# ``.gz`` instead of stopping early at ``.tar``. CJK full-width punctuation accepted as MEDIA path
# terminators, mirroring the ASCII set in the looka below. Chinese-language agent output naturally writes
# ``MEDIA:D:\path\早报.pdf（782.6 KB）`` or ``MEDIA:...pdf：内容`` — without these, the lookahead fails and the
# attachment is silently dropped (#88038).
_MEDIA_CJK_TERMINATORS = "（）〈〉《》：，。；！？、\u201c\u201d\u2018\u2019【】"

MEDIA_TAG_CLEANUP_RE = re.compile(
    r'''[`"'*_]{0,3}MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+?`|"[^"\n]+?"|'[^'\n]+?'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])\S+?(?:[^\S\n]+\S+?)*?\.(?:''' + _MEDIA_EXT_ALTERNATION + r'''))'''
    r'''(?=[\s`"'*_,;:)\]}\[''' + _MEDIA_CJK_TERMINATORS + r''']|MEDIA:|\.(?:\s|$)|$)[`"'*_]{0,3}\.?''',
    re.IGNORECASE)

# Extension-less (Caddyfile) / unknown-ext (.py, .log) tags deliver only after
# ``validate_media_delivery_path`` accepts them, so injected paths that don't validate stay visible.
# The bare path class is whitespace-bounded (a tag glued to the next ``MEDIA:`` or prose must not
# absorb it); spaced paths are recovered by ``_match_extensionless_path`` with on-disk validation.
# Paths NOT covered by MEDIA_TAG_CLEANUP_RE's extension alternation — both extension-less files (Caddyfile,
# Dockerfile, Makefile) and files with an unknown extension (.py, .log, .weirdext, ...) — are validated and
# delivered via MEDIA_EXTENSIONLESS_TAG_RE. Every ``MEDIA:`` path is therefore deliverable regardless of
# file type (#36060): known extensions extract unconditionally via the anchored pattern above, everything
# else extracts only after ``validate_media_delivery_path`` accepts it (exists on disk, not under the
# credential/system denylist, strict-mode rules honored), so prompt-injection paths that do not validate are
# left visible instead of silently dropped. The path class uses a tempered-greedy token (``[^\s\n`"']+?``
# followed by a ``(?=...)`` lookahead) instead of the prior ``[^\s\n`"']+`` so a tag glued to the next
# ``MEDIA:`` keyword (``MEDIA:/a.pngMEDIA:/b.png``) or to arbitrary following text (``MEDIA:/a.pngSome
# text``) cannot silently absorb the next path — that earlier behavior merged the two paths into one invalid
# string and dropped the file (#68773). The bare form stays non-greedy and whitespace-bounded — spaced paths
# are NOT absorbed at the regex level, because greedy space-tolerance would reintroduce the #68773 bug class
# (gluing the next MEDIA: tag or trailing prose into one invalid path). Instead, unknown-extension paths
# containing spaces (``MEDIA:/data/map data.kmz``, ``C:\...\My Documents\x.log``) are recovered by
# ``_match_extensionless_path`` (#24032): when the bare match fails validation, the candidate is
# progressively extended forward across single spaces — bounded, stopping at newline / the next ``MEDIA:``
# keyword — and the first extension that validates on disk wins.
MEDIA_EXTENSIONLESS_TAG_RE = re.compile(
    r'''[`"'*_]{0,3}MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])[^\s\n`"']+?)'''
    r'''(?=[`"'\s,;:)\]}''' + _MEDIA_CJK_TERMINATORS + r''']|MEDIA:|$)'''
    r'''[`"'*_]{0,3}\s*''',
    re.IGNORECASE)


# ┌── 函数 def _match_extensionless_path()
# │  作用：Extensionless MEDIA tag match -> 已校验在…上-disk ``(safe_path, end_offset)`` 或 None:
# │  参数：scan_text: str, match: 're.Match'
# │  返回：Optional[Tuple[str, int]]
# └──────────────────────────────────────────────────────────────
def _match_extensionless_path(scan_text: str, match: "re.Match") -> Optional[Tuple[str, int]]:
    """Extensionless MEDIA tag match -> validated on-disk ``(safe_path, end_offset)`` or None: the
    captured path first, then extended across single spaces (max 8 tokens, never past a newline
    or the next ``MEDIA:``).

    When that fails validation, the candidate is progressively extended forward across single spaces
    (validation-gated, bounded at 8 tokens, never past a newline or a subsequent ``MEDIA:`` keyword) so
    unknown-extension paths containing spaces deliver (#24032). Returns ``(safe_path, end_offset)`` where
    ``end_offset`` is the index in ``scan_text`` just past the matched path, or ``None`` when nothing
    validates.
    """
    path = _normalize_media_tag_path(match.group("path"))
    if not path:
        # ⤷ 出口 return：返回 None
        return None
    safe = validate_media_delivery_path(path)
    if safe:
        # ⤷ 出口 return：返回 (safe, match.end('path'))
        return safe, match.end("path")
    start = match.start("path")
    segment = scan_text[start:].split("\n", 1)[0]
    nxt = segment.find("MEDIA:", 1)
    if nxt != -1:
        segment = segment[:nxt]
    pos = match.end("path") - start
    for _ in range(8):
        token = re.match(r"[ \t]*[^ \t]+", segment[pos:])
        if not token:
            break
        tok_end = pos + token.end()
        safe = validate_media_delivery_path(_normalize_media_tag_path(segment[:tok_end]))
        if safe:
            # ⤷ 出口 return：返回 (safe, start + tok_end)
            return safe, start + tok_end
        pos = tok_end
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _normalize_media_tag_path()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：raw: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    # ⤷ 出口 return：返回 path.lstrip('`"\'').rstrip('`"\',.;:)}]')
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


# ┌── 函数 def _path_lacks_deliverable_extension()
# │  作用：当 ``路径`` 有无 extension 或 one outside MEDIA_DELIVERY_EXTS — such 路径 时返回 True。
# │  参数：path: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _path_lacks_deliverable_extension(path: str) -> bool:
    """True when ``path`` has no extension or one outside MEDIA_DELIVERY_EXTS — such paths
    take the validated delivery pass so nonexistent / denylisted ones stay visible.

    ``path`` — either the basename has no extension at all (Caddyfile, Makefile, …) or the extension is not
    in MEDIA_DELIVERY_EXTS (.py, .log, .weirdext, …). Such paths route through the validated delivery pass
    (``validate_media_delivery_path``) instead of the unconditional one, so every file type is deliverable
    (#36060) while nonexistent / denylisted paths stay visible in the text.
    """
    # ⤷ 出口 return：返回 Path(path).suffix.lower() not in MEDIA_DELIVERY_EXTS
    return Path(path).suffix.lower() not in MEDIA_DELIVERY_EXTS


# ┌── 函数 def _has_media_directives()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：text: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _has_media_directives(text: str) -> bool:
    # ⤷ 出口 return：返回 'MEDIA:' in text or '[[audio_as_voice]]' in text or '[[as_do…
    return "MEDIA:" in text or "[[audio_as_voice]]" in text or "[[as_document]]" in text


# ┌── 函数 def _mask_media_scan_text()
# │  作用：Offset-preserving mask 的 protected spans (code, quotes, JSON 字符串值).
# │  参数：text: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _mask_media_scan_text(text: str) -> str:
    """Offset-preserving mask of protected spans (code, quotes, JSON string values).
    BasePlatformAdapter is defined later in this module; resolved at call time."""
    A = BasePlatformAdapter
    # ⤷ 出口 return：返回 A._mask_json_string_media(A._mask_protected_spans(text))
    return A._mask_json_string_media(A._mask_protected_spans(text))


# ┌── 函数 def _extensionless_media_matches()
# │  作用：Yield ``(match, safe_path, end_offset)`` 用于每个 extension-less / unknown-extension
# │  参数：masked: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _extensionless_media_matches(masked: str):
    """Yield ``(match, safe_path, end_offset)`` for every extension-less / unknown-extension
    MEDIA tag in ``masked`` that ``validate_media_delivery_path`` accepts."""
    for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(masked):
        path = _normalize_media_tag_path(match.group("path"))
        if path and _path_lacks_deliverable_extension(path):
            resolved = _match_extensionless_path(masked, match)
            if resolved is not None:
                # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
                yield match, resolved[0], resolved[1]


# ┌── 函数 def _real_media_tag_spans()
# │  作用：(start, end) spans 的 deliverable MEDIA tags 在…上 masked 复制: known-extension tags
# │  参数：masked: str
# │  返回：list
# └──────────────────────────────────────────────────────────────
def _real_media_tag_spans(masked: str) -> list:
    """(start, end) spans of deliverable MEDIA tags on a masked copy: known-extension tags
    unconditionally, extension-less / unknown ones only if validate_media_delivery_path accepts."""
    spans: list = [m.span() for m in MEDIA_TAG_CLEANUP_RE.finditer(masked)]
    spans.extend((match.start(), end) for match, _, end in _extensionless_media_matches(masked))
    # ⤷ 出口 return：返回 spans
    return spans


_FENCED_CODE_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`[^`\n]+`')


# ┌── 函数 def _code_spans()
# │  作用：(start, end) spans 的 fenced code blocks 且 inline code 在 ``内容``.
# │  参数：content: str
# │  返回：list
# └──────────────────────────────────────────────────────────────
def _code_spans(content: str) -> list:
    """(start, end) spans of fenced code blocks and inline code in ``content``."""
    # ⤷ 出口 return：返回 [m.span() for rx in (_FENCED_CODE_RE, _INLINE_CODE_RE) for m…
    return [m.span() for rx in (_FENCED_CODE_RE, _INLINE_CODE_RE) for m in rx.finditer(content)]


# ┌── 函数 def _blank_spans()
# │  作用：Replace 每个 non-newline char inside ``spans`` 带 space (offsets preserved).
# │  参数：text: str, spans: list
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _blank_spans(text: str, spans: list) -> str:
    """Replace every non-newline char inside ``spans`` with a space (offsets preserved)."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = [c if c == '\n' else ' ' for c in chars[start:end]]
    # ⤷ 出口 return：返回 ''.join(chars)
    return ''.join(chars)


# ┌── 函数 def _delete_spans()
# │  作用：Delete ``spans`` 从 ``文本``, merging overlapping/nested ones first 从而 multi-pattern
# │  参数：text: str, spans: list
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _delete_spans(text: str, spans: list) -> str:
    """Delete ``spans`` from ``text``, merging overlapping/nested ones first so multi-pattern
    matches over the same tag never double-delete adjacent text."""
    if not spans:
        # ⤷ 出口 return：返回 text
        return text
    merged: list = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    chars = list(text)
    for start, end in reversed(merged):
        del chars[start:end]
    # ⤷ 出口 return：返回 ''.join(chars)
    return "".join(chars)


# ┌── 函数 def _strip_media_tag_directives()
# │  作用：移除 MEDIA: tags 且 [[audio_as_voice]] / [[as_document]] markers 从而它们 never render
# │  参数：text: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _strip_media_tag_directives(text: str) -> str:
    """Remove MEDIA: tags and [[audio_as_voice]] / [[as_document]] markers so they never render
    as text (backstop after ``extract_media``). Protected spans are mask-located only — tags
    inside them are neither stripped nor mangled, matching ``extract_media`` so display and
    delivery agree. Empty/None text is returned as-is.

    See #16434.
    """
    if not text or not _has_media_directives(text):
        # ⤷ 出口 return：返回 text
        return text
    cleaned = text.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
    # ⤷ 出口 return：返回 _delete_spans(cleaned, _real_media_tag_spans(_mask_media_sca…
    return _delete_spans(cleaned, _real_media_tag_spans(_mask_media_scan_text(cleaned)))


# ┌── 函数 def cache_document_from_bytes()
# │  作用：Save raw document bytes 缓存作为 ``doc_{uuid12}_{original_name}`` 且返回
# │  参数：data: bytes, filename: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """Save raw document bytes to the cache as ``doc_{uuid12}_{original_name}`` and return
    the absolute path; raises ValueError if the sanitized path escapes the cache directory."""
    cache_dir = get_document_cache_dir()
    # Sanitize: strip directory components, null bytes, and control characters
    safe_name = (Path(filename).name if filename else "document").replace("\x00", "").strip()
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "document"
    filepath = cache_dir / f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    # Final safety check: ensure path stays inside cache dir
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Path traversal rejected: {filename!r}")
    filepath.write_bytes(data)
    # ⤷ 出口 return：返回 str(filepath)
    return str(filepath)


# ┌── 异步函数（协程） def cache_document_from_bytes_async()
# │  作用：缓存 document bytes 不 blocking caller's 事件循环.
# │  参数：data: bytes, filename: str
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_document_from_bytes_async(data: bytes, filename: str) -> str:
    """Cache document bytes without blocking the caller's event loop."""
    # ⤷ 出口 return：返回 await asyncio.to_thread(cache_document_from_bytes, data, fil…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(cache_document_from_bytes, data, filename)


# Unified media caching: classify attachment bytes by ext/MIME, route to cache_*_from_bytes.
# ┌── 类 class CachedMedia
# │  说明：结果的 caching one attachment's bytes.
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class CachedMedia:
    """Result of caching one attachment's bytes."""
    path: str                 # absolute cache path, agent-visible (sandbox-translated)
    media_type: str           # MIME type recorded on the MessageEvent
    kind: str                 # "image" | "video" | "audio" | "document"
    display_name: str         # human-readable name for transcript notes

    # ┌── 函数 def context_note()
    # │  作用：One-line transcript annotation pointing Agent 在文件.
    # │  参数：self
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def context_note(self) -> str:
        """One-line transcript annotation pointing the agent at the file."""
        # ⤷ 出口 return：返回 f"[{self.kind} '{self.display_name}' saved at: {self.path}]"
        return f"[{self.kind} '{self.display_name}' saved at: {self.path}]"


# MIME -> extension reverse lookup; FIRST match across image, video, document tables wins
# (built in reverse so the earliest extension is the one that survives).
_MIME_TO_EXT: Dict[str, str] = {
    mime: ext
    for table in (SUPPORTED_DOCUMENT_TYPES, SUPPORTED_VIDEO_TYPES, SUPPORTED_IMAGE_DOCUMENT_TYPES)
    for ext, mime in reversed(table.items())}


# ┌── 函数 def _resolve_media_ext()
# │  作用：尽最大努力文件 extension 从 filename, 然后 MIME 兜底.
# │  参数：filename: str, mime_type: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _resolve_media_ext(filename: str, mime_type: str) -> str:
    """Best-effort file extension from filename, then MIME fallback."""
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    # ⤷ 出口 return：返回 ext or _MIME_TO_EXT.get((mime_type or '').lower(), '')
    return ext or _MIME_TO_EXT.get((mime_type or "").lower(), "")


# ┌── 函数 def cache_media_bytes()
# │  作用：Classify 且缓存 raw attachment bytes -> CachedMedia (None 仅用于 images failing
# │  参数：data: bytes, filename: str = ..., mime_type: str = ..., default_kind: Optional[str] = ...
# │  返回：Optional[CachedMedia]
# └──────────────────────────────────────────────────────────────
def cache_media_bytes(data: bytes, *, filename: str = "", mime_type: str = "",
                      default_kind: Optional[str] = None) -> Optional[CachedMedia]:
    """Classify and cache raw attachment bytes -> CachedMedia (None only for images failing
    validation). ``default_kind`` biases ambiguous ext/MIME (Telegram native photo, no name);
    anything not image/video/audio is a document."""
    from tools.credential_files import to_agent_visible_cache_path
    ext = _resolve_media_ext(filename, mime_type)
    mime = (mime_type or "").lower()
    display = re.sub(r"[^\w.\- ]", "_", filename) if filename else (ext.lstrip(".") or "file")
    # (kind, ext->mime table, default ext, cache fn, whether a matching caller MIME passes through)
    for kind, table, default_ext, cache_fn, passthrough in (
            ("image", SUPPORTED_IMAGE_DOCUMENT_TYPES, ".jpg", cache_image_from_bytes, True),
            ("video", SUPPORTED_VIDEO_TYPES, ".mp4", cache_video_from_bytes, False),
            ("audio", _AUDIO_MIME_TYPES, ".ogg", cache_audio_from_bytes, True)):
        if not (mime.startswith(f"{kind}/") or ext in table or default_kind == kind):
            continue
        kind_ext = ext if ext in table else default_ext
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            path = cache_fn(data, ext=kind_ext)
        except ValueError:
            if kind != "image":
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise
            # ⤷ 出口 return：返回 None
            return None
        out_mime = mime if passthrough and mime.startswith(f"{kind}/") else table[kind_ext]
        # ⤷ 出口 return：返回 CachedMedia(to_agent_visible_cache_path(path), out_mime, kin…
        return CachedMedia(to_agent_visible_cache_path(path), out_mime, kind, display)
    # Any other type is cached and surfaced as a path — an authorized user's uploads must never be
    # silently dropped. Unknown types get octet-stream so the agent reaches for terminal tools.
    fallback_name = filename or (f"document{ext}" if ext else "document.bin")
    path = cache_document_from_bytes(data, fallback_name)
    out_mime = SUPPORTED_DOCUMENT_TYPES.get(ext) or mime or "application/octet-stream"
    # ⤷ 出口 return：返回 CachedMedia(to_agent_visible_cache_path(path), out_mime, 'do…
    return CachedMedia(to_agent_visible_cache_path(path), out_mime, "document", display or fallback_name)


# ┌── 异步函数（协程） def cache_media_bytes_async()
# │  作用：Classify 且缓存 attachment bytes 不 blocking 事件循环.
# │  参数：data: bytes, filename: str = ..., mime_type: str = ..., default_kind: Optional[str] = ...
# │  返回：Optional[CachedMedia]
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def cache_media_bytes_async(
    data: bytes,
    *,
    filename: str = "",
    mime_type: str = "",
    default_kind: Optional[str] = None,
) -> Optional[CachedMedia]:
    """Classify and cache attachment bytes without blocking the event loop."""
    # ⤷ 出口 return：返回 await asyncio.to_thread(cache_media_bytes, data, filename=fi…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(
        cache_media_bytes,
        data,
        filename=filename,
        mime_type=mime_type,
        default_kind=default_kind,
    )


# ┌── 类 class TextDebounceState
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class TextDebounceState:
    event: MessageEvent
    task: asyncio.Task | None
    first_ts: float
    last_ts: float

    # ┌── 函数 def cancel_timer()
    # │  作用：Cancel pending 冲刷 timer (若当前活跃的且 不 ``unless``).
    # │  参数：self, unless: 'asyncio.Task | None' = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def cancel_timer(self, *, unless: "asyncio.Task | None" = None) -> None:
        """Cancel the pending flush timer (if live and not ``unless``)."""
        if self.task is not None and self.task is not unless and not self.task.done():
            self.task.cancel()


# ┌── 函数 def _append_text()
# │  作用：``existing\nnew`` 当 both non-empty; non-empty one otherwise.
# │  参数：existing: Optional[str], new: Optional[str]
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _append_text(existing: Optional[str], new: Optional[str]) -> str:
    """``existing\\nnew`` when both non-empty; the non-empty one otherwise."""
    # ⤷ 出口 return：返回 f'{existing}\n{new}' if existing else new
    return f"{existing}\n{new}" if existing else new


# ┌── 类 class _ExtractedResponse
# │  说明：Deliverable parts 的 handler 响应 (see ``_extract_response_content``).
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class _ExtractedResponse:
    """Deliverable parts of a handler response (see ``_extract_response_content``)."""
    text_content: str
    images: list
    media_files: list
    local_files: list
    force_document_attachments: bool
    pre_extract: str


_PLAINTEXT_GATEWAY_RESTART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+hermes[.!?\s]*$", re.IGNORECASE))


# ┌── 函数 def coerce_plaintext_gateway_command()
# │  作用：Rewrite tiny 设置的 DM plaintext admin phrases (exact matches 仅) 进入 slash 命令从而
# │  参数：event: 'MessageEvent'
# │  返回：None
# └──────────────────────────────────────────────────────────────
def coerce_plaintext_gateway_command(event: "MessageEvent") -> None:
    """Rewrite a tiny set of DM plaintext admin phrases (exact matches only) into slash commands so
    ``restart gateway`` never reaches the LLM/tool path (a self-restart from inside the running
    agent leaves the gateway stuck in ``draining`` waiting on that agent)."""
    with contextlib.suppress(Exception):
        if event is None or event.message_type != MessageType.TEXT:
            return
        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return
        if getattr(getattr(event, "source", None), "chat_type", None) != "dm":
            return
        if any(pattern.match(text) for pattern in _PLAINTEXT_GATEWAY_RESTART_PATTERNS):
            event.text = "/restart"


# ┌── 类 class ExecApprovalPrompt
# │  说明：One exec-approval 提示词, ready 用于平台 render natively (see
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class ExecApprovalPrompt:
    """One exec-approval prompt, ready for a platform to render natively (see
    ``BasePlatformAdapter.send_exec_approval``). ``actions`` rows are ``(label, choice, style)``
    with ``choice`` in ``once`` / ``session`` / ``always`` / ``deny`` — the vocabulary
    ``tools.approval.resolve_gateway_approval`` accepts — and ``style`` in ``primary`` /
    ``danger`` / ``""``."""
    chat_id: str
    session_key: str
    text: str
    actions: List[Tuple[str, str, str]]
    command: str
    description: str
    smart_denied: bool
    metadata: Optional[Dict[str, Any]] = None

    # ┌── 函数 def choices()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：List[str]
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def choices(self) -> List[str]:
        # ⤷ 出口 return：返回 [choice for _, choice, _ in self.actions]
        return [choice for _, choice, _ in self.actions]


# ┌── 类 class SendResult
# │  说明：结果的 sending 消息.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    # Adapter-specific metadata. Contract: Telegram edit-overflow partials set
    # raw_response["partial_overflow"] so the stream consumer sends the missing tail.
    raw_response: Any = None
    retryable: bool = False  # transient connection error — base retries automatically
    retry_after: Optional[float] = None  # server-requested delay (Telegram FloodWait) beats our backoff
    # Extra ids (send order) when a payload was split; ``message_id`` is then the LAST id so
    # later edits target the newest chunk.
    continuation_message_ids: tuple = ()
    # SEND_ERROR_KINDS member (failures only) via :func:`classify_send_error`, so consumers
    # branch without substring-matching ``error``.
    error_kind: Optional[str] = None


# Longest server ``retry_after`` ``_send_with_retry`` will sleep inline. Longer penalties return the
# typed failure so the delivery ledger owns the wait (#91969: a 97-minute FloodWait slept verbatim
# pinned the send coroutine and froze inbound on every platform).
_SEND_RETRY_INLINE_WAIT_CAP_SECS = 60.0

# Platform-neutral send-failure kinds for ``SendResult.error_kind``: too_long (size cap),
# bad_format (markup rejected; plain-text retry fixes), forbidden (the bot CANNOT reach the user),
# not_found (chat/thread/message gone), rate_limited, transient (connection-level, retry-safe),
# unknown.
SEND_ERROR_KINDS = frozenset(
    {"too_long", "bad_format", "forbidden", "not_found", "rate_limited", "transient", "unknown"})

# ``not_found`` substrings by blast radius: chat-level = target dead; thread/topic/message-level
# leaves the parent chat reachable.
_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS = ("chat not found",)
_SUBCHAT_NOT_FOUND_SUBSTRINGS = (
    "message to edit not found", "message to reply not found", "thread not found", "topic_deleted",
    "message_id_invalid")


# ┌── 函数 def _error_blob()
# │  作用：Lowercased blob (error_text + str(exc) + 异常类 名称) 那个 both
# │  参数：exc: Optional[BaseException] = None, error_text: str = ''
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _error_blob(exc: Optional[BaseException] = None, error_text: str = "") -> str:
    """Lowercased blob (error_text + str(exc) + exception class name) that both
    send-error classifiers match against — one builder so they can never drift."""
    parts = [error_text] if error_text else []
    if exc is not None:
        parts.extend(p for p in (str(exc), exc.__class__.__name__) if p)
    # ⤷ 出口 return：返回 ' '.join(parts).lower()
    return " ".join(parts).lower()


# ┌── 函数 def _any_in()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：blob: str, *needles
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _any_in(blob: str, *needles: str) -> bool:
    # ⤷ 出口 return：返回 any((n in blob for n in needles))
    return any(n in blob for n in needles)


# Ordered (kind, predicate) table for classify_send_error — first match wins.
_SEND_ERROR_CLASSIFIERS: Tuple[Tuple[str, Callable[[str], bool]], ...] = (
    ("too_long", lambda b: _any_in(b, "message_too_long", "too long", "message is too long")),
    ("bad_format", lambda b: (
        _any_in(b, "can't parse entities", "cant parse entities", "can't find end", "unsupported start tag")
        or ("entity" in b and "parse" in b)
        or ("bad request" in b and "entit" in b))),
    ("forbidden", lambda b: _any_in(
        b, "forbidden", "bot was blocked", "blocked by the user", "user is deactivated",
        "not enough rights", "have no rights", "not a member")),
    ("not_found", lambda b: _any_in(b, *_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS, *_SUBCHAT_NOT_FOUND_SUBSTRINGS)),
    ("rate_limited", lambda b: _any_in(b, "flood", "too many requests", "retry after", "rate limit")),
    ("transient", lambda b: _any_in(b, *_RETRYABLE_ERROR_PATTERNS, "connecttimeout")))


# ┌── 函数 def classify_send_error()
# │  作用：Map send 异常 / 错误字符串 :data:`SEND_ERROR_KINDS` 值; anything
# │  参数：exc: Optional[BaseException], error_text: str = ''
# │  返回：str
# └──────────────────────────────────────────────────────────────
def classify_send_error(exc: Optional[BaseException], error_text: str = "") -> str:
    """Map a send exception / error string to a :data:`SEND_ERROR_KINDS` value; anything
    unrecognized is ``"unknown"`` so an unclassified failure is never mistaken for a benign one."""
    blob = _error_blob(exc, error_text)
    # ⤷ 出口 return：返回 next((kind for kind, matches in _SEND_ERROR_CLASSIFIERS if m…
    return next((kind for kind, matches in _SEND_ERROR_CLASSIFIERS if matches(blob)), "unknown")


# ┌── 函数 def is_chat_level_not_found()
# │  作用：是否 ``not_found`` failure means *whole chat* 是 gone (仅那个 marks target。
# │  参数：exc: Optional[BaseException] = None, error_text: str = ''
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_chat_level_not_found(exc: Optional[BaseException] = None, error_text: str = "") -> bool:
    """Whether a ``not_found`` failure means the *whole chat* is gone (only that marks a target
    dead; a deleted topic / edited-away message leaves the parent chat reachable). When both
    markers are present the sub-chat reading wins."""
    blob = _error_blob(exc, error_text)
    # ⤷ 出口 return：返回 not _any_in(blob, *_SUBCHAT_NOT_FOUND_SUBSTRINGS) and _any_i…
    return (not _any_in(blob, *_SUBCHAT_NOT_FOUND_SUBSTRINGS)
            and _any_in(blob, *_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS))


# ┌── 类 class EphemeralReply（继承自：str）
# │  说明：System-提示 reply 那个 auto-deletes 在…之后 ``ttl_seconds`` 在…上平台 implementing
# │  成员：共 2 个方法；对外可用 1 个；含魔术方法 __new__
# └──────────────────────────────────────────────────────────────
class EphemeralReply(str):
    """System-notice reply that auto-deletes after ``ttl_seconds`` on platforms implementing
    ``delete_message`` (others leave it). ``None`` ttl uses ``display.ephemeral_system_ttl``
    (``0`` disables globally). Subclassing ``str`` keeps it transparent to everything that
    treats handler results as text; ``isinstance`` lets the send path schedule deletion."""
    ttl_seconds: Optional[int]

    # ┌── 函数 def __new__()
    # │  作用：魔术方法 __new__（Python 在特定场景自动调用，见下方说明）
    # │  参数：cls, text: str, ttl_seconds: Optional[int] = None
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __new__(cls, text: str, ttl_seconds: Optional[int] = None):
        instance = super().__new__(cls, text)
        instance.ttl_seconds = ttl_seconds
        # ⤷ 出口 return：返回 instance
        return instance

    # ┌── 函数 def text()
    # │  作用：underlying 文本 (explicit 形态的 ``str(reply)``).
    # │  参数：self
    # │  返回：str
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """The underlying text (explicit form of ``str(reply)``)."""
        # ⤷ 出口 return：返回 str.__str__(self)
        return str.__str__(self)


# ┌── 函数 def merge_pending_message_event()
# │  作用：Store 或 merge pending 事件: photo bursts/albums merge 进入 queued 事件从而 next
# │  参数：pending_messages: Dict[str, MessageEvent], session_key: str, event: MessageEvent, merge_text: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def merge_pending_message_event(pending_messages: Dict[str, MessageEvent], session_key: str,
                                event: MessageEvent, *, merge_text: bool = False) -> None:
    """Store or merge a pending event: photo bursts/albums merge into the queued event so the next
    turn sees the whole burst; with ``merge_text`` rapid TEXT follow-ups append instead of
    replace."""
    existing = pending_messages.get(session_key)
    if existing:
        existing_type = getattr(existing, "message_type", None)
        existing_is_photo = existing_type == MessageType.PHOTO
        incoming_is_photo = event.message_type == MessageType.PHOTO
        both_photo = existing_is_photo and incoming_is_photo
        incoming_has_media = bool(event.media_urls)

        def _padded_inline_flags(msg: MessageEvent) -> List[Optional[bool]]:
            flags = list(getattr(msg, "media_text_inlined", []) or [])
            # ⤷ 出口 return：返回 flags + [None] * (len(msg.media_urls) - len(flags))
            return flags + [None] * (len(msg.media_urls) - len(flags))
        incoming_inline_flags: List[Optional[bool]] = []
        if incoming_has_media:
            existing.media_text_inlined = _padded_inline_flags(existing)
            incoming_inline_flags = _padded_inline_flags(event)
        # A photo burst always absorbs; otherwise merge only when media is involved on either
        # side. Captions merge in every absorbing case.
        if both_photo or existing.media_urls or incoming_has_media:
            if both_photo or incoming_has_media:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
                existing.media_text_inlined.extend(incoming_inline_flags)
            if event.text:
                existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
            if existing_is_photo or incoming_is_photo:
                existing.message_type = MessageType.PHOTO
            elif existing_type == MessageType.TEXT and event.message_type != MessageType.TEXT:
                existing.message_type = event.message_type
            # Drop the *derived* STT cache (event changed); the echo ledger must survive or
            # notes echo twice.
            for attr in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts"):
                if hasattr(existing, attr):
                    delattr(existing, attr)
            return
        both_text = existing_type == MessageType.TEXT and event.message_type == MessageType.TEXT
        if merge_text and both_text:
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            return
    pending_messages[session_key] = event


# Transient *connection* failures worth retrying. Plain/read/write "timeout" excluded on purpose:
# the send may have reached the server (retry = duplicate); "connecttimeout" never connected.
_RETRYABLE_ERROR_PATTERNS = (
    "connecterror", "connectionerror", "connectionreset", "connectionrefused", "connecttimeout",
    "network", "broken pipe", "remotedisconnected", "eoferror")

# Handler result: str (reply), ``EphemeralReply`` (auto-delete) or None (already delivered).
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[Union[str, "EphemeralReply"]]]]


# ┌── 函数 def resolve_channel_prompt()
# │  作用：Per-通道 ephemeral 提示词从 ``配置.extra["channel_prompts"]``: exact *channel_id*
# │  参数：config_extra: dict, channel_id: str, parent_id: str | None = None
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def resolve_channel_prompt(config_extra: dict, channel_id: str, parent_id: str | None = None) -> str | None:
    """Per-channel ephemeral prompt from ``config.extra["channel_prompts"]``: exact *channel_id*
    first, then *parent_id* (threads inherit the parent prompt). Blank prompts count as absent."""
    prompts = config_extra.get("channel_prompts") or {}
    if not isinstance(prompts, dict):
        # ⤷ 出口 return：返回 None
        return None
    for key in (channel_id, parent_id):
        prompt = prompts.get(key) if key else None
        if prompt is not None and (prompt := str(prompt).strip()):
            # ⤷ 出口 return：返回 prompt
            return prompt
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def resolve_channel_skills()
# │  作用：Auto-已加载 skill(s) 用于通道/线程从 ``channel_skill_bindings`` (entries
# │  参数：config_extra: dict, channel_id: str, parent_id: str | None = None
# │  返回：list[str] | None
# └──────────────────────────────────────────────────────────────
def resolve_channel_skills(
    config_extra: dict, channel_id: str, parent_id: str | None = None) -> list[str] | None:
    """Auto-loaded skill(s) for a channel/thread from ``channel_skill_bindings`` (entries
    ``{id: "<channel/forum id>", skills: [...]}``; ``skill: "<name>"`` also accepted). Exact
    *channel_id* first, then *parent_id* (threads inherit). Deduplicated ordered list or None."""
    bindings = config_extra.get("channel_skill_bindings") or []
    if not isinstance(bindings, list) or not bindings:
        # ⤷ 出口 return：返回 None
        return None
    ids_to_check = {str(key) for key in (channel_id, parent_id) if key}
    if not ids_to_check:
        # ⤷ 出口 return：返回 None
        return None
    for entry in bindings:
        if not isinstance(entry, dict) or str(entry.get("id", "")) not in ids_to_check:
            continue
        skills = entry.get("skills") or entry.get("skill")
        if isinstance(skills, str):
            # ⤷ 出口 return：返回 [skills.strip()] if skills.strip() else None
            return [skills.strip()] if skills.strip() else None
        if isinstance(skills, list) and skills:
            seen = dict.fromkeys(
                nm for name in skills if isinstance(name, str) and (nm := name.strip()))
            # ⤷ 出口 return：返回 list(seen) or None
            return list(seen) or None
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _split_post_delivery_entry()
# │  作用：``(generation, 回调)`` 从 post-delivery slot; legacy bare 回调有 无
# │  参数：entry: Any
# │  返回：Tuple[Optional[int], Any]
# └──────────────────────────────────────────────────────────────
def _split_post_delivery_entry(entry: Any) -> Tuple[Optional[int], Any]:
    """``(generation, callback)`` from a post-delivery slot; legacy bare callbacks have no
    generation."""
    # ⤷ 出口 return：返回 entry if isinstance(entry, tuple) and len(entry) == 2 else (…
    return entry if isinstance(entry, tuple) and len(entry) == 2 else (None, entry)


# ┌── 函数 def _lazy_attr()
# │  作用：``getattr(obj, 名称)`` 或创建它 via ``factory`` — getattr-护栏用于
# │  参数：obj: Any, name: str, factory: Callable[[], Any]
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _lazy_attr(obj: Any, name: str, factory: Callable[[], Any]) -> Any:
    """``getattr(obj, name)`` or create it via ``factory`` — the getattr-guard for
    tests that build adapters via ``object.__new__`` and never run ``__init__``."""
    value = getattr(obj, name, None)
    if value is None:
        value = factory()
        setattr(obj, name, value)
    # ⤷ 出口 return：返回 value
    return value


_strip_media_directives = _strip_media_tag_directives


# ┌── 类 class BasePlatformAdapter（继承自：ABC）
# │  说明：Base 类用于平台 adapters: connect/auth, receive, send, handle media.
# │  成员：共 172 个方法；对外可用 77 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class BasePlatformAdapter(ABC):
    """Base class for platform adapters: connect/auth, receive, send, handle media."""

    # ``format_message`` renders ``` fences as real code blocks (tool-progress then sends a bare
    # fenced terminal command; plain-text platforms get the preview).
    supports_code_blocks: bool = False
    # Typing indicator renders TEXT (status line); the gateway then feeds set_status_text().
    supports_status_text: bool = False

    # ┌── 函数 def set_status_text()
    # │  作用：设置或 清除 (``None``) 当前活跃的 working-状态 phrase 用于 chat. 在-内存仅:
    # │  参数：self, chat_id: str, text: Optional[str]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_status_text(self, chat_id: str, text: Optional[str]) -> None:
        """Set or clear (``None``) the live working-state phrase for a chat. In-memory only: the
        next typing refresh renders it; a no-op store on adapters that never read
        ``_status_text``."""
        store = _lazy_attr(self, "_status_text", dict)
        if text:
            store[str(chat_id)] = text
        else:
            store.pop(str(chat_id), None)

    # Can wake a fresh turn AFTER a turn ends (detached-subagent completions); False for stateless
    # adapters (API server). Propagated to ``HERMES_SESSION_ASYNC_DELIVERY`` so tools never promise
    # a delivery they can't keep.
    supports_async_delivery: bool = True
    # ``send()`` chunks natively via ``truncate_message()`` -> the router skips its truncation.
    splits_long_messages: bool = False
    # Prefix users can always TYPE for Hermes commands ("!" where the client eats a leading "/").
    typed_command_prefix: str = "/"
    # ``in_channel`` continuable-cron surface: job delivered FLAT, plain replies continue it via
    # the whole-channel bucket ``(platform, chat_id, None)``; needs a flat-reply outbound gate too
    # (Slack ``reply_in_thread: false``). False fails SAFE -> ``thread``.
    supports_inchannel_continuable: bool = False
    # A human can answer "session restored — what next?"; webhook-style platforms set False so
    # auto-resume finishes the work instead of asking nobody.
    # The startup auto-resume turn (``_schedule_resume_pending_sessions`` → the ``_is_resume_pending``
    # branch in ``_handle_message_with_agent``) reads this to pick its guidance: interactive platforms
    # (Telegram, Slack, Discord DMs, …) get "report the restore and ask what the user wants next";
    # non-interactive event platforms (webhook) get "finish the interrupted work" because nobody is there to
    # answer, and an acknowledgement would silently abandon the task (#57056). Read generically via
    # ``getattr(adapter, "interactive_resume", True)`` — no per-platform branching at the call site.
    interactive_resume: bool = True
    # Port-binding adapter that answers ``/p/<profile>/...`` for every served profile on the default
    # listener under ``gateway.multiplex_profiles``. Declared per adapter (not in a central list) so
    # ``hermes gateway migrate`` can tell "URL changes" from "this profile would be skipped" as new
    # HTTP-inbound adapters gain the prefix.
    serves_profile_prefix: bool = False
    # Back-reference to the running ``GatewayRunner`` (set by gateway/run.py); ``build_source``
    # resolves the inbound profile via ``runner._profile_name_for_source``.
    gateway_runner = None  # type: ignore[assignment]

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, config: PlatformConfig, platform: Platform
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
        self._no_message_handler_logged: bool = False
        self._reaction_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        # Runner-owned boundary for normalized events: auth/profile state never lives in an adapter.
        self._platform_event_handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]] = None
        # Rewrites ``event.source.thread_id`` before session keying (Telegram DM topics).
        self._topic_recovery_fn: Optional[Callable[[Any], Optional[str]]] = None
        self._running, self._fatal_error_retryable = False, True
        self._fatal_error_code: Optional[str] = None
        self._fatal_error_message: Optional[str] = None
        self._fatal_error_handler: Optional[Callable[["BasePlatformAdapter"], Awaitable[None] | None]] = None
        # Strong refs to shielded fatal-error handler tasks that outlive their carrier task
        # (asyncio keeps only weak refs); without them the loop can GC the detached handler
        # mid-flight — the "handler killed mid-flight" class (#81335).
        # See #81335.
        self._detached_fatal_tasks: set = set()
        # Lock takeover armed only for the initial connect of ``gateway run --replace``.
        self._platform_lock_takeover_allowed = self._platform_lock_takeover_attempted = False
        # Per-session interrupt Event + owner Task: without the owner map an old task's finally
        # could drop a newer guard.
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._session_tasks: Dict[str, asyncio.Task] = {}
        # Legacy env knob; the runner syncs the busy_input_mode value after construction.
        # Default "interrupt" so a pre-sync read never silently queues.
        self._busy_text_mode: str = (
            os.environ.get("HERMES_GATEWAY_BUSY_TEXT_MODE", "interrupt").strip().lower() or "interrupt")
        self._busy_text_debounce_seconds: float = _float_env("HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS", 0.35)
        self._busy_text_hard_cap_seconds: float = _float_env("HERMES_GATEWAY_BUSY_TEXT_HARD_CAP_SECONDS", 1.0)
        self._text_debounce: dict[str, TextDebounceState] = {}
        # handle_message() tasks; shutdown cancels them so a replaced gateway stops working.
        self._background_tasks: set[asyncio.Task] = set()
        # Post-delivery one-shots per session_key: bare callback (legacy) or ``(generation,
        # callback)`` so a stale run can't clear a fresher run's callback.
        self._post_delivery_callbacks: Dict[str, Any] = {}
        self._expected_cancelled_tasks: set[asyncio.Task] = set()
        self._busy_session_handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]] = None
        # Owning multiplex profile (None on primary); see _session_key_profile.
        self._owner_profile: Optional[str] = None
        # Set by the runner on a secondary's port-binding adapter: serve via the default profile's
        # shared listener (/p/<profile>/...) instead of binding a port (gateway/platforms/shared_ingress.py).
        self._shared_listener_profile: Optional[str] = None
        # Registered by GatewayRunner (see set_authorization_check).
        self._authorization_check: Optional[Callable[[str, Optional[str], Optional[str]], bool]] = None
        # Auto-TTS on voice input: ``voice.auto_tts`` default plus per-chat /voice on|tts / off.
        self._auto_tts_default: bool = False
        self._auto_tts_enabled_chats, self._auto_tts_disabled_chats = set(), set()
        # Turn keys where streaming TTS already delivered audio; whole-file auto-TTS skips them.
        # When the gateway streaming-TTS consumer successfully delivers audio, it adds the turn key here so
        # the base adapter's whole-file auto-TTS path skips the duplicate. Cleared after the turn completes.
        # See #60671.
        self._streaming_tts_completed_turns: set[str] = set()
        # Chats whose typing indicator is paused (approval waits); _keep_typing skips them.
        self._typing_paused: set = set()
        # Per-chat status phrase; the regular _keep_typing refresh renders it (no extra API calls).
        self._status_text: Dict[str, str] = {}

    # ┌── 函数 def message_len_fn()
    # │  作用：Length 函数用于消息 size; 覆盖值何处平台 counts
    # │  参数：self
    # │  返回：Callable[[str], int]
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def message_len_fn(self) -> Callable[[str], int]:
        """Length function for message size; override where the platform counts
        differently from ``len`` (Telegram: UTF-16 code units)."""
        # ⤷ 出口 return：返回 len
        return len

    # ┌── 函数 def max_message_length_for_chat()
    # │  作用：Per-chat max length 在 ``message_len_fn_for_chat`` units: 适配器's
    # │  参数：self, chat_id: str
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def max_message_length_for_chat(self, chat_id: str) -> int:
        """Per-chat max length in ``message_len_fn_for_chat`` units: the adapter's
        ``MAX_MESSAGE_LENGTH`` (4096 when absent); the relay adapter overrides since it fronts N
        platforms with different caps."""
        # ⤷ 出口 return：返回 _or_default(lambda: int(getattr(self, 'MAX_MESSAGE_LENGTH', …
        return _or_default(lambda: int(getattr(self, "MAX_MESSAGE_LENGTH", 4096) or 4096), 4096)

    # ┌── 函数 def message_len_fn_for_chat()
    # │  作用：Per-chat length 函数 (companion max_message_length_for_chat); relay
    # │  参数：self, chat_id: str
    # │  返回：Callable[[str], int]
    # └──────────────────────────────────────────────────────────────
    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        """Per-chat length function (companion to max_message_length_for_chat); the relay
        adapter overrides it per the chat's fronting platform."""
        # ⤷ 出口 return：返回 self.message_len_fn
        return self.message_len_fn

    # ┌── 函数 def enforces_own_access_policy()
    # │  作用：是否 这个适配器强制其 持有配置-driven access 策略在 intake。
    # │  参数：self
    # │  返回：bool
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def enforces_own_access_policy(self) -> bool:
        """Whether this adapter enforces its own config-driven access policy at intake
        (``dm_policy``/``group_policy``/``allow_from``: WeCom, Weixin, QQBot, WhatsApp…). The
        gateway env allowlist runs *after* the adapter; without one it trusts this flag ONLY when
        the effective policy is a real ``"allowlist"`` — never ``"open"`` (the default), which would
        be a network-exposed fail-open (SECURITY.md §2.6). Open access still requires
        ``{PLATFORM}_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOW_ALL_USERS``."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def authorization_is_upstream()
    # │  作用：是否 inbound 是已 authorized 由 TRUSTED UPSTREAM (relay 仅): Team。
    # │  参数：self
    # │  返回：bool
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def authorization_is_upstream(self) -> bool:
        """Whether inbound was already authorized by a TRUSTED UPSTREAM (relay only): the Team
        Gateway connector authenticates the WebSocket and resolves owner-only author binding BEFORE
        delivery, so the no-allowlist default-deny would be wrong. Authorization DELEGATED, not
        ABSENT — every network-exposed direct adapter leaves it False."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def supports_draft_streaming()
    # │  作用：是否 native streaming-draft updates (``send_draft``) work 用于这个 chat 类型;。
    # │  参数：self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, chat_id: Optional[str] = None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def supports_draft_streaming(
        self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None) -> bool:
        """Whether native streaming-draft updates (``send_draft``) work for this chat type;
        ``chat_id`` lets the relay adapter answer per negotiated capabilities. Consumers
        fall back to ``send`` + ``edit_message`` when False or ``send_draft`` raises."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def prefers_fresh_final_streaming()
    # │  作用：是否 stream consumer 应当 finalize 带 *fresh* final 消息 (尽最大努力。
    # │  参数：self, content: str, metadata: Optional[Dict[str, Any]] = None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def prefers_fresh_final_streaming(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Whether the stream consumer should finalize with a *fresh* final message (best-effort
        deleting the preview) instead of final-editing it (Telegram: keeps rich rendering)."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def streaming_overflow_limit()
    # │  作用：Max single-消息 length (``message_len_fn`` units) stream consumer 可以
    # │  参数：self
    # │  返回：Optional[int]
    # └──────────────────────────────────────────────────────────────
    def streaming_overflow_limit(self) -> Optional[int]:
        """Max single-message length (``message_len_fn`` units) the stream consumer may
        accumulate before splitting, for rich send/draft paths exceeding the legacy cap
        (Telegram Rich Messages: 32,768 vs 4,096). ``None`` = use ``MAX_MESSAGE_LENGTH``."""
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 异步函数（协程） def send_draft()
    # │  作用：Send 或更新 animated streaming-draft preview. Reuse one non-zero ``draft_id``
    # │  参数：self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_draft(self, chat_id: str, draft_id: int, content: str,
                         metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send or update an animated streaming-draft preview. Reuse one non-zero ``draft_id``
        across a response so the platform animates (different responses need different ids). Drafts
        have no message_id (no edit/reply/delete) — the final answer goes out as a regular ``send``.
        Must be overridden by adapters returning True from :meth:`supports_draft_streaming`."""
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise NotImplementedError(f"{type(self).__name__} does not implement send_draft")

    # ── Structured stream-event rendering (gateway/stream_events.py): presentation-only,
    # nothing rendered here is persisted, so what an adapter "eats" never changes history.

    # ┌── 函数 def render_message_event()
    # │  作用：把 MessageChunk / MessageStop / Commentary 渲染到输出接收端（sink，……）。
    # │  参数：self, event: Any, sink: Any
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def render_message_event(self, event: Any, sink: Any) -> None:
        """Render a MessageChunk / MessageStop / Commentary onto the sink (a
        GatewayStreamConsumer), mapping 1:1 onto its existing primitives."""
        from gateway.stream_events import MessageChunk, MessageStop, Commentary
        if isinstance(event, MessageChunk) and event.text:
            sink.on_delta(event.text)
        elif isinstance(event, MessageStop) and not event.final:
            # Intermediate stop (text → tool → text) = segment break; the terminal stop is finish().
            sink.on_segment_break()
        elif isinstance(event, Commentary) and event.text:
            sink.on_commentary(event.text)

    # ┌── 函数 def format_tool_event()
    # │  作用：Rendered chrome 用于 ToolCallChunk, 或 None eat 它 (adapters 不 editing/rich
    # │  参数：self, event: Any, mode: str = ..., preview_max_len: int = ...
    # │  返回：Optional[str]
    # └──────────────────────────────────────────────────────────────
    def format_tool_event(self, event: Any, *, mode: str = "all", preview_max_len: int = 40) -> Optional[str]:
        """Rendered chrome for a ToolCallChunk, or None to eat it (adapters without editing/rich
        text override to None). ``mode``: tool-progress mode ("all"/"new"/"verbose");
        ``preview_max_len`` mirrors ``tool_preview_length`` (0 = no cap in verbose)."""
        from gateway.stream_events import ToolCallChunk
        if not isinstance(event, ToolCallChunk):
            # ⤷ 出口 return：返回 None
            return None
        from agent.display import get_tool_emoji, prepare_tool_preview
        head = f"{get_tool_emoji(event.tool_name, default='⚙️')} {event.tool_name}"
        if mode == "verbose" and event.args:
            import json
            args_str = json.dumps(event.args, ensure_ascii=False, default=str)
            if preview_max_len > 0 and len(args_str) > preview_max_len:
                args_str = args_str[:preview_max_len - 3] + "..."
            # ⤷ 出口 return：返回 f'{head}({list(event.args.keys())})\n{args_str}'
            return f"{head}({list(event.args.keys())})\n{args_str}"
        if not event.preview:
            # ⤷ 出口 return：返回 f'{head}...'
            return f"{head}..."
        if mode == "verbose":
            # ⤷ 出口 return：返回 f'{head}: "{event.preview}"'
            return f'{head}: "{event.preview}"'
        # "all" / "new": short capped preview (default 40; progress bubbles persist as messages).
        cap = preview_max_len if preview_max_len > 0 else 40
        prepared = prepare_tool_preview(
            event.tool_name, event.args, fallback=event.preview, max_len=cap)
        # ⤷ 出口 return：返回 f'{head}: "{self.format_tool_preview(prepared)}"'
        return f'{head}: "{self.format_tool_preview(prepared)}"'

    # ┌── 函数 def format_tool_preview()
    # │  作用：平台-native formatting 的 compact 工具 preview; rich-文本 adapters 可以 use
    # │  参数：self, preview: 'ToolPreview'
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def format_tool_preview(self, preview: "ToolPreview") -> str:
        """Platform-native formatting of a compact tool preview; rich-text adapters may use
        the preview's metadata (e.g. a URL shortened for display)."""
        # ⤷ 出口 return：返回 preview.text
        return preview.text

    has_fatal_error = property(lambda self: self._fatal_error_message is not None)
    fatal_error_message = property(lambda self: self._fatal_error_message)
    fatal_error_code = property(lambda self: self._fatal_error_code)
    fatal_error_retryable = property(lambda self: self._fatal_error_retryable)

    # ┌── 函数 def _should_auto_tts_for_chat()
    # │  作用：是否 auto-TTS fires 用于 ``chat_id``: explicit ``/voice 在…上|tts`` wins,。
    # │  参数：self, chat_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        """Whether auto-TTS fires for ``chat_id``: explicit ``/voice on|tts`` wins,
        then explicit ``/voice off``, then the global ``voice.auto_tts`` default.

        Decision layers (Issue #16007): 1. Explicit ``/voice on`` or ``/voice tts`` → always fire (even if
        ``voice.auto_tts`` is False). 2. 3.
        """
        # ⤷ 出口 return：返回 chat_id in self._auto_tts_enabled_chats or (chat_id not in s…
        return chat_id in self._auto_tts_enabled_chats or (
            chat_id not in self._auto_tts_disabled_chats and bool(self._auto_tts_default))

    # ┌── 函数 def set_fatal_error_handler()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, handler: Callable[['BasePlatformAdapter'], Awaitable[None] | None]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_fatal_error_handler(self, handler: Callable[["BasePlatformAdapter"], Awaitable[None] | None]) -> None:
        self._fatal_error_handler = handler

    #: Published when an adapter is installed and running but its receive
    #: path is not yet confirmed (e.g. Telegram polling has not proven a
    #: getUpdates round-trip). Same ``retrying`` platform_state the runner
    #: uses for queued reconnects, so readers see "not delivering" (#101391).
    DEGRADED_STATUS_MESSAGE = "connected but not yet confirmed active; recovering in background"

    # ┌── 函数 def send_path_degraded()
    # │  作用：True 当 connect() succeeded 但 delivery 是不 confirmed.
    # │  参数：self
    # │  返回：bool
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def send_path_degraded(self) -> bool:
        """True while connect() succeeded but delivery is not confirmed.

        Adapters with a separately-proven receive path override this; the
        default adapter is either connected or not.
        """
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def _mark_connected()
    # │  作用：``listener_base`` (``http://宿主:port``) 是 stamped 由 port-binders 在…之后 REAL bind: under
    # │  参数：self, listener_base: Optional[str] = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _mark_connected(self, *, listener_base: Optional[str] = None) -> None:
        """``listener_base`` (``http://host:port``) is stamped by port-binders after a REAL bind: under the
        multiplexer it is the shared listener a served profile's ``/p/<profile>/`` mirror hangs off, and
        what the dashboard/Desktop report as that profile's api_server/webhook URL."""
        self._running = True
        self._fatal_error_code = self._fatal_error_message = None
        self._fatal_error_retryable = True
        if self.send_path_degraded:
            self._mark_degraded()
        else:
            extra = {"listener_base": listener_base} if listener_base else {}
            self._write_runtime_status_safe(
                "connected", platform_state="connected", error_code=None, error_message=None, **extra)

    # ┌── 函数 def _mark_degraded()
    # │  作用：Publish ``retrying`` 用于运行适配器 whose delivery 路径是 unproven.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _mark_degraded(self) -> None:
        """Publish ``retrying`` for a running adapter whose delivery path is unproven."""
        self._write_runtime_status_safe(
            "connected_degraded",
            platform_state="retrying",
            error_code=None,
            error_message=self.DEGRADED_STATUS_MESSAGE,
        )

    # ┌── 函数 def _mark_disconnected()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _mark_disconnected(self) -> None:
        self._running = False
        if not self.has_fatal_error:
            self._write_runtime_status_safe(
                "disconnected", platform_state="disconnected", error_code=None, error_message=None)

    # ┌── 函数 def _set_fatal_error()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, code: str, message: str, retryable: bool
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._running = False
        self._fatal_error_code, self._fatal_error_message = code, message
        self._fatal_error_retryable = retryable
        self._write_runtime_status_safe("fatal", platform_state="fatal", error_code=code, error_message=message)

    # ┌── 函数 def _write_runtime_status_safe()
    # │  作用：写入运行时 status; 日志 first failure per 上下文在 警告, rest 在调试
    # │  参数：self, context: str, **kwargs
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _write_runtime_status_safe(self, context: str, **kwargs) -> None:
        """Write runtime status; log first failure per context at warning, rest at debug
        (failures — permissions, ENOSPC — must neither be silent nor spam reconnect loops)."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from gateway.status import write_runtime_status
            # Multiplexed adapters share the status file; the runner stamps
            # ``<profile>:<platform>``.
            platform_key = getattr(self, "_runtime_status_platform_key", None) or self.platform.value
            write_runtime_status(platform=platform_key, **kwargs)
        except Exception as exc:
            logged = _lazy_attr(self, "_status_write_logged", set)  # object.__new__ in tests
            first = (self.platform.value, context) not in logged
            logged.add((self.platform.value, context))
            (logger.warning if first else logger.debug)(
                "Failed to write runtime status (%s) for %s: %s" + (" (further failures at debug level)" if first else ""),
                context, self.platform.value, exc)

    # ┌── 异步函数（协程） def _notify_fatal_error()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _notify_fatal_error(self) -> None:
        handler = self._fatal_error_handler
        if not handler:
            return
        result = handler(self)
        if asyncio.iscoroutine(result):
            # Detached + shielded: often awaited from an adapter-owned task (e.g. Telegram
            # ``_polling_error_task``) that the gateway fatal handler's ``disconnect()``
            # cancels; unshielded, the handler died mid-flight — adapter popped from the
            # gateway map but never queued for background reconnect, leaving a zombie
            # gateway with no platforms and no pending retries (#81335).
            # See #81335.
            task = asyncio.ensure_future(result)
            # Strong ref: asyncio only keeps weak refs to tasks ("save a reference ... to
            # avoid a task disappearing mid-execution"); matches
            # GatewayRunner._handle_adapter_fatal_error.
            _tasks = _lazy_attr(self, "_detached_fatal_tasks", set)
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
            # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError，出错走兜底而不是崩溃。
            try:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Carrier cancelled (our own teardown inside the handler): let it finish detached.
                if not task.done():
                    task.add_done_callback(_consume_detached_handler_exception)
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise

    # ┌── 函数 def _acquire_platform_lock()
    # │  作用：Acquire scoped 锁用于这个适配器; True 在…上 success. 当前活跃的 cross-HERMES_HOME
    # │  参数：self, scope: str, identity: str, resource_desc: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _acquire_platform_lock(self, scope: str, identity: str, resource_desc: str) -> bool:
        """Acquire a scoped lock for this adapter; True on success. A live cross-HERMES_HOME
        holder is replaced only when the runner armed this adapter for its initial
        ``--replace`` connect (the status module validates ownership and terminates)."""
        from gateway.status import (
            acquire_scoped_lock, scoped_lock_owner_label, take_over_scoped_lock_holder)
        self._platform_lock_scope, self._platform_lock_identity = scope, identity
        lock_meta = {"platform": self.platform.value}
        acquired, existing = acquire_scoped_lock(scope, identity, metadata=lock_meta)
        if acquired:
            # ⤷ 出口 return：返回 True
            return True
        if (self._platform_lock_takeover_allowed and not self._platform_lock_takeover_attempted
                and isinstance(existing, dict)):
            # Consume the authority before any I/O: at most one termination attempt per connect.
            self._platform_lock_takeover_allowed = False
            self._platform_lock_takeover_attempted = True
            owner_pid = take_over_scoped_lock_holder(existing)
            if owner_pid is not None:
                logger.warning(
                    "[%s] %s was held by gateway PID %d — explicit --replace handoff completed",
                    self.name, resource_desc, owner_pid)
                acquired, existing = acquire_scoped_lock(scope, identity, metadata=lock_meta)
                if acquired:
                    logger.info("[%s] Acquired %s after taking over PID %d", self.name, resource_desc, owner_pid)
                    # ⤷ 出口 return：返回 True
                    return True
        owner_pid = existing.get('pid') if isinstance(existing, dict) else None
        # Scoped locks are machine-global: name the owning profile so the operator knows WHICH
        # gateway.
        owner_profile = scoped_lock_owner_label(existing)
        pid_part = f" (PID {owner_pid})" if owner_pid else ""
        holder = f" by the '{owner_profile}' profile gateway{pid_part}" if owner_profile else pid_part
        remedy = (f" Stop that gateway first (hermes --profile {owner_profile} gateway stop)."
                  if owner_profile else " Stop the other gateway first.")
        message = f"{resource_desc} already in use{holder}.{remedy}"
        logger.error('[%s] %s', self.name, message)
        self._set_fatal_error(f'{scope}_lock', message, retryable=True)
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def _release_platform_lock()
    # │  作用：Release scoped 锁 acquired 由 _acquire_platform_lock.
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _release_platform_lock(self) -> None:
        """Release the scoped lock acquired by _acquire_platform_lock."""
        identity = getattr(self, '_platform_lock_identity', None)
        if not identity:
            return
        from gateway.status import release_scoped_lock
        release_scoped_lock(self._platform_lock_scope, identity)
        self._platform_lock_identity = None

    # ┌── 函数 def _wire_plugin_handlers()
    # │  作用：调用插件注册的「原生处理器工厂」（``ctx.register_platform_handler``）。
    # │  参数：self, native: Any = None
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _wire_plugin_handlers(self, native: Any = None) -> None:
        """Invoke plugin-registered native handler factories (``ctx.register_platform_handler``)
        with ``(native, adapter)``; adapters call this from ``connect()`` once the native
        client exists. Each factory is isolated so a bad plugin can't block connecting."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from hermes_cli.plugins import get_plugin_manager
            factories = get_plugin_manager().get_platform_handler_factories(
                getattr(self.platform, "value", str(self.platform)))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[%s] Could not load plugin handler factories: %s", self.name, e)
            return
        for factory, plugin_name in factories:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                factory(native, self)
                logger.info("[%s] Wired native handlers from plugin '%s'", self.name, plugin_name)
            except Exception as exc:
                logger.error("[%s] Plugin '%s' handler factory raised: %s", self.name, plugin_name,
                             exc, exc_info=True)

    # ┌── 函数 def name()
    # │  作用：Human-readable 名称用于这个适配器.
    # │  参数：self
    # │  返回：str
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        # ⤷ 出口 return：返回 self.platform.value.title()
        return self.platform.value.title()

    # ┌── 函数 def is_connected()
    # │  作用：检查若 适配器是 currently connected.
    # │  参数：self
    # │  返回：bool
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def is_connected(self) -> bool:
        """Check if adapter is currently connected."""
        # ⤷ 出口 return：返回 self._running
        return self._running

    # ┌── 函数 def set_message_handler()
    # │  作用：设置 incoming-消息 handler (MessageEvent -> optional 响应 str).
    # │  参数：self, handler: MessageHandler
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the incoming-message handler (MessageEvent -> optional response str)."""
        self._message_handler = handler

    # ┌── 函数 def set_platform_event_handler()
    # │  作用：安装 gateway-持有的已归一化平台-事件 boundary (stable dicts + internal
    # │  参数：self, handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_platform_event_handler(
        self, handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]]) -> None:
        """Install the gateway-owned normalized platform-event boundary (stable dicts + internal
        ``SessionSource``); the runner owns authorization and plugin dispatch: no callback = fail
        closed."""
        self._platform_event_handler = handler

    # ┌── 函数 def set_topic_recovery_fn()
    # │  作用：安装 thread_id-recovery hook (Telegram 私信 topic 模式): called 带 ``事件.source``
    # │  参数：self, fn: Optional[Callable[[Any], Optional[str]]]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_topic_recovery_fn(self, fn: Optional[Callable[[Any], Optional[str]]]) -> None:
        """Install a thread_id-recovery hook (Telegram DM topic mode): called with ``event.source``
        before session keying; a non-None return replaces ``source.thread_id``. ``None`` clears
        it."""
        self._topic_recovery_fn = fn

    # ┌── 函数 def _apply_topic_recovery()
    # │  作用：Rewrite ``事件.source.thread_id`` 在 place 若 hook 返回 one.
    # │  参数：self, event: MessageEvent
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _apply_topic_recovery(self, event: MessageEvent) -> None:
        """Rewrite ``event.source.thread_id`` in place if the hook returns one."""
        recover = getattr(self, "_topic_recovery_fn", None)
        source = getattr(event, "source", None)
        if recover is None or source is None:
            return
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            recovered = recover(source)
            if recovered is None or str(recovered) == str(source.thread_id or ""):
                return
        except Exception:
            logger.debug("topic recovery hook failed", exc_info=True)
            return
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            event.source = dataclasses.replace(source, thread_id=str(recovered))
        except Exception:
            logger.debug("topic recovery rewrite failed", exc_info=True)

    # ┌── 函数 def set_busy_session_handler()
    # │  作用：设置 optional handler 用于消息 arriving 在…期间 active 会话.
    # │  参数：self, handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_busy_session_handler(self, handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]]) -> None:
        """Set an optional handler for messages arriving during active sessions."""
        self._busy_session_handler = handler

    # ┌── 函数 def set_reaction_handler()
    # │  作用：设置 handler 用于平台-native emoji-reaction 事件: normalised 字典
    # │  参数：self, handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_reaction_handler(self, handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]) -> None:
        """Set the handler for platform-native emoji-reaction events: a normalised dict
        (``platform``, ``event_name`` "reaction:added"/"reaction:removed", ``reaction``,
        ``user_id``, ``item_user_id``, ``channel_id``, ``message_ts``, ``event_ts``, ``raw_event``)
        fanned out via ``HookRegistry.emit``."""
        self._reaction_handler = handler

    # ┌── 函数 def set_authorization_check()
    # │  作用：Register ``(user_id, chat_type, chat_id) -> 布尔值``; adapters pulling external 上下文
    # │  参数：self, callback: Optional[Callable[[str, Optional[str], Optional[str]], bool]]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_authorization_check(
        self, callback: Optional[Callable[[str, Optional[str], Optional[str]], bool]]) -> None:
        """Register ``(user_id, chat_type, chat_id) -> bool``; adapters pulling external context
        (Slack thread replies) use it to flag non-allowlisted senders as unverified background."""
        self._authorization_check = callback

    # ┌── 函数 def _is_sender_authorized()
    # │  作用：True/False 从 registered 检查, 或 None 当无 检查 exists ("trust unknown",
    # │  参数：self, user_id: Optional[str], chat_type: Optional[str] = None, chat_id: Optional[str] = None, is_bot: bool = ..., thread_id: Optional[str] = ...
    # │  返回：Optional[bool]
    # └──────────────────────────────────────────────────────────────
    def _is_sender_authorized(self, user_id: Optional[str], chat_type: Optional[str] = None,
                              chat_id: Optional[str] = None, *, is_bot: bool = False,
                              thread_id: Optional[str] = None) -> Optional[bool]:
        """True/False from the registered check, or None when no check exists ("trust unknown",
        legacy). ``is_bot``/``thread_id`` are forwarded as keywords only when set so legacy
        three-positional callbacks keep working. Only literal booleans propagate: a truthy
        non-boolean is "unknown", never an authorization that gates a credentialed side effect."""
        if not user_id or self._authorization_check is None:
            # ⤷ 出口 return：返回 None
            return None
        extra: Dict[str, Any] = {}
        if is_bot:
            extra["is_bot"] = True
        if thread_id is not None:
            extra["thread_id"] = thread_id
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            result = self._authorization_check(user_id, chat_type, chat_id, **extra)
        except Exception:
            logger.warning("[%s] Authorization check raised for user %s; treating as unknown",
                           self.name, user_id, exc_info=True)
            # ⤷ 出口 return：返回 None
            return None
        if result is True or result is False:
            # ⤷ 出口 return：返回 result
            return result
        logger.warning("[%s] Authorization check returned %s for user %s; treating as unknown",
                       self.name, type(result).__name__, user_id)
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def set_session_store()
    # │  作用：设置会话 store (e.g. Slack 检查用于 active 线程会话
    # │  参数：self, session_store: Any
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_session_store(self, session_store: Any) -> None:
        """Set the session store (e.g. Slack checks for an active thread session
        before handling un-mentioned replies)."""
        self._session_store = session_store

    # ┌── 函数 def set_owner_profile()
    # │  作用：Declare 持有 multiplex 配置档 (secondary 配置档仅); 读取由
    # │  参数：self, profile_name: Optional[str]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def set_owner_profile(self, profile_name: Optional[str]) -> None:
        """Declare the owning multiplex profile (secondary profiles only); read by
        :meth:`_session_key_profile` so adapter-level keys leave ``agent:main:``."""
        self._owner_profile = None if (name := (profile_name or "").strip() or None) == "default" else name

    # ┌── 函数 def _session_key_profile()
    # │  作用：配置档 namespace 用于适配器-derived 会话键. Ingress 运行在…之前 runner
    # │  参数：self, source: Optional[Any] = None
    # │  返回：Optional[str]
    # └──────────────────────────────────────────────────────────────
    def _session_key_profile(self, source: Optional[Any] = None) -> Optional[str]:
        """Profile namespace for an adapter-derived session key. Ingress runs BEFORE the runner
        stamps ``source.profile``, so without this every bot in a multiplexed gateway shares one
        ``agent:main:`` lane. Order: ``source.profile`` → ``_owner_profile`` → session-store
        resolver; getattr-guarded (object.__new__ in tests), type-checked (no MagicMock in the
        key)."""
        for candidate in (
            getattr(source, "profile", None) if source is not None else None,
            getattr(self, "_owner_profile", None)):
            if isinstance(candidate, str) and candidate.strip():
                # ⤷ 出口 return：返回 candidate
                return candidate
        store = getattr(self, "_session_store", None)
        resolver = getattr(store, "_resolve_profile_for_key", None) if store else None
        if not callable(resolver):
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            resolved = resolver(source)
        except Exception:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 resolved if isinstance(resolved, str) and resolved.strip() e…
        return resolved if isinstance(resolved, str) and resolved.strip() else None

    # ── Inbound text batching. Chat clients split one long message into several inbound
    # chunks; ``_enqueue_text_event`` merges chunks per session key and ``_flush_text_batch``
    # dispatches after a quiet period (longer when the last chunk sits near the platform's
    # split point, i.e. a continuation is almost certain). Adapters set the delay attrs and
    # ``_SPLIT_THRESHOLD``; ``_text_batch_delay_for`` / ``_pop_text_batch`` /
    # ``_dispatch_text_batch`` are the override seams for platform-specific policy.
    _SPLIT_THRESHOLD: int = 4000
    _text_batch_delay_seconds: float = 0.0
    _text_batch_split_delay_seconds: float = 0.0

    # ┌── 函数 def _event_session_key()
    # │  作用：适配器-级别会话键 用于 ``事件``, 配置档-namespaced like Agent 运行.
    # │  参数：self, event: 'MessageEvent'
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _event_session_key(self, event: "MessageEvent") -> str:
        """Adapter-level session key for ``event``, profile-namespaced like the agent run."""
        # ⤷ 出口 return：返回 self._source_session_key(event.source)
        return self._source_session_key(event.source)

    # ┌── 函数 def _source_session_key()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, source: 'SessionSource'
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _source_session_key(self, source: "SessionSource") -> str:
        extra = self.config.extra
        # ⤷ 出口 return：返回 build_session_key(source, group_sessions_per_user=extra.get(…
        return build_session_key(
            source, group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(source))

    # ┌── 函数 def _text_batch_key()
    # │  作用：会话-scoped 键用于文本 batching (subclasses 可以覆盖值).
    # │  参数：self, event: 'MessageEvent'
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _text_batch_key(self, event: "MessageEvent") -> str:
        """Session-scoped key for text batching (subclasses may override)."""
        # ⤷ 出口 return：返回 self._event_session_key(event)
        return self._event_session_key(event)

    # ┌── 函数 def _enqueue_text_event()
    # │  作用：Buffer 文本事件 (merging 进入 pending one) 且 restart 冲刷 timer.
    # │  参数：self, event: 'MessageEvent'
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _enqueue_text_event(self, event: "MessageEvent") -> None:
        """Buffer a text event (merging into a pending one) and restart the flush timer."""
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        if existing is None:
            existing = self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
        existing._last_chunk_len = len(event.text or "")  # type: ignore[attr-defined]
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(self._flush_text_batch(key))

    # ┌── 函数 def _text_batch_delay_for()
    # │  作用：Quiet period 在…之前 ``pending`` 是 dispatched; near-split chunks wait longer.
    # │  参数：self, pending: Optional['MessageEvent']
    # │  返回：float
    # └──────────────────────────────────────────────────────────────
    def _text_batch_delay_for(self, pending: Optional["MessageEvent"]) -> float:
        """Quiet period before ``pending`` is dispatched; near-split chunks wait longer."""
        last_len = getattr(pending, "_last_chunk_len", 0) if pending is not None else 0
        # ⤷ 出口 return：返回 self._text_batch_split_delay_seconds if last_len >= self._SP…
        return self._text_batch_split_delay_seconds if last_len >= self._SPLIT_THRESHOLD else self._text_batch_delay_seconds

    # ┌── 函数 def _pop_text_batch()
    # │  作用：移除且 返回 pending batch 用于 ``键`` (adapters 带 side tables 覆盖值).
    # │  参数：self, key: str
    # │  返回：Optional['MessageEvent']
    # └──────────────────────────────────────────────────────────────
    def _pop_text_batch(self, key: str) -> Optional["MessageEvent"]:
        """Remove and return the pending batch for ``key`` (adapters with side tables override)."""
        # ⤷ 出口 return：返回 self._pending_text_batches.pop(key, None)
        return self._pending_text_batches.pop(key, None)

    # ┌── 异步函数（协程） def _dispatch_text_batch()
    # │  作用：Hand 已冲刷 batch pipeline (adapters 带 per-chat 护栏覆盖值).
    # │  参数：self, event: 'MessageEvent'
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _dispatch_text_batch(self, event: "MessageEvent") -> None:
        """Hand a flushed batch to the pipeline (adapters with per-chat guards override)."""
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self.handle_message(event)

    # ┌── 异步函数（协程） def _flush_text_batch_now()
    # │  作用：Dispatch pending batch 用于 ``键`` immediately (无 quiet period).
    # │  参数：self, key: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _flush_text_batch_now(self, key: str) -> None:
        """Dispatch the pending batch for ``key`` immediately (no quiet period)."""
        event = self._pop_text_batch(key)
        if event is not None:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._dispatch_text_batch(event)

    # ┌── 异步函数（协程） def _flush_text_batch()
    # │  作用：Wait 用于 quiet period, 然后 dispatch batch 用于 ``键``.
    # │  参数：self, key: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period, then dispatch the batch for ``key``.

        Two races share this body. (1) ``_enqueue_text_event`` cancels the prior flush task
        on each new chunk; when ``Task.cancel()`` lands after ``sleep()`` already completed,
        CancelledError is delivered at the *next* await — after a superseded task would have
        popped the event, so the successor finds nothing and the message is lost. The identity
        check therefore runs synchronously between the sleep and the pop. (2) A cancel that
        lands while the dispatch is in flight would abort the agent turn (#12444), so the
        dispatch is shielded and the outer CancelledError swallowed."""
        current_task = asyncio.current_task()
        # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.sleep(self._text_batch_delay_for(self._pending_text_batches.get(key)))
            owner = self._pending_text_batch_tasks.get(key)
            if owner is not None and owner is not current_task:
                return
            event = self._pop_text_batch(key)
            if event is None:
                return
            logger.info("[%s] Flushing text batch %s (%d chars)", self.name, key, len(event.text or ""))
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.shield(self._dispatch_text_batch(event))
        except asyncio.CancelledError:
            pass
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # ┌── 函数 def _history_media_paths_for_session()
    # │  作用：返回 media 路径已 delivered 在 prior 轮次的 这个会话。
    # │  参数：self, session_key: str
    # │  返回：Optional[set]
    # └──────────────────────────────────────────────────────────────
    def _history_media_paths_for_session(self, session_key: str) -> Optional[set]:
        """Return media paths already delivered in prior turns of this session
        (MEDIA: tags / image_generate payloads), so an echoed old tag isn't re-sent."""
        store = getattr(self, "_session_store", None)
        if not store:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 TranscriptReadError、Exception，出错走兜底而不是崩溃。
        try:
            # Transcripts are keyed by session_id; map via the routing index, else the raw key.
            peek = getattr(store, "peek_session_id", None)
            session_id = peek(session_key) if callable(peek) else None
            transcript = store.load_transcript(session_id or session_key)
        except TranscriptReadError:
            logger.warning(
                "Transcript read failed for session %s; media dedup runs "
                "with no history this turn (#100788)", session_key,
            )
            # ⤷ 出口 return：返回 None
            return None
        except Exception:
            return None
        if not transcript:
            # ⤷ 出口 return：返回 None
            return None
        # Exclude the CURRENT TURN (from the last user message on): rows persist as produced, so a
        # text_to_speech media_tag would otherwise dedup away its own attachment.
        history = list(transcript)
        last_user_idx = next(
            (i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
        if last_user_idx is not None:
            history = history[:last_user_idx]
        else:
            # No user row (unusual store shape): at least drop the trailing reply.
            last_reply = next((msg for msg in reversed(history) if msg.get("role") == "assistant"), None)
            if last_reply is not None:
                history.remove(last_reply)
        if not history:
            # ⤷ 出口 return：返回 None
            return None
        from gateway.run import _collect_history_media_paths  # lazy: gateway.run imports us
        # ⤷ 出口 return：返回 _collect_history_media_paths(history)
        return _collect_history_media_paths(history)

    # ┌── 异步函数（协程） def _bounded_history_media_paths_for_session()
    # │  作用：运行尽最大努力 history lookup 在 bounded isolated daemon 线程.
    # │  参数：self, session_key: str
    # │  返回：Optional[set]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _bounded_history_media_paths_for_session(self, session_key: str) -> Optional[set]:
        """Run best-effort history lookup in a bounded isolated daemon thread."""
        def _fail_open(reason: str, *, exc_info: bool = False) -> None:
            logger.warning(
                "[%s] " + reason + " %s; delivering bare local file path(s) without history dedup",
                self.name, session_key, exc_info=exc_info)
        admission = _HISTORY_MEDIA_LOOKUP_ADMISSION
        if not admission.acquire(blocking=False):
            _fail_open("Media-delivery history lookup capacity exhausted for")
            # ⤷ 出口 return：返回 None
            return None
        loop = asyncio.get_running_loop()
        result_future = loop.create_future()

        def _publish_result(result=None, error=None):
            if not result_future.done():
                (result_future.set_exception(error) if error is not None
                 else result_future.set_result(result))

        def _worker():
            result, error = None, None
            # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
            try:
                result = self._history_media_paths_for_session(session_key)
            except BaseException as exc:
                error = exc
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                with contextlib.suppress(RuntimeError):  # loop already closed (gateway shutdown)
                    loop.call_soon_threadsafe(_publish_result, result, error)
            finally:
                admission.release()
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            threading.Thread(target=_worker, name="media-history-lookup", daemon=True).start()
        except Exception:
            # start() failed (thread exhaustion): the worker never ran, so release the permit here.
            admission.release()
            _fail_open("Could not start media-delivery history lookup worker for", exc_info=True)
            return None
        # ⤷ 关键分支 try/except：捕获 asyncio.TimeoutError、Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 await asyncio.wait_for(result_future, timeout=_HISTORY_MEDIA…
            return await asyncio.wait_for(result_future, timeout=_HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            _fail_open("Timed out loading media-delivery history for")
            return None
        except Exception:
            # Best-effort/fail-open: never let a lookup failure kill media delivery.
            _fail_open("Media-delivery history lookup failed for", exc_info=True)
            return None

    # ┌── 异步函数（协程） def connect()
    # │  作用：Connect 且 start receiving; True 在…上 success. ``is_reconnect``: reconnect watcher 是
    # │  参数：self, is_reconnect: bool = ...
    # │  返回：bool
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect and start receiving; True on success. ``is_reconnect``: the reconnect watcher is
        re-establishing a dropped platform — adapters with a server-side update queue (Telegram)
        must preserve it so outage-time messages aren't discarded."""

    # ┌── 异步函数（协程） def disconnect()
    # │  作用：Disconnect 从平台.
    # │  参数：self
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""

    # ┌── 异步函数（协程） def send()
    # │  作用：Send ``内容`` (可以是 markdown) chat; 返回 SendResult 带消息 id.
    # │  参数：self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send ``content`` (may be markdown) to a chat; returns SendResult with message id."""

    # Surfaces needing an explicit finalize edit (DingTalk AI Cards): the consumer never skips it.
    REQUIRES_EDIT_FINALIZE: bool = False

    # ┌── 异步函数（协程） def create_handoff_thread()
    # │  作用：创建 fresh 线程 under ``parent_chat_id`` 用于 CLI→平台会话交接; 其 id
    # │  参数：self, parent_chat_id: str, name: str
    # │  返回：Optional[str]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a fresh thread under ``parent_chat_id`` for a CLI→platform session handoff; its id
        as str, or None when unsupported/failed (the watcher then uses ``parent_chat_id``
        directly)."""
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 异步函数（协程） def edit_message()
    # │  作用：Edit sent 消息 (optional: success=False makes callers send anew). ``finalize`` marks
    # │  参数：self, chat_id: str, message_id: str, content: str, finalize: bool = ...
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        """Edit a sent message (optional: success=False makes callers send anew). ``finalize`` marks
        the last edit of a streamed response; surfaces with a distinct "in progress" state (DingTalk
        AI Cards) close the message on it and set ``REQUIRES_EDIT_FINALIZE`` so it's routed even
        when content is unchanged."""
        # ⤷ 出口 return：返回 SendResult(success=False, error='Not supported')
        return SendResult(success=False, error="Not supported")

    # ┌── 异步函数（协程） def delete_message()
    # │  作用：Delete sent 消息; True 在…上 success (平台不 deletion API 返回 False 且
    # │  参数：self, chat_id: str, message_id: str
    # │  返回：bool
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a sent message; True on success (platforms without a deletion API return False and
        callers leave it). Used by the stream consumer's fresh-final cleanup to remove stale
        previews.

        Used by the stream consumer's fresh-final cleanup path (see openclaw/openclaw#72038) to remove
        long-lived preview messages after sending the completed reply as a fresh message so the platform's
        visible timestamp reflects completion time.
        """
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def _get_ephemeral_system_ttl_default()
    # │  作用：默认值 :类:`EphemeralReply` TTL 从 ``展示.ephemeral_system_ttl``
    # │  参数：self
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def _get_ephemeral_system_ttl_default(self) -> int:
        """Default :class:`EphemeralReply` TTL from ``display.ephemeral_system_ttl``
        (``0`` = no auto-delete); non-fatal if config is unreadable."""
        # ⤷ 出口 return：返回 _or_default(lambda: int(_config_section('display').get('ephe…
        return _or_default(
            lambda: int(_config_section("display").get("ephemeral_system_ttl", 0)), 0)

    # ┌── 函数 def _schedule_ephemeral_delete()
    # │  作用：Spawn detached task 那个 deletes ``message_id`` 在…之后 ``ttl_seconds``; 尽最大努力
    # │  参数：self, chat_id: str, message_id: str, ttl_seconds: int
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _schedule_ephemeral_delete(self, chat_id: str, message_id: str, ttl_seconds: int) -> None:
        """Spawn a detached task that deletes ``message_id`` after ``ttl_seconds``; best-effort
        (gateway restart, permission denied, Telegram 48h window) swallowed at debug level."""
        async def _run_delete() -> None:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.sleep(max(1, int(ttl_seconds)))
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await self.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception as e:
                logger.debug("[%s] Ephemeral delete failed for %s/%s: %s", self.name, chat_id, message_id, e)
        coro = _run_delete()
        # ⤷ 关键分支 try/except：捕获 RuntimeError，出错走兜底而不是崩溃。
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop (unit tests): close the coroutine to avoid a never-awaited warning.
            coro.close()

    # ── ``_format_exec_approval`` templates; adapters override to keep historical wording.
    _EA_HEADER: str = "⚠️ Command Approval Required\n\n"
    _EA_CODE_OPEN: str = "```\n"
    _EA_CODE_CLOSE: str = "\n```\n"
    _EA_REASON_LABEL: str = "Reason: "
    _EA_SMART_DENY_LINE: str = "\n\nSmart DENY: owner override applies to this one operation only."
    _EA_CMD_BUDGET: int = 3000
    _EA_REASON_BUDGET: int = 0  # 0 = the reason is never truncated

    # ┌── 函数 def _truncate_preview()
    # │  作用：Truncate ``文本`` ``预算`` chars, 追加 ``suffix`` 当 cut.
    # │  参数：text: str, budget: int, suffix: str = '...'
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _truncate_preview(text: str, budget: int, suffix: str = "...") -> str:
        """Truncate ``text`` to ``budget`` chars, appending ``suffix`` when cut."""
        text = str(text or "")
        # ⤷ 出口 return：返回 text[:budget] + suffix if len(text) > budget else text
        return text[:budget] + suffix if len(text) > budget else text

    # ┌── 函数 def _ea_escape()
    # │  作用：Escape hook 用于命令 preview/reason; HTML-模式平台 (Telegram) 覆盖值.
    # │  参数：self, text: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _ea_escape(self, text: str) -> str:
        """Escape hook for command preview/reason; HTML-mode platforms (Telegram) override."""
        # ⤷ 出口 return：返回 text
        return text

    # ┌── 函数 def _exec_approval_cmd_budget()
    # │  作用：Chars 的命令 preview 那个 fit; 平台带 hard 消息 cap compute 它.
    # │  参数：self, description: str, smart_denied: bool
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def _exec_approval_cmd_budget(self, description: str, smart_denied: bool) -> int:
        """Chars of command preview that fit; platforms with a hard message cap compute it."""
        # ⤷ 出口 return：返回 self._EA_CMD_BUDGET
        return self._EA_CMD_BUDGET

    # ┌── 函数 def _format_exec_approval()
    # │  作用：Shared exec-approval 提示词文本: header + fenced (truncated) 命令 + reason,
    # │  参数：self, command: str, description: str = 'dangerous command', smart_denied: bool = False
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _format_exec_approval(
        self, command: str, description: str = "dangerous command", smart_denied: bool = False) -> str:
        """Shared exec-approval prompt text: header + fenced (truncated) command + reason,
        plus the smart-deny line. Buttons/trailing instructions stay platform-local."""
        if self._EA_REASON_BUDGET:
            description = self._truncate_preview(str(description or ""), self._EA_REASON_BUDGET)
        cmd_preview = self._truncate_preview(
            str(command or ""), self._exec_approval_cmd_budget(description, smart_denied))
        text = (f"{self._EA_HEADER}"
                f"{self._EA_CODE_OPEN}{self._ea_escape(cmd_preview)}{self._EA_CODE_CLOSE}"
                f"{self._EA_REASON_LABEL}{self._ea_escape(description)}")
        # ⤷ 出口 return：返回 text + self._EA_SMART_DENY_LINE if smart_denied else text
        return text + self._EA_SMART_DENY_LINE if smart_denied else text

    # ── Exec-approval prompt (template method). The choice set is one rule for every button
    # surface — three separate "same fix × N adapters" commits motivated lifting it here.
    _EA_ACTION_LABELS: Dict[str, str] = {
        "once": "Allow Once", "session": "Allow Session", "always": "Always Allow", "deny": "Deny"}
    _EA_ACTION_STYLES: Dict[str, str] = {"once": "primary", "deny": "danger"}

    # ┌── 函数 def _exec_approval_actions()
    # │  作用：``(label, choice, style)`` rows 用于 approval buttons. smart deny 是 owner
    # │  参数：self, allow_permanent: bool, allow_session: bool, smart_denied: bool
    # │  返回：List[Tuple[str, str, str]]
    # └──────────────────────────────────────────────────────────────
    def _exec_approval_actions(
            self, *, allow_permanent: bool, allow_session: bool, smart_denied: bool) -> List[Tuple[str, str, str]]:
        """``(label, choice, style)`` rows for the approval buttons. A smart deny is an owner
        override for one operation only, so it offers neither the session nor the permanent tier;
        the permanent tier is never offered without the session tier."""
        choices = ["once"]
        if not smart_denied and allow_session:
            choices.append("session")
            if allow_permanent:
                choices.append("always")
        choices.append("deny")
        # ⤷ 出口 return：返回 [(self._EA_ACTION_LABELS[c], c, self._EA_ACTION_STYLES.get(c…
        return [(self._EA_ACTION_LABELS[c], c, self._EA_ACTION_STYLES.get(c, "")) for c in choices]

    # ┌── 函数 def supports_exec_approval_buttons()
    # │  作用：当 适配器 renders native approval buttons (覆盖值提示词 hook); 时返回 True。
    # │  参数：cls
    # │  返回：bool
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def supports_exec_approval_buttons(cls) -> bool:
        """True when the adapter renders native approval buttons (overrides the prompt hook);
        the runner otherwise sends the plain-text ``/approve`` prompt."""
        # ⤷ 出口 return：返回 cls._send_exec_approval_prompt is not BasePlatformAdapter._s…
        return cls._send_exec_approval_prompt is not BasePlatformAdapter._send_exec_approval_prompt

    # ┌── 异步函数（协程） def send_exec_approval()
    # │  作用：Interactive exec-approval 提示词; press resolves via
    # │  参数：self, chat_id: str, command: str, session_key: str, description: str = 'dangerous command', metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True  …共 9 个
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Interactive exec-approval prompt; a press resolves via
        ``tools.approval.resolve_gateway_approval``. Text and choice set are shared; adapters
        render them natively in ``_send_exec_approval_prompt``."""
        prompt = ExecApprovalPrompt(
            chat_id=chat_id, session_key=session_key, metadata=metadata, command=str(command or ""),
            description=description, smart_denied=smart_denied,
            text=self._format_exec_approval(command, description, smart_denied),
            actions=self._exec_approval_actions(
                allow_permanent=allow_permanent, allow_session=allow_session, smart_denied=smart_denied))
        # ⤷ 出口 return：返回 await self._send_exec_approval_prompt(prompt)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self._send_exec_approval_prompt(prompt)

    # ┌── 异步函数（协程） def _send_exec_approval_prompt()
    # │  作用：Render ``提示词`` 带平台's native buttons; 默认值有 None.
    # │  参数：self, prompt: 'ExecApprovalPrompt'
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_exec_approval_prompt(self, prompt: "ExecApprovalPrompt") -> SendResult:
        """Render ``prompt`` with the platform's native buttons; the default has none."""
        # ⤷ 出口 return：返回 SendResult(success=False, error='Not supported')
        return SendResult(success=False, error="Not supported")

    # ┌── 函数 def _format_choice_page()
    # │  作用：Shared picker pagination: clamp ``page``, slice ``选项`` -> ``(page_options, meta)``
    # │  参数：options: list, page: int, per_page: int
    # │  返回：'tuple[list, Dict[str, Any]]'
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _format_choice_page(options: list, page: int, per_page: int) -> "tuple[list, Dict[str, Any]]":
        """Shared picker pagination: clamp ``page``, slice ``options`` -> ``(page_options, meta)``
        with ``page``/``total_pages``/``start``/``end``/``total``/``page_info`` (`` (N–M of T)``,
        empty for one page)."""
        total = len(options)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))
        start, end = page * per_page, min(page * per_page + per_page, total)
        page_info = f" ({start + 1}–{end} of {total})" if total_pages > 1 else ""
        meta: Dict[str, Any] = {"page": page, "total_pages": total_pages, "start": start,
                   "end": end, "total": total, "page_info": page_info}
        # ⤷ 出口 return：返回 (options[start:end], meta)
        return options[start:end], meta

    # ┌── 异步函数（协程） def send_slash_confirm()
    # │  作用：Three-选项 slash-命令 confirmation (e.g. ``/reload-mcp``). Button adapters render
    # │  参数：self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Three-option slash-command confirmation (e.g. ``/reload-mcp``). Button adapters render
        Approve Once / Always Approve / Cancel and MUST resolve via
        ``GatewayRunner._resolve_slash_confirm(confirm_id, "once"|"always"|"cancel")``. Default (not
        supported) falls through to the gateway text fallback
        (``/approve``/``/always``/``/cancel``)."""
        # ⤷ 出口 return：返回 SendResult(success=False, error='Not supported')
        return SendResult(success=False, error="Not supported")

    # ┌── 异步函数（协程） def send_clarify()
    # │  作用：Clarify 提示词; button-capable adapters 应当覆盖值. Multiple choice (``choices``):
    # │  参数：self, chat_id: str, question: str, choices: Optional[list], clarify_id: str, session_key: str, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Clarify prompt; button-capable adapters SHOULD override. Multiple choice (``choices``):
        one button per choice plus "Other"; callbacks MUST resolve via
        ``tools.clarify_gateway.resolve_gateway_clarify(clarify_id, response)``, "Other" calls
        ``mark_awaiting_text(clarify_id)``. Open-ended: send the question as text (the gateway
        text-intercept resolves the next message). Default: numbered list +
        ``mark_awaiting_text``."""
        if choices:
            # Multi-select flag lives on the pending entry (signature stays adapter-compatible).
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                from tools import clarify_gateway as _cg
                with _cg._lock:
                    _is_multi = bool(getattr(_cg._entries.get(clarify_id), "multi_select", False))
            except Exception:
                _is_multi = False
            hint = "Reply with the number, the option text, or your own answer."
            if _is_multi:
                hint = ("Multiple selections allowed — reply with the numbers separated by commas "
                        "or spaces (e.g. \"1, 3\"), the option text, or your own answer.")
            numbered = [f"  {i}. {choice}" for i, choice in enumerate(choices, start=1)]
            text = "\n".join([f"❓ {question}", "", *numbered, "", hint])
            # Text fallback: let the gateway intercept capture the typed reply.
            from tools.clarify_gateway import mark_awaiting_text
            mark_awaiting_text(clarify_id)
        else:
            text = f"❓ {question}"
        # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=text, metadata=meta…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send(chat_id=chat_id, content=text, metadata=metadata)

    # ┌── 异步函数（协程） def send_private_notice()
    # │  作用：Send 提示 privately 当平台 supports 它; 默认值是 normal send.
    # │  参数：self, chat_id: str, user_id: Optional[str], content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_private_notice(
        self, chat_id: str, user_id: Optional[str], content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a notice privately when the platform supports it; default is a normal send."""
        # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=content, reply_to=r…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send(chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata)

    # ┌── 异步函数（协程） def send_typing()
    # │  作用：Send typing indicator; ``metadata`` carries 平台上下文 (Slack thread_id).
    # │  参数：self, chat_id: str, metadata = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator; ``metadata`` carries platform context (Slack thread_id)."""

    # ┌── 异步函数（协程） def stop_typing()
    # │  作用：Stop persistent typing indicator; 覆盖值何处 typing 运行作为循环.
    # │  参数：self, chat_id: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def stop_typing(self, chat_id: str) -> None:
        """Stop a persistent typing indicator; override where typing runs as a loop."""

    # ┌── 函数 def _accepts_kwarg()
    # │  作用：是否 ``fn``'s signature takes keyword ``名称`` (``var_kw``: ``**kwargs`` 也。
    # │  参数：fn: Callable, name: str, var_kw: bool, unknown: bool
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _accepts_kwarg(fn: Callable, name: str, *, var_kw: bool, unknown: bool) -> bool:
        """Whether ``fn``'s signature takes keyword ``name`` (``var_kw``: a ``**kwargs`` also
        counts); ``unknown`` when the signature can't be introspected."""
        # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            # ⤷ 出口 return：返回 unknown
            return unknown
        # ⤷ 出口 return：返回 name in params or (var_kw and any((p.kind is inspect.Paramet…
        return name in params or (var_kw and any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))

    # ┌── 异步函数（协程） def _stop_typing_with_metadata()
    # │  作用：Stop typing, forwarding ``metadata`` 仅若 ``stop_typing`` 接受它 (Slack AI
    # │  参数：self, chat_id: str, metadata = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _stop_typing_with_metadata(self, chat_id: str, metadata=None) -> None:
        """Stop typing, forwarding ``metadata`` only if ``stop_typing`` accepts it (Slack AI
        status is per thread, so dropping metadata could clear a sibling thread; introspecting
        keeps legacy ``stop_typing(chat_id)`` adapters working)."""
        if metadata and self._accepts_kwarg(
                self.stop_typing, "metadata", var_kw=True, unknown=False):
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self.stop_typing(chat_id, metadata=metadata)
            return
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self.stop_typing(chat_id)

    # ┌── 异步函数（协程） def send_multiple_images()
    # │  作用：Send ``(URL, alt)`` images (``http(s)://`` 或 ``文件://``) one 由 one (GIFs via
    # │  参数：self, chat_id: str, images: List[Tuple[str, str]], metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_multiple_images(
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> SendResult:
        """Send ``(url, alt)`` images (``http(s)://`` or ``file://``) one by one (GIFs via
        ``send_animation``, local files via ``send_image_file``); override to bundle natively
        (Signal). Returns success when at least one image was delivered — the outcome
        the turn-level delivery tracker records; every override must return the same
        aggregate, or a media-only turn on that platform reports FAILURE (#106153)."""
        from urllib.parse import unquote as _unquote
        delivered = False
        for image_url, alt_text in images:
            if human_delay > 0:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.sleep(human_delay)
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                logger.info("[%s] Sending image: %s (alt=%s)", self.name,
                            safe_url_for_log(image_url), alt_text[:30] if alt_text else "")
                if image_url.startswith("file://"):
                    sender, url_kw = self.send_image_file, {"image_path": _unquote(image_url[7:])}
                elif self._is_animation_url(image_url):
                    sender, url_kw = self.send_animation, {"animation_url": image_url}
                else:
                    sender, url_kw = self.send_image, {"image_url": image_url}
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                img_result = await sender(
                    chat_id=chat_id, **url_kw, caption=alt_text or None, metadata=metadata)
                if not img_result.success:
                    logger.error("[%s] Failed to send image: %s", self.name, img_result.error)
                else:
                    delivered = True
            except Exception as img_err:
                logger.error("[%s] Error sending image: %s", self.name, img_err, exc_info=True)
        if not images:
            # ⤷ 出口 return：返回 SendResult(success=False, error='no images to send')
            return SendResult(success=False, error="no images to send")
        # ⤷ 出口 return：返回 SendResult(success=delivered, error=None if delivered else '…
        return SendResult(
            success=delivered,
            error=None if delivered else "all images failed to send")

    # ┌── 异步函数（协程） def send_image()
    # │  作用：Send image natively; 默认值 falls back sending URL 作为文本.
    # │  参数：self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image natively; default falls back to sending the URL as text."""
        text = f"{caption}\n{image_url}" if caption else image_url
        # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=text, reply_to=repl…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    # ┌── 异步函数（协程） def send_animation()
    # │  作用：Send GIF 作为 native animation (auto-plays inline); 默认值 falls back
    # │  参数：self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a GIF as a native animation (auto-plays inline); default falls back to
        send_image."""
        # ⤷ 出口 return：返回 await self.send_image(chat_id=chat_id, image_url=animation_u…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send_image(
            chat_id=chat_id, image_url=animation_url, caption=caption, reply_to=reply_to, metadata=metadata)

    # ┌── 函数 def _is_animation_url()
    # │  作用：检查若 URL points animated GIF (vs static image).
    # │  参数：url: str
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_animation_url(url: str) -> bool:
        """Check if a URL points to an animated GIF (vs a static image)."""
        # ⤷ 出口 return：返回 url.lower().split('?')[0].endswith('.gif')
        return url.lower().split('?')[0].endswith('.gif')

    # ┌── 函数 def extract_images()
    # │  作用：Extract ``![alt](URL)`` 且 ``<img src=...>`` image URLs 从响应;
    # │  参数：content: str
    # │  返回：Tuple[List[Tuple[str, str]], str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
        """Extract ``![alt](url)`` and ``<img src=...>`` image URLs from a response;
        returns ``([(url, alt_text), ...], content with those tags removed)``."""
        md_pattern = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
        # <img src="url"> / <img src="url"></img> / <img src="url"/>
        html_pattern = r'<img\s+src=["\']?(https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
        # Only extract URLs that look like actual images.
        markers = ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn',
                   'replicate.delivery')
        images = [(m.group(2), m.group(1)) for m in re.finditer(md_pattern, content)
                  if any(m.group(2).lower().endswith(ext) or ext in m.group(2).lower()
                         for ext in markers)]
        images.extend((match.group(1), "") for match in re.finditer(html_pattern, content))
        if not images:
            # ⤷ 出口 return：返回 (images, content)
            return images, content
        # Remove only the tags we extracted, not every markdown image.
        extracted_urls = {url for url, _ in images}

        def _remove_if_extracted(match):
            url = match.group(2) if match.lastindex >= 2 else match.group(1)
            # ⤷ 出口 return：返回 '' if url in extracted_urls else match.group(0)
            return '' if url in extracted_urls else match.group(0)
        cleaned = content
        for pattern in (md_pattern, html_pattern):
            cleaned = re.sub(pattern, _remove_if_extracted, cleaned)
        # ⤷ 出口 return：返回 (images, re.sub('\\n{3,}', '\n\n', cleaned).strip())
        return images, re.sub(r'\n{3,}', '\n\n', cleaned).strip()  # leftover blank lines

    # ┌── 异步函数（协程） def send_voice()
    # │  作用：Send audio 作为原生语音消息 (Telegram bubble / Discord attachment).
    # │  参数：self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send audio as a native voice message (Telegram bubble / Discord attachment).
        Default: friendly failure notice."""
        # ⤷ 出口 return：返回 await self._send_media_fallback_notice('send_voice', 'audio'…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self._send_media_fallback_notice(
            "send_voice", "audio", audio_path, chat_id, caption, reply_to, metadata)

    # ┌── 异步函数（协程） def _send_media_fallback_notice()
    # │  作用：Shared 默认值用于 send_voice/send_video/send_document/send_image_file. local 路径是
    # │  参数：self, method: str, kind: str, path: str, chat_id: str, caption: Optional[str], reply_to: Optional[str], metadata: Optional[Dict[str, Any]]  …共 9 个
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_media_fallback_notice(
        self, method: str, kind: str, path: str, chat_id: str, caption: Optional[str],
        reply_to: Optional[str], metadata: Optional[Dict[str, Any]], *, file_name: Optional[str] = None,
    ) -> SendResult:
        """Shared default for send_voice/send_video/send_document/send_image_file. The local path is
        logged but NEVER echoed into chat (host layout leak); only the caller's ``file_name`` is
        shown."""
        logger.warning("[%s] %s fallback: native %s send unavailable for %s", self.name, method, kind, path)
        text = _media_failure_text(kind, file_name)
        text = f"{caption}\n{text}" if caption else text
        # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=text, reply_to=repl…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    # ┌── 函数 def prepare_tts_text()
    # │  作用：Chat Markdown -> transcript-like spoken script (推理块已移除,
    # │  参数：self, text: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def prepare_tts_text(self, text: str) -> str:
        """Chat Markdown -> transcript-like spoken script (reasoning blocks removed,
        headings/bullets flattened, units expanded). Chunking and delivery limits are the TTS tool's
        job."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.tts_text_normalize import prepare_spoken_text
            # ⤷ 出口 return：返回 prepare_spoken_text(text, max_chars=None)
            return prepare_spoken_text(text, max_chars=None)
        except Exception:
            # Keep auto-TTS best-effort if the normalizer ever fails.
            text = re.sub(r'<think[\s>].*?</think>', ' ', text, flags=re.DOTALL)
            # ⤷ 出口 return：返回 re.sub('[*_`#\\[\\]()]', '', text).strip()
            return re.sub(r'[*_`#\[\]()]', '', text).strip()

    # ┌── 异步函数（协程） def play_tts()
    # │  作用：Play auto-TTS audio; 覆盖值用于 invisible playback (Web UI). 默认值: send_voice.
    # │  参数：self, chat_id: str, audio_path: str, **kwargs
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Play auto-TTS audio; override for invisible playback (Web UI). Default: send_voice."""
        # ⤷ 出口 return：返回 await self.send_voice(chat_id=chat_id, audio_path=audio_path…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    # ── Streaming TTS contract: voice adapters accept PCM chunks while the LLM generates.
    # Defaults report "unsupported" (whole-file fallback).

    # ------------------------------------------------------------------ Streaming TTS adapter contract
    # (#60671) ------------------------------------------------------------------ Voice-capable adapters
    # (LiveKit, Discord voice, …) override these to accept PCM audio chunks while the LLM is still
    # generating. The default implementations report "unsupported" so existing adapters are
    # source-compatible and keep the whole-file auto-TTS fallback.
    # ┌── 函数 def supports_streaming_tts()
    # │  作用：当 这个适配器可以接受 streaming PCM 用于 *chat_id*. 时返回 True。
    # │  参数：self, chat_id: str, audio_format: AudioFormat
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        """Return True when this adapter can accept streaming PCM for *chat_id*."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 异步函数（协程） def begin_streaming_tts()
    # │  作用：打开 streaming-audio 会话; 返回 opaque handle 用于
    # │  参数：self, chat_id: str, audio_format: AudioFormat, metadata: Optional[Dict[str, Any]] = None
    # │  返回：Optional[StreamingTTSHandle]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def begin_streaming_tts(
        self, chat_id: str, audio_format: AudioFormat, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTTSHandle]:
        """Open a streaming-audio session; returns an opaque handle for the
        write/finish/abort calls, or ``None`` to decline (whole-file fallback)."""
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 异步函数（协程） def write_streaming_tts()
    # │  作用：写入 one PCM chunk 适配器's outbound audio track.
    # │  参数：self, handle: StreamingTTSHandle, chunk: bytes
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        """Write one PCM chunk to the adapter's outbound audio track."""

    # ┌── 异步函数（协程） def finish_streaming_tts()
    # │  作用：信号 normal end 的 audio stream.
    # │  参数：self, handle: StreamingTTSHandle, interrupted: bool = ...
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *, interrupted: bool = False) -> None:
        """Signal normal end of the audio stream."""

    # ┌── 异步函数（协程） def abort_streaming_tts()
    # │  作用：Abort stream due 错误或 cancellation. 必须是 idempotent: late producer
    # │  参数：self, handle: StreamingTTSHandle, error: Optional[str] = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def abort_streaming_tts(self, handle: StreamingTTSHandle, error: Optional[str] = None) -> None:
        """Abort the stream due to an error or cancellation. Must be idempotent: late producer
        chunks after abort are silently dropped, not raised. Restores state to "not streaming"."""

    # ┌── 函数 def _streaming_tts_turn_key()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, session_key: str | None, turn_marker: Any = None, event: Any = ...
    # │  返回：str | None
    # └──────────────────────────────────────────────────────────────
    def _streaming_tts_turn_key(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> str | None:
        # ⤷ 出口 return：返回 streaming_tts_turn_key(session_key, turn_marker, event=event…
        return streaming_tts_turn_key(session_key, turn_marker, event=event)

    # ┌── 函数 def _mark_streaming_tts_completed_turn()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, session_key: str | None, turn_marker: Any = None, event: Any = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _mark_streaming_tts_completed_turn(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> None:
        turn_key = self._streaming_tts_turn_key(session_key, turn_marker, event=event)
        if turn_key is not None:
            _lazy_attr(self, "_streaming_tts_completed_turns", set).add(turn_key)

    # ┌── 函数 def _streaming_tts_turn_completed()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, session_key: str | None, turn_marker: Any = None, event: Any = ...
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _streaming_tts_turn_completed(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> bool:
        # ⤷ 出口 return：返回 streaming_tts_should_skip_whole_file(getattr(self, '_streami…
        return streaming_tts_should_skip_whole_file(
            getattr(self, "_streaming_tts_completed_turns", set()), session_key, turn_marker, event=event)

    # ┌── 异步函数（协程） def send_video()
    # │  作用：Send video natively (inline playable). 默认值: friendly failure 提示.
    # │  参数：self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a video natively (inline playable). Default: friendly failure notice."""
        # ⤷ 出口 return：返回 await self._send_media_fallback_notice('send_video', 'video'…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self._send_media_fallback_notice(
            "send_video", "video", video_path, chat_id, caption, reply_to, metadata)

    # ┌── 异步函数（协程） def send_document()
    # │  作用：Send document/文件 natively. 默认值: friendly failure 提示.
    # │  参数：self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a document/file natively. Default: friendly failure notice."""
        # ⤷ 出口 return：返回 await self._send_media_fallback_notice('send_document', 'fil…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self._send_media_fallback_notice(
            "send_document", "file", file_path, chat_id, caption, reply_to, metadata, file_name=file_name)

    # ┌── 异步函数（协程） def _notify_media_delivery_failure()
    # │  作用：User-visible 提示当 MEDIA attachment upload failed: tag 是
    # │  参数：self, chat_id: str, media_path: str, is_voice: bool = ..., metadata: Optional[Dict[str, Any]] = ...
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _notify_media_delivery_failure(
        self, chat_id: str, media_path: str, *, is_voice: bool = False,
        metadata: Optional[Dict[str, Any]] = None) -> None:
        """User-visible notice when a MEDIA attachment upload failed: the tag was
        already stripped from the text, so silence would be a silent drop.

        The non-streaming dispatch loop strips ``MEDIA:`` tags before sending attachments. When the
        subsequent upload returns ``success=False`` (for example Discord accepted the message but attached
        nothing), the user must see a failure notice instead of a silent drop (#66797).
        """
        ext = Path(media_path).suffix.lower()
        if is_voice or should_send_media_as_audio(self.platform, ext, is_voice=is_voice):
            text = _media_failure_text("audio")
        elif ext in _VIDEO_EXTS:
            text = _media_failure_text("video")
        else:
            text = _media_failure_text("file", os.path.basename(media_path))
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            notice = await self.send(chat_id=chat_id, content=text, metadata=metadata)
            problem = None if notice.success else notice.error
        except Exception as notify_err:
            problem = notify_err
        if problem is not None:
            logger.debug("[%s] Could not send media-delivery-failure notice: %s", self.name, problem)

    # ┌── 异步函数（协程） def send_image_file()
    # │  作用：Send local image 文件 natively (send_image takes URL). 默认值: friendly 提示.
    # │  参数：self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs
    # │  返回：SendResult
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local image file natively (send_image takes a URL). Default: friendly notice."""
        # ⤷ 出口 return：返回 await self._send_media_fallback_notice('send_image_file', 'i…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self._send_media_fallback_notice(
            "send_image_file", "image", image_path, chat_id, caption, reply_to, metadata)

    # ┌── 函数 def validate_media_delivery_path()
    # │  作用：返回 resolved 路径若 它是 safe 用于 native attachment upload.。
    # │  参数：path: str, session_key: str = ''
    # │  返回：Optional[str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def validate_media_delivery_path(path: str, session_key: str = "") -> Optional[str]:
        """Return a resolved path if it is safe for native attachment upload."""
        # ⤷ 出口 return：返回 validate_media_delivery_path(path, session_key=session_key)
        return validate_media_delivery_path(path, session_key=session_key)

    # ┌── 函数 def filter_media_delivery_paths()
    # │  作用：丢弃 unsafe MEDIA 路径且 归一化已接受路径.
    # │  参数：media_files, session_key: str = ''
    # │  返回：List[Tuple[str, bool]]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def filter_media_delivery_paths(media_files, session_key: str = "") -> List[Tuple[str, bool]]:
        """Drop unsafe MEDIA paths and normalize accepted paths."""
        # ⤷ 出口 return：返回 [(safe_path, bool(is_voice)) for media_path, is_voice in med…
        return [
            (safe_path, bool(is_voice)) for media_path, is_voice in media_files or []
            if (safe_path := _validated_delivery_path(media_path, session_key, "MEDIA directive path"))]

    # ┌── 函数 def filter_local_delivery_paths()
    # │  作用：丢弃 unsafe bare local 文件路径且 归一化已接受路径.
    # │  参数：file_paths, session_key: str = ''
    # │  返回：List[str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def filter_local_delivery_paths(file_paths, session_key: str = "") -> List[str]:
        """Drop unsafe bare local file paths and normalize accepted paths."""
        safe_paths = (_validated_delivery_path(p, session_key, "local file path") for p in file_paths or [])
        # ⤷ 出口 return：返回 [p for p in safe_paths if p]
        return [p for p in safe_paths if p]

    # ┌── 函数 def _mask_protected_spans()
    # │  作用：Blank fenced code, inline code 且 blockquotes (length-preserving 从而 regex offsets stay
    # │  参数：content: str
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _mask_protected_spans(content: str) -> str:
        """Blank fenced code, inline code and blockquotes (length-preserving so regex offsets stay
        valid) against MEDIA: false positives; backtick-quoted paths inside MEDIA: tags stay
        scannable."""
        spans: list = [m.span() for m in _FENCED_CODE_RE.finditer(content)]
        for m in _INLINE_CODE_RE.finditer(content):
            start = m.start()
            if re.search(r'MEDIA:\s*$', content[max(0, start - 20):start]):
                continue  # This is a MEDIA path quote, not inline code
            # A whole tag in inline code (`MEDIA:/path.csv`) is a real directive (models format
            # paths as code): deliver IF it validates; non-existent examples stay masked.
            # See #35695.
            inner = m.group(0)[1:-1].strip()
            if inner.upper().startswith("MEDIA:"):
                candidate = _normalize_media_tag_path(inner[6:])
                if candidate and validate_media_delivery_path(candidate):
                    continue  # Real deliverable tag in inline code — keep it scannable
            spans.append((start, m.end()))
        spans.extend(m.span() for m in re.finditer(r'^>.*$', content, re.MULTILINE))
        # ⤷ 出口 return：返回 _blank_spans(content, spans)
        return _blank_spans(content, spans)

    # ┌── 函数 def _mask_json_string_media()
    # │  作用：Blank ``MEDIA:<bare-路径>`` tags inside JSON 字符串 *值* (stored 工具-结果文本
    # │  参数：content: str
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _mask_json_string_media(content: str) -> str:
        """Blank ``MEDIA:<bare-path>`` tags inside JSON string *values* (stored tool-result text
        like ``{"result": "MEDIA:/x/stale.png"}``) so they are never re-delivered. Only
        value-context strings (``:,{[`` before the ``"``) and bare paths (``/``, ``~/``, ``X:\\``)
        count; ``MEDIA:"..."`` quoted tags and line-start/prose tags are untouched. Offsets
        preserved.

        Here the ``MEDIA:`` is part of stored text, not an outbound directive, but the bare-path branch of
        ``MEDIA_TAG_CLEANUP_RE`` would still match it and re-deliver a stale file. (Regression report
        #34375.)
        """
        if '"' not in content or "MEDIA:" not in content:
            # ⤷ 出口 return：返回 content
            return content
        # Value-context string: quote preceded by : , { or [; escape-aware body to the closing
        # quote.
        spans = [
            m.span(1) for m in re.finditer(r'(?<=[:,{\[])\s*"((?:[^"\\\n]|\\.)*)"', content)
            if re.search(r'MEDIA:\s*(?:~/|/|[A-Za-z]:[/\\])', m.group(1))]
        # ⤷ 出口 return：返回 _blank_spans(content, spans)
        return _blank_spans(content, spans)

    # ┌── 函数 def extract_media()
    # │  作用：Extract ``MEDIA:<路径>`` tags 且 strip ``[[audio_as_voice]]`` / ``[[as_document]]`` ->
    # │  参数：content: str
    # │  返回：Tuple[List[Tuple[str, bool]], str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def extract_media(content: str) -> Tuple[List[Tuple[str, bool]], str]:
        """Extract ``MEDIA:<path>`` tags and strip ``[[audio_as_voice]]`` / ``[[as_document]]`` ->
        ``([(path, is_voice), ...], cleaned)``. Both directives are message-global;
        ``[[as_document]]`` (unmodified sendDocument for large images) is detected by dispatch sites
        on the ORIGINAL response and only stripped here."""
        media = []
        has_voice_tag = "[[audio_as_voice]]" in content
        cleaned = content.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
        # Scan a masked copy so example/stored MEDIA paths (code, quotes, JSON values) are never
        # delivered; dedupe on the expanded path so a file referenced twice uploads once.
        scan_content = _mask_media_scan_text(content)
        # - code blocks / inline code / blockquotes hold prose examples (#35695) - serialized JSON string
        #   values hold stored tool-result text (#34375) Both maskers are offset-preserving (chars ->
        #   spaces) so match offsets stay valid; chaining them masks the union of both protected regions.
        # Dedupe on the expanded path (first occurrence wins) so the same file referenced twice in one
        # response — e.g. a MEDIA tag inline AND in a summary footer — is uploaded once, not twice (#29131).
        seen_paths: set = set()

        def _add(path: str) -> None:
            # is_voice only for audio: a voice-flagged image would leave the photo batch.
            if path not in seen_paths:
                seen_paths.add(path)
                media.append((path, has_voice_tag and os.path.splitext(path)[1].lower() in _AUDIO_EXTS))
        for match in MEDIA_TAG_CLEANUP_RE.finditer(scan_content):
            path = _normalize_media_tag_path(match.group("path"))
            if path:
                # ⤷ 关键分支 try/except：捕获 (OSError, RuntimeError, ValueError)，出错走兜底而不是崩溃。
                try:
                    _add(os.path.expanduser(path))
                except (OSError, RuntimeError, ValueError):
                    continue  # crafted ~\x00 path: skip it, keep the rest
        for _, safe_path, _ in _extensionless_media_matches(scan_content):
            _add(safe_path)
        # Locate tag spans on a masked copy, delete them from the unmasked text (protected spans
        # survive).
        if media:
            spans = _real_media_tag_spans(_mask_media_scan_text(cleaned))
            if spans:
                cleaned = re.sub(r'\n{3,}', '\n\n', _delete_spans(cleaned, spans)).strip()
        # ⤷ 出口 return：返回 (media, cleaned)
        return media, cleaned

    # ┌── 函数 def strip_media_directives_for_display()
    # │  作用：Strip MEDIA: directives 从 streamed/展示文本. Known-extension tags 是
    # │  参数：text: str
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def strip_media_directives_for_display(text: str) -> str:
        """Strip MEDIA: directives from streamed/display text. Known-extension tags are
        removed unconditionally (as ``MEDIA_TAG_CLEANUP_RE``); extension-less tags only when
        ``validate_media_delivery_path`` accepts the path, so undeliverable paths stay visible."""
        if not _has_media_directives(text):
            # ⤷ 出口 return：返回 text
            return text
        # ⤷ 出口 return：返回 re.sub('\\n{3,}', '\n\n', _strip_media_tag_directives(text))…
        return re.sub(r'\n{3,}', '\n\n', _strip_media_tag_directives(text)).rstrip()

    # ┌── 函数 def extract_local_files()
    # │  作用：Bare local 文件路径 (absolute, ``~/`` 或 drive-letter) 带 deliverable extensions ->
    # │  参数：content: str
    # │  返回：Tuple[List[str], str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def extract_local_files(content: str) -> Tuple[List[str], str]:
        """Bare local file paths (absolute, ``~/`` or drive-letter) with deliverable extensions ->
        ``(expanded_paths, cleaned_text)``. Candidates must exist on disk (URLs / hallucinated paths
        ignored); paths inside fenced or inline code are skipped so code samples are never
        mutilated. Dispatch by type lives in ``gateway/run.py``."""
        ext_part = '|'.join(e.lstrip('.') for e in MEDIA_DELIVERY_EXTS)
        # Lookbehind rejects URL/relative matches (https://…/img.png, ./foo.png).
        # (?<![/:\w.]) prevents matching inside URLs (e.g. https://…/img.png) and relative paths (./foo.png)
        # (?:~/|/)    anchors to absolute or home-relative Unix paths (?:[A-Za-z]:[/\\]) anchors to Windows
        # drive-letter paths (#34632)
        path_re = re.compile(
            r'(?<![/:\w.])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.(?:' + ext_part + r')\b',
            re.IGNORECASE)
        code_spans = _code_spans(content)
        unique: dict = {}  # expanded_path -> raw_match_text, deduped in discovery order
        for match in path_re.finditer(content):
            if any(s <= match.start() < e for s, e in code_spans):
                continue
            raw = match.group(0)
            expanded = os.path.expanduser(raw)
            if os.path.isfile(expanded):
                unique.setdefault(expanded, raw)
            else:
                # Most common reason a promised file never arrives — log the gap.
                logger.info("Skipping bare file path in reply (no file on disk): %s", _log_safe_path(raw))
        if not unique:
            # ⤷ 出口 return：返回 ([], content)
            return [], content
        cleaned = content
        for raw in unique.values():
            cleaned = cleaned.replace(raw, '')
        # ⤷ 出口 return：返回 (list(unique), re.sub('\\n{3,}', '\n\n', cleaned).strip())
        return list(unique), re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # ┌── 异步函数（协程） def _keep_typing()
    # │  作用：Refresh typing indicator 每个 ``interval`` seconds until cancelled (平台 typing
    # │  参数：self, chat_id: str, interval: float = 2.0, metadata = None, stop_event: asyncio.Event | None = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _keep_typing(self, chat_id: str, interval: float = 2.0, metadata=None,
                           stop_event: asyncio.Event | None = None) -> None:
        """Refresh the typing indicator every ``interval`` seconds until cancelled (platform typing
        state expires after ~5s). Chats in ``_typing_paused`` are skipped (approval waits — Slack's
        setStatus disables the compose box). Each ``send_typing`` is bounded by a sub-interval
        timeout so one slow round-trip is abandoned before the next tick, not the bubble lapsing."""
        _send_typing_timeout = max(0.25, min(1.5, interval - 0.25))
        # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError，出错走兜底而不是崩溃。
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                if chat_id not in self._typing_paused:
                    # ⤷ 关键分支 try/except：捕获 asyncio.TimeoutError、Exception，出错走兜底而不是崩溃。
                    try:
                        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                        await asyncio.wait_for(self.send_typing(chat_id, metadata=metadata),
                                               timeout=_send_typing_timeout)
                    except asyncio.TimeoutError:
                        pass  # Slow network — abandon this tick, stay on schedule.
                    except Exception as typing_err:
                        logger.debug("[%s] send_typing error (non-fatal): %s", self.name, typing_err)
                if stop_event is None:
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await asyncio.sleep(interval)
                    continue
                loop = asyncio.get_running_loop()
                deadline = loop.time() + interval
                while not stop_event.is_set():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    # Poll, not wait_for(stop_event.wait()): cancelling that wedges
                    # 3.11/pytest-asyncio.
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await asyncio.sleep(min(0.25, remaining))
        except asyncio.CancelledError:
            pass  # Normal cancellation when handler completes
        finally:
            # A send_typing after an outer stop_typing() may have recreated the platform loop.
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._stop_typing_quietly(chat_id, metadata)
            self._typing_paused.discard(chat_id)
            # getattr-guard: tests build adapters via object.__new__ without _status_text.
            getattr(self, "_status_text", {}).pop(str(chat_id), None)

    # ┌── 异步函数（协程） def _stop_typing_refresh()
    # │  作用：Stop refresh task 且平台 typing 状态作为 one operation.
    # │  参数：self, chat_id: str, typing_task: asyncio.Task | None = None, metadata = ..., timeout: float = ..., stop_attempts: int = ...
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _stop_typing_refresh(
        self, chat_id: str, typing_task: asyncio.Task | None = None, *, metadata=None,
        timeout: float = 0.5, stop_attempts: int = 2) -> None:
        """Stop the refresh task and platform typing state as one operation."""
        self._typing_paused.add(chat_id)
        # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
        try:
            if typing_task is not None and not typing_task.done():
                typing_task.cancel()
                # Slow adapter cleanup must not block delivery/shutdown.
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await asyncio.wait_for(asyncio.shield(typing_task), timeout=timeout)
            for attempt in range(max(1, stop_attempts)):
                if attempt:
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await asyncio.sleep(0)
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await self._stop_typing_quietly(chat_id, metadata)
        finally:
            self._typing_paused.discard(chat_id)

    # ┌── 异步函数（协程） def _stop_typing_quietly()
    # │  作用：尽最大努力平台 stop_typing; 适配器错误是 swallowed.
    # │  参数：self, chat_id: str, metadata = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _stop_typing_quietly(self, chat_id: str, metadata=None) -> None:
        """Best-effort platform stop_typing; adapter errors are swallowed."""
        with contextlib.suppress(Exception):
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._stop_typing_with_metadata(chat_id, metadata)

    # ┌── 函数 def pause_typing_for_chat()
    # │  作用：Pause typing 用于 chat (approval waits); GIL-safe 从同步 Agent 线程.
    # │  参数：self, chat_id: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def pause_typing_for_chat(self, chat_id: str) -> None:
        """Pause typing for a chat (approval waits); GIL-safe from the sync agent thread."""
        self._typing_paused.add(chat_id)

    # ┌── 函数 def resume_typing_for_chat()
    # │  作用：Resume typing indicator 用于 chat 在…之后 approval resolves.
    # │  参数：self, chat_id: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def resume_typing_for_chat(self, chat_id: str) -> None:
        """Resume typing indicator for a chat after approval resolves."""
        self._typing_paused.discard(chat_id)

    # ┌── 异步函数（协程） def interrupt_session_activity()
    # │  作用：信号 active 会话循环 stop 且清除 typing immediately.
    # │  参数：self, session_key: str, chat_id: str, metadata = None
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def interrupt_session_activity(self, session_key: str, chat_id: str, metadata=None) -> None:
        """Signal the active session loop to stop and clear typing immediately."""
        if session_key and session_key in self._active_sessions:
            self._active_sessions[session_key].set()
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self._stop_typing_quietly(chat_id, metadata)

    # ┌── 函数 def register_post_delivery_callback()
    # │  作用：Register deferred 回调 fire 在…之后 main 响应. Same-键 registrations 是
    # │  参数：self, session_key: str, callback: Callable, generation: int | None = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def register_post_delivery_callback(
        self, session_key: str, callback: Callable, *, generation: int | None = None) -> None:
        """Register a deferred callback to fire after the main response. Same-key registrations are
        chained (both fire in order, per-callback exception isolation) so independent features
        coexist; ``generation`` ties it to a gateway run — stale generations never overwrite a
        fresher slot."""
        if not session_key or not callable(callback):
            return
        existing = self._post_delivery_callbacks.get(session_key)
        if existing is not None:
            existing_gen, existing_cb = _split_post_delivery_entry(existing)
            if existing_gen is not None and generation is not None and int(generation) < int(existing_gen):
                return
            # Same-or-newer generation: chain so both fire in registration order.
            if callable(existing_cb) and (
                existing_gen is None or generation is None or int(existing_gen) == int(generation)):
                callback = self._chain_callbacks(existing_cb, callback)
        self._post_delivery_callbacks[session_key] = (
            callback if generation is None else (int(generation), callback))

    # ┌── 函数 def _chain_callbacks()
    # │  作用：异步包装器运行 ``回调`` 在 order 带 per-回调异常 isolation;
    # │  参数：*callbacks
    # │  返回：Callable[[], Awaitable[None]]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _chain_callbacks(*callbacks: Callable) -> Callable[[], Awaitable[None]]:
        """Async wrapper running ``callbacks`` in order with per-callback exception isolation;
        async so coroutines returned by async hooks are awaited, not dropped."""
        async def _chained() -> None:
            for _cb in callbacks:
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    _result = _cb()
                    if inspect.isawaitable(_result):
                        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                        await _result
                except Exception:
                    logger.debug("Post-delivery callback failed", exc_info=True)
        # ⤷ 出口 return：返回 _chained
        return _chained

    # ┌── 函数 def pop_post_delivery_callback()
    # │  作用：Pop deferred 回调, optionally requiring generation ownership.
    # │  参数：self, session_key: str, generation: int | None = ...
    # │  返回：Callable | None
    # └──────────────────────────────────────────────────────────────
    def pop_post_delivery_callback(
        self, session_key: str, *, generation: int | None = None) -> Callable | None:
        """Pop a deferred callback, optionally requiring generation ownership."""
        entry = self._post_delivery_callbacks.get(session_key) if session_key else None
        if entry is None:
            # ⤷ 出口 return：返回 None
            return None
        entry_generation, callback = _split_post_delivery_entry(entry)
        if generation is not None and (entry_generation is None or int(entry_generation) != int(generation)):
            # ⤷ 出口 return：返回 None
            return None
        self._post_delivery_callbacks.pop(session_key, None)
        # ⤷ 出口 return：返回 callback if callable(callback) else None
        return callback if callable(callback) else None

    # ── Processing lifecycle hooks (Discord 👀/✅/❌ reactions). Adapters exposing
    # ``_add_reaction(chat_id, message_id, emoji)`` / ``_remove_reaction(chat_id, message_id)``
    # can just set the emoji attributes; left ``None`` the hook is a no-op.
    _ACK_EMOJI: Optional[str] = None
    _OK_EMOJI: Optional[str] = None
    _FAIL_EMOJI: Optional[str] = None

    # ┌── 异步函数（协程） def on_processing_start()
    # │  作用：Hook called 当 background processing begins.
    # │  参数：self, event: MessageEvent
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def on_processing_start(self, event: MessageEvent) -> None:
        """Hook called when background processing begins."""

    # ┌── 异步函数（协程） def on_processing_complete()
    # │  作用：Hook called 当 background processing completes. 默认值: opt-在 reaction ack — 带
    # │  参数：self, event: MessageEvent, outcome: ProcessingOutcome
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Hook called when background processing completes. Default: opt-in reaction ack — with
        ``_OK_EMOJI``/``_FAIL_EMOJI`` set and ``_add_reaction``/``_remove_reaction`` present, swap
        the in-progress reaction for the outcome one. Remove-then-add is deterministic whether the
        platform replaces or stacks a sender's reactions. CANCELLED leaves it unreacted."""
        if self._OK_EMOJI is None and self._FAIL_EMOJI is None:
            return
        add: Any = getattr(self, "_add_reaction", None)
        remove: Any = getattr(self, "_remove_reaction", None)
        enabled = getattr(self, "_reactions_enabled", None)
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if (not callable(add) or not callable(remove) or (callable(enabled) and not enabled())
                or not chat_id or not message_id):
            return
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await remove(chat_id, message_id)
        emoji = {ProcessingOutcome.SUCCESS: self._OK_EMOJI,
                 ProcessingOutcome.FAILURE: self._FAIL_EMOJI}.get(outcome)
        if emoji:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await add(chat_id, message_id, emoji)

    # ┌── 异步函数（协程） def _run_processing_hook()
    # │  作用：运行 lifecycle hook 不 letting failures break 消息 flow.
    # │  参数：self, hook_name: str, *args, **kwargs
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s hook failed: %s", self.name, hook_name, e)

    # ┌── 函数 def _is_retryable_error()
    # │  作用：返回 True 若错误字符串 looks like transient network failure.。
    # │  参数：error: Optional[str]
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        """Return True if the error string looks like a transient network failure."""
        lowered = (error or "").lower()
        # ⤷ 出口 return：返回 any((pat in lowered for pat in _RETRYABLE_ERROR_PATTERNS))
        return any(pat in lowered for pat in _RETRYABLE_ERROR_PATTERNS)

    # ┌── 函数 def _is_rate_limited_error()
    # │  作用：返回 True 若错误字符串 classifies 作为 rate 限制 / flood cap.。
    # │  参数：error: Optional[str]
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_rate_limited_error(error: Optional[str]) -> bool:
        """Return True if the error string classifies as a rate limit / flood cap.

        Single wrapper around :func:`classify_send_error` so the call sites in
        :meth:`_send_with_retry` share one notion of "is this a rate limit" instead
        of inline copies that could drift.
        """
        # ⤷ 出口 return：返回 classify_send_error(None, error or '') == 'rate_limited'
        return classify_send_error(None, error or "") == "rate_limited"

    # ┌── 函数 def _is_timeout_error()
    # │  作用：当 读取/写入超时 — 不 retryable 且不 plain-文本 时返回 True。
    # │  参数：error: Optional[str]
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_timeout_error(error: Optional[str]) -> bool:
        """Return True for read/write timeouts — NOT retryable and NOT a plain-text
        fallback trigger, because the request may already have been delivered."""
        lowered = (error or "").lower()
        # ⤷ 出口 return：返回 any((pat in lowered for pat in ('timed out', 'readtimeout', …
        return any(pat in lowered for pat in ("timed out", "readtimeout", "writetimeout"))

    # ┌── 函数 def _unwrap_ephemeral()
    # │  作用：解包 str/None/:类:`EphemeralReply` 响应进入 ``(文本, ttl)``. ``ttl > 0``
    # │  参数：self, response: Any
    # │  返回：Tuple[Optional[str], int]
    # └──────────────────────────────────────────────────────────────
    def _unwrap_ephemeral(self, response: Any) -> Tuple[Optional[str], int]:
        """Unwrap a str/None/:class:`EphemeralReply` response into ``(text, ttl)``. ``ttl > 0``
        means schedule ``_schedule_ephemeral_delete`` after a successful send; forced to 0 when the
        adapter doesn't override ``delete_message`` so non-supporting platforms degrade to normal
        sends."""
        if not isinstance(response, EphemeralReply):
            # ⤷ 出口 return：返回 (response, 0)
            return response, 0
        ttl = response.ttl_seconds
        if ttl is None:
            ttl = _or_default(lambda: int(self._get_ephemeral_system_ttl_default()), 0, Exception)
        if ttl and ttl > 0 and type(self).delete_message is BasePlatformAdapter.delete_message:
            ttl = 0
        # ⤷ 出口 return：返回 (response.text, int(ttl or 0))
        return response.text, int(ttl or 0)

    # ┌── 异步函数（协程） def _dispatch_inline_reply()
    # │  作用：Call handler 且 send 其 reply inline, 带重试, threading 且
    # │  参数：self, event: MessageEvent, log_cmd: Optional[str] = ...
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _dispatch_inline_reply(self, event: MessageEvent, *, log_cmd: Optional[str] = None) -> None:
        """Call the handler and send its reply inline, with retry, threading and
        ephemeral deletion — no session lifecycle (active-session bypass paths)."""
        thread_meta = _thread_metadata_for_event(event)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        response = await self._message_handler(event)
        text, eph_ttl = self._unwrap_ephemeral(response)
        if not text:
            return
        if log_cmd is not None:
            logger.info("[%s] Sending command '/%s' response (%d chars) to %s", self.name, log_cmd,
                        len(text), event.source.chat_id)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        result = await self._send_with_retry(
            chat_id=event.source.chat_id, content=text, reply_to=_reply_anchor_for_event(event),
            metadata=_mark_notify_metadata(thread_meta))
        if eph_ttl > 0 and result.success and result.message_id:
            self._schedule_ephemeral_delete(event.source.chat_id, result.message_id, eph_ttl)

    # ┌── 函数 def _media_delivery_scope()
    # │  作用：runner's ``_media_delivery_scope_for_source`` (routed 配置档's 主目录 + 终端
    # │  参数：self, source: Optional[SessionSource]
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _media_delivery_scope(self, source: Optional[SessionSource]):
        """The runner's ``_media_delivery_scope_for_source`` (routed profile's home + terminal
        policy) for validating outbound paths; a no-op without a runner or outside multiplexing."""
        resolve = getattr(self.gateway_runner, "_media_delivery_scope_for_source", None)
        if not callable(resolve) or source is None:
            # ⤷ 出口 return：返回 contextlib.nullcontext()
            return contextlib.nullcontext()
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 resolve(source)
            return resolve(source)
        except Exception:
            logger.debug("[%s] Failed to resolve media delivery scope", self.name, exc_info=True)
            # ⤷ 出口 return：返回 contextlib.nullcontext()
            return contextlib.nullcontext()

    # ┌── 函数 def _final_delivery_adapter()
    # │  作用：runner's CURRENT 适配器用于 new final-响应 send: reconnect 可以 swap
    # │  参数：self, source: Optional[SessionSource]
    # │  返回：'BasePlatformAdapter'
    # └──────────────────────────────────────────────────────────────
    def _final_delivery_adapter(self, source: Optional[SessionSource]) -> "BasePlatformAdapter":
        """The runner's CURRENT adapter for a new final-response send: a reconnect can swap the
        registry adapter mid-task; an unsent final response belongs on the replacement transport,
        while message IDs, edits and deletes stay owned by the old one (nothing is migrated)."""
        resolve = getattr(self.gateway_runner, "_adapter_for_source", None)
        if not callable(resolve):
            # ⤷ 出口 return：返回 self
            return self
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            live_adapter = resolve(source)
        except Exception:
            logger.debug("[%s] Failed to resolve live adapter for final delivery", self.name)
            # ⤷ 出口 return：返回 self
            return self
        if isinstance(live_adapter, BasePlatformAdapter) and live_adapter.platform == self.platform:
            # ⤷ 出口 return：返回 live_adapter
            return live_adapter
        # ⤷ 出口 return：返回 self
        return self

    # ┌── 异步函数（协程） def _send_with_retry()
    # │  作用：Send 带 exponential-backoff 重试在…上 transient network 错误; permanent
    # │  参数：self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Any = None, max_retries: int = 2, base_delay: float = 2.0
    # │  返回：'SendResult'
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_with_retry(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Any = None,
        max_retries: int = 2, base_delay: float = 2.0) -> "SendResult":
        """Send with exponential-backoff retry on transient network errors; permanent
        failures fall back to a plain-text send, exhausted retries notify the user."""
        async def _send(text: str) -> "SendResult":
            # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=text, reply_to=repl…
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        result = await _send(content)
        if result.success or self._send_retry_is_final(result):
            # ⤷ 出口 return：返回 result
            return result
        error_str = result.error or ""
        # A rate-limited / flood-capped send is transient: it should back off
        # (honoring the server's retry_after when present) rather than fall
        # through to the plain-text fallback, which re-enters the ban and can
        # truncate content.  Gate on the platform-neutral classifier as well so
        # platforms that surface a rate limit without a retry_after field
        # (e.g. Weixin raising a bare RuntimeError) get the same treatment.
        is_rate_limited = self._is_rate_limited_error(error_str)
        is_network = (
            result.retryable
            or is_rate_limited
            or result.retry_after is not None
            or self._is_retryable_error(error_str)
        )
        # Timeouts: not safe to retry (may have delivered) and not a formatting error.
        if not is_network and self._is_timeout_error(error_str):
            # ⤷ 出口 return：返回 result
            return result
        if is_network:
            # A server-requested retry_after (Telegram FloodWait) overrides backoff, once per send.
            server_retry_after = result.retry_after
            for attempt in range(1, max_retries + 1):
                backoff = server_retry_after
                if backoff is None:
                    backoff = base_delay * (2 ** (attempt - 1))
                elif backoff > _SEND_RETRY_INLINE_WAIT_CAP_SECS:
                    # Never hold this coroutine open for a long server penalty: a 97-minute
                    # FloodWait slept verbatim once froze inbound on every platform (#91969).
                    # Return the typed failure; the delivery ledger redelivers after the cooldown.
                    logger.error(
                        "[%s] Server asked to retry after %.0fs (> %.0fs inline cap); returning "
                        "typed failure for redelivery instead of sleeping: %s",
                        self.name, backoff, _SEND_RETRY_INLINE_WAIT_CAP_SECS, error_str,
                    )
                    return result
                delay = backoff + random.uniform(0, 1)
                server_retry_after = None
                logger.warning("[%s] Send failed (attempt %d/%d, retrying in %.1fs): %s", self.name,
                               attempt, max_retries, delay, error_str)
                await asyncio.sleep(delay)
                result = await _send(content)
                if result.success:
                    logger.info("[%s] Send succeeded on retry %d", self.name, attempt)
                    return result
                error_str = result.error or ""
                if self._send_retry_is_final(result):
                    return result
                if result.retry_after is not None:
                    server_retry_after = result.retry_after
                # The failure kind can change between attempts (a transient error may
                # later surface as a flood/rate-limit, or a rate-limited send may give
                # way to a permanent formatting error). Reclassify from the refreshed
                # error_str on every attempt so the break/continue decision below
                # reflects the current attempt, not a stale first-send classification.
                is_rate_limited = self._is_rate_limited_error(error_str)
                if not (
                    result.retryable
                    or is_rate_limited
                    or result.retry_after is not None
                    or self._is_retryable_error(error_str)
                ):
                    break  # error switched to non-transient — fall through to plain-text fallback
            else:
                # All retries exhausted (loop completed without break) — notify user.
                # If the final failure is a rate-limit / still carries a server
                # retry_after, do NOT send the delivery-failure notice now: the notice
                # send would land inside the same flood penalty and re-enter the ban
                # (a fourth send at t=378 in an [0, 189, 378, 378] sequence). Return
                # the typed failure so the delivery ledger owns redelivery after the
                # cooldown instead — no extra sleep or request needed.
                if self._is_rate_limited_error(error_str) or result.retry_after is not None:
                    logger.error(
                        "[%s] Rate-limited send exhausted retries; returning typed failure "
                        "for redelivery (no notice sent inside active flood penalty): %s",
                        self.name, error_str,
                    )
                    return result
                logger.error("[%s] Failed to deliver response after %d retries: %s", self.name, max_retries, error_str)
                notice = (
                    "\u26a0\ufe0f Message delivery failed after multiple attempts. "
                    "Please try again \u2014 your request was processed but the response could not be sent.")
                try:
                    await _send(notice)
                except Exception as notify_err:
                    logger.debug("[%s] Could not send delivery-failure notice: %s", self.name, notify_err)
                return result
        # Non-network / post-retry formatting failure: try plain text as fallback. A
        # rate-limited error never reaches here: it classifies as network above and the
        # loop only breaks on a non-transient, non-rate-limited error.
        logger.warning("[%s] Send failed: %s — trying plain-text fallback", self.name, error_str)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        fallback_result = await self._send_plain_fallback(chat_id, content, reply_to=reply_to, metadata=metadata)
        if not fallback_result.success:
            logger.error("[%s] Fallback send also failed: %s", self.name, fallback_result.error)
        # ⤷ 出口 return：返回 fallback_result
        return fallback_result

    # ┌── 函数 def _send_retry_is_final()
    # │  作用：当 failed send 必须是 returned 作为-是: neither 重试 nor plain-文本 时返回 True。
    # │  参数：self, result: 'SendResult'
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _send_retry_is_final(self, result: "SendResult") -> bool:
        """True when a failed send must be returned as-is: neither a retry nor the plain-text
        fallback can fix it (a structured auth/target refusal). Default: never."""
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 异步函数（协程） def _send_plain_fallback()
    # │  作用：Last-resort send 在…之后 non-transient failure; 平台 whose markup 是不
    # │  参数：self, chat_id: str, content: str, reply_to: Optional[str], metadata: Any
    # │  返回：'SendResult'
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_plain_fallback(
            self, chat_id: str, content: str, *, reply_to: Optional[str], metadata: Any) -> "SendResult":
        """Last-resort send after a non-transient failure; platforms whose markup is not the
        likely culprit override it (Photon drops rich links instead of adding the banner)."""
        # ⤷ 出口 return：返回 await self.send(chat_id=chat_id, content=f'(Response formatt…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await self.send(
            chat_id=chat_id, content=f"(Response formatting failed, plain text:)\n\n{content[:3500]}",
            reply_to=reply_to, metadata=metadata)

    # ┌── 函数 def _merge_caption()
    # │  作用：Merge new caption 进入 existing 文本 unless identical (stripped) caption 已
    # │  参数：existing_text: Optional[str], new_text: str
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _merge_caption(existing_text: Optional[str], new_text: str) -> str:
        """Merge a new caption into existing text unless an identical (stripped) caption already
        exists — exact match per caption, not substring ("Meeting" isn't swallowed by "Meeting
        agenda")."""
        if not existing_text:
            # ⤷ 出口 return：返回 new_text
            return new_text
        if new_text.strip() in [c.strip() for c in existing_text.split("\n\n")]:
            # ⤷ 出口 return：返回 existing_text
            return existing_text
        # ⤷ 出口 return：返回 f'{existing_text}\n\n{new_text}'.strip()
        return f"{existing_text}\n\n{new_text}".strip()

    # ┌── 函数 def _text_debounce_store()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：dict[str, TextDebounceState]
    # └──────────────────────────────────────────────────────────────
    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        # ⤷ 出口 return：返回 _lazy_attr(self, '_text_debounce', dict)
        return _lazy_attr(self, "_text_debounce", dict)

    # ┌── 函数 def _is_queue_text_debounce_candidate()
    # │  作用：当 normal 文本 eligible 用于队列-模式 debounce. 时返回 True。
    # │  参数：self, event: MessageEvent
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _is_queue_text_debounce_candidate(self, event: MessageEvent) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT and not getattr(event, "internal", False)
            and not event.is_command() and bool((event.text or "").strip()))
        if result:
            logger.debug("[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                         self.name, getattr(event, "session_key", "?"), len(event.text or ""))
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def _can_merge_text_debounce_events()
    # │  作用：当 two 文本 debounce 事件 came 从 same sender. 时返回 True。
    # │  参数：self, existing: MessageEvent, event: MessageEvent
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _can_merge_text_debounce_events(self, existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                # ⤷ 出口 return：返回 None
                return None
            platform = _platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
            if sender:
                # ⤷ 出口 return：返回 (platform, str(sender))
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(source, "chat_id", None):
                # ⤷ 出口 return：返回 (platform, 'dm', str(source.chat_id))
                return (platform, "dm", str(source.chat_id))
            # ⤷ 出口 return：返回 None
            return None
        existing_sender = _identity(existing)
        # ⤷ 出口 return：返回 existing_sender is not None and existing_sender == _identity…
        return existing_sender is not None and existing_sender == _identity(event)

    # ┌── 函数 def _text_debounce_delay()
    # │  作用：返回 bounded busy-文本 debounce delay 用于 ``session_key``.。
    # │  参数：self, session_key: str
    # │  返回：float
    # └──────────────────────────────────────────────────────────────
    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            # ⤷ 出口 return：返回 0.0
            return 0.0
        deadline = min(state.last_ts + self._busy_text_debounce_seconds,
                       state.first_ts + self._busy_text_hard_cap_seconds)
        # ⤷ 出口 return：返回 max(0.0, deadline - time.monotonic())
        return max(0.0, deadline - time.monotonic())

    # ┌── 异步函数（协程） def _queue_text_debounce()
    # │  作用：Buffer normal 队列-模式 busy 文本且 schedule bounded 冲刷.
    # │  参数：self, session_key: str, event: MessageEvent
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution: flush the buffer as the next turn, new sender starts
            # fresh.
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    merge_pending_message_event(self._pending_messages, session_key, event, merge_text=True)
                return
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            if event.text:
                state.event.text = _append_text(state.event.text, event.text)
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now
        state.cancel_timer()
        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))

    # ┌── 异步函数（协程） def _flush_text_debounce()
    # │  作用：Timer task 那个冲刷 debounced 文本 buffer.
    # │  参数：self, session_key: str, delay: float
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _flush_text_debounce(self, session_key: str, delay: float) -> None:
        """Timer task that flushes the debounced text buffer."""
        # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.sleep(delay)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._flush_text_debounce_now(session_key)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            state = self._text_debounce_store().get(session_key)
            if state is not None and state.task is current:
                state.task = None

    # ┌── 异步函数（协程） def _flush_text_debounce_now()
    # │  作用：Force-冲刷 one debounced busy-文本 burst 进入 pending slot.
    # │  参数：self, session_key: str
    # │  返回：bool
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _flush_text_debounce_now(self, session_key: str) -> bool:
        """Force-flush one debounced busy-text burst into the pending slot."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            # ⤷ 出口 return：返回 False
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        pending = self._pending_messages.get(session_key)
        if pending is not None and not self._can_merge_text_debounce_events(pending, state.event):
            # ⤷ 出口 return：返回 False
            return False
        store.pop(session_key, None)
        merge_pending_message_event(self._pending_messages, session_key, state.event, merge_text=True)
        # ⤷ 出口 return：返回 True
        return True

    # ┌── 函数 def _discard_text_debounce()
    # │  作用：Cancel 且丢弃 pending 文本 debounce 状态用于 control 命令.
    # │  参数：self, session_key: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()

    # ── Session task + guard ownership helpers: paired with the _session_tasks owner map so
    # reconciliation is deterministic across completion, /stop /new /reset, and stale-lock heal.

    # ┌── 函数 def _release_session_guard()
    # │  作用：Release 会话护栏; 带 ``护栏`` given, 仅若 entry 是仍 那个 exact
    # │  参数：self, session_key: str, guard: Optional[asyncio.Event] = ...
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _release_session_guard(self, session_key: str, *, guard: Optional[asyncio.Event] = None) -> None:
        """Release the session guard; with ``guard`` given, only if the entry is still that exact
        Event (an old task's unwind must not clear the guard a reset-like command swapped in)."""
        current_guard = self._active_sessions.get(session_key)
        if current_guard is None or (guard is not None and current_guard is not guard):
            return
        del self._active_sessions[session_key]

    # ┌── 函数 def _session_task_is_stale()
    # │  作用：True 若 recorded owner task 用于 ``session_key`` 有 exited. 无 owner task 在全部是
    # │  参数：self, session_key: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _session_task_is_stale(self, session_key: str) -> bool:
        """True if the recorded owner task for ``session_key`` has exited. No owner task at all is
        NOT stale (guards installed outside handle_message, as tests do, must not be healed)."""
        done = getattr(self._session_tasks.get(session_key), "done", None)
        # ⤷ 出口 return：返回 bool(done and done())
        return bool(done and done())

    # ┌── 函数 def _heal_stale_session_lock()
    # │  作用：清除卡死的会话锁; True 若 healed. 在…上-entry safety net: 不它 split-brain
    # │  参数：self, session_key: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _heal_stale_session_lock(self, session_key: str) -> bool:
        """Clear a stale session lock; True if healed. On-entry safety net: without it a split-brain
        (guard held, nothing processing) traps the chat in "Interrupting..." until restart."""
        if session_key not in self._active_sessions or not self._session_task_is_stale(session_key):
            # ⤷ 出口 return：返回 False
            return False
        logger.warning("[%s] Healing stale session lock for %s (owner task is done/absent)",
                       self.name, session_key)
        self._active_sessions.pop(session_key, None)
        self._pending_messages.pop(session_key, None)
        self._session_tasks.pop(session_key, None)
        self._discard_text_debounce(session_key)
        # ⤷ 出口 return：返回 True
        return True

    # ┌── 函数 def _start_session_processing()
    # │  作用：Spawn background processing task under 会话护栏; True 在…上 success. 若
    # │  参数：self, event: MessageEvent, session_key: str, interrupt_event: Optional[asyncio.Event] = ...
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _start_session_processing(self, event: MessageEvent, session_key: str, *,
                                  interrupt_event: Optional[asyncio.Event] = None) -> bool:
        """Spawn a background processing task under the session guard; True on success. If
        ``create_task`` is stubbed with a non-Task sentinel (tests), the guard is rolled back
        (False)."""
        guard = interrupt_event or asyncio.Event()
        self._active_sessions[session_key] = guard
        task = asyncio.create_task(self._process_message_background(event, session_key))
        if not self._track_session_task(session_key, task):
            self._session_tasks.pop(session_key, None)
            self._release_session_guard(session_key, guard=guard)
            # ⤷ 出口 return：返回 False
            return False
        # ⤷ 出口 return：返回 True
        return True

    # ┌── 函数 def _track_session_task()
    # │  作用：Record ``task`` 作为会话 owner 且 track 它用于 shutdown; False 当
    # │  参数：self, session_key: str, task: Any
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _track_session_task(self, session_key: str, task: Any) -> bool:
        """Record ``task`` as the session owner and track it for shutdown; False when
        ``create_task`` was stubbed with an unhashable sentinel (tests) — the owner entry is left
        for the caller."""
        self._session_tasks[session_key] = task
        # ⤷ 关键分支 try/except：捕获 TypeError，出错走兜底而不是崩溃。
        try:
            self._background_tasks.add(task)
        except TypeError:
            # ⤷ 出口 return：返回 False
            return False
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)
        # ⤷ 出口 return：返回 True
        return True

    # ┌── 异步函数（协程） def cancel_session_processing()
    # │  作用：Cancel 在-flight processing 用于 one 会话. ``release_guard=False`` 保留护栏从而
    # │  参数：self, session_key: str, release_guard: bool = ..., discard_pending: bool = ...
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def cancel_session_processing(self, session_key: str, *, release_guard: bool = True,
                                        discard_pending: bool = True) -> None:
        """Cancel in-flight processing for one session. ``release_guard=False`` keeps the guard so
        reset-like commands finish atomically; the await is bounded (5s) so a wedged finally can't
        stall."""
        task = self._session_tasks.pop(session_key, None)
        if task is not None and not task.done():
            logger.debug("[%s] Cancelling active processing for session %s", self.name, session_key)
            self._expected_cancelled_tasks.add(task)
            task.cancel()
            # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError、asyncio.TimeoutError、Exception，出错走兜底而不是崩溃。
            try:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("[%s] Cancelled task for %s did not exit within 5s; "
                               "unblocking dispatch and letting the task unwind in the background",
                               self.name, session_key)
            except Exception:
                logger.debug("[%s] Session cancellation raised while unwinding %s", self.name,
                             session_key, exc_info=True)
        if discard_pending:
            self._pending_messages.pop(session_key, None)
            self._discard_text_debounce(session_key)
        if release_guard:
            self._release_session_guard(session_key)

    # ┌── 异步函数（协程） def _drain_pending_after_session_command()
    # │  作用：Tail 的 /stop, /new, /reset: release 命令-scoped 护栏, 然后
    # │  参数：self, session_key: str, command_guard: asyncio.Event
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _drain_pending_after_session_command(
        self, session_key: str, command_guard: asyncio.Event) -> None:
        """Tail of /stop, /new, /reset: release the command-scoped guard, then
        spawn a fresh processing task for any follow-up queued meanwhile."""
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self._flush_text_debounce_now(session_key)
        pending_event = self._pending_messages.pop(session_key, None)
        self._release_session_guard(session_key, guard=command_guard)
        if pending_event is not None:
            self._start_session_processing(pending_event, session_key)

    # ┌── 异步函数（协程） def _dispatch_active_session_command()
    # │  作用：Dispatch reset-like bypass 命令 (/stop, /new, /reset): 保留护栏 installed 当
    # │  参数：self, event: MessageEvent, session_key: str, cmd: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _dispatch_active_session_command(self, event: MessageEvent, session_key: str, cmd: str) -> None:
        """Dispatch a reset-like bypass command (/stop, /new, /reset): keep a guard installed while
        the runner handles it (follow-ups stay queued), cancel the old task AFTER the response,
        drain once."""
        logger.debug("[%s] Command '/%s' bypassing active-session guard for %s", self.name, cmd, session_key)
        current_guard = self._active_sessions.get(session_key)
        command_guard = asyncio.Event()
        self._active_sessions[session_key] = command_guard
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # Send BEFORE cancelling so cancellation side effects can't drop the "/new"
            # confirmation.
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._dispatch_inline_reply(event, log_cmd=cmd)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self.cancel_session_processing(session_key, release_guard=False, discard_pending=False)
        except Exception:
            # On failure restore the original guard so the session isn't left half-reset.
            if self._active_sessions.get(session_key) is command_guard:
                if session_key in self._session_tasks and current_guard is not None:
                    self._active_sessions[session_key] = current_guard
                else:
                    self._release_session_guard(session_key, guard=command_guard)
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self._drain_pending_after_session_command(session_key, command_guard)

    # ┌── 异步函数（协程） def handle_message()
    # │  作用：进程 incoming 消息; 返回 quickly 由 spawning background
    # │  参数：self, event: MessageEvent
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def handle_message(self, event: MessageEvent) -> None:
        """Process an incoming message; returns quickly by spawning a background
        task so new messages (and interrupts) can arrive while an agent runs."""
        event._gateway_accepted = False
        if not self._message_handler:
            # No handler = every inbound silently discarded on an adapter that still polls and sends;
            # say so once per adapter (#102260).
            if not getattr(self, "_no_message_handler_logged", False):
                self._no_message_handler_logged = True
                logger.error(
                    "[%s] Dropping inbound message: no gateway message handler "
                    "is installed on this adapter. The adapter is connected and "
                    "can send, but every inbound message is discarded.",
                    self.name,
                )
            return

        if event.allow_gateway_control:
            coerce_plaintext_gateway_command(event)
        expected_session_key = str((event.metadata or {}).get("gateway_session_key") or "").strip()
        # Explicitly routed events already name their destination; recovering a
        # different topic would redirect them and yield before the session claim.
        if (not expected_session_key and getattr(self, "_topic_recovery_fn", None) is not None
                and event.source.platform == Platform.TELEGRAM and event.source.chat_type == "dm"):
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.to_thread(self._apply_topic_recovery, event)
        session_key = self._event_session_key(event)
        if expected_session_key and session_key != expected_session_key:
            logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                           expected_session_key, session_key)
            return
        # On-entry self-heal: clear a guard whose owner task already exited.
        if session_key in self._active_sessions:
            self._heal_stale_session_lock(session_key)
        if session_key in self._active_sessions:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._handle_message_while_active(event, session_key)
            return
        # Guard installed synchronously BEFORE the task spawns so a second message can't race in.
        event._gateway_accepted = self._start_session_processing(event, session_key)

    # ┌── 异步函数（协程） def _handle_message_while_active()
    # │  作用：路由消息那个 arrived 当 ``session_key`` 是 busy: bypass
    # │  参数：self, event: MessageEvent, session_key: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _handle_message_while_active(self, event: MessageEvent, session_key: str) -> None:
        """Route a message that arrived while ``session_key`` is busy: bypass
        commands / clarify replies dispatch inline, everything else is queued."""
        # Bypass commands run inline: queued they'd leak as user text (/new) or deadlock
        # (/approve, /deny — the agent is blocked on Event.wait).  Dispatch inline by
        # calling the message handler directly and sending the response.  Do NOT use
        # _process_message_background — it manages session lifecycle and its cleanup
        # races with the running task (split-brain, see PR #4926).
        # Certain commands must bypass the active-session guard and be dispatched directly to the gateway
        # runner. Without this, they are queued as pending messages and either: See #4926.
        cmd = event.get_command()
        from hermes_cli.commands import (is_interrupt_then_dispatch, should_bypass_active_session)
        if should_bypass_active_session(cmd):
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                # /stop, /new, /reset: cancel + response + drain; other bypasses don't cancel.
                if cmd and is_interrupt_then_dispatch(cmd):
                    self._discard_text_debounce(session_key)
                    await self._dispatch_active_session_command(event, session_key, cmd)
                else:
                    logger.debug("[%s] Command '/%s' bypassing active-session guard for %s",
                                 self.name, cmd, session_key)
                    await self._dispatch_inline_reply(event)
            except Exception as e:
                logger.error("[%s] Command '/%s' dispatch failed: %s", self.name, cmd, e, exc_info=True)
            return
        # Clarify bypass: while blocked on clarify_tool the next message must reach the
        # text-intercept so numeric/exact/"Other" answers resolve it and unblock the agent.
        # Otherwise it lands in _pending_messages as a follow-up turn and the answer is
        # discarded.  Same shape as the /approve deadlock fix (PR #4926): agent thread
        # blocked on Event.wait, message must reach the resolver before being a new turn.
        # See #4926.
        if not cmd and event.allow_gateway_control:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                from tools import clarify_gateway as _clarify_mod
                _has_text_clarify = _clarify_mod.get_pending_for_session(
                    session_key, include_choice_prompts=True) is not None
            except Exception:
                _has_text_clarify = False
            if _has_text_clarify:
                logger.debug("[%s] Routing message to clarify text-intercept for %s", self.name, session_key)
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    await self._dispatch_inline_reply(event)
                except Exception as e:
                    logger.error("[%s] Clarify text-intercept dispatch failed: %s", self.name, e, exc_info=True)
                return
        if self._busy_session_handler is not None:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                if await self._busy_session_handler(event, session_key):
                    return
            except Exception as e:
                logger.error("[%s] Busy-session handler failed: %s", self.name, e, exc_info=True)
        # Without a runner FIFO, do not merge a wake into an occupied human slot
        # (or collapse distinct wakes into one turn). Its caller can retry admission.
        if event.internal and session_key in self._pending_messages:
            return
        # Photo bursts/albums: queue without interrupting; they run after the current task.
        if event.message_type == MessageType.PHOTO:
            logger.debug("[%s] Queuing photo follow-up for session %s without interrupt", self.name, session_key)
            merge_pending_message_event(self._pending_messages, session_key, event)
            event._gateway_accepted = True
            return
        if self._is_queue_text_debounce_candidate(event):
            logger.debug("[%s] New text message while session %s is active — "
                         "debouncing follow-up (busy_text_mode=queue, window=%.2fs)", self.name,
                         session_key, self._busy_text_debounce_seconds)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._queue_text_debounce(session_key, event)
        else:
            logger.debug("[%s] New message while session %s is active — queuing follow-up "
                         "(no interrupt, will cascade after current turn)", self.name, session_key)
            merge_pending_message_event(self._pending_messages, session_key, event,
                                        merge_text=event.message_type == MessageType.TEXT)
            event._gateway_accepted = True

    # ┌── 函数 def _get_human_delay()
    # │  作用：Random human-like pacing delay (s) 从 HERMES_HUMAN_DELAY_MODE: "off" (默认值) |
    # │  参数：无
    # │  返回：float
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _get_human_delay() -> float:
        """Random human-like pacing delay (s) from HERMES_HUMAN_DELAY_MODE: "off" (default) |
        "natural" 800-2500ms | "custom" via HERMES_HUMAN_DELAY_MIN_MS /
        HERMES_HUMAN_DELAY_MAX_MS."""
        mode = os.getenv("HERMES_HUMAN_DELAY_MODE", "off").lower()
        if mode == "off":
            # ⤷ 出口 return：返回 0.0
            return 0.0
        lo, hi = 800, 2500
        if mode != "natural":  # custom mode tolerates malformed env vars
            lo = _or_default(lambda: int(os.getenv("HERMES_HUMAN_DELAY_MIN_MS", str(lo))), lo)
            hi = _or_default(lambda: int(os.getenv("HERMES_HUMAN_DELAY_MAX_MS", str(hi))), hi)
        # ⤷ 出口 return：返回 random.uniform(lo / 1000.0, hi / 1000.0)
        return random.uniform(lo / 1000.0, hi / 1000.0)

    # ┌── 异步函数（协程） def _synthesize_auto_tts()
    # │  作用：Synthesize auto-TTS audio -> ``(existing_paths, requested_path)``; empty/None 在…上 failure
    # │  参数：self, text_content: str
    # │  返回：Tuple[List[str], Optional[str]]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _synthesize_auto_tts(self, text_content: str) -> Tuple[List[str], Optional[str]]:
        """Synthesize auto-TTS audio -> ``(existing_paths, requested_path)``; empty/None on failure
        (logged, never raised). Path built platform-aware HERE: HERMES_SESSION_PLATFORM is cleared
        post-handler."""
        paths: List[str] = []
        requested_path = None
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.tts_tool import text_to_speech_tool, check_tts_requirements
            if check_tts_requirements():
                import json as _json
                speech_text = self.prepare_tts_text(text_content)
                if not speech_text:
                    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = build_auto_tts_output_path(self.platform)
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                tts_data = _json.loads(await asyncio.to_thread(
                    text_to_speech_tool, text=speech_text, output_path=requested_path))
                if tts_data.get("success", True):
                    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
                    paths = [str(path) for path in raw_tts_paths if path and Path(path).exists()]
        except Exception as tts_err:
            logger.warning("[%s] Auto-TTS failed: %s", self.name, tts_err)
        # ⤷ 出口 return：返回 (paths, requested_path)
        return paths, requested_path

    # ┌── 函数 def _wants_auto_tts()
    # │  作用：Auto-TTS 在…上 voice 输入 (voice-first), gated 由 /voice 或 voice.auto_tts;
    # │  参数：self, event: MessageEvent, session_key: str, interrupt_event: asyncio.Event, text_content: str, media_files: list
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _wants_auto_tts(self, event: MessageEvent, session_key: str, interrupt_event: asyncio.Event,
                        text_content: str, media_files: list) -> bool:
        """Auto-TTS on voice input (voice-first), gated by /voice or voice.auto_tts;
        skipped when streaming TTS already delivered audio this turn."""
        generation = getattr(interrupt_event, "_hermes_run_generation", None)
        # ⤷ 出口 return：返回 bool(self._should_auto_tts_for_chat(event.source.chat_id) an…
        return bool(
            self._should_auto_tts_for_chat(event.source.chat_id)
            and event.message_type == MessageType.VOICE and text_content and not media_files
            and not self._streaming_tts_turn_completed(session_key, generation, event=event))

    # ┌── 异步函数（协程） def _play_tts_file()
    # │  作用：Play one synthesized TTS 文件. 返回 True 当 ORIGINAL reply 文本 rode
    # │  参数：self, event: MessageEvent, text_content: str, tts_path: str, first: bool, metadata: Dict[str, Any], record_delivery: Callable
    # │  返回：bool
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _play_tts_file(
        self, event: MessageEvent, text_content: str, tts_path: str, first: bool,
        metadata: Dict[str, Any], record_delivery: Callable) -> bool:
        """Play one synthesized TTS file. Returns True when the ORIGINAL reply text rode
        along as a Telegram caption (first file, ≤1024 chars) so the text send is skipped."""
        caption = None
        if first and self.platform == Platform.TELEGRAM and text_content and text_content[:1024] == text_content:
            caption = text_content
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        tts_result = await self.play_tts(
            chat_id=event.source.chat_id, audio_path=tts_path, caption=caption, metadata=metadata)
        record_delivery(tts_result)
        # ⤷ 出口 return：返回 bool(caption and getattr(tts_result, 'success', False))
        return bool(caption and getattr(tts_result, "success", False))

    # ┌── 异步函数（协程） def _record_delivery_obligation()
    # │  作用：Ledger final 响应在…之前 send 从而 crash 在…之前平台 ACK redelivers 在…上
    # │  参数：self, event: MessageEvent, session_key: str, text_content: str, delivery_adapter: 'BasePlatformAdapter', is_ephemeral_response: bool
    # │  返回：Optional[str]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _record_delivery_obligation(
        self, event: MessageEvent, session_key: str, text_content: str,
        delivery_adapter: "BasePlatformAdapter", is_ephemeral_response: bool) -> Optional[str]:
        """Ledger the final response BEFORE the send so a crash before platform ACK redelivers on
        next boot; best-effort, skips slash-command and ephemeral replies. Returns the obligation id
        or None."""
        if is_ephemeral_response or str(event.text or "").lstrip().startswith(
            ("/", self.typed_command_prefix or "!")):
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from gateway.delivery_ledger import (
                compute_obligation_id, ledger_enabled, mark_attempting, record_obligation)
            if not await asyncio.to_thread(ledger_enabled):
                # ⤷ 出口 return：返回 None
                return None
            source = event.source
            # ``ledger_message_id`` wins when set: a queued chain's final answers the last message
            # of the chain, not the event that opened it (see ``MessageEvent.ledger_message_id``).
            _ledger_id = getattr(event, "ledger_message_id", None)
            if _ledger_id is None:
                _ledger_id = getattr(event, "message_id", "")
            obligation_id = compute_obligation_id(
                session_key, str(_ledger_id or ""), text_content)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.to_thread(
                record_obligation, obligation_id=obligation_id, session_key=session_key,
                platform=str(getattr(source.platform, "value", source.platform)),
                chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
                content=text_content,
                adapter_profile=getattr(delivery_adapter, "_owner_profile", None))
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.to_thread(mark_attempting, obligation_id)
            # ⤷ 出口 return：返回 obligation_id
            return obligation_id
        except Exception:
            logger.debug("delivery ledger record failed", exc_info=True)
            return None

    # ┌── 异步函数（协程） def _finalize_delivery_obligation()
    # │  作用：Mark ledger row delivered/failed (尽最大努力). 在…上 ``send_path_degraded`` 带
    # │  参数：self, obligation_id: str, result: Any, event: MessageEvent, delivery_adapter: 'BasePlatformAdapter'
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _finalize_delivery_obligation(
        self, obligation_id: str, result: Any, event: MessageEvent,
        delivery_adapter: "BasePlatformAdapter") -> None:
        """Mark the ledger row delivered/failed (best-effort). On ``send_path_degraded`` with a
        replacement adapter live, trigger another redelivery sweep (the watcher's may have run
        before this failure landed; atomic claiming keeps it idempotent). On a flood-control refusal
        arm the runner's timed redelivery, so the reply goes out once the penalty has passed instead
        of waiting for the next restart."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from gateway.delivery_ledger import is_flood_error, mark_delivered, mark_failed
            if getattr(result, "success", False):
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.to_thread(mark_delivered, obligation_id)
                return
            error = str(getattr(result, "error", "") or "")
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await asyncio.to_thread(mark_failed, obligation_id, error)
            if error == "send_path_degraded":
                redeliver = getattr(
                    self.gateway_runner, "_redeliver_failed_obligations_for_platform", None)
                live = self._final_delivery_adapter(event.source)
                if live is not delivery_adapter and callable(redeliver):
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await redeliver(event.source.platform,
                                    profile=getattr(delivery_adapter, "_owner_profile", None))
            elif is_flood_error(error):
                schedule = getattr(self.gateway_runner, "_schedule_flood_redelivery", None)
                if callable(schedule):
                    schedule(event.source.platform,
                             profile=getattr(delivery_adapter, "_owner_profile", None))
        except Exception:
            logger.debug("delivery ledger update failed", exc_info=True)

    # ┌── 异步函数（协程） def _deliver_media_attachments()
    # │  作用：Deliver MEDIA-tag 文件且 检测到 local 文件由 类型: images batched via
    # │  参数：self, event: MessageEvent, media_files: list, local_files: list, force_document_attachments: bool, human_delay: float, metadata: Dict[str, Any], record_delivery: Callable
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _deliver_media_attachments(
        self, event: MessageEvent, media_files: list, local_files: list, *,
        force_document_attachments: bool, human_delay: float, metadata: Dict[str, Any],
        record_delivery: Callable) -> None:
        """Deliver MEDIA-tag files and detected local files by type: images batched via
        ``send_multiple_images`` unless ``[[as_document]]``; otherwise audio → send_voice (MEDIA
        tags only, never bare local files), video → send_video, else send_document. Every failure is
        reported. Each send feeds ``record_delivery`` so media-only turns report SUCCESS."""
        from urllib.parse import quote as _quote

        def _as_image(path: str) -> bool:
            # ⤷ 出口 return：返回 Path(path).suffix.lower() in _IMAGE_EXTS and (not force_docu…
            return Path(path).suffix.lower() in _IMAGE_EXTS and not force_document_attachments
        _image_paths = [p for p, is_voice in media_files if not is_voice and _as_image(p)]
        _image_paths += [p for p in local_files if _as_image(p)]
        if _image_paths:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._send_image_batch(
                event, [(f"file://{_quote(p)}", "") for p in _image_paths], metadata, human_delay,
                record_delivery)
        chat_id = event.source.chat_id

        async def _send_one(path: str, *, is_voice: bool, media_tag: bool) -> SendResult:
            """MEDIA-tag files (``media_tag``) may route to send_voice; bare local files never
            do."""
            ext = Path(path).suffix.lower()
            if media_tag and should_send_media_as_audio(self.platform, ext, is_voice=is_voice):
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                result = await self.send_voice(chat_id=chat_id, audio_path=path, metadata=metadata, is_voice=is_voice)
            elif ext in _VIDEO_EXTS:
                if media_tag:
                    logger.info("[%s] Sending video attachment (%s) to %s", self.name, ext, chat_id)
                result = await self.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
            else:
                result = await self.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            if not result.success:
                logger.warning("[%s] Failed to send %s (%s): %s", self.name,
                               "media" if media_tag else "local file", ext, result.error)
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await self._notify_media_delivery_failure(chat_id, path, is_voice=is_voice, metadata=metadata)
            # ⤷ 出口 return：返回 result
            return result
        queue = [(p, v, True) for p, v in media_files if v or not _as_image(p)]
        if queue:
            logger.info("[%s] Delivering %d non-image MEDIA attachment(s)", self.name, len(queue))
        queue += [(p, False, False) for p in local_files if not _as_image(p)]
        for path, is_voice, media_tag in queue:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                record_delivery(await _send_one(path, is_voice=is_voice, media_tag=media_tag))
            except Exception as err:
                record_delivery(SendResult(success=False, error=str(err)))
                if media_tag:
                    logger.warning("[%s] Error sending media: %s", self.name, err)
                else:
                    logger.error("[%s] Error sending local file %s: %s", self.name, path, err)

    # ┌── 异步函数（协程） def _send_image_batch()
    # │  作用：Batch-send images; failure 是 logged (never 抛出) 从而 other attachments 仍 go.
    # │  参数：self, event: MessageEvent, images: list, metadata: Dict[str, Any], human_delay: float, record_delivery: Callable
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_image_batch(
        self, event: MessageEvent, images: list, metadata: Dict[str, Any], human_delay: float,
        record_delivery: Callable) -> None:
        """Batch-send images; a failure is logged (never raised) so other attachments still go.
        The batch result feeds ``record_delivery`` so media-only turns report their real
        outcome instead of FAILURE."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            result = await self.send_multiple_images(
                chat_id=event.source.chat_id, images=images, metadata=metadata, human_delay=human_delay)
        except Exception as batch_err:
            logger.warning("[%s] Error batching images: %s", self.name, batch_err, exc_info=True)
            record_delivery(SendResult(success=False, error=str(batch_err)))
            return
        record_delivery(result)

    # ┌── 异步函数（协程） def send_final_ledgered()
    # │  作用：delivery-ledger bracket 每个 final 文本 goes through, 在…上 CURRENT transport
    # │  参数：self, event: MessageEvent, session_key: str, text_content: str, metadata: Dict[str, Any], reply_to: Optional[str], is_ephemeral_response: bool = ...
    # │  返回：'tuple[SendResult, BasePlatformAdapter]'
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def send_final_ledgered(
        self, event: MessageEvent, session_key: str, text_content: str, metadata: Dict[str, Any], *,
        reply_to: Optional[str], is_ephemeral_response: bool = False,
    ) -> "tuple[SendResult, BasePlatformAdapter]":
        """The delivery-ledger bracket every final text goes through, on the CURRENT transport
        (a reconnect may have replaced this adapter): record the obligation before the send,
        send with retry, finalize from the result — so a refused final (flood control, a dead
        transport) leaves a ledger row the boot sweep / runtime redelivery can act on. ``event``
        supplies the source and the ledger identity (``ledger_message_id`` or ``message_id``).
        Returns the result with the adapter that sent it: that adapter owns ``result.message_id``
        (an ephemeral delete must go to the same transport)."""
        delivery_adapter = self._final_delivery_adapter(event.source)
        logger.info("[%s] Sending response (%d chars) to %s", delivery_adapter.name,
                    len(text_content), event.source.chat_id)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        obligation_id = await self._record_delivery_obligation(
            event, session_key, text_content, delivery_adapter, is_ephemeral_response)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        result = await delivery_adapter._send_with_retry(
            chat_id=event.source.chat_id, content=text_content, reply_to=reply_to, metadata=metadata)
        if obligation_id is not None:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._finalize_delivery_obligation(obligation_id, result, event, delivery_adapter)
        # ⤷ 出口 return：返回 (result, delivery_adapter)
        return result, delivery_adapter

    # ┌── 异步函数（协程） def _send_final_text()
    # │  作用：Normal-lane final: ledger bracket plus 消息-id owner's ephemeral delete.
    # │  参数：self, event: MessageEvent, session_key: str, text_content: str, metadata: Dict[str, Any], is_ephemeral_response: bool, ephemeral_ttl: int, record_delivery: Callable
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _send_final_text(
        self, event: MessageEvent, session_key: str, text_content: str, metadata: Dict[str, Any],
        is_ephemeral_response: bool, ephemeral_ttl: int, record_delivery: Callable) -> None:
        """Normal-lane final: the ledger bracket plus the message-id owner's ephemeral delete."""
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        result, delivery_adapter = await self.send_final_ledgered(
            event, session_key, text_content, metadata,
            reply_to=_reply_anchor_for_event(event), is_ephemeral_response=is_ephemeral_response)
        record_delivery(result)
        if ephemeral_ttl and ephemeral_ttl > 0 and result.success and result.message_id:
            delivery_adapter._schedule_ephemeral_delete(event.source.chat_id, result.message_id, ephemeral_ttl)

    # ┌── 异步函数（协程） def _notify_turn_error()
    # │  作用：Tell user 轮次 failed rather 比 leaving radio silence (last resort:
    # │  参数：self, event: MessageEvent, e: BaseException
    # │  返回：Optional[dict]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _notify_turn_error(self, event: MessageEvent, e: BaseException) -> Optional[dict]:
        """Tell the user a turn failed rather than leaving radio silence (last resort:
        a failing notice is logged, never raised). Returns the thread metadata used."""
        _thread_metadata = None
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            error_detail = str(e)[:300] if str(e) else "no details available"
            _thread_metadata = _thread_metadata_for_event(event)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self.send(
                chat_id=event.source.chat_id,
                content=(f"Sorry, I encountered an error ({type(e).__name__}).\n{error_detail}\n"
                "Try again or use /reset to start a fresh session."), metadata=_thread_metadata)
        except Exception as notify_err:
            logger.error(
                "[%s] Failed to send error notification to user: %s", self.name, notify_err, exc_info=True)
        # ⤷ 出口 return：返回 _thread_metadata
        return _thread_metadata

    # ┌── 异步函数（协程） def _deliver_attachments()
    # │  作用：Send extracted image URLs, MEDIA 文件且 bare local 文件 (human-paced),
    # │  参数：self, event: MessageEvent, extracted: '_ExtractedResponse', metadata: Dict[str, Any], anything_sent: bool, record_delivery: Callable
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _deliver_attachments(self, event: MessageEvent, extracted: "_ExtractedResponse",
                                   metadata: Dict[str, Any], *, anything_sent: bool,
                                   record_delivery: Callable) -> None:
        """Send extracted image URLs, MEDIA files and bare local files (human-paced),
        then fail loudly if a non-empty response produced nothing deliverable. Attachment
        results feed ``record_delivery`` so the turn outcome reflects them."""
        human_delay = self._get_human_delay()
        images, media_files, local_files = extracted.images, extracted.media_files, extracted.local_files
        if images:
            logger.info("[%s] Extracted %d image(s) to send as attachments", self.name, len(images))
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._send_image_batch(event, images, metadata, human_delay, record_delivery)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        await self._deliver_media_attachments(
            event, media_files, local_files,
            force_document_attachments=extracted.force_document_attachments,
            human_delay=human_delay, metadata=metadata, record_delivery=record_delivery)
        if not (anything_sent or images or local_files or media_files) and extracted.pre_extract.strip():
            logger.error("[%s] response_delivery_dropped: non-empty response "
                         "(%d chars) produced no delivered message or attachment "
                         "for %s (empty after extract, recovery yielded nothing).", self.name,
                         len(extracted.pre_extract), event.source.chat_id)

    # ┌── 函数 def _start_typing_refresh()
    # │  作用：Spawn typing-refresh task, 或 None 当 ``typing_indicator=False``.
    # │  参数：self, event: MessageEvent, interrupt_event: asyncio.Event, metadata: Optional[dict]
    # │  返回：Optional[asyncio.Task]
    # └──────────────────────────────────────────────────────────────
    def _start_typing_refresh(self, event: MessageEvent, interrupt_event: asyncio.Event,
                              metadata: Optional[dict]) -> Optional[asyncio.Task]:
        """Spawn the typing-refresh task, or None when ``typing_indicator=False``.
        ``stop_event`` is passed only when the (possibly overridden) ``_keep_typing`` accepts it."""
        if not getattr(self.config, "typing_indicator", True):
            # ⤷ 出口 return：返回 None
            return None
        kwargs: Dict[str, Any] = {"metadata": metadata}
        if self._accepts_kwarg(self._keep_typing, "stop_event", var_kw=False, unknown=True):
            kwargs["stop_event"] = interrupt_event
        # ⤷ 出口 return：返回 asyncio.create_task(self._keep_typing(event.source.chat_id, …
        return asyncio.create_task(self._keep_typing(event.source.chat_id, **kwargs))

    # ┌── 异步函数（协程） def _extract_response_content()
    # │  作用：Split handler 响应进入 deliverable 文本 + attachments. Order matters: MEDIA tags →
    # │  参数：self, response: str, event: MessageEvent, session_key: str, is_ephemeral_response: bool
    # │  返回：'_ExtractedResponse'
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _extract_response_content(self, response: str, event: MessageEvent, session_key: str,
                                        *, is_ephemeral_response: bool) -> "_ExtractedResponse":
        """Split a handler response into deliverable text + attachments. Order matters: MEDIA tags →
        image URLs → residual directives → bare local paths (skipped for ephemeral notices so config
        paths stay text; unknown-extension MEDIA tags survive for the bare-path detector). History
        dedup is bare-path only, off-loop, fail-open. An emptied non-empty response is recovered."""
        # Captured before extract_media strips it: images then go via send_document (no recompression).
        force_document = "[[as_document]]" in response
        pre_extract = response
        # The handler's routed profile scope is gone by now; Docker MEDIA translation and the
        # bare-path validator infer the sandbox from the ACTIVE profile (#109024).
        with self._media_delivery_scope(event.source):
            media_files, response = self.extract_media(response)
            media_files = self.filter_media_delivery_paths(media_files, session_key=session_key)
            images, text_content = self.extract_images(response)
            # Strip any remaining internal directives from message body (fixes #1561). _strip_media_directives
            # shares MEDIA_TAG_CLEANUP_RE, so a MEDIA: tag with an unknown extension is intentionally left in
            # the body for extract_local_files below to pick up rather than silently dropped (#34517).
            text_content = _strip_media_directives(text_content).strip()
            if images:
                logger.info("[%s] extract_images found %d image(s) in response (%d chars)", self.name, len(images), len(response))
            local_files = []
            if not is_ephemeral_response:
                local_files, text_content = self.extract_local_files(text_content)
                local_files = self.filter_local_delivery_paths(local_files, session_key=session_key)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        history = (await self._bounded_history_media_paths_for_session(session_key)
                   if local_files else None)
        if history:
            suppressed = [p for p in local_files if p in history]
            if suppressed:
                logger.info("[%s] Suppressing %d bare local file path(s) already delivered in "
                            "this session: %s", self.name, len(suppressed), suppressed)
                local_files = [p for p in local_files if p not in history]
        if local_files:
            logger.info("[%s] extract_local_files found %d file(s) in response", self.name, len(local_files))
        # A2 (#29346): extraction can reduce a non-empty response to empty text with no attachment, and the
        # `if text_content` guard below then drops it silently. Recover on every platform (#33842 was
        # Discord-only); the guard avoids duplicating an attachment.
        if not (text_content or images or local_files or media_files):
            _recovered = _strip_media_directives(response).strip()
            if _recovered:
                logger.warning("[%s] response_delivery_recovered: extract pipeline "
                               "reduced a non-empty response (%d chars) to empty with "
                               "no attachment; delivering recovered original to %s", self.name,
                               len(pre_extract), event.source.chat_id)
                text_content = _recovered
        # ⤷ 出口 return：返回 _ExtractedResponse(text_content=text_content, images=images,…
        return _ExtractedResponse(
            text_content=text_content, images=images, media_files=media_files,
            local_files=local_files, force_document_attachments=force_document, pre_extract=pre_extract)

    # ┌── 异步函数（协程） def _fire_post_delivery_callback()
    # │  作用：运行 one-shot post-delivery 回调 (bounded, 错误 swallowed). generation 是
    # │  参数：self, session_key: str, interrupt_event: asyncio.Event
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _fire_post_delivery_callback(self, session_key: str, interrupt_event: asyncio.Event) -> None:
        """Run the one-shot post-delivery callback (bounded, errors swallowed). The generation is
        read HERE — stamped on the interrupt event DURING the handler await; an earlier snapshot
        would let stale runs fire a fresher run's callbacks."""
        _post_cb = self.pop_post_delivery_callback(
            session_key, generation=getattr(interrupt_event, "_hermes_run_generation", None))
        if callable(_post_cb):
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                _post_result = _post_cb()
                if inspect.isawaitable(_post_result):
                    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                    await asyncio.wait_for(_post_result, timeout=_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS)

    # ┌── 函数 def _finish_session_task()
    # │  作用：End-的-task 护栏/ownership reconciliation. late ``_pending_messages`` arrival 必须不
    # │  参数：self, session_key: str, interrupt_event: asyncio.Event
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _finish_session_task(self, session_key: str, interrupt_event: asyncio.Event) -> None:
        """End-of-task guard/ownership reconciliation. A late ``_pending_messages`` arrival must not
        drop: re-queue it if another task already owns the session (drain handoff), else spawn the
        drain task and leave it the guard. Nothing pending: release the guard only if we still own
        it."""
        late_pending = self._pending_messages.pop(session_key, None)
        current_task = asyncio.current_task()
        if late_pending is not None:
            existing_task = self._session_tasks.get(session_key)
            if existing_task is not None and existing_task is not current_task:
                # The in-band drain (or an earlier late-arrival drain) already spawned a follow-up task that
                # owns this session. Re-queue the late-arrival event so that task picks it up — avoids
                # spawning two concurrent _process_message_background tasks for the same key (#17758
                # follow-up: prevents the create_task path from racing with itself across the
                # in-band/finally boundary).
                self._pending_messages[session_key] = late_pending
            else:
                logger.debug(
                    "[%s] Late-arrival pending message during cleanup — spawning drain task",
                    self.name)
                self._spawn_drain_task(late_pending, session_key)
        elif current_task is not None and self._session_tasks.get(session_key) is current_task:
            self._cleanup_finished_session_task(session_key, interrupt_event)

    # ┌── 异步函数（协程） def _process_message_background()
    # │  作用：Background task 那个 actually 进程消息.
    # │  参数：self, event: MessageEvent, session_key: str
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Background task that actually processes the message."""
        delivery_attempted = delivery_succeeded = False  # feeds the processing-complete hook

        def _record_delivery(result):
            # ⤷ nonlocal：声明赋值的是外层函数作用域的变量（闭包常见）。
            nonlocal delivery_attempted, delivery_succeeded
            if result is not None:
                delivery_attempted = True
                delivery_succeeded = delivery_succeeded or bool(getattr(result, "success", False))
        # Reuse the interrupt event handle_message() installed; new Event only if removed externally.
        interrupt_event = self._active_sessions.get(session_key) or asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        _thread_metadata = _thread_metadata_for_event(event)
        typing_task = self._start_typing_refresh(event, interrupt_event, _thread_metadata)
        # ⤷ 关键分支 try/except：捕获 asyncio.CancelledError、BaseException，出错走兜底而不是崩溃。
        try:
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._run_processing_hook("on_processing_start", event)
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            response = await self._message_handler(event)
            is_ephemeral_response = isinstance(response, EphemeralReply)
            # Unwrap EphemeralReply for downstream text processing; TTL applies after send.
            response, _ephemeral_ttl = self._unwrap_ephemeral(response)
            # None/empty is normal (streamed/queued). Suppress a stale response after an interrupt.
            if response and interrupt_event.is_set() and session_key in self._pending_messages:
                logger.info("[%s] Suppressing stale response for interrupted session %s", self.name,
                            session_key)
                response = None
            if not response:
                logger.debug("[%s] Handler returned empty/None response for %s", self.name, event.source.chat_id)
            else:
                extracted = await self._extract_response_content(
                    response, event, session_key, is_ephemeral_response=is_ephemeral_response)
                text_content, media_files = extracted.text_content, extracted.media_files
                # Final content gets notify=True; typing metadata stays unmarked (thread-strict).
                _final_thread_metadata = _mark_notify_metadata(_thread_metadata)
                _tts_paths, _tts_requested_path = [], None
                if self._wants_auto_tts(
                        event, session_key, interrupt_event, text_content, media_files):
                    _tts_paths, _tts_requested_path = await self._synthesize_auto_tts(text_content)
                # TTS plays before text; generated files are removed afterwards.
                _tts_caption_delivered = False
                for _tts_index, _tts_path in enumerate(_tts_paths):
                    try:
                        _tts_caption_delivered |= await self._play_tts_file(
                            event, text_content, _tts_path, _tts_index == 0, _final_thread_metadata,
                            _record_delivery)
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(_tts_path)
                if not _tts_paths and _tts_requested_path is not None:
                    with contextlib.suppress(OSError):
                        os.remove(_tts_requested_path)
                if text_content and not _tts_caption_delivered:
                    await self._send_final_text(
                        event, session_key, text_content, _final_thread_metadata,
                        is_ephemeral_response, _ephemeral_ttl, _record_delivery)
                await self._deliver_attachments(
                    event, extracted, _final_thread_metadata,
                    anything_sent=delivery_attempted or _tts_caption_delivered,
                    record_delivery=_record_delivery)
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            # Clean up the per-turn streaming-TTS flag.
            self._streaming_tts_completed_turns.discard(self._streaming_tts_turn_key(
                session_key, getattr(interrupt_event, "_hermes_run_generation", None),
                event=event) or "")
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._run_processing_hook(
                "on_processing_complete", event,
                ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE)
            # Force-flush an unfired debounce timer so this task hands off to a fresh drain task.
            # Clear the Event BEFORE the stop-typing await so concurrent inbound sees a live guard.
            # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
            await self._flush_text_debounce_now(session_key)
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                logger.debug("[%s] Processing queued follow-up message", self.name)
                self._clear_session_guard(session_key)
                await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
                self._spawn_drain_task(pending_event, session_key)
                return  # Drain task owns the session now.
        except asyncio.CancelledError:
            expected = asyncio.current_task() in self._expected_cancelled_tasks
            await self._run_processing_hook(
                "on_processing_complete", event,
                ProcessingOutcome.CANCELLED if expected else ProcessingOutcome.FAILURE)
            raise
        except BaseException as e:
            await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)
            _thread_metadata = (await self._notify_turn_error(event, e)) or _thread_metadata
            # SystemExit/KeyboardInterrupt propagate; other BaseExceptions are contained.
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
        finally:
            # Stop typing BEFORE the post-delivery callback: a stuck callback must not keep it
            # alive.
            await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
            await self._fire_post_delivery_callback(session_key, interrupt_event)
            # Callback work or a late refresh may have recreated typing — one final bounded stop.
            await self._stop_typing_refresh(
                event.source.chat_id, None, metadata=_thread_metadata, stop_attempts=1)
            # Flush any timer that missed the in-band drain, then reconcile ownership.
            await self._flush_text_debounce_now(session_key)
            self._finish_session_task(session_key, interrupt_event)

    # ┌── 函数 def _spawn_drain_task()
    # │  作用：Hand 会话 fresh task 用于 queued follow-up — never recurse (chained
    # │  参数：self, pending_event: MessageEvent, session_key: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _spawn_drain_task(self, pending_event: MessageEvent, session_key: str) -> None:
        """Hand the session to a fresh task for a queued follow-up — never recurse (chained
        follow-ups grew the C stack to SIGSEGV). Clearing (not deleting) the Event keeps the guard
        live for concurrent inbound; ownership moves so stale-lock detection works."""
        self._clear_session_guard(session_key)
        self._track_session_task(
            session_key,
            asyncio.create_task(self._process_message_background(pending_event, session_key)))

    # ┌── 函数 def _clear_session_guard()
    # │  作用：清除 (不 delete) 会话's interrupt 事件从而护栏 stays 当前活跃的用于 inbound.
    # │  参数：self, session_key: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _clear_session_guard(self, session_key: str) -> None:
        """Clear (not delete) the session's interrupt Event so the guard stays live for inbound."""
        _active = self._active_sessions.get(session_key)
        if _active is not None:
            _active.clear()

    # ┌── 函数 def _cleanup_finished_session_task()
    # │  作用：Release finished owner task's 护栏, 丢弃其 ``_session_tasks`` entry 仅若
    # │  参数：self, session_key: str, interrupt_event: Optional[asyncio.Event]
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _cleanup_finished_session_task(
        self, session_key: str, interrupt_event: Optional[asyncio.Event]) -> None:
        """Release a finished owner task's guard, dropping its ``_session_tasks`` entry ONLY if the
        guard was released: after a concurrent guard swap the done-task entry lets
        ``_session_task_is_stale`` heal the orphan.

        Release-then-conditional-delete is the #48300 fix: when a concurrent path (reset/new command, drain
        handoff) swapped ``_active_sessions[key]`` to a different guard, ``_release_session_guard`` skips on
        the guard mismatch and the lock stays installed. If we deleted ``_session_tasks`` unconditionally
        (the old order), ``_session_task_is_stale`` would later see no owner task and report "not stale", so
        the orphaned guard would never be healed — a permanent session deadlock. Keeping the done-task entry
        when the guard survives lets the on-entry self-heal detect the stale lock and clear it on the next
        inbound message.
        """
        self._release_session_guard(session_key, guard=interrupt_event)
        if session_key not in self._active_sessions:
            self._session_tasks.pop(session_key, None)

    # ┌── 异步函数（协程） def cancel_background_tasks()
    # │  作用：Cancel 在-flight background tasks (shutdown/replacement); 5s bound 每个,
    # │  参数：self
    # │  返回：None
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # └──────────────────────────────────────────────────────────────
    async def cancel_background_tasks(self) -> None:
        """Cancel in-flight background tasks (shutdown/replacement); 5s bound each,
        stragglers are untracked and left to unwind."""
        # Re-drain (max 5 rounds): a message arriving mid-gather spawns a task clear() would
        # untrack.
        for _ in range(5):
            tasks = [task for task in self._background_tasks if not task.done()]
            if not tasks:
                break
            for task in tasks:
                self._expected_cancelled_tasks.add(task)
                task.cancel()
            # ⤷ 关键分支 try/except：捕获 asyncio.TimeoutError，出错走兜底而不是崩溃。
            try:
                # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
                await asyncio.wait_for(
                    asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True),
                    timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[%s] %d background task(s) did not exit within 5s; "
                               "releasing tracking and letting them unwind in the background",
                               self.name, sum(not t.done() for t in tasks))
                break
        with contextlib.suppress(Exception):  # flush pending messages to disk before clearing
            from gateway.shutdown_flush import flush_pending_to_file
            flush_pending_to_file(self._pending_messages, reason="adapter_shutdown")
        for state in self._text_debounce_store().values():
            state.cancel_timer()
        for bucket in (self._background_tasks, self._expected_cancelled_tasks, self._session_tasks,
                       self._pending_messages, self._active_sessions, self._text_debounce_store()):
            bucket.clear()

    # ┌── 函数 def has_pending_interrupt()
    # │  作用：检查若 那里's pending interrupt 用于会话.
    # │  参数：self, session_key: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        # ⤷ 出口 return：返回 session_key in self._active_sessions and self._active_sessio…
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()

    # ┌── 函数 def get_pending_message()
    # │  作用：获取且 清除任意 pending 消息用于会话.
    # │  参数：self, session_key: str
    # │  返回：Optional[MessageEvent]
    # └──────────────────────────────────────────────────────────────
    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        # ⤷ 出口 return：返回 self._pending_messages.pop(session_key, None)
        return self._pending_messages.pop(session_key, None)

    # ┌── 函数 def build_source()
    # │  作用：构建 SessionSource; 带 ``gateway.profile_routes`` configured matching
    # │  参数：self, chat_id: str, chat_name: Optional[str] = None, chat_type: str = 'dm', user_id: Optional[str] = None, user_name: Optional[str] = None, thread_id: Optional[str] = None, chat_topic: Optional[str] = None  …共 18 个
    # │  返回：SessionSource
    # └──────────────────────────────────────────────────────────────
    def build_source(
        self, chat_id: str, chat_name: Optional[str] = None, chat_type: str = "dm",
        user_id: Optional[str] = None, user_name: Optional[str] = None,
        thread_id: Optional[str] = None, chat_topic: Optional[str] = None,
        user_id_alt: Optional[str] = None, chat_id_alt: Optional[str] = None, is_bot: bool = False,
        scope_id: Optional[str] = None, guild_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None, message_id: Optional[str] = None,
        role_authorized: bool = False, auto_thread_created: bool = False,
        auto_thread_initial_name: Optional[str] = None) -> SessionSource:
        """Build a SessionSource; with ``gateway.profile_routes`` configured the matching
        profile is stamped on ``source.profile`` for per-profile HERMES_HOME isolation."""
        def _opt(value) -> Optional[str]:
            # ⤷ 出口 return：返回 str(value) if value else None
            return str(value) if value else None
        fields = dict(
            platform=self.platform, chat_id=str(chat_id), chat_name=chat_name, chat_type=chat_type,
            user_id=_opt(user_id), user_name=user_name, thread_id=_opt(thread_id),
            chat_topic=(chat_topic or "").strip() or None, user_id_alt=user_id_alt,
            chat_id_alt=chat_id_alt, is_bot=is_bot, scope_id=_opt(scope_id),
            guild_id=_opt(guild_id), parent_chat_id=_opt(parent_chat_id),
            message_id=_opt(message_id))
        # Profile from configured routes, else the owning profile of a dedicated secondary bot (so no
        # later ``source.profile``-less fallback can re-route the message through the default bot's routes).
        owner_profile = getattr(self, "_owner_profile", None)
        profile, profile_route_rejected = owner_profile, False
        if self.gateway_runner is not None:
            from gateway.profile_routing import ProfileRouteRejected
            # ⤷ 关键分支 try/except：捕获 ProfileRouteRejected、Exception，出错走兜底而不是崩溃。
            try:
                profile = self.gateway_runner._profile_name_for_source(
                    SessionSource(**fields), adapter_profile=owner_profile) or owner_profile
            except ProfileRouteRejected:
                profile_route_rejected = True
            except Exception:
                logger.warning("Profile resolution failed for %s/%s, defaulting to active profile",
                               self.platform, chat_id, exc_info=True)
        source = SessionSource(**fields, profile=profile, role_authorized=role_authorized,
                               auto_thread_created=auto_thread_created,
                               auto_thread_initial_name=auto_thread_initial_name)
        # Transport-only, kept out of to_dict(): the receiving adapter is authoritative this turn
        # even if profile_routes picks another runtime; the reject flag is consumed before auth.
        source._transport_adapter_ref = weakref.ref(self)
        source.profile_route_rejected = profile_route_rejected
        # ⤷ 出口 return：返回 source
        return source

    # ┌── 异步函数（协程） def get_chat_info()
    # │  作用：获取 information about chat/通道; 字典带 在 least ``名称``
    # │  参数：self, chat_id: str
    # │  返回：Dict[str, Any]
    # │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/channel; dict with at least ``name``
        and ``type`` ("dm", "group", "channel")."""

    # ┌── 函数 def toolsets_for_source()
    # │  作用：Per-source 工具集覆盖值 REPLACING ``platform_toolsets.<平台>``, 或 None
    # │  参数：self, source: 'SessionSource'
    # │  返回：Optional[List[str]]
    # └──────────────────────────────────────────────────────────────
    def toolsets_for_source(self, source: "SessionSource") -> Optional[List[str]]:
        """Per-source toolset override REPLACING ``platform_toolsets.<platform>``, or None
        (default); validated via ``_get_platform_tools`` (webhook adapter pins per-route)."""
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def format_message()
    # │  作用：Format 消息用于这个平台 (覆盖值用于 e.g. Telegram
    # │  参数：self, content: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def format_message(self, content: str) -> str:
        """Format a message for this platform (override for e.g. Telegram
        MarkdownV2); default returns content as-is."""
        # ⤷ 出口 return：返回 content
        return content

    # ┌── 函数 def truncate_message()
    # │  作用：Split long 消息进入 chunks preserving code blocks: split inside fence 关闭它
    # │  参数：content: str, max_length: int = 4096, len_fn: Optional['Callable[[str], int]'] = None
    # │  返回：List[str]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def truncate_message(content: str, max_length: int = 4096,
                         len_fn: Optional["Callable[[str], int]"] = None) -> List[str]:
        """Split a long message into chunks preserving code blocks: a split inside a fence closes it
        at the chunk end and reopens it (same language tag) in the next; multi-chunk output gets
        ``(1/3)`` indicators. ``len_fn`` overrides ``len`` (``utf16_len`` for Telegram)."""
        _len = len_fn or len
        if _len(content) <= max_length:
            # ⤷ 出口 return：返回 [content]
            return [content]
        INDICATOR_RESERVE = 10   # room for " (XX/XX)"
        FENCE_CLOSE = "\n```"
        chunks: List[str] = []
        remaining = content
        carry_lang: Optional[str] = None  # language tag ("" ok) when previous chunk ended mid-fence
        while remaining:
            prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
            # Body budget after prefix/fence/indicator; floored so a tiny max_length can't stall.
            headroom = max_length - INDICATOR_RESERVE - _len(prefix) - _len(FENCE_CLOSE)
            if headroom < 1:
                headroom = max(1, max_length // 2)
            # Remainder fits in one final chunk; close a reopened fence if still open.
            if _len(prefix) + _len(remaining) <= max_length - INDICATOR_RESERVE:
                final_chunk = prefix + remaining
                if carry_lang is not None and fence_state_after(remaining, True, carry_lang)[0]:
                    final_chunk += FENCE_CLOSE
                chunks.append(final_chunk)
                break
            # Natural split (newline, then space); a custom _len budget maps to a codepoint offset.
            _cp_limit = (
                _custom_unit_to_cp(remaining, headroom, _len) if _len is not len else headroom)
            region = remaining[:_cp_limit]
            split_at = region.rfind("\n")
            if split_at < _cp_limit // 2:
                split_at = region.rfind(" ")
            if split_at < 1:
                # Floor at one codepoint: a zero _cp_limit (max_length 0/1, or a surrogate pair
                # wider than the utf16 budget) would never shrink ``remaining``; overshooting beats
                # a hang.
                split_at = max(1, _cp_limit)
            # Don't split inside an inline code span: an unpaired backtick breaks MarkdownV2.
            candidate = remaining[:split_at]
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    safe_split = max(
                        candidate.rfind(" ", 0, last_bt), candidate.rfind("\n", 0, last_bt))
                    if safe_split > _cp_limit // 4:
                        split_at = safe_split
            chunk_body = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()
            full_chunk = prefix + chunk_body
            # Walk only chunk_body (not the prepended prefix) for the fence state.
            in_code, lang = fence_state_after(chunk_body, carry_lang is not None, carry_lang or "")
            carry_lang = lang if in_code else None
            # Close the orphaned fence so the chunk stands alone.
            chunks.append(full_chunk + FENCE_CLOSE if in_code else full_chunk)
        if len(chunks) > 1:
            chunks = [f"{chunk} ({i + 1}/{len(chunks)})" for i, chunk in enumerate(chunks)]
        # ⤷ 出口 return：返回 chunks
        return chunks
