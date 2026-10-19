"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：agent/model_metadata.py（约 2302 行）
# 地位：模型元数据（尤其是上下文窗口长度）和 token 估算。
#
# 为什么需要它：
#   不同模型的上下文长度不同（8K / 128K / 200K / 1M），而且本地模型（Ollama、LM Studio、llama.cpp）
#   的实际窗口要连上服务才能问到。压缩器必须先知道上限，才知道什么时候该压缩。
#
# 主要函数：
#   get_model_context_length(model, base_url, ...)   拿到模型的上下文长度（带多级缓存 + 探测）
#   estimate_tokens_rough(text) / estimate_messages_tokens_rough(messages)   粗略估算 token 数
#   fetch_model_metadata() / fetch_endpoint_model_metadata()   从远端拉模型清单
#   parse_context_limit_from_error(error_msg)   从报错信息里反推出上下文上限
#      （很实用：模型报 context_length_exceeded 时，错误信息里往往带着真实上限）
#
# 工程亮点：
#   - 三级缓存：内存 → 磁盘缓存（带 TTL）→ 真去问服务端
#   - 本地服务探测：_query_ollama_api_show / _lmstudio_context / _llamacpp_context
#   - 坏端点记忆：_note_endpoint_blackholed —— 连不上的地址记下来，别每次都傻等超时
#   - 中文/CJK 字符的 token 密度和英文不同，_is_cjk_token_dense_char 专门处理
# ==========================================================================
#

import base64
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:  # pragma: no cover — runtime import is lazy (see below)
    import requests

from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, base_url_hostname

from hermes_constants import OPENROUTER_MODELS_URL, openrouter_variant_base
from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

logger = logging.getLogger(__name__)

# ``requests`` costs ~27 ms of the `import cli` waterfall, so it is resolved lazily:
# ``_ensure_requests()`` at runtime, PEP 562 ``__getattr__`` for ``patch("agent.model_metadata.requests.get")``.


# ┌── 函数 def _ensure_requests()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _ensure_requests():
    if "requests" not in globals():
        import requests as _requests
        globals()["requests"] = _requests
    # ⤷ 出口 return：返回 globals()['requests']
    return globals()["requests"]


# ┌── 函数 def __getattr__()
# │  作用：魔术方法 __getattr__（Python 在特定场景自动调用，见下方说明）
# │  参数：name: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
# 💡【本项目惯用法】模块级 __getattr__：当外部访问本模块里尚未导入的名字时才会触发，
#    这里用来做「懒加载」——真正用到时才 import，避免启动慢和循环导入。
#    所以你看到 `from xxx import YYY` 在本文件里搜不到定义，是正常的。
def __getattr__(name: str):
    if name == "requests":
        # ⤷ 出口 return：返回 _ensure_requests()
        return _ensure_requests()
    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ┌── 函数 def _resolve_requests_verify()
# │  作用：SSL ``verify`` 用于 ``请求`` probes; 镜像 ``Agent.ssl_verify.resolve_httpx_verify``.
# │  参数：base_url: str = ''
# │  返回：bool | str
# └──────────────────────────────────────────────────────────────
def _resolve_requests_verify(base_url: str = "") -> bool | str:
    """SSL ``verify`` for ``requests`` probes; mirrors ``agent.ssl_verify.resolve_httpx_verify``.
    Priority: per-provider ``ssl_verify: false`` -> per-provider ``ssl_ca_cert`` (else probes log
    spurious CERTIFICATE_VERIFY_FAILED while the httpx chat path succeeds) -> CA env vars -> certifi."""
    if base_url:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from hermes_cli.config import get_custom_provider_tls_settings
            tls = get_custom_provider_tls_settings(base_url)
            if tls.get("ssl_verify") is False:
                # ⤷ 出口 return：返回 False
                return False
            ca = tls.get("ssl_ca_cert")
            if isinstance(ca, str) and ca and os.path.isfile(ca):
                # ⤷ 出口 return：返回 ca
                return ca
        except Exception:
            pass  # fall through to env vars — never break a probe on config lookup
    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            # ⤷ 出口 return：返回 val
            return val
    # ⤷ 出口 return：返回 True
    return True


# Snapshot for callers inspecting this constant; prefix routing queries the registry live.
try:
    from providers import list_providers as _list_providers
except Exception:
    def _list_providers():
        return []

_PROVIDER_PREFIXES: frozenset[str] = frozenset(
    value.lower()
    for profile in _list_providers()
    for value in (profile.name, *profile.aliases)
)
_OLLAMA_TAG_PATTERN = re.compile(r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)", re.IGNORECASE)
# Tailscale CGNAT (RFC 6598): `ipaddress.is_private` excludes it, yet Ollama
# reached over Tailscale must count as local (timeout auto-bumps).
_TAILSCALE_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")


# ┌── 函数 def _strip_provider_prefix()
# │  作用：Strip registry-known 服务商 prefix: ``"local:m"`` -> ``"m"``. Ollama ``模型:tag``
# │  参数：model: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _strip_provider_prefix(model: str) -> str:
    """Strip a registry-known provider prefix: ``"local:m"`` -> ``"m"``. Ollama ``model:tag``
    ids are preserved even when the model half is a provider name (``qwen:0.5b``)."""
    if ":" not in model or model.startswith("http"):
        # ⤷ 出口 return：返回 model
        return model
    prefix, suffix = model.split(":", 1)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from providers import get_provider_profile
        is_provider = get_provider_profile(prefix.strip().lower()) is not None
    except Exception:
        is_provider = False
    if is_provider and not _OLLAMA_TAG_PATTERN.match(suffix.strip()):
        # ⤷ 出口 return：返回 suffix
        return suffix
    # ⤷ 出口 return：返回 model
    return model


_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600
# In-memory memo keyed by (base_url, api-key fingerprint): per-key gateways return a per-key catalog, and
# in a multiplexed process two profiles may share a URL with different keys. The disk memo stays per URL.
_endpoint_model_metadata_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[Tuple[str, str], float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300
# Server-type verdicts (server_type, monotonic_ts): positive ones live an hour so a
# server swap on the same port is re-detected; None gets the short TTL so a
# transient failure recovers in minutes without re-running the waterfall each turn.
_ENDPOINT_PROBE_TTL_SECONDS = 3600.0
# A failed probe verdict (server_type is None — no known endpoint answered) is cached for a much shorter
# window: the in-memory entry exists only to keep one image-bearing turn from re-running the 5-request
# waterfall on every subsequent turn (#89863 — a keyed remote endpoint answered 401 to each leg and the None
# verdict was never cached, so every turn re-probed). Short TTL keeps a transient failure (server starting
# up, key being fixed) recoverable within minutes instead of pinning "undetected" for an hour.
_ENDPOINT_PROBE_FAILURE_TTL_SECONDS = 300.0
_endpoint_probe_path_cache: Dict[str, tuple] = {}
# Routable-but-dead endpoints (corp LAN off-VPN) blackhole TCP: once ANY probe paid
# a full connect timeout, later probes short-circuit for a while.
_ENDPOINT_BLACKHOLE_TTL_SECONDS = 30.0
_endpoint_blackhole_cache: Dict[str, float] = {}  # host:port -> monotonic ts


# ┌── 函数 def _parse_base_url()
# │  作用：``urlparse`` 的已归一化 URL (``scheme`` prepended 当 absent); None 当 empty.
# │  参数：base_url: str, scheme: str = 'http'
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _parse_base_url(base_url: str, scheme: str = "http"):
    """``urlparse`` of the normalized URL (``scheme`` prepended when absent); None when empty."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 urlparse(normalized if '://' in normalized else f'{scheme}:/…
    return urlparse(normalized if "://" in normalized else f"{scheme}://{normalized}")


# ┌── 函数 def _endpoint_host_key()
# │  作用：``宿主:port`` 键 (None 不宿主) 从而每个 probe 路径用于 one 服务端 shares entry.
# │  参数：base_url: str
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _endpoint_host_key(base_url: str) -> Optional[str]:
    """``host:port`` key (None without a host) so every probe path for one server shares an entry."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        parsed = _parse_base_url(base_url)
        if parsed is None or not parsed.hostname:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme ==…
        return f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _note_endpoint_blackholed()
# │  作用：Record 那个 probe ``base_url`` timed out 在…期间 TCP connect.
# │  参数：base_url: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _note_endpoint_blackholed(base_url: str) -> None:
    """Record that a probe to ``base_url`` timed out during TCP connect."""
    key = _endpoint_host_key(base_url)
    if key is not None:
        _endpoint_blackhole_cache[key] = time.monotonic()
        logger.debug("Endpoint %s timed out connecting — skipping further probes for %.0fs", key, _ENDPOINT_BLACKHOLE_TTL_SECONDS)


# ┌── 函数 def _endpoint_blackholed()
# │  作用：True 若 recent probe ``base_url`` timed out 在…期间 TCP connect (缓存 lookup 仅).
# │  参数：base_url: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _endpoint_blackholed(base_url: str) -> bool:
    """True if a recent probe to ``base_url`` timed out during TCP connect (cache lookup only)."""
    if _ENDPOINT_BLACKHOLE_TTL_SECONDS <= 0:
        # ⤷ 出口 return：返回 False
        return False
    key = _endpoint_host_key(base_url)
    seen = _endpoint_blackhole_cache.get(key) if key is not None else None
    if seen is None:
        # ⤷ 出口 return：返回 False
        return False
    if (time.monotonic() - seen) >= _ENDPOINT_BLACKHOLE_TTL_SECONDS:
        del _endpoint_blackhole_cache[key]
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 出口 return：返回 True
    return True


# ┌── 函数 def _note_if_connect_timeout()
# │  作用：Blackhole ``base_url`` 当 ``exc`` 是 connect-阶段超时.
# │  参数：exc: BaseException, base_url: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _note_if_connect_timeout(exc: BaseException, base_url: str) -> None:
    """Blackhole ``base_url`` when ``exc`` is a connect-phase timeout."""
    if _is_connect_timeout(exc):
        _note_endpoint_blackholed(base_url)


# ┌── 函数 def _is_connect_timeout()
# │  作用：当 connect-阶段超时抛出由 httpx 或请求. 读取超时是 时返回 True。
# │  参数：exc: BaseException
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_connect_timeout(exc: BaseException) -> bool:
    """True for connect-phase timeouts raised by httpx or requests. Read timeouts are
    excluded: the server accepted the connection, the opposite of a blackhole."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        import httpx
        from requests.exceptions import ConnectTimeout
        # ⤷ 出口 return：返回 isinstance(exc, (httpx.ConnectTimeout, ConnectTimeout))
        return isinstance(exc, (httpx.ConnectTimeout, ConnectTimeout))
    except Exception:
        # ⤷ 出口 return：返回 False
        return False


# Disk L2 for local-endpoint probes so back-to-back CLI cold starts skip the waterfall.
# Only SUCCESSFUL probes persist (a down server must not pin a negative verdict).
_LOCAL_PROBE_DISK_TTL_SECONDS = 300.0


# ┌── 函数 def _cache_file()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：name: str
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _cache_file(name: str) -> Path:
    from hermes_constants import get_hermes_home
    # ⤷ 出口 return：返回 get_hermes_home() / 'cache' / name
    return get_hermes_home() / "cache" / name


# ┌── 函数 def _load_json_dict()
# │  作用：JSON 对象在 ``路径``, 或 {} 当 missing/invalid.
# │  参数：path: Path
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _load_json_dict(path: Path) -> Dict[str, Any]:
    """JSON object at ``path``, or {} when missing/invalid."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # ⤷ 出口 return：返回 data if isinstance(data, dict) else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        # ⤷ 出口 return：返回 {}
        return {}


# ┌── 函数 def _ttl_memo_get()
# │  作用：Fresh (< ``ttl`` seconds) ``value_key`` 的 ``键`` entry 在 TTL memo 文件, 否则 None.
# │  参数：path: Path, key: str, ttl: float, ts_key: str, value_key: str
# │  返回：Optional[Any]
# └──────────────────────────────────────────────────────────────
def _ttl_memo_get(path: Path, key: str, ttl: float, *, ts_key: str, value_key: str) -> Optional[Any]:
    """Fresh (< ``ttl`` seconds) ``value_key`` of the ``key`` entry in a TTL memo file, else None."""
    entry = _load_json_dict(path).get(key)
    if not isinstance(entry, dict):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        if (time.time() - float(entry[ts_key])) >= ttl:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 entry[value_key]
        return entry[value_key]
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _ttl_memo_put()
# │  作用：写入 ``键`` 进入 TTL memo 文件, 丢弃 expired siblings. 尽最大努力.
# │  参数：path: Path, key: str, value: Any, ttl: float, ts_key: str, value_key: str, what: str, ts_first: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _ttl_memo_put(path: Path, key: str, value: Any, ttl: float, *, ts_key: str, value_key: str, what: str, ts_first: bool = False) -> None:
    """Write ``key`` into a TTL memo file, dropping expired siblings. Best-effort."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        now = time.time()
        data = {k: v for k, v in _load_json_dict(path).items() if isinstance(v, dict) and (now - float(v.get(ts_key, 0))) < ttl}
        data[key] = {ts_key: now, value_key: value} if ts_first else {value_key: value, ts_key: now}
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save %s: %s", what, e)


# ┌── 函数 def _local_probe_disk_cache_path()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _local_probe_disk_cache_path() -> Path:
    # ⤷ 出口 return：返回 _cache_file('local_endpoint_probes.json')
    return _cache_file("local_endpoint_probes.json")


# ┌── 函数 def _local_probe_disk_get()
# │  作用：返回 fresh cached 值用于 ``种类:键``, 否则 None.。
# │  参数：kind: str, key: str
# │  返回：Optional[Any]
# └──────────────────────────────────────────────────────────────
def _local_probe_disk_get(kind: str, key: str) -> Optional[Any]:
    """Return a fresh cached value for ``kind:key``, else None."""
    # ⤷ 出口 return：返回 _ttl_memo_get(_local_probe_disk_cache_path(), f'{kind}:{key}…
    return _ttl_memo_get(_local_probe_disk_cache_path(), f"{kind}:{key}", _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value")


# ┌── 函数 def _local_probe_disk_put()
# │  作用：持久化 successful probe 结果. 尽最大努力; prunes 卡死的 entries.
# │  参数：kind: str, key: str, value: Any
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _local_probe_disk_put(kind: str, key: str, value: Any) -> None:
    """Persist a successful probe result. Best-effort; prunes stale entries."""
    _ttl_memo_put(_local_probe_disk_cache_path(), f"{kind}:{key}", value, _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value", what="local probe disk cache")


# ┌── 函数 def _get_model_metadata_cache_path()
# │  作用：路径 OpenRouter 模型 metadata disk 缓存.
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _get_model_metadata_cache_path() -> Path:
    """Path to the OpenRouter model metadata disk cache."""
    # ⤷ 出口 return：返回 _cache_file('openrouter_model_metadata.json')
    return _cache_file("openrouter_model_metadata.json")


# ┌── 函数 def _model_metadata_disk_cache_age_seconds()
# │  作用：Disk-缓存 age 在 seconds, 或 None 若 freshness 是 unknown.
# │  参数：无
# │  返回：Optional[float]
# └──────────────────────────────────────────────────────────────
def _model_metadata_disk_cache_age_seconds() -> Optional[float]:
    """Disk-cache age in seconds, or None if freshness is unknown."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        age = time.time() - _get_model_metadata_cache_path().stat().st_mtime
        # ⤷ 出口 return：返回 age if age >= 0 else None
        return age if age >= 0 else None
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _load_model_metadata_disk_cache()
# │  作用：Processed OpenRouter metadata 缓存从 disk ({} 在…上任意 failure).
# │  参数：无
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _load_model_metadata_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Processed OpenRouter metadata cache from disk ({} on any failure)."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with _get_model_metadata_cache_path().open("r", encoding="utf-8") as f:
            data = json.load(f)
        # ⤷ 出口 return：返回 {str(key): value for key, value in data.items() if isinstanc…
        return {str(key): value for key, value in data.items() if isinstance(value, dict)} if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("Failed to load OpenRouter model metadata disk cache: %s", e)
        # ⤷ 出口 return：返回 {}
        return {}


# ┌── 函数 def _save_model_metadata_disk_cache()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：data: Dict[str, Dict[str, Any]]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _save_model_metadata_disk_cache(data: Dict[str, Dict[str, Any]]) -> None:
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        atomic_json_write(_get_model_metadata_cache_path(), data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save OpenRouter model metadata disk cache: %s", e)


# ┌── 函数 def _get_endpoint_metadata_cache_path()
# │  作用：在…上-disk memo 的 remote ``/模型`` probes (see ``_endpoint_disk_cache_get``).
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _get_endpoint_metadata_cache_path() -> Path:
    """On-disk memo of remote ``/models`` probes (see ``_endpoint_disk_cache_get``)."""
    # ⤷ 出口 return：返回 _cache_file('endpoint_model_metadata.json')
    return _cache_file("endpoint_model_metadata.json")


# ┌── 函数 def _endpoint_disk_cache_get()
# │  作用：Fresh cross-进程 memo 的 remote ``/模型`` probe (same TTL 作为在-内存): one-shot
# │  参数：normalized: str
# │  返回：Optional[Dict[str, Dict[str, Any]]]
# └──────────────────────────────────────────────────────────────
def _endpoint_disk_cache_get(normalized: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fresh cross-process memo of a remote ``/models`` probe (same TTL as in-memory): one-shot
    runs (``hermes -q``, cron) start cold and Nous bypasses the persistent context cache, so
    without this every launch paid the live probe. Local endpoints are never memoized."""
    models = _ttl_memo_get(_get_endpoint_metadata_cache_path(), normalized, _ENDPOINT_MODEL_CACHE_TTL, ts_key="at", value_key="models")
    # ⤷ 出口 return：返回 models if isinstance(models, dict) else None
    return models if isinstance(models, dict) else None


# ┌── 函数 def _endpoint_disk_cache_put()
# │  作用：记忆化 successful remote ``/模型`` probe; expired siblings 是已丢弃.
# │  参数：normalized: str, cache: Dict[str, Dict[str, Any]]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _endpoint_disk_cache_put(normalized: str, cache: Dict[str, Dict[str, Any]]) -> None:
    """Memoize a successful remote ``/models`` probe; expired siblings are dropped."""
    _ttl_memo_put(
        _get_endpoint_metadata_cache_path(), normalized, cache, _ENDPOINT_MODEL_CACHE_TTL,
        ts_key="at", value_key="models", what="endpoint model metadata disk cache", ts_first=True)


# Descending probe tiers for unknown models; tier[0] is also the default fallback.
CONTEXT_PROBE_TIERS = [256_000, 128_000, 64_000, 32_000, 16_000, 8_000]
DEFAULT_FALLBACK_CONTEXT = CONTEXT_PROBE_TIERS[0]
_FALLBACK_WARNED: set = set()  # the fallback is never cached, so dedupe its warning per (model, base_url)


# ┌── 函数 def _warn_context_length_fallback()
# │  作用：警告一旦 per 模型+endpoint 那个 detection failed 且 hard 默认值是 used,
# │  参数：model: str, base_url: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _warn_context_length_fallback(model: str, base_url: str) -> None:
    """Warn once per model+endpoint that detection failed and the hard default is used,
    so small-context models (8K, 32K) don't silently get 256K and fail at the API."""
    key = (model, base_url or "")
    if key not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(key)
        logger.warning(
            "Could not determine context length for model %r (base_url=%s) — falling back to %s tokens. "
            "Set model.context_length in config.yaml to override.",
            model, base_url or "default", f"{DEFAULT_FALLBACK_CONTEXT:,}",
        )


# Sessions, model switches and cron jobs reject models below this (too little working memory).
MINIMUM_CONTEXT_LENGTH = 64_000
# In-process (model, base_url) -> (result, monotonic_ts) memo for local probes: one
# startup resolves the same model several times (banner, /model, compressor). Never persisted.
_LOCAL_CTX_PROBE_TTL_SECONDS = 30.0
_LOCAL_CTX_PROBE_CACHE: Dict[tuple, tuple] = {}
# Family-pattern fallbacks, used only when provider-aware sources all miss.
# Lookups are longest-key-first substring matches, so dict order is cosmetic
# and a specific key must be STRICTLY longer than its catch-all.
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic — bare ids only (prefixed ids resolve via OpenRouter/models.dev
    # and would collide: "anthropic/claude-sonnet-4" ⊂ "anthropic/claude-sonnet-4.6").
    "claude-fable-5": 1000000, "claude-fable": 1000000, "claude-opus-5": 1000000, "claude-sonnet-5": 1000000,
    "claude-opus-4-8": 1000000, "claude-opus-4.8": 1000000, "claude-opus-4-7": 1000000, "claude-opus-4.7": 1000000,
    "claude-opus-4-6": 1000000, "claude-sonnet-4-6": 1000000, "claude-opus-4.6": 1000000, "claude-sonnet-4.6": 1000000,
    "claude": 200000,  # catch-all for older Claude models
    # OpenAI — direct-API windows (Codex OAuth caps gpt-5.4+/5.5/5.6 at 272K, resolved by
    # its own branch). 5.4-nano/-mini are 400k, not 1.05M; gpt-5.3-codex-spark is
    # Codex-OAuth-only and listed so "gpt-5" (400k) doesn't win.
    "gpt-6-astra": 1050000,  # also matches -pro (verified live on OpenRouter)
    "gpt-5.6-luna": 1050000, "gpt-5.6-terra": 1050000, "gpt-5.6-sol": 1050000, "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000, "gpt-5.4-mini": 400000, "gpt-5.4": 1050000,
    "gpt-5.3-codex-spark": 128000, "gpt-5.1-chat": 128000, "gpt-5": 400000,
    "gpt-4.1": 1047576, "gpt-4": 128000,
    # Google / Gemma ("gemma4" is Ollama-style naming, e.g. gemma4:31b-cloud)
    "gemini": 1048576,
    "gemma-4": 256000, "gemma4": 256000, "gemma-4-31b": 256000, "gemma-3": 131072, "gemma": 8192,
    # DeepSeek — V4 family is 1M; deepseek-chat/-reasoner alias v4-flash modes. ``deepseek-flash``
    # (version-less canonical id, 2026-09 Flash refresh) needs a discrete entry or the
    # longest-key-first scan falls through to the 128K ``deepseek`` catch-all below.
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-v4-pro": 1_000_000, "deepseek-v4.1-flash": 1_000_000, "deepseek-v4-flash": 1_000_000, "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000, "deepseek-flash": 1_000_000, "deepseek": 128000,
    # Meta; Muse Spark family (1.1/1.2/1.3, -contributor(-free), meta/ prefixed) is 1M per OpenRouter,
    # models.dev and api.commandcode.ai /models — keep the "muse-spark" prefix (bare "muse" would match
    # muse-image/muse-voice). Thinking Machines inkling (covers inkling-small and :free/:batch variants)
    "llama": 131072, "muse-spark-1.3": 1_048_576, "muse-spark": 1_048_576, "inkling": 1_048_576,
    # Qwen — https://help.aliyun.com/zh/model-studio/developer-reference/ (3.8-max/flash
    # 1M verified on OpenRouter & Nous portal 2026-08; qwen3-max = 256K Coding Plan snapshot)
    "qwen3.8-max": 1_000_000, "qwen3.8-flash": 1_000_000, "qwen3.6-plus": 1048576, "qwen3.7-plus": 1048576,
    "qwen3-coder-plus": 1000000, "qwen3-coder": 262144, "qwen3-max": 262144, "qwen": 131072,
    # MiniMax — M3 is 1M; M2.x is 204,800. https://platform.minimax.io/docs/api-reference/text-chat-openai
    "minimax-m3": 1000000, "minimax": 204800,
    # GLM — Nous + OpenRouter /v1/models (2026-09-09): 5.3 / 5.3-flash 1,310,720 (:batch/:US 1,048,576);
    # 5.2 1,048,576; 5 / 5.1 / 4.7 / 4.6 204,800; *-turbo / 4.7-flash 202,752 (the catch-all).
    # The OpenRouter :free variant is capped; the longer key wins.
    "glm-5.3": 1_310_720, "glm-5.3-flash": 1_310_720, "glm-5.3:batch": 1_048_576, "glm-5.3:us": 1_048_576,
    "glm-5.3-flash:batch": 1_048_576, "glm-5.3-flash:us": 1_048_576,
    "glm-5.2": 1_048_576, "glm-5.2:free": 256_000,
    "glm-5.1": 204_800, "glm-5-turbo": 202752, "glm-5v-turbo": 202752, "glm-5": 204_800,
    "glm-4.7-flash": 202752, "glm-4.7": 204_800, "glm-4.6v": 131072, "glm-4.6": 204_800, "glm": 202752,
    # xAI — /v1/models returns no context_length, so these prevent probe-down on api.x.ai
    # custom providers (docs.x.ai). grok-composer(-2.5-fast, Grok Build CLI) is OAuth-only:
    # 200k usable (the /v1/responses ~262144 input+output budget is a separate limit).
    # grok-build-latest aliases grok-4.5; grok-4-fast / grok-4.20 also match their
    # -(non-)reasoning and -multi-agent variants; "grok" is the catch-all.
    "grok-composer": 200000, "grok-build-latest": 500000, "grok-build": 256000, "grok-code-fast": 256000,
    "grok-2-vision": 8192, "grok-4-fast": 2000000, "grok-4.20": 2000000,
    "grok-4.6": 500000, "grok-4.5": 500000, "grok-4.3": 1000000, "grok-4": 256000,
    "grok-3": 131072, "grok-2": 131072, "grok": 131072,
    # Kimi — K3 is 1 Mi (matches the endpoint-scoped override); older Kimi 256K.
    "kimi-k3": 1_048_576, "kimi": 262144,
    # Upstage Solar — /v1/models returns no context_length; dated variants resolve via prefix.
    "solar-open2": 262144, "solar-pro3": 131072, "solar-pro2": 65536, "solar-mini": 32768,
    # Tencent Hunyuan (262144 = 256 × 1024, aligned with OpenRouter live metadata)
    "hy4-preview": 1_048_576, "hy3-preview": 262144, "hy3": 262144,
    # "Ox Alpha" stealth model (OpenCode Zen / OpenRouter slugs); NVIDIA Nemotron (128K
    # except 3.5 Lightning); Poolside Laguna 2.1 (:free / -free slugs); Arcee; OpenRouter.
    "x-preview-f": 1_048_576, "ox-alpha": 1_048_576,
    "nemotron-3.5-lightning": 1_000_000, "nemotron": 131072,
    "laguna-s-2.1": 262144, "laguna-xs-2.1": 262144, "trinity": 262144, "elephant": 262144,
    # Hugging Face Inference Providers — model IDs use org/name format
    "Qwen/Qwen3.5-397B-A17B": 131072, "Qwen/Qwen3.5-35B-A3B": 131072, "deepseek-ai/DeepSeek-V3.2": 65536,
    "moonshotai/Kimi-K2.5": 262144, "moonshotai/Kimi-K2.6": 262144, "moonshotai/Kimi-K2-Thinking": 262144,
    "MiniMaxAI/MiniMax-M2.5": 204800, "XiaomiMiMo/MiMo-V2-Flash": 262144,
    "mimo-v2-pro": 1048576, "mimo-v2.5-pro": 1048576, "mimo-v2.5": 1048576, "mimo-v2-omni": 262144, "mimo-v2-flash": 262144,
    "zai-org/GLM-5": 202752,
}
# xAI Grok models that ACCEPT `reasoning.effort` (verified live against
# /v1/responses). Unlisted Grok models still reason natively but 400 on the
# parameter, so callers must send no `reasoning` key rather than a default `medium`.
# grok-4.5/4.6 accept low/medium/high (default high) but REJECT "none", unlike grok-4.3.
_GROK_EFFORT_CAPABLE_PREFIXES = ("grok-3-mini", "grok-4.20-multi-agent", "grok-4.3", "grok-4.5", "grok-4.6")


# ┌── 函数 def grok_supports_reasoning_effort()
# │  作用：白名单检查 (aggregator prefixes like ``x-ai/`` stripped); unknown Grok 模型获取无 effort dial.
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def grok_supports_reasoning_effort(model: str) -> bool:
    """Allowlist check (aggregator prefixes like ``x-ai/`` stripped); unknown Grok models get no effort dial."""
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    # ⤷ 出口 return：返回 bool(name) and any((name.startswith(prefix) for prefix in _G…
    return bool(name) and any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


# ┌── 函数 def is_grok_46_family()
# │  作用：是否 *模型* 是 Grok 4.6 family identifier.。
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_grok_46_family(model: str) -> bool:
    """Whether *model* is a Grok 4.6 family identifier."""
    name = (model or "").strip().lower().replace("_", "-").rsplit("/", 1)[-1]
    # ⤷ 出口 return：返回 name == 'grok-4.6' or name.startswith('grok-4.6-')
    return name == "grok-4.6" or name.startswith("grok-4.6-")


_CONTEXT_LENGTH_KEYS = (
    "context_length", "context_window", "context_size", "max_context_length", "max_position_embeddings",
    "max_model_len", "max_input_tokens", "max_sequence_length", "max_seq_len", "n_ctx_train", "n_ctx", "ctx_size",
)
_MAX_COMPLETION_KEYS = ("max_completion_tokens", "max_output_tokens", "max_tokens")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
# Docker / Podman / Lima DNS names that resolve to the host machine
_CONTAINER_LOCAL_SUFFIXES = (".docker.internal", ".containers.internal", ".lima.internal")


# ┌── 函数 def _normalize_base_url()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：base_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _normalize_base_url(base_url: str) -> str:
    # ⤷ 出口 return：返回 (base_url or '').strip().rstrip('/')
    return (base_url or "").strip().rstrip("/")


# ┌── 函数 def _auth_headers()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：api_key: object = ''
# │  返回：Dict[str, str]
# └──────────────────────────────────────────────────────────────
def _auth_headers(api_key: object = "") -> Dict[str, str]:
    from agent.command_token_source import materialize_probe_api_key
    token = materialize_probe_api_key(api_key)
    # ⤷ 出口 return：返回 {'Authorization': f'Bearer {token}'} if token else {}
    return {"Authorization": f"Bearer {token}"} if token else {}


# ┌── 函数 def _is_custom_endpoint()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：base_url: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_custom_endpoint(base_url: str) -> bool:
    normalized = _normalize_base_url(base_url)
    # ⤷ 出口 return：返回 bool(normalized) and (not base_url_host_matches(normalized, …
    return bool(normalized) and not base_url_host_matches(normalized, "openrouter.ai")


# Host substring -> provider. ".githubcopilot.com" covers api.enterprise./api.business.
# hosts; models.inference.ai.azure.com (GitHub Models free tier, ~8K per-request cap)
# is mapped so a targeted hint fires instead of the custom-endpoint path.
_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai", "chatgpt.com": "openai", "api.anthropic.com": "anthropic",
    "api.z.ai": "zai", "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding", "api.moonshot.cn": "kimi-coding-cn", "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun", "api.stepfun.com": "stepfun", "api.arcee.ai": "arcee", "api.minimax": "minimax",
    "dashscope.aliyuncs.com": "alibaba", "dashscope-intl.aliyuncs.com": "alibaba", "portal.qwen.ai": "qwen-oauth",
    "openrouter.ai": "openrouter", "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous", "api.deepseek.com": "deepseek",
    "api.githubcopilot.com": "copilot", ".githubcopilot.com": "copilot", "models.github.ai": "copilot",
    "models.inference.ai.azure.com": "copilot",
    "api.fireworks.ai": "fireworks", "opencode.ai": "opencode-go", "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia", "api.xiaomimimo.com": "xiaomi", "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi", "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub", "api.lkeap.cloud.tencent.com": "tencent-tokenplan",
    "ollama.com": "ollama-cloud",
}

# Auto-extend with provider-profile hostnames not already mapped.
try:
    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:
    pass


# ┌── 函数 def _infer_provider_from_url()
# │  作用：模型.dev 服务商名称用于 base URL (custom endpoints need 无 explicit 服务商).
# │  参数：base_url: str
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """models.dev provider name for a base URL (custom endpoints need no explicit provider)."""
    parsed = _parse_base_url(base_url, "https")
    if parsed is None:
        # ⤷ 出口 return：返回 None
        return None
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            # ⤷ 出口 return：返回 provider
            return provider
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _strip_openrouter_routing_variant()
# │  作用：Strip OpenRouter routing-variant suffix 用于 catalog lookup.
# │  参数：model: str, base_url: str = '', provider: str = ''
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _strip_openrouter_routing_variant(
    model: str, base_url: str = "", provider: str = ""
) -> str:
    """Strip an OpenRouter routing-variant suffix for catalog lookup.

    ``:nitro`` / ``:floor`` / ``:exacto`` / ``:online`` are request-time
    routing modifiers, NOT catalog entries — OpenRouter's ``/models`` lists
    only the base id, and a variant shares the base model's context window.
    Without this, every lookup below misses and the resolver falls through to
    a generic family default (``x-ai/grok-4.6:nitro`` → the 131K ``grok``
    catch-all instead of its real 2M window).

    Only the id used for LOOKUP is rewritten. The suffixed id the caller holds
    stays on the wire, so the routing opt-in is preserved — the same rule
    :func:`hermes_cli.models.validate_requested_model` applies. Sharing the
    base's cache key is intentional: the window is identical, so a variant and
    its base must never disagree.

    Narrow by design: only applied when the request actually routes through
    OpenRouter, so a local ``model:tag`` that happens to end in one of these
    words is untouched.
    """
    if not model:
        # ⤷ 出口 return：返回 model
        return model
    is_openrouter = (provider or "").strip().lower() == "openrouter" or (
        bool(base_url) and _infer_provider_from_url(base_url) == "openrouter"
    )
    if not is_openrouter:
        # ⤷ 出口 return：返回 model
        return model
    base = openrouter_variant_base(model)
    if base is None:
        # ⤷ 出口 return：返回 model
        return model
    logger.debug(
        "Resolving context length for OpenRouter routing variant %r via base id %r",
        model, base,
    )
    # ⤷ 出口 return：返回 base
    return base


# ┌── 函数 def _is_known_provider_base_url()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：base_url: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_known_provider_base_url(base_url: str) -> bool:
    # ⤷ 出口 return：返回 _infer_provider_from_url(base_url) is not None
    return _infer_provider_from_url(base_url) is not None


# ┌── 函数 def _lmstudio_server_root()
# │  作用：LM Studio 服务端 root 用于 native ``/API/v1`` endpoints.
# │  参数：base_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _lmstudio_server_root(base_url: str) -> str:
    """LM Studio server root for native ``/api/v1`` endpoints."""
    root = _normalize_base_url(base_url)
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            # ⤷ 出口 return：返回 root[:-len(suffix)].rstrip('/')
            return root[: -len(suffix)].rstrip("/")
    # ⤷ 出口 return：返回 root
    return root


# ┌── 函数 def _server_root()
# │  作用：Probe root 用于 local 服务端: IPv4-resolved, ``/v1`` suffix stripped.
# │  参数：base_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _server_root(base_url: str) -> str:
    """Probe root for a local server: IPv4-resolved, ``/v1`` suffix stripped."""
    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    # ⤷ 出口 return：返回 server_url[:-3] if server_url.endswith('/v1') else server_ur…
    return server_url[:-3] if server_url.endswith("/v1") else server_url


# ┌── 函数 def _catalog_key_matches()
# │  作用：Substring match 带版本 separators normalised 在…上 both sides, 从而 relay slug like
# │  参数：key: str, model_lower: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _catalog_key_matches(key: str, model_lower: str) -> bool:
    """Substring match with version separators normalised on both sides, so a relay slug like
    ``z-ai-glm-5-3`` still hits the ``glm-5.3`` entry instead of the ``glm`` catch-all (#97398)."""
    # ⤷ 出口 return：返回 key in model_lower or _normalize_model_version(key) in _norm…
    return key in model_lower or _normalize_model_version(key) in _normalize_model_version(model_lower)


# ┌── 函数 def _longest_key_match()
# │  作用：First ``(键, 值)`` whose 键是 substring 的 ``model_lower``, longest 键 first 从而
# │  参数：table: Dict[str, int], model_lower: str
# │  返回：Optional[Tuple[str, int]]
# └──────────────────────────────────────────────────────────────
def _longest_key_match(table: Dict[str, int], model_lower: str) -> Optional[Tuple[str, int]]:
    """First ``(key, value)`` whose key is a substring of ``model_lower``, longest key first so
    specific entries (``gpt-5.4-mini``) beat their family catch-all (``gpt-5``); ties keep table order."""
    for key, value in sorted(table.items(), key=lambda x: len(x[0]), reverse=True):
        if _catalog_key_matches(key, model_lower):
            # ⤷ 出口 return：返回 (key, value)
            return key, value
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _ollama_show_context()
# │  作用：上下文 length 从 Ollama ``/API/show`` payload. ``参数.num_ctx`` 是运行时
# │  参数：data: Dict[str, Any], gguf_first: bool, minimum: Optional[int] = ...
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _ollama_show_context(data: Dict[str, Any], *, gguf_first: bool, minimum: Optional[int] = None) -> Optional[int]:
    """Context length from an Ollama ``/api/show`` payload. ``parameters.num_ctx`` is the RUNTIME
    window (Modelfile override), ``model_info.*.context_length`` the GGUF training max. Local users
    control num_ctx (prefer it); hosted operators may cap it arbitrarily (``gguf_first``)."""
    def _num_ctx():
        for line in data.get("parameters", "").split("\n"):
            parts = line.strip().split()
            if "num_ctx" in line and len(parts) >= 2:
                # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
                try:
                    # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
                    yield int(parts[-1])
                except ValueError:
                    continue
    def _gguf():
        for key, value in data.get("model_info", {}).items():
            if "context_length" in key and isinstance(value, (int, float)):
                # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
                yield int(value)
    for reader in ((_gguf, _num_ctx) if gguf_first else (_num_ctx, _gguf)):
        for ctx in reader():
            if minimum is None or ctx >= minimum:
                # ⤷ 出口 return：返回 ctx
                return ctx
    # ⤷ 出口 return：返回 None
    return None


# (host, canonical paths, model ids, context) — see _endpoint_scoped_context_length.
_ENDPOINT_SCOPED_CONTEXT = (
    ("api.kimi.com", {"/coding", "/coding/v1"}, {"k3", "kimi-k3", "kimi-k3-cot"}, 1_048_576),
    ("integrate.api.nvidia.com", {"/v1"}, {"deepseek-ai/deepseek-v4-pro"}, 262_144),
)


# ┌── 函数 def _endpoint_scoped_context_length()
# │  作用：上下文 confirmed 用于 one 服务商 endpoint 仅 (see _ENDPOINT_SCOPED_CONTEXT): Kimi Coding
# │  参数：model: str, base_url: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _endpoint_scoped_context_length(model: str, base_url: str) -> Optional[int]:
    """Context confirmed for one provider endpoint only (see _ENDPOINT_SCOPED_CONTEXT): Kimi Coding
    serves K3 at 1 Mi only on the canonical ``api.kimi.com/coding`` host (legacy Moonshot keys do
    not); NVIDIA NIM serves deepseek-v4-pro at 262,144 while DeepSeek's native endpoint is 1M."""
    # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
    try:
        parsed = urlparse(_normalize_base_url(base_url))
        port = parsed.port
    except ValueError:
        # ⤷ 出口 return：返回 None
        return None
    # Only canonical https://host[:443]/path with no credentials/query/fragment.
    if parsed.scheme.lower() != "https" or port not in (None, 443) or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        # ⤷ 出口 return：返回 None
        return None
    host, path, model_key = (parsed.hostname or "").lower(), parsed.path.rstrip("/"), model.strip().lower()
    # ⤷ 出口 return：返回 next((ctx for scoped_host, paths, models, ctx in _ENDPOINT_S…
    return next((ctx for scoped_host, paths, models, ctx in _ENDPOINT_SCOPED_CONTEXT if host == scoped_host and path in paths and model_key in models), None)


# ┌── 函数 def _skip_persistent_context_cache()
# │  作用：服务商 whose disk 上下文缓存必须不 short-circuit probing: LM Studio (已加载
# │  参数：base_url: str, provider: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _skip_persistent_context_cache(base_url: str, provider: str) -> bool:
    """Providers whose disk context cache must not short-circuit probing: LM Studio (loaded
    context is transient), Codex OAuth (entitlement-specific window; a persisted fallback would suppress revalidation)."""
    # ⤷ 出口 return：返回 (provider or '').strip().lower() in {'lmstudio', 'openai-cod…
    return (provider or "").strip().lower() in {"lmstudio", "openai-codex"}


# ┌── 函数 def _save_unless_skipped()
# │  作用：持久化 ``ctx`` unless 服务商 opts out 的 disk 缓存.
# │  参数：model: str, base_url: str, ctx: int, provider: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _save_unless_skipped(model: str, base_url: str, ctx: int, provider: str) -> None:
    """Persist ``ctx`` unless the provider opts out of the disk cache."""
    if not _skip_persistent_context_cache(base_url, provider):
        save_context_length(model, base_url, ctx)


# ┌── 函数 def _maybe_cache_local_context_length()
# │  作用：持久化 probed local window 仅在/above MINIMUM_CONTEXT_LENGTH: sub-minimum Windows 是
# │  参数：model: str, base_url: str, length: int
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _maybe_cache_local_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a probed local window only at/above MINIMUM_CONTEXT_LENGTH: sub-minimum windows are
    still returned so agent_init can reject them, but must not be blessed into the disk cache."""
    if length >= MINIMUM_CONTEXT_LENGTH:
        save_context_length(model, base_url, length)


# ┌── 函数 def _probe_local_context_length()
# │  作用：当前活跃的 local probe; 持久化 positive 结果 unless 服务商 opts out 的 disk 缓存.
# │  参数：model: str, base_url: str, api_key: str, provider: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _probe_local_context_length(model: str, base_url: str, api_key: str, provider: str) -> Optional[int]:
    """Live local probe; persists a positive result unless the provider opts out of the disk cache."""
    local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if not (local_ctx and local_ctx > 0):
        # ⤷ 出口 return：返回 None
        return None
    if not _skip_persistent_context_cache(base_url, provider):
        _maybe_cache_local_context_length(model, base_url, local_ctx)
    # ⤷ 出口 return：返回 local_ctx
    return local_ctx


# ┌── 函数 def _reconcile_local_cached_context_length()
# │  作用：*cached* unless 当前活跃的 local probe reports different 限制 (operators restart
# │  参数：model: str, base_url: str, cached: int, api_key: str = ''
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _reconcile_local_cached_context_length(model: str, base_url: str, cached: int, api_key: str = "") -> int:
    """*cached* unless a live local probe reports a different limit (operators restart
    vLLM/Ollama with a new --max-model-len / num_ctx under the same id). A failed
    probe keeps the disk entry; sub-minimum live windows invalidate but are not persisted."""
    live_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if not (live_ctx and live_ctx > 0 and live_ctx != cached):
        # ⤷ 出口 return：返回 cached
        return cached
    if live_ctx < MINIMUM_CONTEXT_LENGTH:
        logger.info("Live local probe for %s@%s reports %s (< minimum %s); invalidating stale cache — agent init should reject", model, base_url, f"{live_ctx:,}", f"{MINIMUM_CONTEXT_LENGTH:,}")
    else:
        logger.info("Reconciling stale local cache entry %s@%s: %s -> %s (live probe)", model, base_url, f"{cached:,}", f"{live_ctx:,}")
    _invalidate_cached_context_length(model, base_url)
    if live_ctx >= MINIMUM_CONTEXT_LENGTH:
        _maybe_cache_local_context_length(model, base_url, live_ctx)
    # ⤷ 出口 return：返回 live_ctx
    return live_ctx


# ┌── 函数 def is_local_endpoint()
# │  作用：当 loopback, container-internal DNS, unqualified 宿主, RFC-1918, 时返回 True。
# │  参数：base_url: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_local_endpoint(base_url: str) -> bool:
    """True for loopback, container-internal DNS, unqualified hosts, RFC-1918,
    link-local and Tailscale CGNAT (so a trusted Ollama box over Tailscale gets
    the same timeout auto-bumps as localhost)."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        parsed = _parse_base_url(base_url)
        host = parsed.hostname or "" if parsed is not None else None
    except Exception:
        return False
    if host is None:
        # ⤷ 出口 return：返回 False
        return False
    # Unqualified hostnames (no dots) are local by definition — Docker Compose service names, /etc/hosts
    # entries, mDNS — as is `*.local` (RFC 6762 mDNS, LAN-only). IPv6 literals have no dots either, so
    # they are excluded here and classified by scope below (a global address is not local).
    if host in _LOCAL_HOSTS or host.endswith(_CONTAINER_LOCAL_SUFFIXES) or host.endswith(".local") or (host and "." not in host and ":" not in host):
        # ⤷ 出口 return：返回 True
        return True
    # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or (isinstance(addr, ipaddress.IPv4Address) and addr in _TAILSCALE_CGNAT):
            return True
    except ValueError:
        pass
    # Dotted quad that ipaddress rejected but still looks like a private range
    # (e.g. 172.26.x.x for WSL) or Tailscale CGNAT (100.64.x.x–100.127.x.x).
    parts = host.split(".")
    if len(parts) != 4:
        return False
    # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    # ⤷ 出口 return：返回 first == 10 or (first == 172 and 16 <= second <= 31) or (fir…
    return first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168) or (first == 100 and 64 <= second <= 127)


# ┌── 函数 def _localhost_to_ipv4()
# │  作用：``localhost`` 宿主 -> ``127.0.0.1`` (Windows dual-stack resolves ::1 first 且 pays ~2s IPv6
# │  参数：url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _localhost_to_ipv4(url: str) -> str:
    """``localhost`` HOST -> ``127.0.0.1`` (Windows dual-stack resolves ::1 first and pays a ~2s IPv6
    connect timeout on IPv4-only servers). Scheme-anchored so ``?upstream=http://localhost`` is untouched."""
    if not url or not isinstance(url, str):
        # ⤷ 出口 return：返回 url
        return url  # non-string values (test doubles, lazy config) pass through
    # ⤷ 出口 return：返回 re.sub('^(https?://)localhost(?=[:/]|$)', '\\g<1>127.0.0.1',…
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url, count=1)


# ┌── 函数 def detect_local_server_type()
# │  作用：Probe known endpoints: "Ollama", "lm-studio", "vllm", "llamacpp", 或 None (TTL-cached).
# │  参数：base_url: str, api_key: str = ''
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def detect_local_server_type(base_url: str, api_key: str = "") -> Optional[str]:
    """Probe known endpoints: "ollama", "lm-studio", "vllm", "llamacpp", or None (TTL-cached)."""
    import httpx
    # IPv4-resolve BEFORE deriving server/LM Studio URLs and the cache lookup, so localhost and 127.0.0.1 share a cache entry.
    normalized = _localhost_to_ipv4(_normalize_base_url(base_url))
    server_url = _server_root(normalized)
    lmstudio_url = _lmstudio_server_root(normalized)
    cached = _endpoint_probe_path_cache.get(server_url)
    if cached is not None and (time.monotonic() - cached[1]) < (_ENDPOINT_PROBE_TTL_SECONDS if cached[0] is not None else _ENDPOINT_PROBE_FAILURE_TTL_SECONDS):
        # ⤷ 出口 return：返回 cached[0]
        return cached[0]
    # Blackholed host: skip the waterfall. Deliberately NOT written to the
    # hour-long verdict cache, which would pin "undetected" after it comes back.
    if _endpoint_blackholed(server_url):
        # ⤷ 出口 return：返回 None
        return None
    disk_hit = _local_probe_disk_get("server_type", server_url)
    if isinstance(disk_hit, str):
        _endpoint_probe_path_cache[server_url] = (disk_hit, time.monotonic())
        # ⤷ 出口 return：返回 disk_hit
        return disk_hit
    # Most specific first: (name, paths tried until one answers 200, body check). LM Studio answers /api/tags
    # with {"error": ...} and 200, so Ollama's body must carry "models"; older llama.cpp builds have no /v1 prefix.
    waterfall = (
        ("lm-studio", (f"{lmstudio_url}/api/v1/models",), lambda r: True),
        ("ollama", (f"{server_url}/api/tags",), lambda r: "models" in r.json()),
        ("llamacpp", (f"{server_url}/v1/props", f"{server_url}/props"), lambda r: "default_generation_settings" in r.text),
        ("vllm", (f"{server_url}/version",), lambda r: "version" in r.json()),
    )
    result: Optional[str] = None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with httpx.Client(timeout=2.0, headers=_auth_headers(api_key)) as client:
            for name, urls, check in waterfall:
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    for url in urls:
                        r = client.get(url)
                        if r.status_code == 200:
                            break
                    if r.status_code == 200 and check(r):
                        result = name
                        break
                except Exception as exc:
                    # A connect timeout condemns the host: skip the remaining legs.
                    if _is_connect_timeout(exc):
                        _note_endpoint_blackholed(server_url)
                        raise
    except Exception:
        pass
    # Negative verdict in memory only (never on disk — failures are often transient).
    _endpoint_probe_path_cache[server_url] = (result, time.monotonic())
    if result is not None:
        _local_probe_disk_put("server_type", server_url, result)
    # ⤷ 出口 return：返回 result
    return result


# Cache the negative verdict in memory only (never on disk — a failure is often transient: server starting,
# key being fixed) so the very next turn does not re-run the whole waterfall against an endpoint that just
# answered nothing (#89863).
# ┌── 函数 def _iter_nested_dicts()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


# ┌── 函数 def _coerce_reasonable_int()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any, minimum: int = 1024, maximum: int = 10000000
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    if isinstance(value, bool):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
    try:
        result = int(value.strip().replace(",", "") if isinstance(value, str) else value)
    except (TypeError, ValueError):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 result if minimum <= result <= maximum else None
    return result if minimum <= result <= maximum else None


# ┌── 函数 def _extract_first_int()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：payload: Dict[str, Any], keys: tuple[str, ...]
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _extract_first_int(payload: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() in keyset:
                coerced = _coerce_reasonable_int(value)
                if coerced is not None:
                    # ⤷ 出口 return：返回 coerced
                    return coerced
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _extract_flat_context_length()
# │  作用：Top-级别-仅上下文窗口读取 (无 nested walk, 从而 unrelated nested section 无法 leak
# │  参数：payload: Dict[str, Any]
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _extract_flat_context_length(payload: Dict[str, Any]) -> Optional[int]:
    """Top-level-only context WINDOW read (no nested walk, so an unrelated nested section can't leak
    a same-named key). ``max_tokens`` is NOT a window key: on OpenAI-compatible passthroughs it is the max OUTPUT."""
    # ⤷ 出口 return：返回 next((c for c in map(_coerce_reasonable_int, (payload.get(k)…
    return next((c for c in map(_coerce_reasonable_int, (payload.get(k) for k in _CONTEXT_LENGTH_KEYS)) if c is not None), None)


# ┌── 函数 def _context_length_from_model_payload()
# │  作用：上下文窗口从 ``/v1/模型`` 对象: window 键 first, ``max_tokens`` last (Anthropic
# │  参数：payload: Dict[str, Any]
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _context_length_from_model_payload(payload: Dict[str, Any]) -> Optional[int]:
    """Context window from a ``/v1/models`` object: window keys first, ``max_tokens`` last (Anthropic
    payloads carry ``max_input_tokens`` = 1M window AND ``max_tokens`` = 128k OUTPUT cap)."""
    if not isinstance(payload, dict):
        # ⤷ 出口 return：返回 None
        return None
    ctx = _extract_flat_context_length(payload)
    if ctx is not None:
        # ⤷ 出口 return：返回 ctx
        return ctx
    raw = payload.get("max_tokens")
    # ⤷ 出口 return：返回 int(raw) if isinstance(raw, (int, float)) and int(raw) > 0 e…
    return int(raw) if isinstance(raw, (int, float)) and int(raw) > 0 else None


# ┌── 函数 def _extract_pricing()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：payload: Dict[str, Any]
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _per_token(source: Dict[str, Any], fields: Dict[str, str], scale) -> Dict[str, Any]:
        # Provider $/MTok (or Novita's 1/10_000-$ per M) -> per-token strings, the same path usage_pricing uses for OpenRouter.
        # ⤷ 出口 return：返回 {target: str(scale(float(source[key]))) for target, key in f…
        return {target: str(scale(float(source[key]))) for target, key in fields.items() if source.get(key) is not None}
    novita_fields = {"prompt": "input_token_price_per_m", "completion": "output_token_price_per_m"}
    if any(payload.get(k) is not None for k in novita_fields.values()):
        # ⤷ 出口 return：返回 _per_token(payload, novita_fields, lambda v: v / 10000 / 100…
        return _per_token(payload, novita_fields, lambda v: v / 10_000 / 1_000_000)
    # DeepInfra ships pricing under ``metadata.pricing`` in $/MTok.
    metadata = payload.get("metadata")
    deepinfra_pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
    deepinfra_fields = {"prompt": "input_tokens", "completion": "output_tokens", "cache_read": "cache_read_tokens"}
    if isinstance(deepinfra_pricing, dict) and any(k in deepinfra_pricing for k in deepinfra_fields.values()):
        # ⤷ 出口 return：返回 _per_token(deepinfra_pricing, deepinfra_fields, lambda v: v …
        return _per_token(deepinfra_pricing, deepinfra_fields, lambda v: v / 1_000_000)
    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        pricing: Dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in {None, ""}:
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            # ⤷ 出口 return：返回 pricing
            return pricing
    # ⤷ 出口 return：返回 {}
    return {}


# ┌── 函数 def _add_model_aliases()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：cache: Dict[str, Dict[str, Any]], model_id: str, entry: Dict[str, Any]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _add_model_aliases(cache: Dict[str, Dict[str, Any]], model_id: str, entry: Dict[str, Any]) -> None:
    cache[model_id] = entry
    if "/" in model_id:
        cache.setdefault(model_id.split("/", 1)[1], entry)


# ┌── 函数 def fetch_model_metadata()
# │  作用：Fetch 模型 metadata 从 OpenRouter (cached 用于 1 hour).
# │  参数：force_refresh: bool = False
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from OpenRouter (cached for 1 hour)."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _model_metadata_cache, _model_metadata_cache_time
    if not force_refresh:
        if _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
            # ⤷ 出口 return：返回 _model_metadata_cache
            return _model_metadata_cache
        disk_age = _model_metadata_disk_cache_age_seconds()
        if disk_age is not None and disk_age < _MODEL_CACHE_TTL:
            disk_cache = _load_model_metadata_disk_cache()
            if disk_cache:
                _model_metadata_cache = disk_cache
                _model_metadata_cache_time = time.time() - disk_age
                # ⤷ 出口 return：返回 _model_metadata_cache
                return _model_metadata_cache
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        _ensure_requests()
        # (connect, read) tuple: a flat timeout lets urllib3 block per retry stage through proxies that 403 CONNECT.
        # See #46620.
        response = requests.get(OPENROUTER_MODELS_URL, timeout=(5, 10), verify=_resolve_requests_verify())
        response.raise_for_status()
        cache = {}
        for model in response.json().get("data", []):
            model_id = model.get("id", "")
            entry = {
                "context_length": model.get("context_length", 128000),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id), "pricing": model.get("pricing", {}),
            }
            canonical = model.get("canonical_slug", "")
            for alias in ((model_id, canonical) if canonical and canonical != model_id else (model_id,)):
                _add_model_aliases(cache, alias, entry)
        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        _save_model_metadata_disk_cache(cache)
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        # ⤷ 出口 return：返回 cache
        return cache
    except Exception as e:
        logger.warning("Failed to fetch model metadata from OpenRouter: %s", e)
        if _model_metadata_cache:
            return _model_metadata_cache
        disk_cache = _load_model_metadata_disk_cache()
        if disk_cache:
            _model_metadata_cache = disk_cache
            disk_age = _model_metadata_disk_cache_age_seconds()
            stale_by = min(disk_age, _MODEL_CACHE_TTL) if disk_age is not None else _MODEL_CACHE_TTL - 1
            _model_metadata_cache_time = time.time() - stale_by
            return _model_metadata_cache
        # ⤷ 出口 return：返回 {}
        return {}


# ┌── 函数 def _endpoint_model_entry()
# │  作用：缓存 entry 用于 one ``/模型`` item; optional 键是 设置仅 当 known.
# │  参数：model: Dict[str, Any], model_id: str, context_length: Optional[int]
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _endpoint_model_entry(model: Dict[str, Any], model_id: str, context_length: Optional[int]) -> Dict[str, Any]:
    """Cache entry for one ``/models`` item; optional keys are set only when known."""
    optional = (("context_length", context_length), ("max_completion_tokens", _extract_first_int(model, _MAX_COMPLETION_KEYS)), ("pricing", _extract_pricing(model) or None))
    # ⤷ 出口 return：返回 {'name': model.get('name', model_id), **{k: v for k, v in op…
    return {"name": model.get("name", model_id), **{k: v for k, v in optional if v is not None}}


# ┌── 函数 def _lmstudio_loaded_context()
# │  作用：上下文的 first 已加载 LM Studio instance ( 运行时值), 否则 None.
# │  参数：model: Dict[str, Any]
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _lmstudio_loaded_context(model: Dict[str, Any]) -> Optional[int]:
    """Context of the first loaded LM Studio instance (the runtime value), else None."""
    for inst in model.get("loaded_instances", []) or []:
        cfg = inst.get("config", {}) if isinstance(inst, dict) else None
        ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
        if isinstance(ctx, int) and ctx > 0:
            # ⤷ 出口 return：返回 ctx
            return ctx
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _lmstudio_native_models()
# │  作用：LM Studio ``/API/v1/模型`` → 缓存; 上下文 comes 从 first 已加载 instance.
# │  参数：normalized: str, headers: Dict[str, str]
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _lmstudio_native_models(normalized: str, headers: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """LM Studio ``/api/v1/models`` → cache; context comes from the first loaded instance."""
    response = requests.get(_lmstudio_server_root(normalized).rstrip("/") + "/api/v1/models", headers=headers, timeout=(5, 10), verify=_resolve_requests_verify(normalized))
    response.raise_for_status()
    cache: Dict[str, Dict[str, Any]] = {}
    for model in response.json().get("models", []):
        model_id = (model.get("key") or model.get("id")) if isinstance(model, dict) else None
        if not model_id:
            continue
        entry = _endpoint_model_entry(model, model_id, _lmstudio_loaded_context(model))
        _add_model_aliases(cache, model_id, entry)
        alt_id = model.get("id")
        if isinstance(alt_id, str) and alt_id and alt_id != model_id:
            _add_model_aliases(cache, alt_id, entry)
    # ⤷ 出口 return：返回 cache
    return cache


# ┌── 函数 def _apply_llamacpp_props()
# │  作用：Overwrite ``context_length`` 带 llama.cpp's allocated ``n_ctx`` 从 /props (``/v1/props``, 然后
# │  参数：cache: Dict[str, Dict[str, Any]], request_candidate: str, headers: Dict[str, str], verify
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _apply_llamacpp_props(cache: Dict[str, Dict[str, Any]], request_candidate: str, headers: Dict[str, str], verify) -> None:
    """Overwrite ``context_length`` with llama.cpp's allocated ``n_ctx`` from /props (``/v1/props``, then
    ``/props`` for older builds). In router mode the bare endpoint 400s, so each LOADED child is read
    via ``/props?model=``; unloaded children are skipped — probing could autoload them."""
    base = request_candidate.rstrip("/").replace("/v1", "")
    def _props(params=None):
        resp = requests.get(base + "/v1/props", params=params, headers=headers, timeout=5, verify=verify)
        if not resp.ok:
            resp = requests.get(base + "/props", params=params, headers=headers, timeout=5, verify=verify)
        # ⤷ 出口 return：返回 resp
        return resp
    def _n_ctx(props: Dict[str, Any]) -> Any:
        # ⤷ 出口 return：返回 (props.get('default_generation_settings') or {}).get('n_ctx'…
        return (props.get("default_generation_settings") or {}).get("n_ctx")
    props_resp = _props()
    if props_resp.ok:
        props = props_resp.json()
        n_ctx, model_alias = _n_ctx(props), props.get("model_alias", "")
        if n_ctx and model_alias and model_alias in cache:
            cache[model_alias]["context_length"] = n_ctx
        return
    native = requests.get(base + "/models", headers=headers, timeout=5, verify=verify)
    if not native.ok:
        return
    for child in (native.json() or {}).get("data", [])[:16]:
        child_id = child.get("id") if isinstance(child, dict) else None
        if not child_id or child_id not in cache or (child.get("status") or {}).get("value") not in ("loaded", "ready"):
            continue
        pr = _props({"model": child_id})
        child_ctx = _n_ctx(pr.json()) if pr.ok else None
        if child_ctx:
            cache[child_id]["context_length"] = child_ctx


# ┌── 函数 def _endpoint_memo_key()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：normalized: str, api_key: object
# │  返回：Tuple[str, str]
# └──────────────────────────────────────────────────────────────
def _endpoint_memo_key(normalized: str, api_key: object) -> Tuple[str, str]:
    from agent.credential_persistence import fingerprint_secret_value
    # Callable (minted) keys are not fingerprinted here: doing so would mint on every cache hit.
    # ⤷ 出口 return：返回 (normalized, fingerprint_secret_value(api_key) or '' if isin…
    return normalized, (fingerprint_secret_value(api_key) or "") if isinstance(api_key, str) else ""


# ┌── 函数 def _remember_endpoint_models()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：memo_key: Tuple[str, str], cache: Dict[str, Dict[str, Any]]
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _remember_endpoint_models(memo_key: Tuple[str, str], cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    _endpoint_model_metadata_cache[memo_key] = cache
    _endpoint_model_metadata_cache_time[memo_key] = time.time()
    # ⤷ 出口 return：返回 cache
    return cache


# ┌── 函数 def _parse_models_payload()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：payload: Dict[str, Any]
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _parse_models_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for model in payload.get("data", []):
        model_id = model.get("id") if isinstance(model, dict) else None
        if model_id:
            _add_model_aliases(cache, model_id, _endpoint_model_entry(model, model_id, _extract_first_int(model, _CONTEXT_LENGTH_KEYS)))
    # ⤷ 出口 return：返回 cache
    return cache


# ┌── 函数 def fetch_endpoint_model_metadata()
# │  作用：模型 metadata 从 OpenAI-compatible ``/模型`` endpoint (cached per base URL).
# │  参数：base_url: str, api_key: str = '', force_refresh: bool = False
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def fetch_endpoint_model_metadata(base_url: str, api_key: str = "", force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Model metadata from an OpenAI-compatible ``/models`` endpoint (cached per base URL)."""
    normalized = _normalize_base_url(base_url)
    if not normalized or base_url_host_matches(normalized, "openrouter.ai"):
        # ⤷ 出口 return：返回 {}
        return {}
    _ensure_requests()
    local = is_local_endpoint(normalized)
    memo_key = _endpoint_memo_key(normalized, api_key)
    if not force_refresh:
        cached = _endpoint_model_metadata_cache.get(memo_key)
        if cached is not None and (time.time() - _endpoint_model_metadata_cache_time.get(memo_key, 0)) < _ENDPOINT_MODEL_CACHE_TTL:
            # ⤷ 出口 return：返回 cached
            return cached
        memo = _endpoint_disk_cache_get(normalized) if not local else None
        if memo is not None:
            return _remember_endpoint_models(memo_key, memo)
    # Blackholed: return empty WITHOUT caching so it is retried once the entry expires.
    if _endpoint_blackholed(normalized):
        # ⤷ 出口 return：返回 {}
        return {}
    alternate = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates = [normalized] + ([alternate] if alternate != normalized else [])
    headers = _auth_headers(api_key)
    verify = _resolve_requests_verify(normalized)
    last_error: Optional[Exception] = None
    if local:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            if detect_local_server_type(normalized, api_key=api_key) == "lm-studio":
                return _remember_endpoint_models(memo_key, _lmstudio_native_models(normalized, headers))
        except Exception as exc:
            last_error = exc
            _note_if_connect_timeout(exc, normalized)
    for candidate in candidates:
        # A connect timeout condemns the host, not the path.
        if _endpoint_blackholed(normalized):
            break
        # Cache keys stay unrewritten; only the outbound target is IPv4-resolved.
        request_candidate = _localhost_to_ipv4(candidate)
        url = request_candidate.rstrip("/") + "/models"
        response = None
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            response = requests.get(url, headers=headers, timeout=(5, 10), verify=verify, stream=True)
            if response.status_code in (401, 403):
                logger.debug("Model metadata probe received HTTP %s from %s; stopping candidate probing", response.status_code, url)
                break
            response.raise_for_status()
            payload = response.json()
            cache = _parse_models_payload(payload)
            if any(m.get("owned_by") == "llamacpp" for m in payload.get("data", []) if isinstance(m, dict)):
                with contextlib.suppress(Exception):
                    _apply_llamacpp_props(cache, request_candidate, headers, verify)
            if cache and not local:
                _endpoint_disk_cache_put(normalized, cache)
            return _remember_endpoint_models(memo_key, cache)
        except Exception as exc:
            last_error = exc
            _note_if_connect_timeout(exc, normalized)
        finally:
            if response is not None:
                response.close()
    if last_error:
        logger.debug("Failed to fetch model metadata from %s/models: %s", normalized, last_error)
    # ⤷ 出口 return：返回 _remember_endpoint_models(memo_key, {})
    return _remember_endpoint_models(memo_key, {})


# ┌── 函数 def _resolve_endpoint_context_length()
# │  作用：Resolve 上下文 length 从 endpoint's 当前活跃的 ``/模型`` metadata.
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _resolve_endpoint_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Resolve context length from an endpoint's live ``/models`` metadata."""
    endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    matched = endpoint_metadata.get(model)
    if not matched and len(endpoint_metadata) == 1:
        matched = next(iter(endpoint_metadata.values()))
    elif not matched and model:  # substring match; "" would match EVERY key and poison the window
        matched = next((entry for key, entry in endpoint_metadata.items() if model in key or key in model), None)
    context_length = matched.get("context_length") if matched else None
    # ⤷ 出口 return：返回 context_length if isinstance(context_length, int) else None
    return context_length if isinstance(context_length, int) else None


# ┌── 函数 def _get_context_cache_path()
# │  作用：路径 persistent 上下文 length 缓存文件.
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _get_context_cache_path() -> Path:
    """Path to the persistent context length cache file."""
    from hermes_constants import get_hermes_home
    # ⤷ 出口 return：返回 get_hermes_home() / 'context_length_cache.yaml'
    return get_hermes_home() / "context_length_cache.yaml"


# ┌── 函数 def _load_context_cache()
# │  作用：加载模型+服务商 -> context_length 缓存从 disk.
# │  参数：无
# │  返回：Dict[str, int]
# └──────────────────────────────────────────────────────────────
def _load_context_cache() -> Dict[str, int]:
    """Load the model+provider -> context_length cache from disk."""
    path = _get_context_cache_path()
    if not path.exists():
        # ⤷ 出口 return：返回 {}
        return {}
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with open(path, encoding="utf-8") as f:
            # ⤷ 出口 return：返回 (yaml.safe_load(f) or {}).get('context_lengths') or {}
            return (yaml.safe_load(f) or {}).get("context_lengths") or {}
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        # ⤷ 出口 return：返回 {}
        return {}


# ┌── 函数 def _write_context_cache()
# │  作用：Atomic 写入 ( truncating 写入 killed mid-dump leaves partial 文件那个
# │  参数：cache: Dict[str, int]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _write_context_cache(cache: Dict[str, int]) -> None:
    """Atomic write (a truncating write killed mid-dump leaves a partial file that
    _load_context_cache() swallows as {}, wiping EVERY cached length). Raises on failure."""
    atomic_yaml_write(_get_context_cache_path(), {"context_lengths": cache})


# ┌── 函数 def _context_cache_key()
# │  作用：Canonical ``模型@base_url`` 键; trailing slashes stripped 从而 ``/v1`` 且 ``/v1/`` share one entry.
# │  参数：model: str, base_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _context_cache_key(model: str, base_url: str) -> str:
    """Canonical ``model@base_url`` key; trailing slashes stripped so ``/v1`` and ``/v1/`` share one entry."""
    # ⤷ 出口 return：返回 f"{model}@{(base_url or '').rstrip('/')}"
    return f"{model}@{(base_url or '').rstrip('/')}"


# ┌── 函数 def save_context_length()
# │  作用：持久化 discovered 上下文 length under ``模型@base_url`` (same 模型, different 服务商, different 限制).
# │  参数：model: str, base_url: str, length: int
# │  返回：None
# └──────────────────────────────────────────────────────────────
def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length under ``model@base_url`` (same model, different providers, different limits)."""
    # 0/negative is always a bug and would make get_model_context_length() return 0 (`0 is not None`).
    if length <= 0:
        logger.warning("Refusing to cache non-positive context length %s -> %s tokens", f"{model}@{base_url}", length)
        return
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        _write_context_cache(cache)
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


# ┌── 函数 def get_cached_context_length()
# │  作用：Look up previously discovered 上下文 length 用于模型+服务商.
# │  参数：model: str, base_url: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Legacy rows may carry a trailing slash, so probe the canonical key, the literal form and the slashed canonical form.
    # ⤷ 出口 return：返回 next((hit for hit in map(cache.get, (key, f'{model}@{base_ur…
    return next((hit for hit in map(cache.get, (key, f"{model}@{base_url}", f"{key}/")) if hit is not None), None)


# ┌── 函数 def _invalidate_cached_context_length()
# │  作用：丢弃卡死的缓存 entry 从而它 获取 re-resolved 在…上 next lookup.
# │  参数：model: str, base_url: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _invalidate_cached_context_length(model: str, base_url: str) -> None:
    """Drop a stale cache entry so it gets re-resolved on the next lookup."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Also drop the in-memory TTL probe entries, or the next resolution inside the TTL window reuses the stale value.
    bare, stripped = _strip_provider_prefix(model), (base_url or "").rstrip("/")
    _LOCAL_CTX_PROBE_CACHE.pop((bare, stripped), None)
    _LOCAL_CTX_PROBE_CACHE.pop(("ollama_show", bare, stripped), None)
    # Every key shape get_cached_context_length consults.
    stale_keys = {key, f"{model}@{base_url}", f"{key}/"}
    if not any(k in cache for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        _write_context_cache(cache)
    except Exception as e:
        logger.debug("Failed to invalidate context length cache entry %s: %s", key, e)


# ┌── 函数 def get_next_probe_tier()
# │  作用：Next lower probe tier, 或 None 若已 在 minimum.
# │  参数：current_length: int
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Next lower probe tier, or None if already at minimum."""
    # ⤷ 出口 return：返回 next((tier for tier in CONTEXT_PROBE_TIERS if tier < current…
    return next((tier for tier in CONTEXT_PROBE_TIERS if tier < current_length), None)


# ┌── 函数 def parse_context_limit_from_error()
# │  作用：上下文限制 quoted 在服务商错误 ("maximum 上下文 length 是 32768 令牌"), 若任意.
# │  参数：error_msg: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any.

    A message about only an OUTPUT cap ("... model output limit of 16384") never says "context";
    bail out so the generic "limit ... of N" pattern can't cache the output cap as the window."""
    error_lower = error_msg.lower()
    if ("output limit" in error_lower or "output tokens" in error_lower or "output token" in error_lower) and "context" not in error_lower:
        # ⤷ 出口 return：返回 None
        return None
    patterns = (
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
        # Gemini: "input token count is 32825 but model only supports up to
        # 32768" — anchor on the phrase so the input count isn't captured.
        r'supports?\s+(?:only\s+)?up\s+to\s+(\d{4,})',
    )
    for match in filter(None, (re.search(pattern, error_lower) for pattern in patterns)):
        limit = int(match.group(1))
        if 1024 <= limit <= 10_000_000:  # sanity: must be a plausible window
            # ⤷ 出口 return：返回 limit
            return limit
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def get_context_length_from_provider_error()
# │  作用：服务商-reported 限制 LOWER 比 current window, 否则 None. Overflow recovery 必须
# │  参数：error_msg: str, current_context_length: int
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def get_context_length_from_provider_error(error_msg: str, current_context_length: int) -> Optional[int]:
    """Provider-reported limit LOWER than the current window, else None. Overflow recovery must
    not invent a window: when the provider only says the input is too long, callers keep the
    configured length and compress rather than stepping down guessed probe tiers."""
    parsed_limit = parse_context_limit_from_error(error_msg)
    # ⤷ 出口 return：返回 parsed_limit if parsed_limit is not None and parsed_limit < …
    return parsed_limit if parsed_limit is not None and parsed_limit < current_context_length else None


# ┌── 函数 def parse_available_output_tokens_from_error()
# │  作用：Available 输出令牌从 "max_tokens too large" 错误, 或 None. Distinct 从 "提示词
# │  参数：error_msg: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Available OUTPUT tokens from a "max_tokens too large" error, or None. Distinct from "prompt
    too long" (-> compress): here input + requested_output > window, so the fix is a smaller
    max_tokens for this call and context_length must NOT be touched."""
    error_lower = error_msg.lower()
    if not _any_phrase_group(error_lower, _PARSEABLE_OUTPUT_CAP_SIGNALS):
        # ⤷ 出口 return：返回 None
        return None
    # Direct cap figures, most specific first: "exceeds model's maximum output tokens (65536)", "Range of
    # max_tokens should be [1, 65536]" (upper bound is the cap), Anthropic "= available_tokens: 10000", last "= N".
    for pattern in (
        r'exceeds model(?:\'s)? maximum output tokens\s*\(?\s*(\d+)\s*\)?',
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        r'available_tokens[:\s]+(\d+)',
        r'available\s+tokens[:\s]+(\d+)',
        # Switchyard: "max_tokens cannot exceed the configured model output limit of 16384".
        r'output limit (?:of|is)\s*(\d+)',
        r'=\s*(\d+)\s*$',
    ):
        match = re.search(pattern, error_lower)
        if match and int(match.group(1)) >= 1:
            # ⤷ 出口 return：返回 int(match.group(1))
            return int(match.group(1))
    # OpenRouter/Nous: "maximum context length is N … (A of text input, B of tool input, C in the output)" -> ctx - A - B.
    _m_ctx = re.search(r'maximum context length is (\d+)', error_lower)
    _m_parts = re.search(r'\((\d+)\s+of text input,\s*(\d+)\s+of tool input,\s*(\d+)\s+in the output\)', error_lower)
    if _m_ctx and _m_parts:
        _available = int(_m_ctx.group(1)) - int(_m_parts.group(1)) - int(_m_parts.group(2))
        if _available >= 1:
            # ⤷ 出口 return：返回 _available
            return _available
    # LM Studio / llama.cpp: window in tokens, prompt in CHARACTERS; ~3 chars/token over-reserves the input.
    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)
    _m_chars = re.search(r'prompt contains (\d+)\s*character', error_lower)
    if _m_ctx_tok and _m_chars:
        _available = int(_m_ctx_tok.group(1)) - (int(_m_chars.group(1)) + 2) // 3
        if _available >= 1:
            # ⤷ 出口 return：返回 _available
            return _available
    # vLLM: window and prompt both in TOKENS; available = window - input (None when the input alone
    # overflows -> compress). When max_tokens is the BINDING constraint vLLM reports "at least N input
    # tokens" with N == window + 1 - requested_output, so window - N == requested_output - 1 and each
    # retry walks the cap down by the safety margin without ever fitting: halve the cap instead.
    _m_vllm_input = re.search(r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower)
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        _m_requested_out = re.search(r'requested (\d+)\s*output tokens', error_lower)
        if 'at least' in error_lower and _m_requested_out:
            _requested_out = int(_m_requested_out.group(1))
            if _available >= _requested_out - 1:
                # The budget is derived from the constraint, not measured.
                return max(1, _requested_out // 2)
        if _available >= 1:
            # ⤷ 出口 return：返回 _available
            return _available
    # ⤷ 出口 return：返回 None
    return None


# Each entry is a phrase group; the group matches when ALL phrases are present.
# DashScope, Anthropic, OpenRouter/Nous, LM Studio/llama.cpp, generic "should be <= N", OpenAI-compat relays.
_OUTPUT_CAP_SIGNALS = (
    ("range of max_tokens should be",), ("available_tokens",), ("available tokens",),
    ("in the output", "maximum context length"), ("requested", "output tokens"),
    ("should be",), ("less than or equal",), ("must be",), ("exceeds model", "maximum output tokens"),
    ("output limit",),
)
_INPUT_OVERFLOW_SIGNALS = (
    "prompt is too long", "prompt too long", "input is too long", "input token",
    "prompt length", "prompt contains", "reduce the length",
)
# Narrower than _OUTPUT_CAP_SIGNALS: only phrasings we can extract a number from.
# "requested N output tokens" means the OUTPUT cap is the problem (the input fits) —
# reduce max_tokens, don't compress. DashScope's bounded range upper bound IS the
# real max-output cap ("Range of max_tokens should be [1, 65536]").
_PARSEABLE_OUTPUT_CAP_SIGNALS = (
    ("max_tokens", "available_tokens"), ("max_tokens", "available tokens"),
    ("in the output", "maximum context length"),
    ("maximum context length", "requested", "output tokens"),
    ("range of max_tokens should be",), ("exceeds model", "maximum output tokens"),
    ("output limit",),
)


# ┌── 函数 def _any_phrase_group()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：text: str, groups: tuple
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _any_phrase_group(text: str, groups: tuple) -> bool:
    # ⤷ 出口 return：返回 any((all((p in text for p in group)) for group in groups))
    return any(all(p in text for p in group) for group in groups)


# ┌── 函数 def is_output_cap_error()
# │  作用：Yes/无 sibling 的 :func:`parse_available_output_tokens_from_error` 用于 unparseable wordings.
# │  参数：error_msg: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_output_cap_error(error_msg: str) -> bool:
    """Yes/no sibling of :func:`parse_available_output_tokens_from_error` for unparseable wordings. An
    output-cap 400 misclassified as context overflow death-loops the compressor (same max_tokens, same
    rejection). Signal: talks about max_tokens as a cap/range/limit and NOT about an oversized input."""
    error_lower = error_msg.lower()
    # An error that ALSO describes an oversized INPUT is a genuine overflow — compression can fix it.
    # ⤷ 出口 return：返回 any((p in error_lower for p in ('max_tokens', 'max_output_to…
    return (
        any(p in error_lower for p in ("max_tokens", "max_output_tokens", "max_completion_tokens"))
        and _any_phrase_group(error_lower, _OUTPUT_CAP_SIGNALS)
        and not any(p in error_lower for p in _INPUT_OVERFLOW_SIGNALS)
    )


# ┌── 函数 def _model_id_matches()
# │  作用：Exact match, 或 ``publisher/slug`` (LM Studio native ids) whose slug equals configured 名称.
# │  参数：candidate_id: str, lookup_model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _model_id_matches(candidate_id: str, lookup_model: str) -> bool:
    """Exact match, or ``publisher/slug`` (LM Studio native ids) whose slug equals the configured name."""
    # ⤷ 出口 return：返回 candidate_id == lookup_model or ('/' in candidate_id and can…
    return candidate_id == lookup_model or ("/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model)


# ┌── 函数 def _ollama_show()
# │  作用：Ollama ``/API/show`` JSON 用于 ``bare_model``, 或 None 在…上任意 failure (``note_blackhole``: connect 超时 condemn 宿主).
# │  参数：server_url: str, api_key: str, bare_model: str, timeout: float = 3.0, note_blackhole: bool = ...
# │  返回：Optional[Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _ollama_show(server_url: str, api_key: str, bare_model: str, timeout: float = 3.0, *, note_blackhole: bool = False) -> Optional[Dict[str, Any]]:
    """Ollama ``/api/show`` JSON for ``bare_model``, or None on any failure (``note_blackhole``: connect timeouts condemn the host)."""
    import httpx
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            # ⤷ 出口 return：返回 resp.json() if resp.status_code == 200 else None
            return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        if note_blackhole:
            _note_if_connect_timeout(exc, server_url)
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _is_ollama_server()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：base_url: str, api_key: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_ollama_server(base_url: str, api_key: str) -> bool:
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # Forward the API key: a remote API-keyed endpoint answers the probe waterfall with 401s without it,
        # and an unauthorized probe can never produce a positive verdict (#89863).
        # ⤷ 出口 return：返回 detect_local_server_type(base_url, api_key=api_key) == 'olla…
        return detect_local_server_type(base_url, api_key=api_key) == "ollama"
    except Exception:
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def query_ollama_num_ctx()
# │  作用：Ollama ``/API/show`` 上下文 (Modelfile num_ctx, 否则 GGUF max); 值 send 作为 ``num_ctx``.
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def query_ollama_num_ctx(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Ollama ``/api/show`` context (Modelfile num_ctx, else GGUF max); the value to send as ``num_ctx``."""
    bare_model, server_url = _strip_provider_prefix(model), _server_root(base_url)
    if not _is_ollama_server(base_url, api_key):
        # ⤷ 出口 return：返回 None
        return None
    _disk_key = f"{server_url}|{bare_model}"
    disk_hit = _local_probe_disk_get("ollama_num_ctx", _disk_key)
    if isinstance(disk_hit, int) and disk_hit > 0:
        # ⤷ 出口 return：返回 disk_hit
        return disk_hit
    data = _ollama_show(server_url, api_key, bare_model)
    ctx = _ollama_show_context(data, gguf_first=False) if data is not None else None
    if ctx is not None:
        _local_probe_disk_put("ollama_num_ctx", _disk_key, ctx)
    # ⤷ 出口 return：返回 ctx
    return ctx


# ┌── 函数 def query_ollama_supports_vision()
# │  作用：True/False 当 Ollama ``/API/show`` reports vision support (``capabilities`` 在…上 0.6.0+, 否则
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[bool]
# └──────────────────────────────────────────────────────────────
def query_ollama_supports_vision(model: str, base_url: str, api_key: str = "") -> Optional[bool]:
    """True/False when Ollama ``/api/show`` reports vision support (``capabilities`` on 0.6.0+, else
    ``model_info.*.vision.block_count``); None when unreachable, not Ollama, or model unknown."""
    bare_model = _strip_provider_prefix(model)
    if not bare_model or not base_url or not _is_ollama_server(base_url, api_key):
        # ⤷ 出口 return：返回 None
        return None
    data = _ollama_show(_server_root(base_url), api_key, bare_model)
    if data is None:
        # ⤷ 出口 return：返回 None
        return None
    caps = data.get("capabilities")
    if isinstance(caps, list) and caps:
        # ⤷ 出口 return：返回 any((str(cap).lower() == 'vision' for cap in caps))
        return any(str(cap).lower() == "vision" for cap in caps)
    model_info = data.get("model_info")
    if isinstance(model_info, dict) and any("vision.block_count" in str(key).lower() for key in model_info):
        # ⤷ 出口 return：返回 True
        return True
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _memo_local_probe()
# │  作用：Short-TTL, positive-仅 memo 的 local probe (see _LOCAL_CTX_PROBE_CACHE): failure 在…期间
# │  参数：cache_key: tuple, probe: Callable[[], Optional[int]]
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _memo_local_probe(cache_key: tuple, probe: Callable[[], Optional[int]]) -> Optional[int]:
    """Short-TTL, positive-only memo of a local probe (see _LOCAL_CTX_PROBE_CACHE): a failure during
    a startup race must not suppress the retry once the server is up, so only truthy results memoize."""
    now = time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        # ⤷ 出口 return：返回 cached[0]
        return cached[0]
    result = probe()
    if result:
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    # ⤷ 出口 return：返回 result
    return result


# ┌── 函数 def _query_ollama_api_show()
# │  作用：服务商-agnostic Ollama ``/API/show`` probe (任意 hostname; non-Ollama 服务端 404 fast).
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _query_ollama_api_show(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Provider-agnostic Ollama ``/api/show`` probe (any hostname; non-Ollama servers 404 fast).
    GGUF-first (hosted users can't set num_ctx) — the reverse of query_ollama_num_ctx(), hence the namespaced memo key."""
    # ⤷ 出口 return：返回 _memo_local_probe(('ollama_show', _strip_provider_prefix(mod…
    return _memo_local_probe(("ollama_show", _strip_provider_prefix(model), base_url.rstrip("/")), lambda: _query_ollama_api_show_uncached(model, base_url, api_key=api_key))


# ┌── 函数 def _query_ollama_api_show_uncached()
# │  作用：Uncached body 的 ``_query_ollama_api_show`` — one POST ``/API/show``.
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _query_ollama_api_show_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Uncached body of ``_query_ollama_api_show`` — one POST to ``/api/show``."""
    server_url = _server_root(base_url)
    if _endpoint_blackholed(server_url):
        # ⤷ 出口 return：返回 None
        return None
    data = _ollama_show(server_url, api_key, model, timeout=5.0, note_blackhole=True)
    # Hosted Ollama: the GGUF max is authoritative (the operator may have capped num_ctx arbitrarily).
    # ⤷ 出口 return：返回 _ollama_show_context(data, gguf_first=True, minimum=1024) if…
    return _ollama_show_context(data, gguf_first=True, minimum=1024) if data is not None else None


# ┌── 函数 def _model_name_suggests_kimi()
# │  作用：Kimi family (``kimi-*``, ``moonshotai/*``) — 护栏 against 卡死的 32K underreports.
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _model_name_suggests_kimi(model: str) -> bool:
    """Kimi family (``kimi-*``, ``moonshotai/*``) — guard against stale 32K underreports."""
    lower = model.lower()
    # ⤷ 出口 return：返回 lower.startswith('kimi') or 'moonshot' in lower
    return lower.startswith("kimi") or "moonshot" in lower


# ┌── 函数 def _model_name_suggests_minimax_m3()
# │  作用：MiniMax M3 在…上任意上报 — 模型.dev underreport 护栏且 agent_runtime_helpers 缓存-control gating.
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _model_name_suggests_minimax_m3(model: str) -> bool:
    """MiniMax M3 on any surface — models.dev underreport guard and agent_runtime_helpers cache-control gating."""
    # ⤷ 出口 return：返回 'minimax-m3' in model.lower()
    return "minimax-m3" in model.lower()


# Catalog keys added AFTER the model was reachable via a shorter catch-all (or the
# 256K fallback): older builds persisted that smaller value and the step-1 cache hit
# would pin it forever. Only list keys whose catalog value is STRICTLY ABOVE every
# shorter matching key and the 256K fallback — the threshold is inferred from them.
_PRE_CATALOG_STALE_KEYS = frozenset({
    "minimax-m3",  # 1M; "minimax" catch-all persisted 204,800
    "muse-spark-1.3", "muse-spark",  # 1M; pre-entry builds fell through to the 256K fallback
    "grok-4.3", "grok-4.6",  # 1M / 500K; "grok-4" catch-all persisted 256,000
    "grok-4-fast", "grok-4.20",  # 2M; fell through to the 256K fallback
    "qwen3.6-plus",  # 1M; "qwen" catch-all persisted 131,072
    # V4 / V4.1 Flash: 1M. Pre-entry builds matched the family catch-all and persisted 128K.
    "deepseek-flash", "deepseek-v4.1-flash", "deepseek-v4-flash", "deepseek-v4-pro",
    "deepseek-chat", "deepseek-reasoner",
})


# ┌── 函数 def _stale_pre_catalog_cache_entry()
# │  作用：当 已持久化的 window 是 pre-catalog leftover: 模型 resolves (longest-键-first) 时返回 True。
# │  参数：model: str, cached: int
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _stale_pre_catalog_cache_entry(model: str, cached: int) -> bool:
    """True when a persisted window is a pre-catalog leftover: the model resolves (longest-key-first) to a
    _PRE_CATALOG_STALE_KEYS key and the cached value is <= the largest shorter matching catch-all (or 256K)."""
    model_lower = model.lower()
    matches = [(key, value) for key, value in DEFAULT_CONTEXT_LENGTHS.items() if _catalog_key_matches(key, model_lower)]
    if not matches:
        # ⤷ 出口 return：返回 False
        return False
    specific_key, specific_value = max(matches, key=lambda kv: len(kv[0]))
    if specific_key not in _PRE_CATALOG_STALE_KEYS or cached >= specific_value:
        # ⤷ 出口 return：返回 False
        return False
    shorter_values = [v for k, v in matches if len(k) < len(specific_key)]
    # ⤷ 出口 return：返回 cached <= max(shorter_values, default=DEFAULT_FALLBACK_CONTE…
    return cached <= max(shorter_values, default=DEFAULT_FALLBACK_CONTEXT)


# ┌── 函数 def _model_name_suggests_minimax()
# │  作用：MiniMax family (``minimax*``, ``minimaxai/*``) — 护栏 against 卡死的 32K underreports (real: 204.8K).
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _model_name_suggests_minimax(model: str) -> bool:
    """MiniMax family (``minimax*``, ``minimaxai/*``) — guard against stale 32K underreports (real: 204.8K)."""
    lower = model.lower()
    # ⤷ 出口 return：返回 lower.startswith('minimax') or 'minimaxai/' in lower
    return lower.startswith("minimax") or "minimaxai/" in lower


# ┌── 函数 def _model_name_suggests_stale_32k_underreport()
# │  作用：模型 families known 是 wrongly underreported 作为 32K.
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _model_name_suggests_stale_32k_underreport(model: str) -> bool:
    """Model families known to be wrongly underreported as 32K."""
    # ⤷ 出口 return：返回 _model_name_suggests_kimi(model) or _model_name_suggests_min…
    return _model_name_suggests_kimi(model) or _model_name_suggests_minimax(model)


# ┌── 函数 def _query_local_context_length()
# │  作用：Local-服务端上下文 probe, short-TTL cached (see _LOCAL_CTX_PROBE_CACHE).
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _query_local_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Local-server context probe, short-TTL cached (see _LOCAL_CTX_PROBE_CACHE)."""
    # ⤷ 出口 return：返回 _memo_local_probe((_strip_provider_prefix(model), base_url.r…
    return _memo_local_probe((_strip_provider_prefix(model), base_url.rstrip("/")), lambda: _query_local_context_length_uncached(model, base_url, api_key=api_key))


# ┌── 函数 def _positive_int()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _positive_int(value: Any) -> Optional[int]:
    # ⤷ 出口 return：返回 int(value) if isinstance(value, (int, float)) and value else…
    return int(value) if isinstance(value, (int, float)) and value else None


# ┌── 函数 def _lmstudio_context()
# │  作用：LM Studio native /API/v1/模型 ( OpenAI-compat 列表 omits 上下文);
# │  参数：client, lmstudio_url: str, model: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _lmstudio_context(client, lmstudio_url: str, model: str) -> Optional[int]:
    """LM Studio native /api/v1/models (the OpenAI-compat list omits context);
    loaded-instance config is the runtime value."""
    resp = client.get(f"{lmstudio_url}/api/v1/models")
    if resp.status_code != 200:
        # ⤷ 出口 return：返回 None
        return None
    for m in resp.json().get("models", []):
        if _model_id_matches(m.get("key", ""), model) or _model_id_matches(m.get("id", ""), model):
            # ⤷ 出口 return：返回 next((ctx for ctx in (_positive_int(inst.get('config', {}).g…
            return next((ctx for ctx in (_positive_int(inst.get("config", {}).get("context_length")) for inst in m.get("loaded_instances", [])) if ctx is not None), None)
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _llamacpp_context()
# │  作用：llama.cpp /props: 运行时 n_ctx, answered 由 router even 用于不-yet-已加载模型
# │  参数：client, server_url: str, model: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _llamacpp_context(client, server_url: str, model: str) -> Optional[int]:
    """llama.cpp /props: the RUNTIME n_ctx, answered by the router even for a not-yet-loaded model
    (while /v1/models has meta=null), so a lazily-loaded model doesn't fall to a family catch-all."""
    import httpx
    for props_path in (f"/props?model={model}", "/props"):
        # ⤷ 关键分支 try/except：捕获 httpx.HTTPError，出错走兜底而不是崩溃。
        try:
            resp = client.get(f"{server_url}{props_path}")
        except httpx.HTTPError:
            # ⤷ 出口 return：返回 None
            return None
        if resp.status_code != 200:
            continue
        n_ctx = _positive_int((resp.json().get("default_generation_settings") or {}).get("n_ctx"))
        if n_ctx is not None:
            # ⤷ 出口 return：返回 n_ctx
            return n_ctx
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _openai_models_list_context()
# │  作用：/v1/模型列表: match 由 id, 否则 sole 模型在…上 single-模型服务端 (llama.cpp reports GGUF 路径作为 id).
# │  参数：client, server_url: str, model: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _openai_models_list_context(client, server_url: str, model: str) -> Optional[int]:
    """/v1/models list: match by id, else the sole model on single-model servers (llama.cpp reports a GGUF path as id)."""
    resp = client.get(f"{server_url}/v1/models")
    if resp.status_code != 200:
        # ⤷ 出口 return：返回 None
        return None
    models_list = resp.json().get("data", [])
    matched = next((m for m in models_list if isinstance(m, dict) and _model_id_matches(m.get("id", ""), model)), None)
    if matched is None and len(models_list) == 1:
        matched = models_list[0]
    if matched is None:
        # ⤷ 出口 return：返回 None
        return None
    # Runtime n_ctx (llama.cpp nests it under meta) beats n_ctx_train, which can exceed what the server allocates.
    sources = [s for s in (matched, matched.get("meta") or {}) if isinstance(s, dict)]
    for reader in (lambda src: _positive_int(src.get("n_ctx")), _context_length_from_model_payload):
        for source in sources:
            ctx = reader(source)
            if ctx is not None:
                # ⤷ 出口 return：返回 ctx
                return ctx
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _query_local_context_length_uncached()
# │  作用：Query local 服务端用于模型's 上下文 length.
# │  参数：model: str, base_url: str, api_key: str = ''
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _query_local_context_length_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length."""
    import httpx
    model = _strip_provider_prefix(model)
    server_url = _server_root(base_url)
    lmstudio_url = _localhost_to_ipv4(_lmstudio_server_root(base_url))
    if _endpoint_blackholed(server_url):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        server_type = None
    def _ollama_ctx(client) -> Optional[int]:
        # Ollama: num_ctx (runtime window) before the GGUF training max, or conversations
        # grow past what Ollama allocated and silently truncate. Matches query_ollama_num_ctx().
        resp = client.post(f"{server_url}/api/show", json={"name": model})
        # ⤷ 出口 return：返回 _ollama_show_context(resp.json(), gguf_first=False) if resp.…
        return _ollama_show_context(resp.json(), gguf_first=False) if resp.status_code == 200 else None
    def _model_detail_ctx(client) -> Optional[int]:
        # LM Studio / vLLM / llama.cpp / Anthropic-compat proxies: /v1/models/{model}
        resp = client.get(f"{server_url}/v1/models/{model}")
        # ⤷ 出口 return：返回 _context_length_from_model_payload(resp.json()) if resp.stat…
        return _context_length_from_model_payload(resp.json()) if resp.status_code == 200 else None
    typed = {
        "ollama": _ollama_ctx,
        "lm-studio": lambda client: _lmstudio_context(client, lmstudio_url, model),
        "llamacpp": lambda client: _llamacpp_context(client, server_url, model),
    }.get(server_type)
    probes = ([typed] if typed else []) + [_model_detail_ctx, lambda client: _openai_models_list_context(client, server_url, model)]
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        with httpx.Client(timeout=3.0, headers=_auth_headers(api_key)) as client:
            return next((ctx for ctx in (probe(client) for probe in probes) if ctx is not None), None)
    except Exception as exc:
        _note_if_connect_timeout(exc, server_url)
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _normalize_model_version()
# │  作用：Dots -> dashes 从而 Nous ids (claude-opus-4-6) compare 带 OpenRouter's (claude-opus-4.6).
# │  参数：model: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _normalize_model_version(model: str) -> str:
    """Dots -> dashes so Nous ids (claude-opus-4-6) compare with OpenRouter's (claude-opus-4.6)."""
    # ⤷ 出口 return：返回 model.replace('.', '-')
    return model.replace(".", "-")


# ┌── 函数 def _query_anthropic_context_length()
# │  作用：Anthropic /v1/模型 max_input_tokens; OAuth 令牌 (sk-ant-oat*) 401 且是 skipped.
# │  参数：model: str, base_url: str, api_key: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _query_anthropic_context_length(model: str, base_url: str, api_key: str) -> Optional[int]:
    """Anthropic /v1/models max_input_tokens; OAuth tokens (sk-ant-oat*) 401 and are skipped."""
    if not api_key or api_key.startswith("sk-ant-oat"):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        base = base_url.rstrip("/").removesuffix("/v1")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        _ensure_requests()
        resp = requests.get(f"{base}/v1/models?limit=1000", headers=headers, timeout=(5, 10), verify=_resolve_requests_verify(base_url))
        if resp.status_code != 200:
            # ⤷ 出口 return：返回 None
            return None
        for m in resp.json().get("data", []):
            ctx = m.get("max_input_tokens") if m.get("id") == model else None
            if isinstance(ctx, int) and ctx > 0:
                # ⤷ 出口 return：返回 ctx
                return ctx
    except Exception as e:
        logger.debug("Anthropic /v1/models query failed: %s", e)
    # ⤷ 出口 return：返回 None
    return None


# Codex OAuth `context_window` values (what Codex enforces — lower than the direct API for the same
# slugs). Fallback when the live probe fails; longest-key-first. gpt-5.3-codex-spark is listed so "gpt-5.3-codex" doesn't win.
_CODEX_OAUTH_CONTEXT_FALLBACK: Dict[str, int] = {
    "gpt-6-astra": 272_000,
    "gpt-5.1-codex-max": 272_000, "gpt-5.1-codex-mini": 272_000, "gpt-5.3-codex": 272_000,
    "gpt-5.3-codex-spark": 128_000, "gpt-5.2-codex": 272_000, "gpt-5.4-mini": 272_000,
    "gpt-5.6-sol": 272_000, "gpt-5.6-terra": 272_000, "gpt-5.6-luna": 272_000, "gpt-daybreak-blue-latest": 272_000,
    "gpt-5.5": 272_000, "gpt-5.4": 272_000, "gpt-5.2": 272_000, "gpt-5": 272_000,
}
# Codex OAuth advertises 272K for these families but ACCEPTS ~900K+ (verified live; gpt-5.5 and
# gpt-5.4-mini genuinely reject >272K). 900K keeps ≥11K margin. OPT-IN ONLY via explicit ``-900k``
# picker variants (a 900K default burned subscription usage); the suffix is stripped before the wire.
# The bump fires ONLY when the resolved value is exactly the stale 272,000. ``gpt-5.6`` is a FAMILY
# PREFIX (``-pro`` slugs aren't routable on Codex); ``gpt-5.4`` is EXACT because gpt-5.4-mini enforces 272K.
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES: Dict[str, int] = {"gpt-5.6": 900_000}  # sol / terra / luna
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT: Dict[str, int] = {
    "gpt-5.4": 900_000, "gpt-daybreak-blue-latest": 900_000,
    "gpt-6-astra": 900_000,  # advertised 272K; 920,043 input OK, 1,000,043 rejected (live 2026-09-04)
}
_CODEX_OAUTH_STALE_ADVERTISED_CTX = 272_000  # the only advertised value the bump may override
CODEX_CONTEXT_VARIANT_SUFFIX = "-900k"  # picker-only opt-in suffix; never sent on the wire
# The ONLY bases eligible for ``-900k``: routable, live-verified. No family prefixing (it would synthesize
# dead ``-pro`` variants); dated snapshots of the 5.6 bases are allowed. gpt-daybreak-blue-latest is a verified Sol alias.
_CODEX_900K_SNAPSHOT_BASES = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
_CODEX_900K_ELIGIBLE_BASES = frozenset({*_CODEX_900K_SNAPSHOT_BASES, "gpt-5.4", "gpt-daybreak-blue-latest", "gpt-6-astra"})
_CODEX_900K_SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ┌── 函数 def _bare_codex_slug()
# │  作用：Lowercased slug 不 ``vendor/`` (展示/辅助的 callers pass ``OpenAI/gpt-5.6-sol-900k``).
# │  参数：model: Optional[str]
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _bare_codex_slug(model: Optional[str]) -> str:
    """Lowercased slug without ``vendor/`` (display/auxiliary callers pass ``openai/gpt-5.6-sol-900k``).

    Display/auxiliary callers pass ids like ``openai/gpt-5.6-sol-900k``; the main-agent path normalizes the
    namespace away earlier, but this resolver must accept both shapes (#92797 review).
    """
    # ⤷ 出口 return：返回 (model or '').strip().lower().rsplit('/', 1)[-1]
    return (model or "").strip().lower().rsplit("/", 1)[-1]


# ┌── 函数 def is_codex_900k_base()
# │  作用：Single source 的 truth 用于 ``-900k`` eligibility (picker, resolution, /模型 validation, wire stripping).
# │  参数：model: Optional[str]
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_codex_900k_base(model: Optional[str]) -> bool:
    """Single source of truth for ``-900k`` eligibility (picker, resolution, /model validation, wire stripping)."""
    slug = _bare_codex_slug(model)
    if not slug or slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        # ⤷ 出口 return：返回 False
        return False
    # Dated snapshots of the routable 5.6 bases (gpt-5.6-sol-2026-07-09) also qualify.
    # ⤷ 出口 return：返回 slug in _CODEX_900K_ELIGIBLE_BASES or any((slug.startswith(b…
    return slug in _CODEX_900K_ELIGIBLE_BASES or any(
        slug.startswith(base + "-") and _CODEX_900K_SNAPSHOT_RE.match(slug[len(base) + 1:]) for base in _CODEX_900K_SNAPSHOT_BASES
    )


# ┌── 函数 def _codex_variant_base()
# │  作用：Lowercased eligible base 的 VALID ``-900k`` variant, 否则 None (``gpt-5.5-900k`` 是 invalid 别名).
# │  参数：model: Optional[str]
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _codex_variant_base(model: Optional[str]) -> Optional[str]:
    """Lowercased eligible base of a VALID ``-900k`` variant, else None (``gpt-5.5-900k`` is an invalid alias)."""
    slug = _bare_codex_slug(model)
    base = slug[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)] if slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX) else None
    # ⤷ 出口 return：返回 base if base is not None and is_codex_900k_base(base) else N…
    return base if base is not None and is_codex_900k_base(base) else None


# ┌── 函数 def is_codex_context_variant()
# │  作用：Suffix 且 eligible base — ``gpt-5.5-900k`` 是 invalid 别名, 不 variant.
# │  参数：model: Optional[str]
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_codex_context_variant(model: Optional[str]) -> bool:
    """Suffix AND eligible base — ``gpt-5.5-900k`` is an invalid alias, not a variant."""
    # ⤷ 出口 return：返回 _codex_variant_base(model) is not None
    return _codex_variant_base(model) is not None


# ┌── 函数 def strip_codex_context_variant_suffix()
# │  作用：Wire-safe slug 带 VALID ``-900k`` suffix 已移除 (vendor prefix 保留); ineligible 别名是
# │  参数：model: Optional[str]
# │  返回：str
# └──────────────────────────────────────────────────────────────
def strip_codex_context_variant_suffix(model: Optional[str]) -> str:
    """Wire-safe slug with a VALID ``-900k`` suffix removed (vendor prefix kept); an ineligible alias is
    returned unchanged so it fails honestly at the API instead of running as another model."""
    raw = (model or "").strip()
    # ⤷ 出口 return：返回 raw[:-len(CODEX_CONTEXT_VARIANT_SUFFIX)] if _codex_variant_b…
    return raw[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)] if _codex_variant_base(raw) is not None else raw


# ┌── 函数 def has_codex_context_variant()
# │  作用：Picker-side 别名的 :func:`is_codex_900k_base`.
# │  参数：model_bare: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def has_codex_context_variant(model_bare: str) -> bool:
    """Picker-side alias of :func:`is_codex_900k_base`."""
    # ⤷ 出口 return：返回 is_codex_900k_base(model_bare)
    return is_codex_900k_base(model_bare)


# ┌── 函数 def _verified_codex_ctx_for_slug()
# │  作用：当前活跃的-verified cap 用于 VALID ``-900k`` variant 仅; base slugs 且 ineligible 别名 -> None.
# │  参数：model_bare: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _verified_codex_ctx_for_slug(model_bare: str) -> Optional[int]:
    """Live-verified cap for a VALID ``-900k`` variant only; base slugs and ineligible aliases -> None."""
    base = _codex_variant_base(model_bare)
    if base is None:
        # ⤷ 出口 return：返回 None
        return None
    exact = _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT.get(base)
    if exact is not None:
        # ⤷ 出口 return：返回 exact
        return exact
    # ⤷ 出口 return：返回 next((ctx for key, ctx in _CODEX_OAUTH_VERIFIED_ABOVE_ADVERT…
    return next((ctx for key, ctx in _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES.items() if base == key or base.startswith((key + "-", key + "."))), None)


_codex_oauth_context_cache: Dict[str, Tuple[Dict[str, int], float]] = {}
_CODEX_OAUTH_CONTEXT_CACHE_TTL = 3600  # 1 hour
# The Codex models endpoint reads ``client_version`` as a Codex CLI compatibility version and
# hides models whose ``minimal_client_version`` is newer, so a made-up version (the old
# "1.0.0") silently drops future models. "0.0.0" is the backend's ungated sentinel returning
# the full account catalog; other out-of-sequence values return an empty catalog and omitting
# the parameter is HTTP 400.
CODEX_UNGATED_CLIENT_VERSION = "0.0.0"
CODEX_MODELS_CATALOG_URL = f"https://chatgpt.com/backend-api/codex/models?client_version={CODEX_UNGATED_CLIENT_VERSION}"


# ┌── 函数 def _codex_oauth_token_fingerprint()
# │  作用：Non-secret 缓存键 用于 Codex OAuth access 令牌.
# │  参数：access_token: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _codex_oauth_token_fingerprint(access_token: str) -> str:
    """Non-secret cache key for a Codex OAuth access token."""
    # ⤷ 出口 return：返回 hashlib.sha256(access_token.encode('utf-8')).hexdigest()[:16…
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


# ┌── 函数 def _extract_chatgpt_account_id()
# │  作用：``chatgpt_account_id`` 从 Codex OAuth JWT, 或 None 在…上任意解析错误. 不
# │  参数：access_token: str
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    """``chatgpt_account_id`` from the Codex OAuth JWT, or None on any parse error. Without the
    ``ChatGPT-Account-Id`` header /backend-api/codex/models returns ``{"models":[]}`` (HTTP 200)
    and the probe silently falls back. Mirrors auxiliary_client.py."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        payload_b64 = access_token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id") if isinstance(claims, dict) else None
        # ⤷ 出口 return：返回 acct_id if isinstance(acct_id, str) and acct_id else None
        return acct_id if isinstance(acct_id, str) and acct_id else None
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _fetch_codex_oauth_context_lengths_with_source()
# │  作用：Codex catalogue ``{slug: context_window}`` plus whether 它 came 从 HTTP. Cached per 令牌
# │  参数：access_token: str
# │  返回：Tuple[Dict[str, int], bool]
# └──────────────────────────────────────────────────────────────
def _fetch_codex_oauth_context_lengths_with_source(access_token: str) -> Tuple[Dict[str, int], bool]:
    """Codex catalogue ``{slug: context_window}`` plus whether it came from HTTP. Cached per token
    fingerprint (windows vary by entitlement). An in-process hit reports False: not a fresh
    provider confirmation, must not drive persistent writes."""
    now = time.time()
    cache_key = _codex_oauth_token_fingerprint(access_token)
    cached = _codex_oauth_context_cache.get(cache_key)
    if cached is not None and now - cached[1] < _CODEX_OAUTH_CONTEXT_CACHE_TTL:
        # ⤷ 出口 return：返回 (cached[0], False)
        return cached[0], False
    headers = {"Authorization": f"Bearer {access_token}"}
    acct_id = _extract_chatgpt_account_id(access_token)
    if acct_id:
        headers["ChatGPT-Account-Id"] = acct_id
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        _ensure_requests()
        resp = requests.get(CODEX_MODELS_CATALOG_URL, headers=headers, timeout=(5, 10), verify=_resolve_requests_verify())
        if resp.status_code != 200:
            logger.debug("Codex /models probe returned HTTP %s; falling back to hardcoded defaults", resp.status_code)
            # ⤷ 出口 return：返回 ({}, False)
            return {}, False
        data = resp.json()
    except Exception as exc:
        logger.debug("Codex /models probe failed: %s", exc)
        # ⤷ 出口 return：返回 ({}, False)
        return {}, False
    result: Dict[str, int] = {}
    for item in data.get("models", []) if isinstance(data, dict) else []:
        slug, ctx = (item.get("slug"), item.get("context_window")) if isinstance(item, dict) else (None, None)
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx
    if result:
        _codex_oauth_context_cache[cache_key] = (result, now)
    # ⤷ 出口 return：返回 (result, True)
    return result, True


# ┌── 函数 def _resolve_codex_oauth_context_length_with_source()
# │  作用：``(context_length, source)`` 用于 Codex OAuth slug. source: "当前活跃的" (fresh authenticated probe —
# │  参数：model: str, access_token: str = ''
# │  返回：Tuple[Optional[int], str]
# └──────────────────────────────────────────────────────────────
def _resolve_codex_oauth_context_length_with_source(model: str, access_token: str = "") -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Codex OAuth slug. source: "live" (fresh authenticated probe —
    the only one eligible for persistent writes), "memory" (same-token in-process hit), "fallback"
    (static table), or "" when unresolved."""
    model_bare = _strip_provider_prefix(model).strip()
    if not model_bare:
        # ⤷ 出口 return：返回 (None, '')
        return None, ""
    def _apply_verified_bump(ctx: int, source: str) -> Tuple[int, str]:
        """Lift an EXACT stale 272K advertisement to the verified cap for opted-in ``-900k`` variants only."""
        bumped = _verified_codex_ctx_for_slug(model_bare)
        if bumped is not None and ctx == _CODEX_OAUTH_STALE_ADVERTISED_CTX:
            logger.debug("Codex OAuth context for %s: advertised %d raised to live-verified %d", model_bare, ctx, bumped)
            # ⤷ 出口 return：返回 (bumped, source)
            return bumped, source
        # ⤷ 出口 return：返回 (ctx, source)
        return ctx, source
    # The Codex catalog only knows the base slug (no -900k, no vendor/).
    # ``-900k`` variants are Hermes picker aliases — the Codex catalog only knows the base slug, so resolve
    # against the stripped id. Also drop any ``vendor/`` namespace (``openai/gpt-5.6-sol-900k``): the
    # main-agent path normalizes it away before reaching here, but display/auxiliary callers pass it through
    # (#92797 review).
    lookup_bare = _bare_codex_slug(strip_codex_context_variant_suffix(model_bare))
    if access_token:
        live, fresh_probe = _fetch_codex_oauth_context_lengths_with_source(access_token)
        # Exact slug, then case-insensitive in case casing drifts.
        hit = live.get(lookup_bare)
        if hit is None:
            hit = next((ctx for slug, ctx in live.items() if slug.lower() == lookup_bare.lower()), None)
        if hit is not None:
            # ⤷ 出口 return：返回 _apply_verified_bump(hit, 'live' if fresh_probe else 'memory…
            return _apply_verified_bump(hit, "live" if fresh_probe else "memory")
    hit = _longest_key_match(_CODEX_OAUTH_CONTEXT_FALLBACK, lookup_bare.lower())
    # ⤷ 出口 return：返回 _apply_verified_bump(hit[1], 'fallback') if hit else (None, …
    return _apply_verified_bump(hit[1], "fallback") if hit else (None, "")


# ┌── 函数 def _resolve_nous_context_length()
# │  作用：``(context_length, source)`` 用于 Nous Portal 模型: portal /v1/模型是 authoritative
# │  参数：model: str, base_url: str = '', api_key: str = ''
# │  返回：Tuple[Optional[int], str]
# └──────────────────────────────────────────────────────────────
def _resolve_nous_context_length(model: str, base_url: str = "", api_key: str = "") -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Nous Portal model: portal /v1/models is authoritative
    ("portal"). Fallback matches OR's prefixed ids against the bare Nous id with dot/dash
    normalisation ("openrouter" — callers must NOT persist it, or a portal blip freezes the wrong value)."""
    if base_url:
        portal_ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if portal_ctx is not None:
            # ⤷ 出口 return：返回 (portal_ctx, 'portal')
            return portal_ctx, "portal"
    metadata = fetch_model_metadata()
    def _safe_ctx(or_id: str, entry: dict) -> Optional[int]:
        """Context length minus the known stale 32K underreports (same guard as step 6)."""
        ctx = entry.get("context_length")
        if ctx is not None and ctx <= 32768 and _model_name_suggests_stale_32k_underreport(or_id):
            logger.info("Rejecting OpenRouter metadata context=%s for %r (known 32K underreport, Nous path); falling through to hardcoded defaults", ctx, or_id)
            return None
        # ⤷ 出口 return：返回 ctx
        return ctx
    model_lower, normalized = model.lower(), _normalize_model_version(model).lower()
    def _pairs(or_id: str):
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        # ⤷ 出口 return：返回 ((bare.lower(), model_lower), (_normalize_model_version(bare…
        return ((bare.lower(), model_lower), (_normalize_model_version(bare).lower(), normalized))
    def _exact(or_id: str) -> bool:
        # ⤷ 出口 return：返回 any((candidate == query for candidate, query in _pairs(or_id…
        return any(candidate == query for candidate, query in _pairs(or_id))
    def _prefix(or_id: str) -> bool:
        # ⤷ 出口 return：返回 any((candidate.startswith(query) and (len(candidate) == len(…
        return any(candidate.startswith(query) and (len(candidate) == len(query) or candidate[len(query)] in "-:.") for candidate, query in _pairs(or_id))
    # Direct id, then exact bare-id match (with dot/dash normalisation), then prefix match on a
    # separator boundary — separate passes so any exact hit beats every prefix hit.
    for matcher in (lambda or_id: or_id == model, _exact, _prefix):
        for or_id, entry in metadata.items():
            ctx = _safe_ctx(or_id, entry) if matcher(or_id) else None
            if ctx is not None:
                return ctx, "openrouter"
    # ⤷ 出口 return：返回 (None, '')
    return None, ""


# ┌── 函数 def _validate_cached_context_length()
# │  作用：步骤 1 的 get_model_context_length: 接受, repair, 或丢弃已持久化的 entry. 返回
# │  参数：model: str, base_url: str, cached: int, is_bedrock_context: bool, api_key: str = ...
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _validate_cached_context_length(model: str, base_url: str, cached: int, is_bedrock_context: bool, *, api_key: str = "") -> Optional[int]:
    """Step 1 of get_model_context_length: accept, repair, or drop a persisted entry. Returns the
    value to use, or None to fall through to live resolution. Order matters: a value must be
    rejected as bogus before any provider-specific handling."""
    # Drop rules: (predicate, log level, message, shown value). 0/negative is always a bug (`0 is not
    # None` would hand the compressor a zero window); Kimi/MiniMax are underreported as 32K by stale
    # third-party metadata; pre-catalog leftovers persisted a shorter catch-all (see _PRE_CATALOG_STALE_KEYS).
    drop_rules = (
        (cached <= 0, logger.warning, "Dropping non-positive cache entry %s@%s -> %s; re-resolving", cached),
        (cached <= 32768 and _model_name_suggests_stale_32k_underreport(model), logger.info,
         "Dropping stale cached context entry %s@%s -> %s (known 32K underreport); re-resolving via hardcoded defaults", f"{cached:,}"),
        (_stale_pre_catalog_cache_entry(model, cached), logger.info,
         "Dropping stale pre-catalog cache entry %s@%s -> %s; re-resolving via hardcoded defaults", f"{cached:,}"),
    )
    for hit, log, msg, shown in drop_rules:
        if hit:
            log(msg, model, base_url, shown)
            _invalidate_cached_context_length(model, base_url)
            # ⤷ 出口 return：返回 None
            return None
    # Nous Portal: /v1/models is authoritative. Bypass (don't drop) the cache so step
    # 5b reconciles OR-seeded entries without touching disk when the portal is down.
    if _infer_provider_from_url(base_url) == "nous":
        logger.debug("Bypassing persistent cache for %s@%s (Nous portal authoritative)", model, base_url)
        # ⤷ 出口 return：返回 None
        return None
    if is_bedrock_context:
        # Bedrock: the static table is a FLOOR — probe-derived entries may legitimately exceed it.
        # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
        try:
            from agent.bedrock_adapter import get_bedrock_context_length
            bedrock_ctx = get_bedrock_context_length(model)
        except ImportError:
            return cached
        if cached < bedrock_ctx:
            logger.info("Dropping stale Bedrock cache entry %s@%s -> %s; using static Bedrock table value %s", model, base_url, f"{cached:,}", f"{bedrock_ctx:,}")
            _invalidate_cached_context_length(model, base_url)
            return bedrock_ctx
        # ⤷ 出口 return：返回 cached
        return cached
    # For local endpoints, run the probe that respects configured Modelfile context values first.
    # _query_local_context_length prefers num_ctx from Modelfile, while _query_ollama_api_show returns the
    # GGUF training max first which can be larger and would create a false-safe window for compression
    # (#63122). Non-local endpoints preserve the existing GGUF-first behavior.
    if is_local_endpoint(base_url):
        # ⤷ 出口 return：返回 _reconcile_local_cached_context_length(model, base_url, cach…
        return _reconcile_local_cached_context_length(model, base_url, cached, api_key=api_key)
    # ⤷ 出口 return：返回 cached
    return cached


# ┌── 函数 def _resolve_bedrock_context_length()
# │  作用：步骤 1b: Bedrock static table + one cached 当前活跃的 probe (Bedrock exposes 无上下文窗口 via
# │  参数：model: str, base_url: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _resolve_bedrock_context_length(model: str, base_url: str) -> Optional[int]:
    """Step 1b: Bedrock static table + one cached live probe (Bedrock exposes no context window via
    metadata APIs); None when boto3 is absent. Cached per model under base_url, else a synthetic
    bedrock:// key so display/offline paths share it."""
    # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
    try:
        from agent.bedrock_adapter import get_bedrock_context_length, resolve_bedrock_region
    except ImportError:
        # ⤷ 出口 return：返回 None
        return None  # boto3 not installed — fall through to generic resolution
    cache_key_url = base_url or "bedrock://"
    cached = get_cached_context_length(model, cache_key_url)
    if cached is not None:
        # ⤷ 出口 return：返回 cached
        return cached
    # Region from the base_url host first, then the standard AWS chain. An empty region disables probing (table only).
    _m = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url) if base_url else None
    region = _m.group(1) if _m else ""
    if not region:
        with contextlib.suppress(Exception):
            region = resolve_bedrock_region()
    ctx = get_bedrock_context_length(model, region=region, probe=bool(region))
    # Only persist probe-derived values (region present); a pure table fallback must not poison the cache.
    if ctx and region:
        save_context_length(model, cache_key_url, ctx)
    # ⤷ 出口 return：返回 ctx
    return ctx


# ┌── 函数 def _resolve_custom_endpoint_context_length()
# │  作用：步骤 2-3 用于 truly custom endpoint: /模型, local probes, Ollama /API/show, catalog, 默认值.
# │  参数：model: str, base_url: str, api_key: str, provider: str
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _resolve_custom_endpoint_context_length(model: str, base_url: str, api_key: str, provider: str) -> int:
    """Steps 2-3 for a truly custom endpoint: /models, local probes, Ollama /api/show, catalog, default."""
    context_length = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
    if context_length is not None:
        # ⤷ 出口 return：返回 context_length
        return context_length
    # Local endpoints: the num_ctx-aware probe first — _query_ollama_api_show is GGUF-first, which
    # can be larger and create a false-safe compression window.
    local_ctx = _probe_local_context_length(model, base_url, api_key, provider) if is_local_endpoint(base_url) else None
    if local_ctx:
        # ⤷ 出口 return：返回 local_ctx
        return local_ctx
    # 2b. Ollama native /api/show (GGUF-first for non-local). Non-Ollama servers 404/405 quickly.
    ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
    if ctx is not None:
        _save_unless_skipped(model, base_url, ctx, provider)
        # ⤷ 出口 return：返回 ctx
        return ctx
    # 3. Probe-down fallback after endpoint-specific detection failed
    logger.info(
        "Could not detect context length for model %r at %s — defaulting to %s tokens (probe-down). "
        "Set model.context_length in config.yaml to override.",
        model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )
    # 3b. Hardcoded catalog as a last resort: a proxied Anthropic gateway fails the probes above
    # but its model name still matches DEFAULT_CONTEXT_LENGTHS.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        logger.info("Using hardcoded context length %s for model %r (custom endpoint, catalog match on %r)", f"{hit[1]:,}", model, hit[0])
        # ⤷ 出口 return：返回 hit[1]
        return hit[1]
    # Same silent-256K bug class as the step-9 fallback — warn here too.
    _warn_context_length_fallback(model, base_url)
    # ⤷ 出口 return：返回 DEFAULT_FALLBACK_CONTEXT
    return DEFAULT_FALLBACK_CONTEXT


# ┌── 函数 def _resolve_moa_context_length()
# │  作用：步骤 0a: MoA virtual 服务商 — ``模型`` 是 preset 名称, 从而每个 probe 会 miss. Resolve
# │  参数：model: str, custom_providers: list | None
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _resolve_moa_context_length(model: str, custom_providers: list | None) -> Optional[int]:
    """Step 0a: MoA virtual provider — ``model`` is a preset name, so every probe would miss. Resolve
    the aggregator's real provider+model (references are advisory). None on any failure."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config
        from hermes_cli.moa_config import resolve_moa_preset
        from hermes_cli.runtime_provider import resolve_runtime_provider
        config = load_config()
        if custom_providers is None:
            custom_providers = get_compatible_custom_providers(config)
        agg = resolve_moa_preset(config.get("moa") or {}, model).get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_model and agg_provider and agg_provider.lower() != "moa":
            rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
            # ⤷ 出口 return：返回 get_model_context_length(agg_model, base_url=rt.get('base_ur…
            return get_model_context_length(
                agg_model, base_url=rt.get("base_url", "") or "", api_key=rt.get("api_key", "") or "",
                provider=rt.get("provider") or agg_provider, custom_providers=custom_providers,
            )
    except Exception:
        logger.debug("MoA aggregator context-length resolution failed", exc_info=True)
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _config_override_context_length()
# │  作用：步骤 0b-0c: 配置-仅覆盖值 (never touch network). 0b: EXPLICIT model_overrides
# │  参数：model: str, base_url: str, provider: str, custom_providers: list | None
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _config_override_context_length(model: str, base_url: str, provider: str, custom_providers: list | None) -> Optional[int]:
    """Steps 0b-0c: config-only overrides (never touch the network). 0b: EXPLICIT model_overrides
    only — fill-gap _default entries apply inside lookup_models_dev_context once the catalog has
    missed, so a _default can never preempt custom_providers or live probes. 0c: custom_providers."""
    # This is the supported self-unblock path for models with wrong context in models.dev (#84482) and for
    # custom/local models (#8731).
    if provider and model:
        with contextlib.suppress(Exception):  # fall through to other resolution paths
            from agent.models_dev import _override_context_window
            mo_ctx = _override_context_window(provider, model)
            if mo_ctx is not None and mo_ctx > 0:
                # ⤷ 出口 return：返回 mo_ctx
                return mo_ctx
    # 0c. custom_providers per-model override — check before any probe. This closes the gap where /model
    # switch and display paths used to fall back to 128K despite the user having a per-model context_length
    # set. See #15779.
    if custom_providers and base_url and model:
        with contextlib.suppress(Exception):  # fall through to probing
            from hermes_cli.config import get_custom_provider_context_length
            cp_ctx = get_custom_provider_context_length(model=model, base_url=base_url, custom_providers=custom_providers)
            if cp_ctx:
                # ⤷ 出口 return：返回 cp_ctx
                return cp_ctx
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _resolve_provider_aware_context_length()
# │  作用：步骤 5: 服务商-specific sources, tried 在 order; None 当全部 miss.
# │  参数：model: str, base_url: str, api_key: str, provider: str, effective_provider: str
# │  返回：Optional[int]
# └──────────────────────────────────────────────────────────────
def _resolve_provider_aware_context_length(model: str, base_url: str, api_key: str, provider: str, effective_provider: str) -> Optional[int]:
    """Step 5: provider-specific sources, tried in order; None when all miss."""
    # 5a. Copilot live /models — account-specific models (claude-opus-4.6-1m) absent from
    # models.dev, and the provider-enforced limit for the rest.
    if effective_provider in {"copilot", "copilot-acp", "github-copilot"}:
        with contextlib.suppress(Exception):  # fall through to models.dev
            from hermes_cli.models import get_copilot_model_context
            ctx = get_copilot_model_context(model, api_key=api_key)
            if ctx:
                return ctx
    # 5b/5c. Nous portal and Codex OAuth (lower limits than the direct API for the same slug; its
    # own /models is authoritative). Persist ONLY the authoritative source ("portal" / "live"): an
    # OR-fallback or static-table value cached on a blip would be frozen in by step 1 forever.
    sourced = {
        "nous": lambda: _resolve_nous_context_length(model, base_url=base_url or "", api_key=api_key or "") + ("portal",),
        "openai-codex": lambda: _resolve_codex_oauth_context_length_with_source(model, access_token=api_key or "") + ("live",),
    }.get(effective_provider)
    if sourced is not None:
        ctx, source, persist_on = sourced()
        if ctx:
            if base_url and source == persist_on:
                save_context_length(model, base_url, ctx)
            # ⤷ 出口 return：返回 ctx
            return ctx
    if effective_provider in {"gmi", "commandcode", "commandcode-anthropic"} and base_url:
        # GMI and CommandCode expose authoritative context_length via /models (e.g. muse-spark 1M) but are
        # not in models.dev, and as known providers they skip step 2's probe — else they fell to 256K.
        ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if ctx is not None:
            # ⤷ 出口 return：返回 ctx
            return ctx
    # 5e. Ollama native /api/show for any base_url that is not a known non-Ollama provider (there
    # the POST always 404s and cost ~300ms on the first turn).
    if base_url:
        inferred = _infer_provider_from_url(base_url)
        ctx = _query_ollama_api_show(model, base_url, api_key=api_key) if inferred is None or "ollama" in inferred else None
        if ctx is not None:
            _save_unless_skipped(model, base_url, ctx, provider)
            # ⤷ 出口 return：返回 ctx
            return ctx
    # 5f. OpenRouter live /models — authoritative for OR-routed models, so it must win over models.dev
    # and the family catch-all (a brand-new slug would otherwise fall to the generic "claude": 200K).
    if effective_provider == "openrouter":
        or_ctx = (fetch_model_metadata().get(model) or {}).get("context_length")
        # Guard against the known OpenRouter Kimi-family 32k underreport.
        if isinstance(or_ctx, int) and or_ctx > 0 and not (or_ctx == 32768 and _model_name_suggests_kimi(model)):
            # ⤷ 出口 return：返回 or_ctx
            return or_ctx
    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            # MiniMax M3: models.dev reports 512K but actual context is 1M — prefer the hardcoded catalog.
            catalog = DEFAULT_CONTEXT_LENGTHS.get("minimax-m3") if _model_name_suggests_minimax_m3(model) else None
            if catalog and ctx < catalog:
                logger.info("Rejecting models.dev context=%s for %r (MiniMax-M3 underreport); using hardcoded default %s", ctx, model, f"{catalog:,}")
                ctx = catalog
            # ⤷ 出口 return：返回 ctx
            return ctx
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def get_model_context_length()
# │  作用：上下文 length 用于模型. Resolution order: 0 配置覆盖值 / MoA aggregator /
# │  参数：model: str, base_url: str = '', api_key: str = '', config_context_length: int | None = None, provider: str = '', custom_providers: list | None = None
# │  返回：int
# └──────────────────────────────────────────────────────────────
def get_model_context_length(
    model: str, base_url: str = "", api_key: str = "", config_context_length: int | None = None,
    provider: str = "", custom_providers: list | None = None,
) -> int:
    """Context length for a model. Resolution order: 0 config override / MoA aggregator /
    model_overrides / custom_providers / endpoint-scoped; 1 persistent cache (Nous, LM
    Studio, Codex OAuth bypass it) and Bedrock; 2-3 custom endpoints (/models, local
    probe, Ollama); 4 Anthropic /v1/models (API keys only); 5 provider-aware (Copilot,
    Nous, Codex OAuth, GMI, Ollama, OpenRouter live, models.dev); 6 OpenRouter for
    unknown providers; 7 local server; 8 hardcoded defaults; 9 256K fallback."""
    # 0. Explicit config override — user knows best
    if isinstance(config_context_length, int) and config_context_length > 0:
        # ⤷ 出口 return：返回 config_context_length
        return config_context_length
    if (provider or "").strip().lower() == "moa":
        ctx = _resolve_moa_context_length(model, custom_providers)
        if ctx is not None:
            return ctx
    ctx = _config_override_context_length(model, base_url, provider, custom_providers)
    if ctx is not None:
        # ⤷ 出口 return：返回 ctx
        return ctx
    # Malformed URLs (unmatched IPv6 bracket) make urllib.parse raise; treat them as unknown so
    # the inference layer reports the configuration error itself.
    if base_url:
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            _ = urlparse(_normalize_base_url(base_url)).port
        except ValueError:
            base_url = ""
    # A blank model id would fuzzy-match an arbitrary catalog entry (`"" in key` is vacuously
    # true) and persist it under a junk "@<base_url>" cache key.
    if not str(model or "").strip():
        logger.info("No model id provided for context length resolution — defaulting to %s tokens.", f"{DEFAULT_FALLBACK_CONTEXT:,}")
        # ⤷ 出口 return：返回 DEFAULT_FALLBACK_CONTEXT
        return DEFAULT_FALLBACK_CONTEXT
    model = _strip_provider_prefix(model)  # "local:x" -> "x"; Ollama "model:tag" colons preserved
    # OpenRouter routing variants (":nitro", ":floor", ...) are request-time
    # modifiers, not catalog entries — resolve the window from the BASE id.
    # Deliberately placed AFTER the explicit config overrides above (0b/0c) so
    # a user who pinned the fully-suffixed id keeps winning, and BEFORE every
    # cache/catalog lookup below so the base's real window is found instead of
    # a generic family default. Mirrors the validation path's base/suffix split
    # in hermes_cli.models.validate_requested_model.
    model = _strip_openrouter_routing_variant(model, base_url=base_url, provider=provider)
    # Endpoint-scoped metadata goes AHEAD of the persistent cache so a value learned on a
    # multiplexed provider's other endpoint cannot override it.
    endpoint_context = _endpoint_scoped_context_length(model, base_url)
    if endpoint_context is not None:
        # ⤷ 出口 return：返回 endpoint_context
        return endpoint_context
    is_bedrock_context = provider == "bedrock" or (
        base_url and base_url_hostname(base_url).startswith("bedrock-runtime.") and base_url_host_matches(base_url, "amazonaws.com")
    )
    # 1. Persistent cache (LM Studio / Codex OAuth excluded — see _skip_persistent_context_cache).
    cached = get_cached_context_length(model, base_url) if base_url and not _skip_persistent_context_cache(base_url, provider) else None
    validated = _validate_cached_context_length(model, base_url, cached, is_bedrock_context, api_key=api_key) if cached is not None else None
    if validated is not None:
        return validated
    # 1b. AWS Bedrock. Must run BEFORE the custom-endpoint step: bedrock-runtime.* is not in
    # _URL_TO_PROVIDER and would fail the /models probe into the default.
    ctx = _resolve_bedrock_context_length(model, base_url) if is_bedrock_context else None
    if ctx is not None:
        return ctx
    if provider == "novita" or (base_url and base_url_host_matches(base_url, "api.novita.ai")):
        ctx = _resolve_endpoint_context_length(model, base_url or "https://api.novita.ai/openai/v1", api_key=api_key)
        if ctx is not None:
            if base_url:
                save_context_length(model, base_url, ctx)
            return ctx
    # 2. Live /models for truly custom endpoints. Known providers skip this: their /models may
    # report a provider-imposed limit (Copilot: 128k) rather than the window.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        return _resolve_custom_endpoint_context_length(model, base_url, api_key, provider)
    # 4. Anthropic /v1/models API (only for regular API keys, not OAuth)
    if provider == "anthropic" or (base_url and base_url_hostname(base_url) == "api.anthropic.com"):
        ctx = _query_anthropic_context_length(model, base_url or "https://api.anthropic.com", api_key)
        if ctx:
            return ctx
    # 5. Provider-aware lookups — before the generic OR cache, since the same model has
    # different limits per provider. Generic providers are inferred from the URL.
    effective_provider = provider
    if base_url and (not effective_provider or effective_provider in {"openrouter", "custom"}):
        effective_provider = _infer_provider_from_url(base_url) or effective_provider
    ctx = _resolve_provider_aware_context_length(model, base_url, api_key, provider, effective_provider)
    if ctx is not None:
        return ctx
    # 6. OpenRouter metadata, provider-unaware fallback — only when the provider is unknown (OR
    # data is community-maintained); 32K underreport guard.
    if not effective_provider:
        metadata = fetch_model_metadata()
        if model in metadata:
            or_ctx = metadata[model].get("context_length", DEFAULT_FALLBACK_CONTEXT)
            if or_ctx == 32768 and _model_name_suggests_stale_32k_underreport(model):
                logger.info("Rejecting OpenRouter metadata context=%s for %r (known 32K underreport); falling through to hardcoded defaults", or_ctx, model)
            else:
                return or_ctx
    # 7. Local server before hardcoded defaults — ``Hermes-3-Llama-3.1-70B`` matches ``llama``
    # (131072) even when vLLM runs at a lower ``--max-model-len``.
    local_ctx = _probe_local_context_length(model, base_url, api_key, provider) if base_url and is_local_endpoint(base_url) else None
    if local_ctx:
        return local_ctx
    # 8. Hardcoded defaults: `key in model` only — the reverse would let "claude-sonnet-4" match "claude-sonnet-4-6" and return 1M.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        return hit[1]
    # 9. Default fallback — warn (deduped) so small-context models don't silently get 256K.
    _warn_context_length_fallback(model, base_url)
    # ⤷ 出口 return：返回 DEFAULT_FALLBACK_CONTEXT
    return DEFAULT_FALLBACK_CONTEXT


# ┌── 异步函数（协程） def get_model_context_length_async()
# │  作用：get_model_context_length 在…上工作线程 (其 blocking HTTP 会 stall 事件循环).
# │  参数：model: str, base_url: str = '', api_key: str = '', config_context_length: int | None = None, provider: str = '', custom_providers: list | None = None
# │  返回：int
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def get_model_context_length_async(model: str, base_url: str = "", api_key: str = "", config_context_length: int | None = None, provider: str = "", custom_providers: list | None = None) -> int:
    """get_model_context_length on a worker thread (its blocking HTTP would stall the event loop)."""
    import asyncio
    # ⤷ 出口 return：返回 await asyncio.to_thread(get_model_context_length, model, bas…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await asyncio.to_thread(
        get_model_context_length, model, base_url=base_url, api_key=api_key,
        config_context_length=config_context_length, provider=provider, custom_providers=custom_providers)


# CJK/Hangul/Kana codepoints (~1 token each), counted in one C-level regex pass: Hangul
# Jamo (+Ext-A), CJK radicals/ideographs (+compat), Hangul syllables, fullwidth/halfwidth.
# Rough chars-per-token ratio for ASCII text; the single source for every "N tokens ≈ N*4 chars"
# budget conversion (context files, tool-output budgets, whisper prompt cap, compressor metadata).
CHARS_PER_TOKEN = 4

_CJK_DENSE_RE = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


# ┌── 函数 def _is_cjk_token_dense_char()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：ch: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_cjk_token_dense_char(ch: str) -> bool:
    # ⤷ 出口 return：返回 _CJK_DENSE_RE.fullmatch(ch) is not None
    return _CJK_DENSE_RE.fullmatch(ch) is not None


# ┌── 函数 def estimate_tokens_rough()
# │  作用：粗略的令牌估算: CJK/Hangul/Kana codepoints ~1 令牌每个; everything 否则 ceil(UTF-8 bytes/4).
# │  参数：text: str
# │  返回：int
# └──────────────────────────────────────────────────────────────
def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate: CJK/Hangul/Kana codepoints ~1 token each; everything else ceil(UTF-8 bytes/4).
    Ceiling keeps short texts from estimating 0. Runs on every preflight walk, so all-ASCII stays O(1).

    Byte-counting (not chars) is the corrective for non-CJK, non-ASCII text: Cyrillic/Greek/Arabic are 2
    bytes/char so count ~chars/2, matching real BPE cost (~2-3 chars/token) where chars/4 under-counted
    ~2x and let sessions ride the provider ceiling below the compaction threshold. Calibrated vs
    cl100k/o200k/Qwen2.5 (estimate/real): Russian 0.67->1.24, Arabic 0.53->0.96, Hindi 0.34->0.90,
    Greek 0.37->0.68; accented Latin barely moves (French 1.02->1.03). errors="replace": lone surrogates
    (routine in tool output; see message_sanitization) must not turn an estimate into a raise."""
    if not text:
        # ⤷ 出口 return：返回 0
        return 0
    text = str(text)
    if text.isascii():  # flag check on CPython; ASCII cannot contain token-dense CJK
        # ⤷ 出口 return：返回 (len(text) + 3) // CHARS_PER_TOKEN
        return (len(text) + 3) // CHARS_PER_TOKEN
    stripped = _CJK_DENSE_RE.sub("", text)
    dense = len(text) - len(stripped)
    # ⤷ 出口 return：返回 dense + (len(stripped.encode('utf-8', 'replace')) + 3) // CH…
    return dense + ((len(stripped.encode("utf-8", "replace")) + 3) // CHARS_PER_TOKEN)


# ┌── 函数 def estimate_messages_tokens_rough()
# │  作用：粗略的令牌估算用于消息列表 (pre-flight 仅). Images cost per-image price
# │  参数：messages: List[Dict[str, Any]], charge_stale_thinking: bool = ...
# │  返回：int
# └──────────────────────────────────────────────────────────────
def estimate_messages_tokens_rough(messages: List[Dict[str, Any]], *, charge_stale_thinking: bool = True) -> int:
    """Rough token estimate for a message list (pre-flight only). Images cost the per-image price
    learned from provider usage (``agent.image_token_cost``; flat default before calibration)
    rather than their base64 length. ``charge_stale_thinking=False`` mirrors the tail-budget
    walk (``context_compressor._estimate_msg_budget_tokens``): on non-echo routes stale reasoning
    rides the wire only for the NEWEST assistant turn, so excluding it keeps the compaction TRIGGER
    in the same size class as the walk — otherwise reasoning-heavy sessions fire preflight forever."""
    from agent.image_token_cost import current_image_token_cost

    image_cost = current_image_token_cost()
    if not charge_stale_thinking:
        messages = _strip_stale_thinking_for_estimate(messages)
    # ⤷ 出口 return：返回 sum((_estimate_message_tokens_cached(msg, image_cost) for ms…
    return sum(_estimate_message_tokens_cached(msg, image_cost) for msg in messages)


# Thinking-text keys replayed for at most the newest assistant turn on non-echo routes — must stay
# in lockstep with ``context_compressor._NEWEST_TURN_ONLY_BUDGET_KEYS``.
_STALE_THINKING_ESTIMATE_KEYS = ("reasoning", "reasoning_content")


# ┌── 函数 def _strip_stale_thinking_for_estimate()
# │  作用：复制的 ``消息`` 带卡死的 thinking 键已移除 (newest 保留). Shallow stripped 复制
# │  参数：messages: List[Dict[str, Any]]
# │  返回：List[Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def _strip_stale_thinking_for_estimate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of ``messages`` with stale thinking keys removed (newest kept). Shallow stripped copies
    share the original value objects, so the per-message memo still hits for the stripped shape."""
    def _is_assistant(m: Any) -> bool:
        # ⤷ 出口 return：返回 isinstance(m, dict) and m.get('role') == 'assistant'
        return isinstance(m, dict) and m.get("role") == "assistant"
    newest = next((i for i in range(len(messages) - 1, -1, -1) if _is_assistant(messages[i])), -1)
    # ⤷ 出口 return：返回 [{k: v for k, v in m.items() if k not in _STALE_THINKING_EST…
    return [
        {k: v for k, v in m.items() if k not in _STALE_THINKING_ESTIMATE_KEYS}
        if i != newest and _is_assistant(m) and any(m.get(k) for k in _STALE_THINKING_ESTIMATE_KEYS) else m
        for i, m in enumerate(messages)
    ]


# Per-message token-estimate memo keyed by an exact value fingerprint: strings by ``id()`` AND
# pinned (strong ref in the entry, so the id can't be reused and immutability makes id-equality
# value-equality); numbers/bools/None by value; dicts/lists structurally in key order (``str(shadow)``
# depends on it); any other type aborts the memo. api_messages shallow-copies dicts but shares the strings.
# ``estimate_messages_tokens_rough`` is called on the full history every loop iteration (conversation_loop
# preflight), repeatedly during compaction telemetry, and inside an O(n^2) shrink loop in moa_loop. The
# per-message helpers are pure functions of the message's value, so a memo keyed on a fingerprint that
# uniquely determines the value is exactly equivalent. Fingerprint design (soundness argument): While the
# entry lives, that id cannot be reused by another object, so id-equality implies object-equality — strings
# are immutable, so value-equality too (no #50372-style aliasing). Equal fingerprints therefore imply
# deep-equal messages built from identical immutable leaves ⇒ identical ``str(shadow)`` bytes ⇒ identical
# estimate. Because the api_messages build shallow-copies history dicts each iteration, the copies share the
# same content strings — so unchanged history messages hit the memo even though the outer dicts are fresh
# objects every turn.
_MSG_TOKENS_CACHE: Dict[Any, Tuple[list, int, int]] = {}  # pins, text tokens, image count
_MSG_TOKENS_CACHE_MAX = 4096


# ┌── 函数 def _msg_fingerprint()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：value: Any, pins: list
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def _msg_fingerprint(value: Any, pins: list) -> Any:
    if value is None or value is True or value is False:
        # ⤷ 出口 return：返回 value
        return value
    t = type(value)
    if t is str:
        pins.append(value)
        # ⤷ 出口 return：返回 ('s', id(value))
        return ("s", id(value))
    if t is int or t is float:
        # ⤷ 出口 return：返回 ('n', t.__name__, value)
        return ("n", t.__name__, value)
    if t is dict:
        # ⤷ 出口 return：返回 ('d', tuple(((_msg_fingerprint(k, pins), _msg_fingerprint(v,…
        return ("d", tuple((_msg_fingerprint(k, pins), _msg_fingerprint(v, pins)) for k, v in value.items()))
    if t is list or t is tuple:
        # ⤷ 出口 return：返回 ('l' if t is list else 't', tuple((_msg_fingerprint(v, pins)…
        return ("l" if t is list else "t", tuple(_msg_fingerprint(v, pins) for v in value))
    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
    raise ValueError("unfingerprintable message value")


# ┌── 函数 def _estimate_message_tokens_cached()
# │  作用：文本令牌 + images x ``image_cost``; memo holds 文本且 image COUNT 从而 recalibrated
# │  参数：msg: Any, image_cost: int
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _estimate_message_tokens_cached(msg: Any, image_cost: int) -> int:
    """Text tokens + images x ``image_cost``; the memo holds text and image COUNT so a recalibrated
    per-image price re-prices cached rows without invalidating them."""
    def _compute() -> Tuple[int, int]:
        # ⤷ 出口 return：返回 (_estimate_message_tokens_without_images(msg), _count_image_…
        return _estimate_message_tokens_without_images(msg), _count_image_tokens(msg, 1)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        pins: list = []
        key = _msg_fingerprint(msg, pins)
        hash(key)
    except Exception:
        text, images = _compute()
        # ⤷ 出口 return：返回 text + images * image_cost
        return text + images * image_cost
    cached = _MSG_TOKENS_CACHE.get(key)
    if cached is not None:
        # ⤷ 出口 return：返回 cached[1] + cached[2] * image_cost
        return cached[1] + cached[2] * image_cost
    text, images = _compute()
    tokens = text + images * image_cost
    _MSG_TOKENS_CACHE[key] = (pins, text, images)
    while len(_MSG_TOKENS_CACHE) > _MSG_TOKENS_CACHE_MAX:
        # ⤷ 关键分支 try/except：捕获 (StopIteration, KeyError, RuntimeError)，出错走兜底而不是崩溃。
        try:
            _MSG_TOKENS_CACHE.pop(next(iter(_MSG_TOKENS_CACHE)))
        except (StopIteration, KeyError, RuntimeError):
            break
    # ⤷ 出口 return：返回 tokens
    return tokens


# ┌── 函数 def _count_parts()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：parts: Any, types: set
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _count_parts(parts: Any, types: set) -> int:
    # ⤷ 出口 return：返回 sum((1 for part in parts if isinstance(part, dict) and part.…
    return sum(1 for part in parts if isinstance(part, dict) and part.get("type") in types) if isinstance(parts, list) else 0


# ┌── 函数 def _count_image_tokens()
# │  作用：Count image-like 内容 parts 在消息; 返回其 令牌 cost.
# │  参数：msg: Dict[str, Any], cost_per_image: int
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    if not isinstance(msg, dict):
        # ⤷ 出口 return：返回 0
        return 0
    content = msg.get("content")
    count = _count_parts(content, {"image", "image_url", "input_image"})
    count += _count_parts(msg.get("_anthropic_content_blocks"), {"image"})
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        count += _count_parts(content.get("content"), {"image", "image_url"})
    # ⤷ 出口 return：返回 count * cost_per_image
    return count * cost_per_image


# ┌── 函数 def strip_opaque_replay_items()
# │  作用：``codex_reasoning_items`` 带 ``encrypted_content`` blanked 用于 local 令牌 estimation.
# │  参数：items: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def strip_opaque_replay_items(items: Any) -> Any:
    """``codex_reasoning_items`` with ``encrypted_content`` blanked for local token estimation.
    The ciphertext is priced by the provider's own count, never by its bytes (a compaction
    checkpoint alone can be 5M chars, #100611); only real usage prices it."""
    if not isinstance(items, list):
        # ⤷ 出口 return：返回 items
        return items
    # ⤷ 出口 return：返回 [{k: '' if k == 'encrypted_content' else v for k, v in item.…
    return [
        {k: ("" if k == "encrypted_content" else v) for k, v in item.items()} if isinstance(item, dict) else item
        for item in items
    ]


# ┌── 函数 def _wire_message_shadow()
# │  作用：Shadow 的消息 holding 仅什么服务商 actually receives.
# │  参数：msg: Dict[str, Any]
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _wire_message_shadow(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow of a message holding only what the provider actually receives.
    * ``api_content`` SUBSTITUTES ``content`` (mirrors ``turn_context.substitute_api_content`` exactly):
      only a non-empty STRING sidecar on a user/assistant row displaces content; substituting any
      other shape would UNDERcount — the dangerous direction.
    * Base64 images become a placeholder; ``_count_image_tokens`` charges them flat.
    * ``reasoning`` never ships as-is (request builds pop it after optionally promoting it into
      ``reasoning_content``); counting both inflated estimates up to +53%.
    * Opaque provider blobs (``encrypted_content`` on codex reasoning / compaction items) are
      ciphertext the provider prices by its OWN token count, never by bytes; a native compaction
      checkpoint alone can be 5M chars (#100611). They contribute 0 here: only real usage ever
      prices them, and the usage anchor carries that price forward."""
    sidecar = msg.get("api_content")
    sidecar_wins = isinstance(sidecar, str) and bool(sidecar) and msg.get("role") in ("user", "assistant")
    _rc = msg.get("reasoning_content")
    drop_reasoning_dup = isinstance(_rc, str) and bool(_rc.strip())
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k in ("_anthropic_content_blocks", "reasoning_details") or k in PERSISTENCE_ONLY_MESSAGE_FIELDS or (k == "reasoning" and drop_reasoning_dup):
            continue
        if k == "api_content":
            if sidecar_wins:
                shadow["content"] = v
        elif k == "content" and sidecar_wins:
            continue
        elif k == "content" and isinstance(v, list):
            shadow[k] = [
                {"type": part.get("type"), "image": "[stripped]"}
                if isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"} else part
                for part in v
            ]
        elif k == "content" and isinstance(v, dict) and v.get("_multimodal"):
            shadow[k] = v.get("text_summary", "")
        elif k == "codex_reasoning_items":
            shadow[k] = strip_opaque_replay_items(v)
        elif k == "encrypted_content":  # a Responses reasoning/compaction item passed as a row
            shadow[k] = ""
        else:
            shadow[k] = v
    # ⤷ 出口 return：返回 shadow
    return shadow


# ┌── 函数 def _estimate_message_tokens_without_images()
# │  作用：令牌估算用于消息 shadow 带 image payloads stripped.
# │  参数：msg: Dict[str, Any]
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _estimate_message_tokens_without_images(msg: Dict[str, Any]) -> int:
    """Token estimate for a message shadow with image payloads stripped."""
    # ⤷ 出口 return：返回 estimate_tokens_rough(str(_wire_message_shadow(msg) if isins…
    return estimate_tokens_rough(str(_wire_message_shadow(msg) if isinstance(msg, dict) else msg))


# ┌── 函数 def estimate_request_tokens_rough()
# │  作用：粗略的令牌估算用于 full 请求: system 提示词 + 消息 + 工具结构定义 (50+ 工具
# │  参数：messages: List[Dict[str, Any]], system_prompt: str = ..., tools: Optional[List[Dict[str, Any]]] = ..., charge_stale_thinking: bool = ...
# │  返回：int
# └──────────────────────────────────────────────────────────────
def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]], *, system_prompt: str = "", tools: Optional[List[Dict[str, Any]]] = None, charge_stale_thinking: bool = True,
) -> int:
    """Rough token estimate for a full request: system prompt + messages + tool schemas (50+ tools
    add 20-30K on their own). ``charge_stale_thinking`` is forwarded — pass False when the route
    provably strips stale thinking (``message_sanitization.stale_thinking_reaches_wire``)."""
    total = estimate_tokens_rough(system_prompt) if system_prompt else 0
    if messages:
        # Positional call: test seams and plugin engines monkeypatch estimate_messages_tokens_rough with (messages)-only signatures.
        total += estimate_messages_tokens_rough(messages) if charge_stale_thinking else estimate_messages_tokens_rough(messages, charge_stale_thinking=False)
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    # ⤷ 出口 return：返回 total
    return total


# Keyed by ``id(tools)``; bounded, oldest-first eviction. Repeated ``str(tools)`` on
# large schemas stalls GUI event loops under GIL pressure.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


# ┌── 函数 def _tool_name_for_cache()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：tool: Any
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        # ⤷ 出口 return：返回 ''
        return ""
    fn = tool.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    name = name if isinstance(name, str) else tool.get("name")
    # ⤷ 出口 return：返回 name if isinstance(name, str) else ''
    return name if isinstance(name, str) else ""


# ┌── 函数 def _estimate_tools_tokens_rough()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：tools: List[Dict[str, Any]]
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        # ⤷ 出口 return：返回 0
        return 0
    key = id(tools)
    signature = (len(tools), _tool_name_for_cache(tools[0]), _tool_name_for_cache(tools[-1]))
    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None and cached[:3] == signature:
        # ⤷ 出口 return：返回 cached[3]
        return cached[3]
    # Sum the major schema fields (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        src = fn if isinstance(fn, dict) else tool
        params = src.get("parameters") or {}
        total_chars += sum(len(v) for v in (src.get("name") or "", src.get("description") or "") if isinstance(v, str))
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:  # JSON is closer to wire size than repr()
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))
    tokens = (total_chars + 3) // 4
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (*signature, tokens)
    # ⤷ 出口 return：返回 tokens
    return tokens
