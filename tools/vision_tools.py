#!/usr/bin/env python3
"""Vision tools: ``vision_analyze`` (image) and ``video_analyze``.

Images resolve through :mod:`tools.image_source`, are normalized to a provider-supported format
(:mod:`tools.vision_tools_image_prep`), then either attach natively to a vision-capable main
model (multimodal tool-result envelope) or are described by the auxiliary vision LLM router.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：tools/vision_tools.py（约 1068 行）
# 地位：视觉能力——看图（vision_analyze）和看视频（video_analyze）。
#
# 两种处理路径：
#   1. 原生路径：如果主模型本身支持视觉，就把图片直接作为多模态内容附在工具结果里
#      （_build_native_vision_tool_result / _should_use_native_vision_fast_path）
#   2. 辅助路径：主模型不支持，就先让一个便宜的视觉模型把图「描述」成文字，再给主模型
#      （_call_vision_llm / _run_analysis）
#
# 图片预处理（很实用的一块）：
#   _download_image / _download_media        下载（带重试和 SSRF 重定向守卫）
#   _resize_image_for_vision                 压缩尺寸（大图费 token 还可能被拒）
#   _image_to_base64_data_url                转成 base64 data URL
#   _is_image_size_error / _too_large_message 处理「图片太大」的错误并给用户提示
#
# 注意：大量函数是 async def（异步），因为下载和编码是 IO 密集操作。
#   _run_encode_on_cpu_executor 会把 CPU 密集的编码丢到线程池，避免阻塞事件循环。
# ==========================================================================
#

import base64
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, NamedTuple, Optional
from urllib.parse import urlparse
import httpx

# ``agent.auxiliary_client`` costs ~50 ms cold; only the handlers need it. Both names stay
# module attributes so tests can patch ``tools.vision_tools.async_call_llm`` (truthy-skip: mocks win).
async_call_llm: Any = None
extract_content_or_reasoning: Any = None


# ┌── 函数 def _load_auxiliary_client()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _load_auxiliary_client() -> None:
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global async_call_llm, extract_content_or_reasoning
    if async_call_llm is None or extract_content_or_reasoning is None:
        from agent import auxiliary_client as _aux
        async_call_llm = async_call_llm or _aux.async_call_llm
        extract_content_or_reasoning = extract_content_or_reasoning or _aux.extract_content_or_reasoning


from hermes_constants import get_hermes_dir
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access
from tools.vision_tools_image_prep import (
    _VISION_MAX_VALIDATED_AGGREGATE_PIXELS,
    _VISION_MAX_VALIDATED_FRAME_COUNT,
    _crop_image_region,
    _determine_mime_type,
    _image_exceeds_dimension,
    _normalize_to_supported_image,
    _validate_raster_image_decodable)

logger = logging.getLogger(__name__)

_debug = DebugSession("vision_tools", env_var="VISION_TOOLS_DEBUG")


# ┌── 函数 def _cfg_auxiliary()
# │  作用：``辅助的.<键...>`` 从配置.YAML; ``默认值`` 当配置是 unavailable.
# │  参数：*keys, default = ...
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _cfg_auxiliary(*keys: str, default=None):
    """``auxiliary.<keys...>`` from config.yaml; ``default`` when config is unavailable."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.config import cfg_get, load_config
        # ⤷ 出口 return：返回 cfg_get(load_config(), 'auxiliary', *keys, default=default)
        return cfg_get(load_config(), "auxiliary", *keys, default=default)
    except Exception:
        # ⤷ 出口 return：返回 default
        return default


# ┌── 函数 def _read_vision_setting()
# │  作用：环境变量 → ``辅助的.vision.<键>`` → None. 值那个 fail ``cast`` 或 fall below
# │  参数：env_var: str, key: str, cast, minimum = None
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _read_vision_setting(env_var: str, key: str, cast, minimum=None):
    """Env var → ``auxiliary.vision.<key>`` → None. Values that fail ``cast`` or fall below
    ``minimum`` are skipped in favor of the next source (a cap can never be disabled by a bad value)."""
    def _accept(raw):
        # ⤷ 关键分支 try/except：捕获 (TypeError, ValueError)，出错走兜底而不是崩溃。
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 val if minimum is None or val >= minimum else None
        return val if minimum is None or val >= minimum else None
    val = _accept(os.getenv(env_var, "").strip() or None)
    # ⤷ 出口 return：返回 val if val is not None else _accept(_cfg_auxiliary('vision',…
    return val if val is not None else _accept(_cfg_auxiliary("vision", key))


# HTTP download timeout (separate from ``auxiliary.vision.timeout``, which governs the LLM call).
_VISION_DOWNLOAD_TIMEOUT = _read_vision_setting("HERMES_VISION_DOWNLOAD_TIMEOUT", "download_timeout", float)
if _VISION_DOWNLOAD_TIMEOUT is None:
    _VISION_DOWNLOAD_TIMEOUT = 30.0

# Hard cap on downloaded media (50 MB): bounds memory/disk against attacker-hosted files.
_VISION_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


# CPU-burst cap: a turn can fan out dozens of vision_analyze calls (base64 encode + Pillow
# resize each) that would saturate every core and starve the shared event loop. Only the CPU
# burst is capped (LLM calls stay concurrent), on a dedicated executor sized to usable cores.
# Must be a threading primitive — each call runs via model_tools._run_async on a PER-THREAD
# loop, so an asyncio semaphore cannot coordinate; the default executor is shared with the
# gateway/web server and is deliberately NOT used.
# ┌── 函数 def _detect_host_cpus()
# │  作用：Usable CPU count (``sched_getaffinity`` honors cpuset pinning), 在 least 1.
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _detect_host_cpus() -> int:
    """Usable CPU count (``sched_getaffinity`` honors cpuset pinning), at least 1."""
    # ⤷ 关键分支 try/except：捕获 (AttributeError, OSError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 max(1, len(os.sched_getaffinity(0)))
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        # ⤷ 出口 return：返回 max(1, os.cpu_count() or 1)
        return max(1, os.cpu_count() or 1)


# ┌── 函数 def _resolve_vision_cpu_workers()
# │  作用：HERMES_VISION_MAX_CONCURRENCY → ``辅助的.vision.max_concurrency`` → 宿主 cores (< 1 ignored).
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _resolve_vision_cpu_workers() -> int:
    """HERMES_VISION_MAX_CONCURRENCY → ``auxiliary.vision.max_concurrency`` → host cores (< 1 ignored)."""
    val = _read_vision_setting("HERMES_VISION_MAX_CONCURRENCY", "max_concurrency", int, minimum=1)
    # ⤷ 出口 return：返回 val or _detect_host_cpus()
    return val or _detect_host_cpus()


_VISION_CPU_WORKERS = _resolve_vision_cpu_workers()

_vision_cpu_executor = ThreadPoolExecutor(
    max_workers=_VISION_CPU_WORKERS, thread_name_prefix="vision-encode",
)


# ┌── 异步函数（协程） def _run_encode_on_cpu_executor()
# │  作用：运行同步 encode/resize callable 在…上 bounded vision CPU executor (never LLM call).
# │  参数：fn, *args, **kwargs
# │  返回：（未标注类型）
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _run_encode_on_cpu_executor(fn, *args, **kwargs):
    """Run a sync encode/resize callable on the bounded vision CPU executor (never the LLM call)."""
    import functools
    loop = asyncio.get_running_loop()
    # ⤷ 出口 return：返回 await loop.run_in_executor(_vision_cpu_executor, functools.p…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await loop.run_in_executor(_vision_cpu_executor, functools.partial(fn, *args, **kwargs))


# ┌── 异步函数（协程） def _validate_image_url_async()
# │  作用：HTTP(S) 形态检查 (scheme, netloc; extension-less CDN URLs pass) + SSRF 护栏带 DNS
# │  参数：url: str
# │  返回：bool
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _validate_image_url_async(url: str) -> bool:
    """HTTP(S) shape check (scheme, netloc; extension-less CDN URLs pass) + SSRF guard with DNS
    off the event loop."""
    if not (isinstance(url, str) and url.startswith(("http://", "https://")) and urlparse(url).netloc):
        # ⤷ 出口 return：返回 False
        return False
    from tools.url_safety import async_is_safe_url
    # ⤷ 出口 return：返回 await async_is_safe_url(url)
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await async_is_safe_url(url)


# ┌── 函数 def _is_retryable_download_error()
# │  作用：Transient failures 仅: 429, 5xx, transport 错误且 anything unclassified. Fail-fast 在…上
# │  参数：error: Exception
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_retryable_download_error(error: Exception) -> bool:
    """Transient failures only: 429, 5xx, transport errors and anything unclassified. Fail-fast on
    other 4xx, PermissionError (policy/SSRF block) and ValueError (too large / blocked redirect)."""
    if isinstance(error, (PermissionError, ValueError)):
        # ⤷ 出口 return：返回 False
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        # ⤷ 出口 return：返回 status == 429 or not 400 <= status < 500
        return status == 429 or not 400 <= status < 500
    # ⤷ 出口 return：返回 True
    return True


_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ┌── 异步函数（协程） def _ssrf_redirect_guard()
# │  作用：Re-校验每个 redirect target ( public URL 302ing 169.254.169.254 会 otherwise
# │  参数：response
# │  返回：（未标注类型）
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _ssrf_redirect_guard(response):
    """Re-validate each redirect target (a public URL 302ing to 169.254.169.254 would otherwise
    bypass the pre-flight check). Async because httpx.AsyncClient awaits hooks."""
    from tools.url_safety import async_is_safe_url, redirect_target_from_response
    redirect_url = redirect_target_from_response(response)
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    if redirect_url and not await async_is_safe_url(redirect_url):
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise ValueError(f"Blocked redirect to private/internal address: {redirect_url}")


# ┌── 异步函数（协程） def _download_media()
# │  作用：SSRF-safe streaming download 带 exponential backoff (2s/4s/8s). ``retry_all=False`` (images)
# │  参数：url: str, destination: Path, max_retries: int, media_label: str, accept: str, max_bytes: int, timeout: float, retry_all: bool
# │  返回：Path
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _download_media(
    url: str, destination: Path, max_retries: int, *,
    media_label: str, accept: str, max_bytes: int, timeout: float, retry_all: bool) -> Path:
    """SSRF-safe streaming download with exponential backoff (2s/4s/8s). ``retry_all=False`` (images)
    retries only :func:`_is_retryable_download_error` errors — a 404/403 never succeeds on retry."""
    from utils import atomic_replace
    from tools.url_safety import create_ssrf_safe_async_client
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(max_retries):
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            blocked = check_website_access(url)
            if blocked:
                # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                raise PermissionError(blocked["message"])

            # follow_redirects for CDNs; the client validates DNS at connect
            # time and the hook re-validates each redirect target.
            async with create_ssrf_safe_async_client(
                timeout=timeout, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client, client.stream(
                "GET", url, headers={"User-Agent": _DOWNLOAD_USER_AGENT, "Accept": accept},
            ) as response:
                response.raise_for_status()

                # Content-Length gives an early reject but servers can omit or lie,
                # so the streaming cap below is authoritative.
                cl = response.headers.get("content-length")
                # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
                try:
                    declared_size = int(cl) if cl else None
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise ValueError(f"{media_label} too large ({declared_size} bytes, max {max_bytes})")
                blocked = check_website_access(str(response.url))
                if blocked:
                    raise PermissionError(blocked["message"])

                # Stream to a temp file, atomically moved onto destination on success.
                tmp_destination = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                bytes_written = 0
                try:
                    with tmp_destination.open("wb") as f:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            bytes_written += len(chunk)
                            if bytes_written > max_bytes:
                                raise ValueError(
                                    f"{media_label} too large ({bytes_written} bytes, max {max_bytes})"
                                )
                            f.write(chunk)
                    atomic_replace(tmp_destination, destination)
                except Exception:
                    try:
                        tmp_destination.unlink(missing_ok=True)
                    except OSError:
                        logger.debug("Could not delete partial download: %s", tmp_destination, exc_info=True)
                    raise
            # ⤷ 出口 return：返回 destination
            return destination
        except Exception as e:
            last_error = e
            if attempt >= max_retries - 1 or (not retry_all and not _is_retryable_download_error(e)):
                logger.error("%s download failed after %s attempt(s): %s",
                             media_label, attempt + 1, str(e)[:100], exc_info=True)
                if not retry_all:
                    raise
                break
            wait_time = 2 ** (attempt + 1)
            logger.warning("%s download failed (attempt %s/%s): %s",
                           media_label, attempt + 1, max_retries, str(e)[:50])
            if not retry_all:
                logger.warning("Retrying in %ss...", wait_time)
            await asyncio.sleep(wait_time)
    # Reaching here means max_retries was non-positive (or video exhausted retries).
    if last_error is not None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise last_error
    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
    raise RuntimeError(
        f"_download_{media_label.lower()} exited retry loop without attempting (max_retries={max_retries})")


# ┌── 异步函数（协程） def _download_image()
# │  作用：Download image 带 SSRF protection 且错误-类-aware 重试.
# │  参数：image_url: str, destination: Path, max_retries: int = 3
# │  返回：Path
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _download_image(image_url: str, destination: Path, max_retries: int = 3) -> Path:
    """Download an image with SSRF protection and error-class-aware retry."""
    # ⤷ 出口 return：返回 await _download_media(image_url, destination, max_retries, m…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _download_media(
        image_url, destination, max_retries,
        media_label="Image", accept="image/*,*/*;q=0.8",
        max_bytes=_VISION_MAX_DOWNLOAD_BYTES, timeout=_VISION_DOWNLOAD_TIMEOUT, retry_all=False)


# ┌── 函数 def _image_to_base64_data_url()
# │  作用：``data:<mime>;base64,...`` 用于文件 (MIME 从 extension 当不 given).
# │  参数：image_path: Path, mime_type: Optional[str] = None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _image_to_base64_data_url(image_path: Path, mime_type: Optional[str] = None) -> str:
    """``data:<mime>;base64,...`` for a file (MIME from extension when not given)."""
    mime = mime_type or _determine_mime_type(image_path)
    # ⤷ 出口 return：返回 f"data:{mime};base64,{base64.b64encode(image_path.read_bytes…
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


# Absolute hard ceiling for vision payloads (20 MB): no major provider accepts more.
_MAX_BASE64_BYTES = 20 * 1024 * 1024

# Proactive embed caps for history reuse: the native path bakes the data URL into the tool
# result, re-sent every later turn (a 4 MB embed cost ~100-260K billed tokens). Anthropic
# downsamples to a 1568px long edge anyway, so pixels past that cost wire bytes for no fidelity.
# The 20 MB hard ceiling / Anthropic 5 MB reject-cap still apply as safety nets; those are one-shot viewing
# limits, not history-reuse sizes. A 4 MB / 7900px embed was observed at ~400K chars and ~100–260K billed
# tokens per image (#92699), so we size for model reading instead: 256 KB keeps a 1568px screenshot cheap
# enough to ride the session (PNGs that exceed it are downscaled further by the byte-budget ladder), well
# under every provider's per-image limit.
_EMBED_TARGET_BYTES = 256 * 1024
_EMBED_MAX_DIMENSION = 1568

# Target when auto-resizing after a provider size rejection (retry once).
_RESIZE_TARGET_BYTES = 5 * 1024 * 1024

_SIZE_ERROR_HINTS = (
    "too large", "payload", "413", "content_too_large",
    "request_too_large", "exceeds", "size limit",
)


# ┌── 函数 def _is_image_size_error()
# │  作用：检测若 API 错误是 related image 或 payload size.
# │  参数：error: Exception
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_image_size_error(error: Exception) -> bool:
    """Detect if an API error is related to image or payload size."""
    err_str = str(error).lower()
    # ⤷ 出口 return：返回 any((hint in err_str for hint in _SIZE_ERROR_HINTS + ('image…
    return any(hint in err_str for hint in _SIZE_ERROR_HINTS + ("image_url", "invalid_request"))


# ┌── 函数 def _build_scale_note()
# │  作用：Coordinate-mapping disclosure 用于 downscale 且/或 region crop; ``None`` 当 neither applied.
# │  参数：scale_info: Optional[dict], crop_offset: Optional[dict]
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _build_scale_note(scale_info: Optional[dict], crop_offset: Optional[dict]) -> Optional[str]:
    """Coordinate-mapping disclosure for downscale and/or region crop; ``None`` when neither applied."""
    parts = []
    if scale_info:
        ow, oh = scale_info["orig_width"], scale_info["orig_height"]
        nw, nh = scale_info["new_width"], scale_info["new_height"]
        fx, fy = (ow / nw if nw else 1.0), (oh / nh if nh else 1.0)
        axes = (f"any coordinates you report by {fx:.2f}" if f"{fx:.2f}" == f"{fy:.2f}"
                else f"any x coordinates you report by {fx:.2f} and any y coordinates by {fy:.2f}")
        parts.append(f"Image downscaled from {ow}x{oh} to {nw}x{nh} for vision; "
                     f"multiply {axes} to map back to the original image.")
    if crop_offset:
        parts.append(
            f"Analysis was performed on a cropped region of the original "
            f"image starting at offset ({crop_offset['x']}, "
            f"{crop_offset['y']}); coordinates are relative to that crop "
            f"origin — add the offset to map back to the full image.")
    # ⤷ 出口 return：返回 ' '.join(parts) if parts else None
    return " ".join(parts) if parts else None


# ┌── 函数 def _import_pillow_for_resize()
# │  作用：返回 ``PIL.Image`` 或 None (Pillow 是 lazy-installable soft dependency). ``提示词=False``:。
# │  参数：无
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _import_pillow_for_resize():
    """Return ``PIL.Image`` or None (Pillow is a lazy-installable soft dependency). ``prompt=False``:
    a blocking input() deadlocks the CLI where prompt_toolkit owns stdin; the install is already
    gated by security.allow_lazy_installs."""
    # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
    try:
        from PIL import Image
    except ImportError:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.lazy_deps import ensure as _ensure_dep
            # prompt=False: never raise a blocking input() prompt mid-session. Under the interactive CLI
            # prompt_toolkit owns stdin, so a bare input() deadlocks the terminal (#40490). The install is
            # already gated by security.allow_lazy_installs, so reaching here is opt-in.
            _ensure_dep("tool.vision", prompt=False)
            from PIL import Image
        except Exception:
            # ⤷ 出口 return：返回 None
            return None
    # ⤷ 出口 return：返回 Image
    return Image


# ┌── 函数 def _resize_image_for_vision()
# │  作用：Base64 data URL, progressively downscaled 带 Pillow 当 over 预算.
# │  参数：image_path: Path, mime_type: Optional[str] = None, max_base64_bytes: int = _RESIZE_TARGET_BYTES, max_dimension: Optional[int] = None, scale_out: Optional[dict] = None, force_jpeg: bool = False
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _resize_image_for_vision(image_path: Path, mime_type: Optional[str] = None,
                              max_base64_bytes: int = _RESIZE_TARGET_BYTES,
                              max_dimension: Optional[int] = None,
                              scale_out: Optional[dict] = None,
                              force_jpeg: bool = False) -> str:
    """Base64 data URL, progressively downscaled with Pillow while over budget.

    Halves dimensions (aspect-preserving, 64px floor) up to 4 times; JPEG also walks a
    quality ladder (85/70/50) per step. Without Pillow, or if it still doesn't fit, returns
    the best attempt (or raw bytes) and lets the caller apply the size check.
    ``max_dimension``: force a downscale above this long edge even when bytes fit
    (Anthropic's 8000px cap is independent of bytes). ``force_jpeg``: re-encode PNG as JPEG
    when resizing — halving PNG dimensions destroys text legibility on dense screenshots.

    Args: max_dimension: If set, images whose longest side exceeds this pixel count are forcibly downscaled
    even if they're under the byte budget. Anthropic enforces an 8000 px per-side cap independently of the 5
    MB byte cap. force_jpeg: Re-encode as JPEG even for PNG input when a resize is needed. History-reuse
    embeds (#92699) opt in so a text-heavy screenshot keeps its readable resolution and shrinks via JPEG
    quality instead. Images already under both caps are returned unchanged (still PNG).
    """
    file_size = image_path.stat().st_size
    estimated_b64 = (file_size * 4) // 3 + 100  # base64 ~4/3 + data URL header
    data_url = None
    if estimated_b64 <= max_base64_bytes and not (
        max_dimension is not None and _image_exceeds_dimension(image_path, max_dimension)
    ):
        data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        if len(data_url) <= max_base64_bytes:
            # ⤷ 出口 return：返回 data_url
            return data_url

    def _raw() -> str:
        # ⤷ 出口 return：返回 data_url or _image_to_base64_data_url(image_path, mime_type=…
        return data_url or _image_to_base64_data_url(image_path, mime_type=mime_type)
    Image = _import_pillow_for_resize()
    if Image is None:
        logger.info("Pillow not installed — cannot auto-resize oversized image")
        # ⤷ 出口 return：返回 _raw()
        return _raw()  # caller will raise the size error
    logger.info("Image file is %.1f MB (estimated base64 %.1f MB, limit %.1f MB, max_dimension=%s), auto-resizing...",
                file_size / (1024 * 1024), estimated_b64 / (1024 * 1024),
                max_base64_bytes / (1024 * 1024), max_dimension)
    # JPEG for photos (smaller), PNG for transparency — unless force_jpeg.
    is_png = (mime_type or _determine_mime_type(image_path)) == "image/png" and not force_jpeg
    pil_format, out_mime = ("PNG", "image/png") if is_png else ("JPEG", "image/jpeg")
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.info("Pillow cannot open image for resizing: %s", exc)
        return _raw()

    # JPEG cannot encode alpha/palette modes (force_jpeg routes PNGs here).
    if not is_png and img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")
    quality_steps = (None,) if is_png else (85, 70, 50)
    orig_dims = prev_dims = (img.width, img.height)
    candidate = None

    def _record_scale(w: int, h: int) -> None:
        if scale_out is not None and (w, h) != orig_dims:
            scale_out.update(orig_width=orig_dims[0], orig_height=orig_dims[1], new_width=w, new_height=h)
    for attempt in range(5):
        if attempt > 0:
            # Halve, then re-derive from whichever axis hit the 64px floor so both shrink equally.
            new_w, new_h = max(int(img.width * 0.5), 64), max(int(img.height * 0.5), 64)
            if new_w == 64 and img.width > 0:
                new_h = max(int(img.height * (64 / img.width)), 64)
            elif new_h == 64 and img.height > 0:
                new_w = max(int(img.width * (64 / img.height)), 64)
            if (new_w, new_h) == prev_dims:
                break
            img = img.resize((new_w, new_h), Image.LANCZOS)
            prev_dims = (new_w, new_h)
            logger.info("Resized to %dx%d (attempt %d)", new_w, new_h, attempt)
        dims_ok = max_dimension is None or max(img.width, img.height) <= max_dimension
        for q in quality_steps:
            buf = BytesIO()
            img.save(buf, format=pil_format, **({} if q is None else {"quality": q}))
            candidate = f"data:{out_mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
            if len(candidate) <= max_base64_bytes and dims_ok:
                logger.info("Auto-resized image fits: %.1f MB (quality=%s, %dx%d)",
                            len(candidate) / (1024 * 1024), q, img.width, img.height)
                _record_scale(img.width, img.height)
                return candidate
    if candidate is None:
        # ⤷ 出口 return：返回 _raw()
        return _raw()
    logger.warning("Auto-resize could not fit image under %.1f MB (best: %.1f MB)",
                   max_base64_bytes / (1024 * 1024), len(candidate) / (1024 * 1024))
    _record_scale(img.width, img.height)
    # ⤷ 出口 return：返回 candidate
    return candidate


# Native fast path: a vision-capable main model gets the image bytes as a multimodal
# tool-result envelope (the agent loop unwraps it into an OpenAI-style content list on the
# `tool` role); no aux LLM, no information loss. Providers whose tool results accept image
# content: Anthropic Messages (and aggregators proxying Claude — assume support), OpenAI
# Chat/Responses. Gemini is gated on model: only 3.x supports multimodal functionResponse.
_TOOL_RESULT_MEDIA_PROVIDERS = frozenset({
    "openrouter", "nous", "vertex", "bedrock", "anthropic-vertex", "google-vertex",
    "anthropic", "claude", "anthropic-direct",
    "openai", "openai-chat", "openai-codex", "azure-openai",
})
_GEMINI_PROVIDERS = frozenset({"google", "gemini", "google-gemini", "google-vertex-gemini"})


# ┌── 函数 def _profile_rejects_tool_media()
# │  作用：Hard veto: 服务商's ``ProviderProfile`` declares
# │  参数：provider: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _profile_rejects_tool_media(provider: str) -> bool:
    """Hard veto: the provider's ``ProviderProfile`` declares
    ``supports_vision_tool_messages=False`` — images are accepted in user
    messages but list-type tool-result content is rejected with 400
    (xiaomi/MiMo "text is not set"). ``supports_vision`` alone must not
    override this, or the multimodal tool-result envelope 400s every turn
    and the image never enters context (#89981).
    """
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(str(provider or "").strip().lower())
        # ⤷ 出口 return：返回 profile is not None and profile.supports_vision_tool_message…
        return profile is not None and profile.supports_vision_tool_messages is False
    except Exception:
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _supports_media_in_tool_results()
# │  作用：是否 服务商+模型接受 image 内容 inside 工具-结果消息. Unknown。
# │  参数：provider: str, model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _supports_media_in_tool_results(provider: str, model: str) -> bool:
    """Whether provider+model accepts image content inside a tool-result message. Unknown
    providers are False (caller falls back to aux-LLM text) unless their ``ProviderProfile``
    declares ``supports_vision``; ``supports_vision_tool_messages=False`` is a hard veto."""
    p = provider.strip().lower() if isinstance(provider, str) else ""
    if not p or _profile_rejects_tool_media(p):
        # ⤷ 出口 return：返回 False
        return False
    if p in _TOOL_RESULT_MEDIA_PROVIDERS:
        # ⤷ 出口 return：返回 True
        return True
    if p in _GEMINI_PROVIDERS:
        m = model.strip().lower() if isinstance(model, str) else ""
        # ⤷ 出口 return：返回 any((tag in m for tag in ('gemini-3', 'gemini-pro-3', 'gemin…
        return any(tag in m for tag in ("gemini-3", "gemini-pro-3", "gemini-flash-3"))
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(p)
        # ⤷ 出口 return：返回 profile is not None and bool(profile.supports_vision)
        return profile is not None and bool(profile.supports_vision)
    except Exception:
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _should_use_native_vision_fast_path()
# │  作用：当 image routing resolves ``native`` 且服务商接受 images 在工具 时返回 True。
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _should_use_native_vision_fast_path() -> bool:
    """True when image routing resolves to ``native`` AND the provider accepts images in tool
    results, or the user set the ``model.supports_vision`` override (escape hatch for
    custom/local providers). Any failure → False."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from agent.auxiliary_client import _read_main_provider, _read_main_model
        from agent.image_routing import decide_image_input_mode, _lookup_supports_vision
        from hermes_cli.config import load_config
        provider = _read_main_provider()
        model = _read_main_model()
        cfg = load_config()
        if decide_image_input_mode(provider, model, cfg) != "native":
            # ⤷ 出口 return：返回 False
            return False
        # The profile veto applies ahead of the capability lookup too: a
        # model marked vision-capable by models.dev / custom_providers must
        # not re-open the multimodal-envelope route the profile rejects.
        if _profile_rejects_tool_media(provider):
            # ⤷ 出口 return：返回 False
            return False
        # ⤷ 出口 return：返回 _supports_media_in_tool_results(provider, model) or _lookup_…
        return (
            _supports_media_in_tool_results(provider, model)
            or _lookup_supports_vision(provider, model, cfg) is True)
    except Exception as exc:
        logger.debug("Native vision fast-path check failed: %s", exc)
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _build_native_vision_tool_result()
# │  作用：Multimodal 工具-结果 envelope. 文本 part 是 intentionally minimal ( 模型已
# │  参数：image_url: str, question: str, image_data_url: str, image_size_bytes: int, scale_note: Optional[str] = None
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def _build_native_vision_tool_result(
    image_url: str, question: str, image_data_url: str, image_size_bytes: int,
    scale_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Multimodal tool-result envelope. The text part is intentionally minimal (the model already
    has the question); ``text_summary`` is the fallback for providers without multimodal tool results."""
    text_part = (
        "Image loaded into your context — you can see it natively now. "
        "Use your built-in vision to answer the user.")
    if isinstance(question, str) and question.strip():
        text_part += f"\n\nQuestion: {question.strip()}"
    if scale_note:
        text_part += f"\n\nNote: {scale_note}"
    # ⤷ 出口 return：返回 {'_multimodal': True, 'content': [{'type': 'text', 'text': t…
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": image_data_url}}],
        "text_summary": (
            f"Image attached natively for the main model ({image_size_bytes / 1024:.1f} KB). "
            "Answer using built-in vision."),
        "meta": {"image_url": image_url[:200], "size_bytes": image_size_bytes, "native_vision": True}}


# ┌── 函数 def _unlink_quietly()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：path: Optional[Path]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _unlink_quietly(path: Optional[Path]) -> None:
    if path is not None:
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# ┌── 类 class _ImagePrepError（继承自：ValueError）
# │  说明：抛出由 :func:`_prepare_image`; 消息是 面向用户的.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _ImagePrepError(ValueError):
    """Raised by :func:`_prepare_image`; the message is user-facing."""


# ┌── 类 class _PreparedImage（继承自：NamedTuple）
# │  说明：Temp image ready encode; ``路径`` 是持有的由 caller (delete 它).
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class _PreparedImage(NamedTuple):
    """Temp image ready to encode; ``path`` is owned by the caller (delete it)."""
    path: Path
    mime: Optional[str]
    size_bytes: int
    crop_offset: dict


# ┌── 异步函数（协程） def _prepare_image()
# │  作用：Resolve → materialize → 归一化 → (校验) → (crop). 抛出 ``_ImagePrepError``.
# │  参数：image_url: str, task_id: Optional[str], region: Optional[list], validate_decode: bool
# │  返回：_PreparedImage
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _prepare_image(
    image_url: str, task_id: Optional[str], region: Optional[list], *, validate_decode: bool,
) -> _PreparedImage:
    """Resolve → materialize → normalize → (validate) → (crop). Raises ``_ImagePrepError``.
    Unsupported formats (SVG, BMP) become PNG BEFORE encoding — an unsupported media_type baked
    into immutable history would 400 on every resume. The crop runs BEFORE any downscale so the
    region keeps the full resolution budget. On error no temp file is left."""
    from tools.image_source import ImageResolutionError, ResolveContext, resolve_image_source
    # ⤷ 关键分支 try/except：捕获 ImageResolutionError，出错走兜底而不是崩溃。
    try:
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        resolved = await resolve_image_source(image_url, ResolveContext(task_id=task_id))
    except ImageResolutionError as exc:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise _ImagePrepError(str(exc)) from exc
    temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"temp_image_{uuid.uuid4()}.img"
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    await asyncio.to_thread(path.write_bytes, resolved.data)
    mime, size_bytes, crop_offset = resolved.mime, len(resolved.data), {}
    # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
    try:
        normalized_path, mime, norm_err = await asyncio.to_thread(_normalize_to_supported_image, path, mime)
        if norm_err or normalized_path is None:
            raise _ImagePrepError(norm_err or "Image normalization failed.")
        if normalized_path != path:
            _unlink_quietly(path)
            path = normalized_path
            size_bytes = path.stat().st_size
        if validate_decode:
            decode_error = await _run_encode_on_cpu_executor(
                _validate_raster_image_decodable, path,
                _VISION_MAX_VALIDATED_FRAME_COUNT, _VISION_MAX_VALIDATED_AGGREGATE_PIXELS)
            if decode_error:
                raise _ImagePrepError(decode_error)
        if region is not None:
            cropped_path, cropped_mime, crop_err = await asyncio.to_thread(
                _crop_image_region, path, region, offset_out=crop_offset)
            if crop_err or cropped_path is None:
                raise _ImagePrepError(crop_err or "Region crop failed.")
            _unlink_quietly(path)
            path, mime, size_bytes = cropped_path, cropped_mime, cropped_path.stat().st_size
    except BaseException:
        _unlink_quietly(path)
        raise
    # ⤷ 出口 return：返回 _PreparedImage(path, mime, size_bytes, crop_offset)
    return _PreparedImage(path, mime, size_bytes, crop_offset)


# ┌── 函数 def _too_large_message()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：image_data_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _too_large_message(image_data_url: str) -> str:
    # ⤷ 出口 return：返回 f'Image too large for vision API: base64 payload is {len(ima…
    return (
        f"Image too large for vision API: base64 payload is {len(image_data_url) / (1024 * 1024):.1f} MB "
        f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) even after resizing. Install Pillow "
        f"(`pip install Pillow`) for better auto-resize, or compress the image manually.")


# ┌── 异步函数（协程） def _resize_prepared()
# │  作用：运行 :func:`_resize_image_for_vision` 在…上 CPU executor 用于 prepared image.
# │  参数：prepared: _PreparedImage, scale_info: dict, **kwargs
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _resize_prepared(prepared: _PreparedImage, scale_info: dict, **kwargs) -> str:
    """Run :func:`_resize_image_for_vision` on the CPU executor for a prepared image."""
    # ⤷ 出口 return：返回 await _run_encode_on_cpu_executor(_resize_image_for_vision, …
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _run_encode_on_cpu_executor(
        _resize_image_for_vision, prepared.path, mime_type=prepared.mime, scale_out=scale_info, **kwargs,
    )


# ┌── 异步函数（协程） def _vision_analyze_native()
# │  作用：Fast 路径用于 vision-capable main 模型: ``_multimodal`` envelope 字典在…上 success,
# │  参数：image_url: str, question: str, task_id: Optional[str] = None, region: Optional[list] = None
# │  返回：Any
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _vision_analyze_native(
    image_url: str, question: str, task_id: Optional[str] = None, region: Optional[list] = None,
) -> Any:
    """Fast path for vision-capable main models: a ``_multimodal`` envelope dict on success,
    or a JSON error string (the normal tool-result contract) on failure."""
    if not isinstance(image_url, str) or not image_url.strip():
        # ⤷ 出口 return：返回 tool_error('image_url is required', success=False)
        return tool_error("image_url is required", success=False)
    prepared: Optional[_PreparedImage] = None
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            # ⤷ 出口 return：返回 tool_error('Interrupted', success=False)
            return tool_error("Interrupted", success=False)
        # ⤷ 关键分支 try/except：捕获 _ImagePrepError，出错走兜底而不是崩溃。
        try:
            prepared = await _prepare_image(image_url, task_id, region, validate_decode=True)
        except _ImagePrepError as exc:
            return tool_error(str(exc), success=False)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url, prepared.path, mime_type=prepared.mime)
        # Proactive embed cap: this image is re-sent on every later turn, so resize DOWN to the
        # history-reuse target whenever the byte or long-edge cap is exceeded, not just at 20 MB.
        _scale_info: dict = {}
        # Anthropic still rejects >5 MB / >8000px with a non-retryable 400, but those are one-shot viewing
        # limits — history embeds are sized smaller so repeated vision_analyze turns don't blow the context
        # (#92699).
        _over_dims = await _run_encode_on_cpu_executor(
            _image_exceeds_dimension, prepared.path, _EMBED_MAX_DIMENSION)
        if len(image_data_url) > _EMBED_TARGET_BYTES or _over_dims:
            image_data_url = await _resize_prepared(
                prepared, _scale_info,
                max_base64_bytes=_EMBED_TARGET_BYTES, max_dimension=_EMBED_MAX_DIMENSION, force_jpeg=True)
            # Reject rather than embed a session-wedging payload.
            if len(image_data_url) > _MAX_BASE64_BYTES:
                return tool_error(_too_large_message(image_data_url), success=False)
        # ⤷ 出口 return：返回 _build_native_vision_tool_result(image_url=image_url, questi…
        return _build_native_vision_tool_result(
            image_url=image_url, question=question, image_data_url=image_data_url,
            image_size_bytes=prepared.size_bytes,
            scale_note=_build_scale_note(_scale_info or None, prepared.crop_offset or None))
    except Exception as exc:
        logger.warning("Native vision fast path failed: %s", exc)
        return tool_error(f"Native vision failed: {exc}", success=False)
    finally:
        # Only delete temp files we created — never user-provided paths.
        if prepared is not None:
            _unlink_quietly(prepared.path)


# ┌── 函数 def _aux_call_kwargs()
# │  作用：``async_call_llm`` kwargs 带 ``辅助的.vision.超时`` / ``.temperature`` 从配置.
# │  参数：messages: list, model: Optional[str], default_timeout: float, min_timeout: Optional[float] = ...
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _aux_call_kwargs(messages: list, model: Optional[str], default_timeout: float, *,
                     min_timeout: Optional[float] = None) -> dict:
    """``async_call_llm`` kwargs with ``auxiliary.vision.timeout`` / ``.temperature`` from config.
    Local vision models (llama.cpp, ollama) can take well over 30s, hence generous defaults
    (temperature 0.1); ``min_timeout`` lets video enforce a floor."""
    timeout, temperature = default_timeout, 0.1
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        _vision_cfg = _cfg_auxiliary("vision", default={}) or {}
        if _vision_cfg.get("timeout") is not None:
            timeout = max(float(_vision_cfg["timeout"]), float("-inf") if min_timeout is None else min_timeout)
        if _vision_cfg.get("temperature") is not None:
            temperature = float(_vision_cfg["temperature"])
    except Exception:
        pass
    # ⤷ 出口 return：返回 {'task': 'vision', 'messages': messages, 'temperature': temp…
    return {"task": "vision", "messages": messages, "temperature": temperature, "timeout": timeout,
            **({"model": model} if model else {})}


# ┌── 函数 def _media_messages()
# │  作用：Single user 消息: 文本 + one ``image_url``/``video_url`` data-URL part.
# │  参数：user_prompt: str, part_type: str, data_url: str
# │  返回：list
# └──────────────────────────────────────────────────────────────
def _media_messages(user_prompt: str, part_type: str, data_url: str) -> list:
    """Single user message: text + one ``image_url``/``video_url`` data-URL part."""
    # ⤷ 出口 return：返回 [{'role': 'user', 'content': [{'type': 'text', 'text': user_…
    return [{"role": "user", "content": [
        {"type": "text", "text": user_prompt}, {"type": part_type, part_type: {"url": data_url}}]}]


# Aux-LLM error classification: first matching hint set wins (billing → capability → size/format).
_BILLING_HINTS = ("402", "insufficient", "payment required", "credits", "billing")
_IMAGE_ERROR_RULES = (
    (_BILLING_HINTS,
     "Insufficient credits or payment required. Please top up your "
     "API provider account and try again. Error: {e}"),
    (("does not support", "not support image", "content_policy", "multimodal",
      "unrecognized request argument", "image input"),
     "{model} does not support vision or our request was not "
     "accepted by the server. Error: {e}"),
    (("invalid_request", "image_url"),
     "The vision API rejected the image. This can happen when the "
     "image is in an unsupported format, corrupted, or still too "
     "large after auto-resize. Try a smaller JPEG/PNG and retry. "
     "Error: {e}"),
)
_VIDEO_ERROR_RULES = (
    (_BILLING_HINTS, _IMAGE_ERROR_RULES[0][1]),
    (("does not support", "not support video", "content_policy", "multimodal",
      "unrecognized request argument", "video input", "video_url"),
     "The model does not support video analysis or the request was "
     "rejected. Ensure you're using a video-capable model "
     "(e.g. google/gemini-2.5-flash). Error: {e}"),
    (_SIZE_ERROR_HINTS,
     "The video is too large for the API. Try compressing or trimming "
     "the video (max ~50 MB). Error: {e}"),
)
# kind -> (debug tool name, error rules)
_ANALYSIS_KINDS = {
    "image": ("vision_analyze_tool", _IMAGE_ERROR_RULES),
    "video": ("video_analyze_tool", _VIDEO_ERROR_RULES)}


# ┌── 函数 def _debug_call_data()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：kind: str, source: str, user_prompt: str, model
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _debug_call_data(kind: str, source: str, user_prompt: str, model) -> dict:
    prompt = user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt
    # ⤷ 出口 return：返回 {'parameters': {f'{kind}_url': source, 'user_prompt': prompt…
    return {
        "parameters": {f"{kind}_url": source, "user_prompt": prompt, "model": model},
        "error": None, "success": False, "analysis_length": 0, "model_used": model, f"{kind}_size_bytes": 0}


# ┌── 异步函数（协程） def _call_vision_llm()
# │  作用：Aux vision LLM analysis 文本, retrying 一旦在…上 empty 内容 (reasoning-仅响应).
# │  参数：call_kwargs: dict, empty_log: str, response = None
# │  返回：（未标注类型）
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _call_vision_llm(call_kwargs: dict, empty_log: str, response=None):
    """Aux vision LLM analysis text, retrying once on empty content (reasoning-only response).
    ``response``: an already-made first call (the image path handles size-error retries itself)."""
    _load_auxiliary_client()
    if response is None:
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        response = await async_call_llm(**call_kwargs)
    analysis = extract_content_or_reasoning(response)
    if not analysis:
        logger.warning(empty_log)
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        analysis = extract_content_or_reasoning(await async_call_llm(**call_kwargs))
    # ⤷ 出口 return：返回 analysis
    return analysis


# ┌── 异步函数（协程） def _run_analysis()
# │  作用：Aux-LLM analysis skeleton 用于 image/video: interrupt 检查 → ``阶段`` → JSON 结果.
# │  参数：kind: str, source: str, user_prompt: str, model: Optional[str], stage: Callable[[str, dict, list], Awaitable[tuple]]
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _run_analysis(
    kind: str, source: str, user_prompt: str, model: Optional[str],
    stage: Callable[[str, dict, list], Awaitable[tuple]]) -> str:
    """Aux-LLM analysis skeleton for image/video: interrupt check → ``stage`` → JSON result.

    ``stage(user_prompt, debug_call_data, temp_paths)`` returns ``(analysis, scale_note)``; temp
    files it appends to ``temp_paths`` are deleted in ``finally``. Returns JSON
    ``{"success": bool, "analysis": str}`` (``analysis`` carries the error explanation on failure).
    """
    tool_name, rules = _ANALYSIS_KINDS[kind]
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data = _debug_call_data(kind, source, user_prompt, model)
    temp_paths: list = []

    def finish(result: dict) -> str:
        _debug.log_call(tool_name, debug_call_data)
        _debug.save()
        # ⤷ 出口 return：返回 json.dumps(result, indent=2, ensure_ascii=False)
        return json.dumps(result, indent=2, ensure_ascii=False)

    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            # ⤷ 出口 return：返回 tool_error('Interrupted', success=False)
            return tool_error("Interrupted", success=False)
        logger.info("Analyzing %s: %s", kind, source[:60])
        logger.info("User prompt: %s", user_prompt[:100])
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        analysis, scale_note = await stage(user_prompt, debug_call_data, temp_paths)
        analysis_length = len(analysis) if analysis else 0
        logger.info("%s analysis completed (%s characters)", kind.capitalize(), analysis_length)
        analysis = analysis or f"There was a problem with the request and the {kind} could not be analyzed."
        result = {"success": True, "analysis": f"[{scale_note}] {analysis}" if scale_note else analysis}
        if scale_note:
            result["scale_note"] = scale_note
        debug_call_data.update(success=True, analysis_length=analysis_length)
        # ⤷ 出口 return：返回 finish(result)
        return finish(result)
    except Exception as e:
        error_msg = f"Error analyzing {kind}: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        err_str = str(e).lower()
        template = next(
            (tpl for hints, tpl in rules if any(hint in err_str for hint in hints)),
            f"There was a problem with the request and the {kind} could not be analyzed. Error: {{e}}")
        analysis = template.format(e=e, model=model)
        debug_call_data["error"] = error_msg
        # ⤷ 出口 return：返回 finish({'success': False, 'error': error_msg, 'analysis': an…
        return finish({"success": False, "error": error_msg, "analysis": analysis})
    finally:
        for path in temp_paths:
            if path and path.exists():
                try:
                    path.unlink()
                    logger.debug("Cleaned up temporary %s file", kind)
                except Exception as cleanup_error:
                    logger.warning("Could not delete temporary file: %s", cleanup_error, exc_info=True)


# ┌── 异步函数（协程） def vision_analyze_tool()
# │  作用：描述 image (URL, local 路径, data: URL) 带辅助的 vision LLM. ``user_prompt``
# │  参数：image_url: str, user_prompt: str, model: str = None, task_id: Optional[str] = None, region: Optional[list] = None
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def vision_analyze_tool(
    image_url: str, user_prompt: str, model: str = None,
    task_id: Optional[str] = None, region: Optional[list] = None) -> str:
    """Describe an image (URL, local path, data: URL) with the auxiliary vision LLM. ``user_prompt``
    is pre-formatted by the caller. Temp images live under $HERMES_HOME/cache/vision/."""
    async def stage(prompt: str, debug_call_data: dict, temp_paths: list) -> tuple:
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        prepared = await _prepare_image(image_url, task_id, region, validate_decode=False)
        temp_paths.append(prepared.path)
        logger.info("Image ready (%.1f KB)", prepared.size_bytes / 1024)
        # Send at full resolution first; on a size rejection, downscale and retry.
        logger.info("Converting image to base64...")
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url, prepared.path, mime_type=prepared.mime)
        logger.info("Image converted to base64 (%.1f KB)", len(image_data_url) / 1024)
        _scale_info: dict = {}
        if len(image_data_url) > _MAX_BASE64_BYTES:
            image_data_url = await _resize_prepared(prepared, _scale_info)
            if len(image_data_url) > _MAX_BASE64_BYTES:
                raise ValueError(_too_large_message(image_data_url))
        debug_call_data["image_size_bytes"] = prepared.size_bytes
        messages = _media_messages(prompt, "image_url", image_data_url)
        logger.info("Processing image with vision model...")
        call_kwargs = _aux_call_kwargs(messages, model, 120.0)
        _load_auxiliary_client()
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            response = await async_call_llm(**call_kwargs)
        except Exception as _api_err:
            if not (_is_image_size_error(_api_err) and len(image_data_url) > _RESIZE_TARGET_BYTES):
                raise
            logger.info(
                "API rejected image (%.1f MB, likely too large); auto-resizing to ~%.0f MB and retrying...",
                len(image_data_url) / (1024 * 1024), _RESIZE_TARGET_BYTES / (1024 * 1024))
            image_data_url = await _resize_prepared(prepared, _scale_info)
            messages[0]["content"][1]["image_url"]["url"] = image_data_url
            response = await async_call_llm(**call_kwargs)
        analysis = await _call_vision_llm(
            call_kwargs, "Vision LLM returned empty content, retrying once", response)
        # ⤷ 出口 return：返回 (analysis, _build_scale_note(_scale_info or None, prepared.c…
        return analysis, _build_scale_note(_scale_info or None, prepared.crop_offset or None)
    # ⤷ 出口 return：返回 await _run_analysis('image', image_url, user_prompt, model, …
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _run_analysis("image", image_url, user_prompt, model, stage)


# ┌── 函数 def check_video_requirements()
# │  作用：当 ``call_llm(task="vision")`` 可以 resolve 客户端. 时返回 True。
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def check_video_requirements() -> bool:
    """True when ``call_llm(task="vision")`` could resolve a client.

    Mirrors its fallback chain: explicit ``auxiliary.vision.provider``, then auto (main
    provider → openrouter → nous) — without the auto step the tool would vanish whenever
    the explicit name was unresolvable. Probe mode skips real SDK client construction.

    See #31179.
    """
    from agent.auxiliary_client import aux_probe_mode, resolve_vision_provider_client
    # No blanket except: a resolver crash must reach the registry, which logs it with a
    # traceback; a swallowed exception reads as "no vision backend configured" (#87950).
    with aux_probe_mode():
        # ⤷ 出口 return：返回 any((resolve_vision_provider_client(**kw)[1] is not None for…
        return any(
            resolve_vision_provider_client(**kw)[1] is not None for kw in ({}, {"provider": "auto"})
        )


# ┌── 函数 def check_vision_requirements()
# │  作用：Image gate (``vision_analyze``, ``browser_vision``): aux vision 客户端或 native fast
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def check_vision_requirements() -> bool:
    """Image gate (``vision_analyze``, ``browser_vision``): an aux vision client OR the native fast
    path. Both handlers attach pixels straight to a vision-capable main model, so a main model on a
    provider the aux resolver does not know (OAuth, local vLLM) must not hide a working tool (#47149).
    ``video_analyze`` keeps the aux-only gate — its handler has no native path.
    """
    # ⤷ 出口 return：返回 _should_use_native_vision_fast_path() or check_video_require…
    return _should_use_native_vision_fast_path() or check_video_requirements()


from tools.registry import registry, tool_error

VISION_ANALYZE_SCHEMA = {
    "name": "vision_analyze",
    # Routing mechanics are deliberately absent (the route is automatic and the
    # native result says so itself); region keeps its pre-effect guidance — a
    # model that doesn't know crops keep full resolution never zooms.
    "description": (
        # Dieted (#95681): routing mechanics (native attach vs aux-model text fallback) removed — the route
        # is automatic and the native path's own tool result says "you can see it natively now"; the schema
        # doesn't need to predict plumbing.
        "Load an image into the conversation so you can see it. Call it "
        "any time the user references an image — then answer from what "
        "you see."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Image URL (http/https), local file path, or data: URL to load."
            },
            "question": {
                "type": "string",
                "description": "Your question or request about the image."
            },
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional [x1, y1, x2, y2] crop in ORIGINAL-image pixel "
                    "coordinates, applied before any downscaling — the crop "
                    "keeps full resolution. Load the full image first, then "
                    "re-call with a region to zoom into small text or fine "
                    "detail."
                )
            }
        },
        "required": ["image_url", "question"]
    }
}


# ┌── 函数 def _configured_aux_model()
# │  作用：First non-empty ``辅助的.<section>.模型`` 从配置.YAML, 否则 first non-empty
# │  参数：sections: tuple, env_vars: tuple
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _configured_aux_model(sections: tuple, env_vars: tuple) -> Optional[str]:
    """First non-empty ``auxiliary.<section>.model`` from config.yaml, else the first non-empty
    env var (legacy override), else None."""
    for section in sections:
        _vmodel = _cfg_auxiliary(section, "model")
        if _vmodel:
            if str(_vmodel).strip():
                # ⤷ 出口 return：返回 str(_vmodel).strip()
                return str(_vmodel).strip()
            break
    # ⤷ 出口 return：返回 next((v for v in (os.getenv(e, '').strip() for e in env_vars…
    return next((v for v in (os.getenv(e, "").strip() for e in env_vars) if v), None)


# ┌── 异步函数（协程） def _handle_vision_analyze()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: Dict[str, Any], **kw
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _handle_vision_analyze(args: Dict[str, Any], **kw: Any) -> str:
    image_url, question, region = args.get("image_url", ""), args.get("question", ""), args.get("region")
    task_id = kw.get("task_id")
    # No concurrency gate around the whole analysis — the CPU burst is bounded inside the
    # encode/resize step, so multi-image fan-out keeps full request concurrency.
    if _should_use_native_vision_fast_path():
        logger.info("vision_analyze: native fast path")
        # ⤷ 出口 return：返回 await _vision_analyze_native(image_url, question, task_id=ta…
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        return await _vision_analyze_native(image_url, question, task_id=task_id, region=region)

    # Legacy path: aux LLM describes the image and we return its text.
    full_prompt = (
        "Fully describe and explain everything about this image, then answer the "
        f"following question:\n\n{question}")
    model = _configured_aux_model(("vision",), ("AUXILIARY_VISION_MODEL",))
    # ⤷ 出口 return：返回 await vision_analyze_tool(image_url, full_prompt, model, tas…
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await vision_analyze_tool(image_url, full_prompt, model, task_id=task_id, region=region)


registry.register(
    name="vision_analyze",
    toolset="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=_handle_vision_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="👁️")


# --- video_analyze --------------------------------------------------------
# Extension → MIME. avi/mkv fall back to mp4.
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/mov",
    ".avi": "video/mp4",
    ".mkv": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}

_MAX_VIDEO_BASE64_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
_VIDEO_SIZE_WARN_BYTES = 20 * 1024 * 1024


# ┌── 函数 def _detect_video_mime_type()
# │  作用：Video MIME 类型从 extension, 或 None 若 unsupported.
# │  参数：video_path: Path
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _detect_video_mime_type(video_path: Path) -> Optional[str]:
    """Video MIME type from extension, or None if unsupported."""
    # ⤷ 出口 return：返回 _VIDEO_MIME_TYPES.get(video_path.suffix.lower())
    return _VIDEO_MIME_TYPES.get(video_path.suffix.lower())


# ┌── 函数 def _unsupported_video_format()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：suffix: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _unsupported_video_format(suffix: str) -> str:
    # ⤷ 出口 return：返回 f"Unsupported video format: '{suffix}'. Supported: {', '.joi…
    return f"Unsupported video format: '{suffix}'. Supported: {', '.join(sorted(_VIDEO_MIME_TYPES.keys()))}"


# ┌── 函数 def _video_to_base64_data_url()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：video_path: Path, mime_type: Optional[str] = None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _video_to_base64_data_url(video_path: Path, mime_type: Optional[str] = None) -> str:
    mime = mime_type or _detect_video_mime_type(video_path) or "video/mp4"
    # ⤷ 出口 return：返回 f"data:{mime};base64,{base64.b64encode(video_path.read_bytes…
    return f"data:{mime};base64,{base64.b64encode(video_path.read_bytes()).decode('ascii')}"


# ┌── 异步函数（协程） def _materialize_video()
# │  作用：Local video 路径用于终端-backend 路径, local 文件, 或 HTTP(S) URL. 仅文件已创建
# │  参数：video_url: str, task_id: Optional[str], temp_paths: list
# │  返回：Path
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def _materialize_video(video_url: str, task_id: Optional[str], temp_paths: list) -> Path:
    """Local video path for a terminal-backend path, local file, or HTTP(S) URL. Only files created
    here are appended to ``temp_paths`` — never user-provided paths. Terminal-backend reads use the
    shared media resolver with ``permitted=("video",)`` — the exact pipeline vision_analyze uses
    (media-cache host reads, bounded in-sandbox exec-read, credential-read guard, 50MB cap)."""
    from tools.image_source import (
        ImageResolutionError, ResolveContext, _is_local_terminal_backend, resolve_image_source,
    )
    source = video_url.removeprefix("file://")
    local_path = Path(os.path.expanduser(source))
    lowered = (video_url or "").strip().lower()
    path_like = bool(lowered) and not lowered.startswith(("http://", "https://", "data:"))
    if not _is_local_terminal_backend() and path_like:
        logger.info("Reading video source via terminal backend: %s", video_url)
        suffix = Path(source).suffix.lower()
        if suffix not in _VIDEO_MIME_TYPES:
            raise ValueError(_unsupported_video_format(suffix))
        # ⤷ 关键分支 try/except：捕获 ImageResolutionError，出错走兜底而不是崩溃。
        try:
            resolved = await resolve_image_source(
                video_url, ResolveContext(task_id=task_id), permitted=("video",))
        except ImageResolutionError as exc:
            raise ValueError(f"Could not read video from terminal backend: {exc}") from exc
        temp_dir = get_hermes_dir("cache/video", "temp_video_files")
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"terminal_video_{uuid.uuid4()}{suffix}"
        path.write_bytes(resolved.data)
        temp_paths.append(path)
        # ⤷ 出口 return：返回 path
        return path
    if local_path.is_file():
        from agent.file_safety import raise_if_read_blocked
        raise_if_read_blocked(str(local_path))
        logger.info("Using local video file: %s", video_url)
        # ⤷ 出口 return：返回 local_path
        return local_path
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    if await _validate_image_url_async(video_url):
        blocked = check_website_access(video_url)
        if blocked:
            raise PermissionError(blocked["message"])
        path = get_hermes_dir("cache/video", "temp_video_files") / f"temp_video_{uuid.uuid4()}.mp4"
        temp_paths.append(path)
        # Video downloads retry every failure class (legacy behavior).
        await _download_media(
            video_url, path, 3, media_label="Video", accept="video/*,*/*;q=0.8",
            max_bytes=_MAX_VIDEO_BASE64_BYTES, timeout=60.0, retry_all=True)
        # ⤷ 出口 return：返回 path
        return path
    # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
    raise ValueError("Invalid video source. Provide an HTTP/HTTPS URL or a valid local file path.")


# ┌── 异步函数（协程） def video_analyze_tool()
# │  作用：Analyze video via multimodal LLM. 返回 JSON {success, analysis}.
# │  参数：video_url: str, user_prompt: str, model: str = None, task_id: Optional[str] = None
# │  返回：str
# │  【语法】async def 定义的是协程函数：调用它得到协程对象，必须 await 或交给事件循环才会真正执行。
# └──────────────────────────────────────────────────────────────
async def video_analyze_tool(
    video_url: str, user_prompt: str, model: str = None, task_id: Optional[str] = None) -> str:
    """Analyze a video via multimodal LLM. Returns JSON {success, analysis}."""
    async def stage(prompt: str, debug_call_data: dict, temp_paths: list) -> tuple:
        # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
        temp_video_path = await _materialize_video(video_url, task_id, temp_paths)
        video_size_bytes = temp_video_path.stat().st_size
        video_size_mb = video_size_bytes / (1024 * 1024)
        logger.info("Video ready (%.1f MB)", video_size_mb)
        detected_mime = _detect_video_mime_type(temp_video_path)
        if not detected_mime:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError(_unsupported_video_format(temp_video_path.suffix))
        if video_size_bytes > _VIDEO_SIZE_WARN_BYTES:
            logger.warning("Video is %.1f MB — may be slow or rejected", video_size_mb)
        video_data_url = _video_to_base64_data_url(temp_video_path, mime_type=detected_mime)
        if len(video_data_url) > _MAX_VIDEO_BASE64_BYTES:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise ValueError(
                f"Video too large for API: base64 payload is {len(video_data_url) / (1024 * 1024):.1f} MB "
                f"(limit {_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                f"Compress or trim the video and retry.")
        debug_call_data["video_size_bytes"] = video_size_bytes
        messages = _media_messages(prompt, "video_url", video_data_url)
        call_kwargs = _aux_call_kwargs(messages, model, 180.0, min_timeout=180.0)
        analysis = await _call_vision_llm(call_kwargs, "Empty video response, retrying once")
        # ⤷ 出口 return：返回 (analysis, None)
        return analysis, None
    # ⤷ 出口 return：返回 await _run_analysis('video', video_url, user_prompt, model, …
    # ⤷ await：在这里让出控制权等异步结果回来，期间不阻塞整个线程。
    return await _run_analysis("video", video_url, user_prompt, model, stage)


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video from a URL or local file path using a multimodal AI model. "
        "Sends the video to a video-capable model (e.g. Gemini) for understanding. "
        "Use this for video files — for images, use vision_analyze instead. "
        "Supports mp4, webm, mov, avi, mkv, mpeg formats. "
        "Note: large videos (>20 MB) may be slow; max ~50 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "Video URL (http/https) or local file path to analyze.",
            },
            "question": {
                "type": "string",
                "description": "Your specific question about the video. The AI will describe what happens in the video and answer your question.",
            },
        },
        "required": ["video_url", "question"],
    },
}


# ┌── 函数 def _handle_video_analyze()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: Dict[str, Any], **kw
# │  返回：Awaitable[str]
# └──────────────────────────────────────────────────────────────
def _handle_video_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    video_url, question = args.get("video_url", ""), args.get("question", "")
    full_prompt = (
        "Fully describe and explain everything happening in this video, "
        "including visual content, motion, audio cues, text overlays, and scene "
        f"transitions. Then answer the following question:\n\n{question}")
    model = _configured_aux_model(("video", "vision"), ("AUXILIARY_VIDEO_MODEL", "AUXILIARY_VISION_MODEL"))
    # ⤷ 出口 return：返回 video_analyze_tool(video_url, full_prompt, model, task_id=kw…
    return video_analyze_tool(video_url, full_prompt, model, task_id=kw.get("task_id"))


registry.register(
    name="video_analyze",
    toolset="video",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    check_fn=check_video_requirements,
    is_async=True,
    emoji="🎬")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import contextlib  # noqa: F401,E402
import sys  # noqa: F401,E402
import threading  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
