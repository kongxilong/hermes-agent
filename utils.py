"""Shared utility functions for hermes-agent."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：utils.py（约 599 行）
# 地位：全项目通用的小工具函数库（和 Agent 逻辑无关，纯粹是工程实用代码）。
#
# 分三大块：
#   1. 原子写文件：atomic_write_text / atomic_write_bytes / atomic_json_write / atomic_yaml_write
#      「原子写」= 先写临时文件，再原子替换目标文件。这样断电/崩溃时不会留下半个损坏的文件。
#      这是配置文件、状态文件必须的写法，非常值得学。
#   2. 环境变量读取：env_int / env_float / env_bool / env_var_enabled
#      统一处理「环境变量是字符串，但我要布尔/整数」的转换和默认值。
#   3. URL 处理：normalize_proxy_url / base_url_hostname / base_url_origin / base_url_host_matches
#      判断一个 API 地址属于哪个服务商。
#
# 新手推荐先读：atomic_replace() 和 _atomic_write()——体会什么叫「写文件要防崩溃」。
# ==========================================================================
#

import errno
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


# ┌── 函数 def is_truthy_value()
# │  作用：Coerce 布尔值-ish 值 using 投射's shared truthy 字符串设置.
# │  参数：value: Any, default: bool = False
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        # ⤷ 出口 return：返回 default
        return default
    if isinstance(value, str):
        # ⤷ 出口 return：返回 value.strip().lower() in TRUTHY_STRINGS
        return value.strip().lower() in TRUTHY_STRINGS
    # ⤷ 出口 return：返回 bool(value)
    return bool(value)


# ┌── 函数 def env_var_enabled()
# │  作用：当 environment variable 是设置 truthy 值. 时返回 True。
# │  参数：name: str, default: str = ''
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    # ⤷ 出口 return：返回 is_truthy_value(os.getenv(name, default), default=False)
    return is_truthy_value(os.getenv(name, default), default=False)


# ┌── 函数 def _preserve_file_mode()
# │  作用：Permission bits 的 *路径* 若它 exists, 否则 ``None``.
# │  参数：path: Path
# │  返回：'int | None'
# └──────────────────────────────────────────────────────────────
def _preserve_file_mode(path: Path) -> "int | None":
    """Permission bits of *path* if it exists, else ``None``."""
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _preserve_file_owner()
# │  作用：持有 ``(uid, gid)`` 的 *路径* 在…上 POSIX, 否则 ``None``.
# │  参数：path: Path
# │  返回：'tuple[int, int] | None'
# └──────────────────────────────────────────────────────────────
def _preserve_file_owner(path: Path) -> "tuple[int, int] | None":
    """Owning ``(uid, gid)`` of *path* on POSIX, else ``None``."""
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        st = path.stat() if os.name == "posix" else None
    except OSError:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 (st.st_uid, st.st_gid) if st else None
    return (st.st_uid, st.st_gid) if st else None


# ┌── 函数 def _restore_file_metadata()
# │  作用：尽最大努力 re-apply 的 uid/gid 且 permission bits 在…之后 atomic replace.
# │  参数：path: Path, owner: 'tuple[int, int] | None', mode: 'int | None'
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _restore_file_metadata(path: Path, owner: "tuple[int, int] | None", mode: "int | None") -> None:
    """Best-effort re-apply of uid/gid and permission bits after an atomic replace.

    Docker/NAS installs often run some commands as root on a volume owned by the runtime user;
    ``os.replace`` swaps in the temp file's owner, so privileged callers chown it back. ``mkstemp``
    creates 0o600 files; without re-applying *mode* the target would inherit that and break
    volume mounts relying on broader permissions.
    """
    if owner is not None and hasattr(os, "chown"):
        with suppress(OSError):
            os.chown(path, owner[0], owner[1])
    if mode is not None:
        with suppress(OSError):
            os.chmod(path, mode)


# ┌── 函数 def default_new_file_mode()
# │  作用：模式 ``打开(路径, "w")`` gives 文件它 有创建 (``0o666 & ~umask``); ``None``
# │  参数：无
# │  返回：'int | None'
# └──────────────────────────────────────────────────────────────
def default_new_file_mode() -> "int | None":
    """The mode ``open(path, "w")`` gives a file it has to create (``0o666 & ~umask``); ``None``
    when the umask cannot be read or on non-POSIX hosts (Windows mode bits are synthesized).

    ``mkstemp`` always creates at 0600, so publishing a *new* non-secret file through a temp
    file would tighten it to owner-only — the Docker/NAS volume-mount hazard
    :func:`_restore_file_metadata` documents. The transient mask is 0o077: a thread that opens
    a file in the read window gets a tighter file, never a looser one.
    """
    if os.name != "posix":
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        current = os.umask(0o077)
        os.umask(current)
    except OSError:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 438 & ~current
    return 0o666 & ~current


# ┌── 函数 def _restore_file_owner()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：path: Path, owner: 'tuple[int, int] | None'
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _restore_file_owner(path: Path, owner: "tuple[int, int] | None") -> None:
    _restore_file_metadata(path, owner, None)


# ┌── 函数 def _restore_file_mode()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：path: Path, mode: 'int | None'
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _restore_file_mode(path: Path, mode: "int | None") -> None:
    _restore_file_metadata(path, None, mode)


_IS_WINDOWS = os.name == "nt"
# Windows rename failures possibly caused by another handle on the target. CPython opens files
# without FILE_SHARE_DELETE, so ``os.replace`` onto an open file is denied with 5 ERROR_ACCESS_DENIED
# (what a held *target* handle actually reports — measured: a plain reader yields 5, NOT 32),
# 32 ERROR_SHARING_VIOLATION (the *source* temp file is held) or 33 ERROR_LOCK_VIOLATION (byte-range
# lock on the target). Ambiguous (a real ACL denial is also 5), so recovery is bounded and a
# still-failing write is re-raised unchanged rather than classified up front.
_WINDOWS_CONTENDED_REPLACE_ERRORS = frozenset({5, 32, 33})
# Retry budget for the atomic rename. A rename that wins here keeps the write fully atomic, so the
# budget covers a realistic hold (desktop auth-init holds auth.json >100 ms): ~200 ms recovered
# atomically, ~310 ms worst case. The cap matters as much as the count — gateway_state.json is
# rewritten every turn, so a permanently-held target pays the full budget per write. Jittered so
# concurrent writers don't retry in lockstep.
_REPLACE_RETRY_ATTEMPTS = 4
_REPLACE_RETRY_BASE_DELAY_S = 0.02
_REPLACE_RETRY_MAX_DELAY_S = 0.1
_CROSS_DEVICE_ERRNOS = (errno.EXDEV, errno.EBUSY)


# ┌── 函数 def _is_contended_windows_replace_error()
# │  作用：Candidate-仅: winerror 5 也 covers genuine ACL denial.
# │  参数：exc: OSError
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_contended_windows_replace_error(exc: OSError) -> bool:
    """Candidate-only: winerror 5 also covers a genuine ACL denial."""
    # ⤷ 出口 return：返回 _IS_WINDOWS and getattr(exc, 'winerror', None) in _WINDOWS_C…
    return _IS_WINDOWS and getattr(exc, "winerror", None) in _WINDOWS_CONTENDED_REPLACE_ERRORS


# ┌── 函数 def _rewrite_in_place()
# │  作用：Overwrite *real_path* through existing 文件 — last resort 用于仍-held target.
# │  参数：tmp_str: str, real_path: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _rewrite_in_place(tmp_str: str, real_path: str) -> None:
    """Overwrite *real_path* through the existing file — last resort for a still-held target.

    Not atomic (a smaller window than a copy, not none), so it runs only after the rename has
    genuinely failed. Writing through the target also preserves its ACL, which ``os.replace``
    does not (the temp file's inherited ACL wins there).
    """
    with open(tmp_str, "rb") as src:
        data = src.read()
    fd = os.open(real_path, os.O_WRONLY | getattr(os, "O_BINARY", 0))
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)
    os.unlink(tmp_str)


# ┌── 函数 def _copy_fallback()
# │  作用：复制/fsync/unlink 兜底用于 cross-device 且 bind-mount renames.
# │  参数：tmp_str: str, real_path: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _copy_fallback(tmp_str: str, real_path: str) -> None:
    """Copy/fsync/unlink fallback for cross-device and bind-mount renames."""
    shutil.copyfile(tmp_str, real_path)
    with suppress(OSError):
        shutil.copystat(tmp_str, real_path)
    with suppress(OSError), open(real_path, "rb") as f:
        os.fsync(f.fileno())
    os.unlink(tmp_str)


# ┌── 函数 def atomic_replace()
# │  作用：把 *tmp_path* 原子地移动到 *target* 上，并保留符号链接。
# │  参数：tmp_path: Union[str, Path], target: Union[str, Path]
# │  返回：str
# └──────────────────────────────────────────────────────────────
def atomic_replace(tmp_path: Union[str, Path], target: Union[str, Path]) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    Resolves a symlink first so ``os.replace`` writes the real file in place and the symlink
    survives. Otherwise identical to ``os.replace`` unless the rename fails with EXDEV/EBUSY
    (cross-device, bind-mount, busy file: copy/fsync/unlink immediately — these never clear on
    retry) or a Windows rename contended by another open handle (winerror 5/32/33: bounded retry,
    then in-place rewrite).
    """
    target_str = str(target)
    real_path = os.path.realpath(target_str) if os.path.islink(target_str) else target_str
    tmp_str = str(tmp_path)
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        os.replace(tmp_str, real_path)
        # ⤷ 出口 return：返回 real_path
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in _CROSS_DEVICE_ERRNOS and not contended:
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        if contended:
            # Lazy: keeps ``utils`` free of a package-level dependency on ``agent``.
            from agent.retry_utils import jittered_backoff
            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(jittered_backoff(attempt, base_delay=_REPLACE_RETRY_BASE_DELAY_S, max_delay=_REPLACE_RETRY_MAX_DELAY_S))
                # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
                try:
                    os.replace(tmp_str, real_path)
                    # ⤷ 出口 return：返回 real_path
                    return real_path
                except OSError as retry_exc:
                    exc = retry_exc
                    if retry_exc.errno in _CROSS_DEVICE_ERRNOS:
                        contended = False  # not contention after all — stop burning the budget
                        break
                    if not _is_contended_windows_replace_error(retry_exc):
                        raise
        logger.debug("atomic_replace: %s -> %s failed with %s; falling back to %s", tmp_str, real_path,
                     getattr(exc, "winerror", None) or errno.errorcode.get(exc.errno or 0, exc.errno),
                     "in-place rewrite" if contended else "copy")
        # The rewrite re-raises its own error, so an ACL denial is reported as such, not as contention.
        (_rewrite_in_place if contended else _copy_fallback)(tmp_str, real_path)
    # ⤷ 出口 return：返回 real_path
    return real_path


# ┌── 函数 def fsync_directory()
# │  作用：尽最大努力 fsync 的 directory entry 从而 just-renamed 文件 survives power loss.
# │  参数：path: Union[str, Path]
# │  返回：None
# └──────────────────────────────────────────────────────────────
def fsync_directory(path: Union[str, Path]) -> None:
    """Best-effort fsync of a directory entry so a just-renamed file survives power loss.

    No-op on Windows (directories can't be opened with ``os.open``; the file fsync still applies)
    and on any OSError — durability of the directory entry is never worth failing a write that
    has already been replaced into place.
    """
    if os.name == "nt":
        return
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
    try:
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


# ┌── 函数 def _atomic_write()
# │  作用：Temp 文件 + fsync + :func:`atomic_replace`, 然后 re-apply owner/模式.
# │  参数：path: Path, write, prefix: str, encoding: str = ..., mode: 'int | None' = ..., preserve_owner: bool = ..., binary: bool = ..., fsync_dir: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _atomic_write(path: Path, write, *, prefix: str, encoding: str = "utf-8", mode: "int | None" = None,
                  preserve_owner: bool = True, binary: bool = False, fsync_dir: bool = False) -> None:
    """Temp file + fsync + :func:`atomic_replace`, then re-apply owner/mode.

    *write(f)* emits the payload into the open handle (text, or bytes when *binary*). The temp file
    is created by ``mkstemp`` — ``O_CREAT|O_EXCL`` at 0600 regardless of umask — so a secret is
    never readable at process umask, not even between create and chmod. *mode* is fchmod'd onto
    the temp fd BEFORE the replace so the target never transits through mkstemp's 0600 (fchmod is
    Unix-only; the post-replace chmod is the sole path on Windows). With no *mode* a NEW target
    gets what ``open(path, "w")`` would have given it (process umask) — the callers this replaced
    wrote at umask, and silently tightening every fresh cache/state file to 0600 breaks shared
    volume mounts; an existing target with no *mode* keeps mkstemp's bits, as before. *fsync_dir*
    also fsyncs the parent so the rename itself is durable. The temp file is removed on any
    failure — ``BaseException`` on purpose, so KeyboardInterrupt / SystemExit still clean up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None and not path.exists():
        mode = default_new_file_mode()
    original_owner = _preserve_file_owner(path) if preserve_owner else None
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    # ⤷ 关键分支 try/except：捕获 BaseException，出错走兜底而不是崩溃。
    try:
        with os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        _restore_file_metadata(Path(atomic_replace(tmp_path, path)), original_owner, mode)  # symlink-preserving
        if fsync_dir:
            fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise


# ┌── 函数 def _mode_for_write()
# │  作用：Existing permission bits 的 *路径* (当 *preserve*), 否则 *create_mode* 用于 new 文件.
# │  参数：path: Path, create_mode: 'int | None', preserve: bool = True
# │  返回：'int | None'
# └──────────────────────────────────────────────────────────────
def _mode_for_write(path: Path, create_mode: "int | None", preserve: bool = True) -> "int | None":
    """Existing permission bits of *path* (when *preserve*), else *create_mode* for a new file."""
    mode = _preserve_file_mode(path) if preserve else None
    # ⤷ 出口 return：返回 mode if mode is not None or path.exists() else create_mode
    return mode if mode is not None or path.exists() else create_mode


# ┌── 函数 def atomic_write_text()
# │  作用：写入 *内容* *路径* via temp 文件 + fsync + atomic rename.
# │  参数：path: Union[str, Path], content: str, encoding: str = ..., tmp_prefix: str = ..., preserve_mode: bool = ..., create_mode: 'int | None' = ..., mode: 'int | None' = ..., fsync_dir: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8", tmp_prefix: str = ".tmp_",
                      preserve_mode: bool = False, create_mode: "int | None" = None, mode: "int | None" = None,
                      fsync_dir: bool = False) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    The target is never left partially written on crash/interrupt. Shared by every destructive
    file rewrite (memory store, skill manager, agent importer, ...). *mode* forces the final
    permission bits (secret files: ``0o600``) regardless of what exists; *create_mode* applies only
    when the target is new and *preserve_mode* carries an existing file's bits and owner across.
    """
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, encoding=encoding,
                  mode=mode if mode is not None else _mode_for_write(path, create_mode, preserve=preserve_mode),
                  preserve_owner=preserve_mode, fsync_dir=fsync_dir)


# ┌── 函数 def atomic_write_bytes()
# │  作用：Bytes variant 的 :func:`atomic_write_text` (encrypted blobs, 键 material).
# │  参数：path: Union[str, Path], content: bytes, tmp_prefix: str = ..., mode: 'int | None' = ..., fsync_dir: bool = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_write_bytes(path: Union[str, Path], content: bytes, *, tmp_prefix: str = ".tmp_",
                       mode: "int | None" = None, fsync_dir: bool = False) -> None:
    """Bytes variant of :func:`atomic_write_text` (encrypted blobs, key material)."""
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, binary=True, preserve_owner=False,
                  mode=mode if mode is not None else _preserve_file_mode(path), fsync_dir=fsync_dir)


# ┌── 函数 def _dump_json()
# │  作用：``JSON.dump`` 那个 survives surrogate-escaped strings.
# │  参数：data: Any, f, indent: 'int | None', ensure_ascii: bool, dump_kwargs: dict
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _dump_json(data: Any, f, *, indent: "int | None", ensure_ascii: bool, dump_kwargs: dict) -> None:
    """``json.dump`` that survives surrogate-escaped strings.

    ``os.fsdecode`` of a non-UTF-8 filename/argv yields lone surrogates (``'\\udcff'``); a utf-8
    text handle rejects them with ``UnicodeEncodeError`` — a ValueError, which callers guarding
    ``except OSError`` never see. ``ensure_ascii=True`` escapes them as ``\\udcff`` and
    ``json.loads`` restores the identical str, so the retry round-trips; ``surrogateescape``
    would emit a raw 0xFF byte that the reader's utf-8 decode rejects. Serializing to a str first
    keeps the failure before any byte reaches the file, so no partial payload is left behind.
    """
    text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
    # ⤷ 关键分支 try/except：捕获 UnicodeEncodeError，出错走兜底而不是崩溃。
    try:
        f.write(text)
    except UnicodeEncodeError:
        f.write(json.dumps(data, indent=indent, ensure_ascii=True, **dump_kwargs))


# ┌── 函数 def atomic_json_write()
# │  作用：写入 JSON *路径* atomically (temp 文件 + fsync + replace).
# │  参数：path: Union[str, Path], data: Any, indent: int = ..., mode: int | None = ..., ensure_ascii: bool = ..., fsync_dir: bool = ..., **dump_kwargs
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_json_write(
    path: Union[str, Path], data: Any, *, indent: int = 2, mode: int | None = None,
    ensure_ascii: bool = False, fsync_dir: bool = False, **dump_kwargs: Any,
) -> None:
    """Write JSON to *path* atomically (temp file + fsync + replace).

    Surrogate-escaped strings (non-UTF-8 argv/paths) are always persisted: the write falls back
    to ``ensure_ascii=True`` escapes for that payload only, so normal content keeps its raw UTF-8
    bytes. ``mode=0o600`` is the private-credential form: the temp file is 0600 from creation
    (mkstemp), so the payload is never umask-readable.
    """
    path = Path(path)
    _atomic_write(path, lambda f: _dump_json(data, f, indent=indent, ensure_ascii=ensure_ascii, dump_kwargs=dump_kwargs),
                  prefix=f".{path.stem}_", mode=mode if mode is not None else _preserve_file_mode(path),
                  fsync_dir=fsync_dir)


# ┌── 函数 def read_json_or_empty()
# │  作用：JSON 对象在 *路径*, 或 ``{}`` 当文件是 missing, unreadable, 格式非法的或
# │  参数：path: Union[str, Path]
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def read_json_or_empty(path: Union[str, Path]) -> dict:
    """The JSON object at *path*, or ``{}`` when the file is missing, unreadable, malformed or
    not an object. The read half of every ``read → merge → atomic_json_write`` config store
    (memory-provider ``save_config``), so a corrupt sidecar degrades to defaults instead of
    taking the provider down."""
    # ⤷ 关键分支 try/except：捕获 (OSError, ValueError)，出错走兜底而不是崩溃。
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ⤷ 出口 return：返回 {}
        return {}
    # ⤷ 出口 return：返回 data if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}


# ┌── 函数 def warn_if_credential_file_broadly_readable()
# │  作用：警告当 credential 文件是 group/world-readable; True 当警告是 emitted.
# │  参数：path: Union[str, Path], label: str = ..., log: logging.Logger | None = ...
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def warn_if_credential_file_broadly_readable(path: Union[str, Path], *, label: str = "", log: logging.Logger | None = None) -> bool:
    """Warn when a credential file is group/world-readable; True when a warning was emitted.

    Hand-made secret files (or ones older Hermes wrote without an explicit mode) commonly end up
    0o644 under the default umask; call this before loading any token/credential file. No-op on
    non-POSIX (Windows ACLs don't map onto group/other bits; st_mode there is synthesized), when
    the file is missing, or when permissions are already tight.
    """
    p = Path(path)
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        file_mode = p.stat().st_mode
    except OSError:
        # ⤷ 出口 return：返回 False
        return False
    if os.name != "posix" or not (file_mode & (stat.S_IRGRP | stat.S_IROTH)):
        # ⤷ 出口 return：返回 False
        return False
    (log or logger).warning("%s%s is group/world-readable (mode 0%o) and contains secrets. Run: chmod 600 %s",
                            f"{label} " if label else "", p.name, stat.S_IMODE(file_mode), p)
    # ⤷ 出口 return：返回 True
    return True


# ┌── 类 class IndentDumper（继承自：yaml.SafeDumper）
# │  说明：PyYAML dumper 那个 indents 列表 items under mapping 键 (2-space).
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
class IndentDumper(yaml.SafeDumper):
    """PyYAML dumper that indents list items under mapping keys (2-space).

    PyYAML emits "indentless" sequences while ruamel (:func:`atomic_roundtrip_yaml_update`)
    indents them; mixing both in one ``config.yaml`` makes stricter parsers like ``js-yaml``
    reject it, so every write path is forced to the same shape.

    Forcing ``indentless=False`` aligns the two serializers so all write paths emit byte-identical layouts
    (#31999).
    """

    # ┌── 函数 def increase_indent()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, flow = False, indentless = False
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def increase_indent(self, flow=False, indentless=False):  # noqa: ARG002
        # ⤷ 出口 return：返回 super().increase_indent(flow, False)
        return super().increase_indent(flow, False)


# ┌── 函数 def atomic_yaml_write()
# │  作用：写入 YAML *路径* atomically (temp 文件 + fsync + replace).
# │  参数：path: Union[str, Path], data: Any, default_flow_style: bool = ..., sort_keys: bool = ..., extra_content: str | None = ..., create_mode: 'int | None' = ...
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_yaml_write(path: Union[str, Path], data: Any, *, default_flow_style: bool = False, sort_keys: bool = False,
                      extra_content: str | None = None, create_mode: "int | None" = None) -> None:
    """Write YAML to *path* atomically (temp file + fsync + replace)."""
    path = Path(path)

    def _write(f) -> None:
        # allow_unicode=True writes emoji/kaomoji as real UTF-8. Without it PyYAML emits astral
        # chars as `\UXXXXXXXX` escapes inside `\`-continued double-quoted strings — a structure
        # stricter parsers and hand-edits routinely break into unclosed quotes, corrupting the config.
        yaml.dump(data, f, Dumper=IndentDumper, default_flow_style=default_flow_style, sort_keys=sort_keys, allow_unicode=True)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=_mode_for_write(path, create_mode))


# ┌── 函数 def _roundtrip_load()
# │  作用：``(yaml_rt, CommentedMap)``: ruamel 轮-trip loader 保留 quotes/Unicode 带 2-space
# │  参数：path: Path
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _roundtrip_load(path: Path):
    """``(yaml_rt, CommentedMap)``: a ruamel round-trip loader keeping quotes/Unicode with 2-space
    indents, plus *path* loaded through it (empty map when missing/blank)."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    data = yaml_rt.load(path.read_text(encoding="utf-8")) if path.exists() else None
    # ⤷ 出口 return：返回 (yaml_rt, data if isinstance(data, CommentedMap) else Commen…
    return yaml_rt, data if isinstance(data, CommentedMap) else CommentedMap(data or {})


# ┌── 函数 def _roundtrip_dump()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：path: Path, yaml_rt, config
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _roundtrip_dump(path: Path, yaml_rt, config) -> None:
    _atomic_write(path, lambda f: yaml_rt.dump(config, f), prefix=f".{path.stem}_", mode=_preserve_file_mode(path))


# ┌── 函数 def atomic_roundtrip_yaml_update()
# │  作用：更新 one dotted YAML 键当 preserving comments, ordering, quoting 且 Unicode.
# │  参数：path: Union[str, Path], key_path: str, value: Any
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_roundtrip_yaml_update(path: Union[str, Path], key_path: str, value: Any) -> None:
    """Update one dotted YAML key while preserving comments, ordering, quoting and Unicode.

    Narrower than :func:`atomic_yaml_write` on purpose: for user-edited config files where a
    single setting mutation must not disturb the rest. Still writes via temp file + atomic replace.
    ``value=None`` removes the key (a ``key: null`` leftover reads as absent everywhere but
    litters the file and diverges from whole-document writers that drop the key).
    """
    from ruamel.yaml.comments import CommentedMap
    # Honor escaped dots and prefer existing literal dotted keys (model IDs like ``glm-5.3``) over
    # blind splitting — same navigation as ``hermes config set``'s ``_set_nested``; otherwise
    # /model + TUI persistence wrote ``glm-5: {'3': ...}`` phantom siblings.
    # See #91607.
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_rt, config = _roundtrip_load(path)
    current = config
    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = _greedy_literal_match(dict(current), remaining) or (remaining[0], 1)
        if i + consumed == len(keys):
            if value is None:
                current.pop(seg, None)
            else:
                current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            if value is None:
                return  # nothing to remove under a missing/scalar parent
            next_value = CommentedMap()
            current[seg] = next_value
        current = next_value
        i += consumed
    _roundtrip_dump(path, yaml_rt, config)


# ruamel's round-trip dumper resolves plain scalars under YAML 1.2, where only true/false/null are
# reserved — so a str like "off" or "yes" is emitted unquoted. Every other config reader here
# (PyYAML, yaml.safe_load sites) parses under YAML 1.1, where on/off/yes/no are booleans: an
# unquoted ``approvals.mode: off`` would silently round-trip back as ``False``.
_YAML11_AMBIGUOUS_WORDS = frozenset({"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"})


# ┌── 函数 def atomic_roundtrip_yaml_save()
# │  作用：持久化 full 配置-状态字典当 preserving comments 且 ordering.
# │  参数：path: Union[str, Path], new_state: dict
# │  返回：None
# └──────────────────────────────────────────────────────────────
def atomic_roundtrip_yaml_save(path: Union[str, Path], new_state: dict) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    Comment-safe replacement for ``yaml.safe_dump(cfg, f)``: writes the whole file from
    ``new_state`` through ruamel round-trip mode so existing comments, key order, quotes and
    readable Unicode survive.
    """
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString
    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(path)
    yaml_rt, existing = _roundtrip_load(path)

    def _merge(dst: CommentedMap, src: dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict):
                current = dst.get(key)
                if not isinstance(current, CommentedMap):
                    current = CommentedMap()
                    dst[key] = current
                _merge(current, value)
            elif isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
                dst[key] = DoubleQuotedScalarString(value)
            else:
                dst[key] = value
        # Keys missing from src are deleted: ``cfg.pop("custom_prompt")`` then save must remove
        # the key from disk ("explicit absence" semantics of the old _save_cfg pattern).
        for key in [k for k in dst if k not in src]:
            del dst[key]

    _merge(existing, new_state)
    _roundtrip_dump(path, yaml_rt, existing)


# ┌── 函数 def safe_json_loads()
# │  作用：解析 JSON, 返回 *默认值* 在…上任意解析错误.
# │  参数：text: str, default: Any = None
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error."""
    # ⤷ 关键分支 try/except：捕获 (json.JSONDecodeError, TypeError, ValueError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 json.loads(text)
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        # ⤷ 出口 return：返回 default
        return default


# libyaml's CSafeLoader is ~8x faster than the pure-Python SafeLoader and a true drop-in for
# ``safe_load`` (same restricted tag set); startup parses config.yaml and every plugin manifest,
# so the slow path cost ~0.9 s of cold start.
_fast_yaml_loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader


# ┌── 函数 def fast_safe_load()
# │  作用：``YAML.safe_load`` (same 输入, same 结果) using libyaml C loader 当 available.
# │  参数：stream: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def fast_safe_load(stream: Any) -> Any:
    """``yaml.safe_load`` (same inputs, same result) using the libyaml C loader when available."""
    # ⤷ 出口 return：返回 yaml.load(stream, Loader=_fast_yaml_loader)
    return yaml.load(stream, Loader=_fast_yaml_loader)


# ┌── 函数 def _env_number()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：key: str, default, cast
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    # ⤷ 关键分支 try/except：捕获 (ValueError, TypeError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 cast(raw) if raw else default
        return cast(raw) if raw else default
    except (ValueError, TypeError):
        # ⤷ 出口 return：返回 default
        return default


# ┌── 函数 def env_int()
# │  作用：读取 environment variable 作为整数, 带兜底.
# │  参数：key: str, default: int = 0
# │  返回：int
# └──────────────────────────────────────────────────────────────
def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    # ⤷ 出口 return：返回 _env_number(key, default, int)
    return _env_number(key, default, int)


# ┌── 函数 def env_float()
# │  作用：读取 environment variable 作为浮点数, 带兜底.
# │  参数：key: str, default: float = 0.0
# │  返回：float
# └──────────────────────────────────────────────────────────────
def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    # ⤷ 出口 return：返回 _env_number(key, default, float)
    return _env_number(key, default, float)


# ┌── 函数 def env_bool()
# │  作用：读取 environment variable 作为布尔值.
# │  参数：key: str, default: bool = False
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    # ⤷ 出口 return：返回 is_truthy_value(os.getenv(key, ''), default=default)
    return is_truthy_value(os.getenv(key, ""), default=default)


_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


# ┌── 函数 def normalize_proxy_url()
# │  作用：归一化 proxy URLs 用于 httpx/aiohttp: WSL/Clash export ``socks://``, httpx needs ``socks5://``.
# │  参数：proxy_url: str | None
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp: WSL/Clash export ``socks://``, httpx needs ``socks5://``."""
    candidate = str(proxy_url or "").strip()
    if candidate.lower().startswith("socks://"):
        # ⤷ 出口 return：返回 f"socks5://{candidate[len('socks://'):]}"
        return f"socks5://{candidate[len('socks://'):]}"
    # ⤷ 出口 return：返回 candidate or None
    return candidate or None


# ┌── 函数 def normalize_proxy_env_vars()
# │  作用：Rewrite supported proxy env vars canonical URL 形态在-place.
# │  参数：无
# │  返回：None
# └──────────────────────────────────────────────────────────────
def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized


# ┌── 函数 def _parse_base_url()
# │  作用：``urlparse`` 那个 tolerates bare ``宿主[:port][/路径]`` (无 scheme).
# │  参数：base_url: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _parse_base_url(base_url: str):
    """``urlparse`` that tolerates a bare ``host[:port][/path]`` (no scheme)."""
    raw = (base_url or "").strip()
    # ⤷ 出口 return：返回 urlparse(raw if '://' in raw else f'//{raw}') if raw else No…
    return urlparse(raw if "://" in raw else f"//{raw}") if raw else None


# ┌── 函数 def _hostname_of()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：parsed
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _hostname_of(parsed) -> str:
    # ⤷ 出口 return：返回 (parsed.hostname or '').lower().rstrip('.') if parsed else '…
    return (parsed.hostname or "").lower().rstrip(".") if parsed else ""


# ┌── 函数 def base_url_hostname()
# │  作用：Lowercased hostname 用于 base URL, 或 ``""`` 若 absent.
# │  参数：base_url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def base_url_hostname(base_url: str) -> str:
    """Lowercased hostname for a base URL, or ``""`` if absent.

    Compare exact hostnames against provider hosts instead of substring-matching the raw URL:
    ``https://api.openai.com.example/v1`` or ``https://proxy.test/api.openai.com/v1`` would
    otherwise pass as native endpoints and mis-route api_mode and auth.
    """
    # ⤷ 出口 return：返回 _hostname_of(_parse_base_url(base_url))
    return _hostname_of(_parse_base_url(base_url))


# ┌── 函数 def model_forces_max_completion_tokens()
# │  作用：当 OpenAI families 那个拒绝 ``max_tokens`` (HTTP 400 ``unsupported_parameter``). 时返回 True。
# │  参数：model: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def model_forces_max_completion_tokens(model: str) -> bool:
    """True for OpenAI families that reject ``max_tokens`` (HTTP 400 ``unsupported_parameter``)."""
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    # ⤷ 出口 return：返回 m.startswith(('gpt-4o', 'gpt-4.1', 'gpt-5', 'o1', 'o3', 'o4'…
    return m.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


# ┌── 函数 def base_url_origin()
# │  作用：``(scheme, hostname, effective_port)`` 用于 base URL; ``("", "", 0)`` 在…上无 宿主/bad port.
# │  参数：base_url: str
# │  返回：tuple[str, str, int]
# └──────────────────────────────────────────────────────────────
def base_url_origin(base_url: str) -> tuple[str, str, int]:
    """``(scheme, hostname, effective_port)`` for a base URL; ``("", "", 0)`` on no host/bad port.

    Origin, not just host: ``https://h`` vs ``http://h`` and two ports on one host are different
    trust boundaries, so handing a bearer secret to a new URL must compare all three — hostname
    alone would authorise an HTTPS→HTTP downgrade. Port defaults to 443/80 so ``https://h``
    equals ``https://h:443``.
    """
    parsed = _parse_base_url(base_url)
    hostname = _hostname_of(parsed)
    if not hostname:
        # ⤷ 出口 return：返回 ('', '', 0)
        return ("", "", 0)
    scheme = (parsed.scheme or "").lower()
    # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
    try:
        port = parsed.port
    except ValueError:  # out-of-range or non-numeric port — not a usable origin
        # ⤷ 出口 return：返回 ('', '', 0)
        return ("", "", 0)
    # ⤷ 出口 return：返回 (scheme, hostname, {'https': 443, 'http': 80}.get(scheme, 0)…
    return (scheme, hostname, {"https": 443, "http": 80}.get(scheme, 0) if port is None else port)


# ┌── 函数 def base_url_host_matches()
# │  作用：当 base URL's hostname 是 ``domain`` 或 subdomain. 时返回 True。
# │  参数：base_url: str, domain: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def base_url_host_matches(base_url: str, domain: str) -> bool:
    """True when the base URL's hostname is ``domain`` or a subdomain.

    Safer than ``domain in base_url`` (``evil.com/moonshot.ai`` / ``moonshot.ai.evil`` must not
    match). Accepts bare hosts, full URLs, and URLs with paths.
    """
    hostname = base_url_hostname(base_url)
    domain = (domain or "").strip().lower().rstrip(".")
    # ⤷ 出口 return：返回 bool(hostname and domain) and (hostname == domain or hostnam…
    return bool(hostname and domain) and (hostname == domain or hostname.endswith("." + domain))
