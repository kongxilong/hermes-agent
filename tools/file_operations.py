#!/usr/bin/env python3
"""File operations (read, write, patch, search) over any terminal backend.

Every operation is a shell command run through the backend's ``execute()``, so one
implementation serves every environment (local, docker, ssh, modal, ...). Companions:
``file_operations_common`` (result dataclasses, text helpers), ``file_operations_lint``
(LintMixin), ``file_operations_search`` (SearchMixin).
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：tools/file_operations.py（约 1458 行）
# 地位：文件操作的「后端实现层」。
#
# 核心思想（很妙的一点）：
#   每一个文件操作都被实现成一条 shell 命令，然后通过执行后端的 execute() 跑。
#   这样「一份实现」就能服务所有环境：本机、docker、ssh、modal ... 全都通用。
#   —— 这是典型的「用抽象换可移植性」。
#
# 主要类型：
#   class FileOperations(ABC)         抽象基类，定义 read_file / write_file / patch_replace /
#                                     patch_v4a / delete_file / move_file / search
#   class ShellFileOperations         shell 实现（继承 LintMixin、SearchMixin）
#                                     50 个方法，比如 _cat / _head / _run_python_snippet /
#                                     _add_line_numbers / _escape_shell_arg
#
# 配套模块：file_operations_common（结果数据类）、file_operations_lint、file_operations_search
#
# 细节亮点：
#   identify_binary_bytes() / describe_binary_file() —— 判断是不是二进制文件，是就不当文本读
#   _new_sentinel / _split_segments —— 用哨兵字符串切分命令输出，避免分隔符被内容污染
# ==========================================================================
#

import base64
import binascii
import os
import re
import sys
import difflib
import hashlib
import json
import logging
import secrets
import unicodedata
from abc import ABC, abstractmethod
from typing import Optional, Dict
from pathlib import Path

from tools.binary_extensions import BINARY_EXTENSIONS
from agent.file_safety import get_write_denied_error
from tools.file_operations_common import (
    ExecuteResult, PatchResult, ReadResult, SearchResult, WriteResult,
    _UTF8_BOM, _detect_line_ending, _has_bom, _normalize_line_endings, _strip_bom,
    _strip_terminal_fence_leaks, normalize_read_pagination, normalize_search_pagination)
from tools.file_operations_lint import LINTERS_INPROC, LintMixin, _FAIL_CLOSED_INPROC_EXTS
from tools.file_operations_search import SearchMixin

logger = logging.getLogger(__name__)

# Controller home; SearchMixin reads it (tests monkeypatch it here).
_HOME = str(Path.home())

# --- Binary-content identification -------------------------------------------

_MAGIC_SIGNATURES: tuple = (
    # (prefix bytes, human name) — ordered, first match wins. Longest
    # prefixes for a shared first byte come first.
    (b"\x89PNG\r\n\x1a\n", "PNG image data"),
    (b"\xff\xd8\xff", "JPEG image data"),
    (b"GIF87a", "GIF image data"),
    (b"GIF89a", "GIF image data"),
    (b"RIFF", "RIFF container (WAV/AVI/WebP family)"),
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (also docx/xlsx/jar/apk)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip compressed data"),
    (b"BZh", "bzip2 compressed data"),
    (b"\xfd7zXZ\x00", "xz compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows PE executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"OggS", "Ogg container"),
    (b"fLaC", "FLAC audio"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00", "ISO media container (MP4/MOV family)"),  # ftyp at +4
    (b"BM", "BMP image data"),
    (b"II*\x00", "TIFF image data (little-endian)"),
    (b"MM\x00*", "TIFF image data (big-endian)"),
)


# ┌── 函数 def identify_binary_bytes()
# │  作用：尽最大努力 human 名称用于 binary 内容从 其 magic bytes; never 抛出.
# │  参数：sample: bytes
# │  返回：str
# └──────────────────────────────────────────────────────────────
def identify_binary_bytes(sample: bytes) -> str:
    """Best-effort human name for binary content from its magic bytes; never raises.
    The ISO-media entry additionally requires ``ftyp`` at offset 4 (three leading
    NULs alone are too weak a signature)."""
    for prefix, name in _MAGIC_SIGNATURES:
        if sample.startswith(prefix):
            if name.startswith("ISO media") and sample[4:8] != b"ftyp":
                continue
            # ⤷ 出口 return：返回 name
            return name
    # ⤷ 出口 return：返回 'unknown binary'
    return "unknown binary"


# ┌── 函数 def describe_binary_file()
# │  作用：One-line binary-文件 refusal naming 类型 ("PNG image data, 4.1 KB"), 从而
# │  参数：sample: Optional[bytes], file_size: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def describe_binary_file(sample: Optional[bytes], file_size: int) -> str:
    """One-line binary-file refusal naming the TYPE ("PNG image data, 4.1 KB"), so the
    model gets what-is-this in one read instead of hunting for tools it may lack."""
    kind = identify_binary_bytes(sample or b"")
    if file_size >= 1024 * 1024:
        size = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size >= 1024:
        size = f"{file_size / 1024:.1f} KB"
    else:
        size = f"{file_size} bytes"
    # ⤷ 出口 return：返回 f'Binary file ({kind}, {size}) — cannot display as text.'
    return f"Binary file ({kind}, {size}) — cannot display as text."


# ┌── 类 class FileOperations（继承自：ABC）
# │  说明：Abstract interface 用于文件 operations across 终端 backends.
# │  成员：共 8 个方法；对外可用 8 个
# └──────────────────────────────────────────────────────────────
class FileOperations(ABC):
    """Abstract interface for file operations across terminal backends."""

    # ┌── 函数 def read_file()
    # │  作用：读取文件带 pagination support.
    # │  参数：self, path: str, offset: int = 1, limit: int = 2000
    # │  返回：ReadResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """Read a file with pagination support."""

    # ┌── 函数 def read_file_raw()
    # │  作用：Whole 文件作为 plain 字符串: 无 pagination, line numbers 或 clamping.
    # │  参数：self, path: str
    # │  返回：ReadResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def read_file_raw(self, path: str) -> ReadResult:
        """Whole file as a plain string: no pagination, line numbers or clamping."""

    # ┌── 函数 def write_file()
    # │  作用：写入内容文件, 创建 directories 作为 needed.
    # │  参数：self, path: str, content: str, pre_content: Optional[str] = None
    # │  返回：WriteResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def write_file(self, path: str, content: str, pre_content: Optional[str] = None) -> WriteResult:
        """Write content to a file, creating directories as needed."""

    # ┌── 函数 def patch_replace()
    # │  作用：Replace 文本在 文件 using fuzzy matching.
    # │  参数：self, path: str, old_string: str, new_string: str, replace_all: bool = False
    # │  返回：PatchResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching."""

    # ┌── 函数 def patch_v4a()
    # │  作用：应用 V4A 格式补丁。
    # │  参数：self, patch_content: str
    # │  返回：PatchResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""

    # ┌── 函数 def delete_file()
    # │  作用：Delete 文件. 返回 WriteResult 带 .错误设置在…上 failure.
    # │  参数：self, path: str
    # │  返回：WriteResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def delete_file(self, path: str) -> WriteResult:
        """Delete a file. Returns WriteResult with .error set on failure."""

    # ┌── 函数 def move_file()
    # │  作用：Move/rename 文件. 返回 WriteResult 带 .错误设置在…上 failure.
    # │  参数：self, src: str, dst: str
    # │  返回：WriteResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move/rename a file. Returns WriteResult with .error set on failure."""

    # ┌── 函数 def search()
    # │  作用：Search 用于内容或 文件.
    # │  参数：self, pattern: str, path: str = '.', target: str = 'content', file_glob: Optional[str] = None, limit: int = 50, offset: int = 0, output_mode: str = 'content'  …共 10 个
    # │  返回：SearchResult
    # │  【语法】@abstractmethod 抽象方法：父类只声明不实现，子类必须重写。
    # └──────────────────────────────────────────────────────────────
    @abstractmethod
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0,
               order: str = "discovery") -> SearchResult:
        """Search for content or files."""


# --- Shell-based implementation ----------------------------------------------

# Image extensions (subset of binary that we can return as base64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}

# Echoed by the size probe when the path exists but is not a regular file.
# `wc -c` prints only digits, so this can never collide with a real size.
NOT_REGULAR_SENTINEL = "__hermes_not_regular__"

# Echoed by the compound read/write probes when the path does not exist. A
# compound command only reports its *last* exit status, so the missing-file
# signal that ``_probe_regular_file`` carries in ``exit 1`` travels in-band.
MISSING_SENTINEL = "__hermes_missing__"

_READ_SENTINEL_PREFIX = "__HERMES_RF_"
_WRITE_SENTINEL_PREFIX = "__HERMES_WF_"


# ┌── 函数 def _new_sentinel()
# │  作用：每次调用的 separator line 用于 compound shell probe. 128 random bits make
# │  参数：prefix: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _new_sentinel(prefix: str) -> str:
    """Per-call separator line for a compound shell probe. 128 random bits make a
    collision with file content negligible; the underscores keep the token outside
    the base64 alphabet, so a sentinel leaking into a sample segment fails base64
    validation instead of decoding into bytes."""
    # ⤷ 出口 return：返回 f'{prefix}{secrets.token_hex(16)}__'
    return f"{prefix}{secrets.token_hex(16)}__"


# ┌── 函数 def _split_segments()
# │  作用：Split compound-probe 标准输出在…上其 sentinel lines. 每个 producer (``wc``,
# │  参数：output: str, sentinel: str
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _split_segments(output: str, sentinel: str) -> list[str]:
    """Split compound-probe stdout on its sentinel lines. Every producer (``wc``,
    ``base64``, ``cut``) newline-terminates or prints nothing, so the separator is
    always ``sentinel + "\n"`` on its own line; the text after the final sentinel
    is the status segment."""
    # ⤷ 出口 return：返回 output.split(sentinel + '\n')
    return output.split(sentinel + "\n")


# ┌── 类 class ShellFileOperations（继承自：LintMixin, SearchMixin, FileOperations）
# │  说明：文件 operations over 任意终端 backend exposing ``执行(命令, cwd)``
# │  【语法】Mixin 是一种「插件式父类」：本身不能单独用，靠多继承把一组方法拼进主类。看到 xxxMixin 就当成「这批方法从别的文件来」。
# │  成员：共 50 个方法；对外可用 9 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class ShellFileOperations(LintMixin, SearchMixin, FileOperations):
    """File operations over any terminal backend exposing ``execute(command, cwd)``
    returning ``{"output": str, "returncode": int}``.

    cwd rule: every ``_exec`` prefers the LIVE ``env.cwd`` so a ``cd`` run via the
    terminal tool is picked up immediately; the init-time ``self.cwd`` is only a
    fallback for envs that don't track cwd (using it for every call once made
    patches "succeed" with a plausible diff while landing in the wrong directory).
    """

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, terminal_env, cwd: str = None
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __init__(self, terminal_env, cwd: str = None):
        self.env = terminal_env
        # Never os.getcwd(): that is the HOST path, absent inside container backends.
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or "/"
        # Ordinary executables: bool cache (hits AND misses). rg is special — it has
        # an off-PATH resolver and may be installed mid-session — so only successful
        # rg resolutions are cached (see SearchMixin._resolve_command).
        self._command_cache: Dict[str, bool] = {}
        self._rg_resolution_cache: Dict[str, str] = {}
        self._rg_modified_capability: Dict[str, Optional[str]] = {}

    # ┌── 函数 def _exec()
    # │  作用：运行 ``命令`` 在…上 backend. cwd: explicit arg → 当前活跃的 ``env.cwd`` →
    # │  参数：self, command: str, cwd: str = None, timeout: int = None, stdin_data: str = None
    # │  返回：ExecuteResult
    # └──────────────────────────────────────────────────────────────
    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Run ``command`` on the backend. cwd: explicit arg → live ``env.cwd`` →
        init-time ``self.cwd``. ``stdin_data`` is piped (bypasses ARG_MAX)."""
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data
        effective_cwd = cwd or getattr(self.env, 'cwd', None) or self.cwd
        result = self.env.execute(command, cwd=effective_cwd, **kwargs)
        exit_code = result.get("returncode", 0)
        # A stdin write failure with a clean child exit is still a failure: the
        # child never received the input.
        if result.get("stdin_error") and exit_code == 0:
            exit_code = 1
        # ⤷ 出口 return：返回 ExecuteResult(stdout=result.get('output', ''), exit_code=exi…
        return ExecuteResult(stdout=result.get("output", ""), exit_code=exit_code)

    # ┌── 函数 def _has_command()
    # │  作用：检查若 命令 exists 在 environment (cached); rg goes through
    # │  参数：self, cmd: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _has_command(self, cmd: str) -> bool:
        """Check if a command exists in the environment (cached); rg goes through
        the resolver so a mid-session install becomes visible."""
        if cmd == "rg":
            # ⤷ 出口 return：返回 self._resolve_command(cmd) is not None
            return self._resolve_command(cmd) is not None
        if cmd not in self._command_cache:
            result = self._exec(f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'")
            self._command_cache[cmd] = result.stdout.strip() == 'yes'
        # ⤷ 出口 return：返回 self._command_cache[cmd]
        return self._command_cache[cmd]

    # ┌── 函数 def _cat()
    # │  作用：``cat`` 文件带 标准错误 silenced (missing 文件 → non-zero exit).
    # │  参数：self, path: str
    # │  返回：ExecuteResult
    # └──────────────────────────────────────────────────────────────
    def _cat(self, path: str) -> ExecuteResult:
        """``cat`` the file with stderr silenced (missing file → non-zero exit)."""
        # ⤷ 出口 return：返回 self._exec(f'cat {self._escape_shell_arg(path)} 2>/dev/null'…
        return self._exec(f"cat {self._escape_shell_arg(path)} 2>/dev/null")

    # ┌── 函数 def _head()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, path: str, nbytes: int
    # │  返回：ExecuteResult
    # └──────────────────────────────────────────────────────────────
    def _head(self, path: str, nbytes: int) -> ExecuteResult:
        # ⤷ 出口 return：返回 self._exec(f'head -c {nbytes} {self._escape_shell_arg(path)}…
        return self._exec(f"head -c {nbytes} {self._escape_shell_arg(path)} 2>/dev/null")

    # ┌── 函数 def _run_python_snippet()
    # │  作用：运行 ``snippet`` via backend's ``python3``, retrying 带 ``python``
    # │  参数：self, snippet: str
    # │  返回：ExecuteResult
    # └──────────────────────────────────────────────────────────────
    def _run_python_snippet(self, snippet: str) -> ExecuteResult:
        """Run ``snippet`` via the backend's ``python3``, retrying with ``python``
        when only that name exists (Windows / older systems)."""
        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def _sample_file_bytes()
    # │  作用：First ``length`` raw bytes, base64-wrapped 从而它们 survive 终端
    # │  参数：self, path: str, length: int = 1000
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _sample_file_bytes(self, path: str, length: int = 1000):
        """First ``length`` raw bytes, base64-wrapped so they survive the terminal
        transport (which decodes stdout with ``errors="replace"`` and manufactures
        U+FFFD for every undecodable byte, including a multibyte char cut in half
        by ``head -c``). None when no clean base64 came back (no ``base64`` binary);
        callers then fall back to the text heuristic.

        Wrapping the sample in base64 lets the original bytes survive the transport, so binary detection can
        happen at the byte layer where it is well-defined (#80308 and friends).
        """
        result = self._exec(f"head -c {length} {self._escape_shell_arg(path)} 2>/dev/null | base64")
        if result.exit_code != 0:
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 self._decode_base64_sample(result.stdout)
        return self._decode_base64_sample(result.stdout)

    # ┌── 函数 def _decode_base64_sample()
    # │  作用：Decode one ``head -c N | base64`` sample. Whitespace-joins whole 文本
    # │  参数：text: str
    # │  返回：Optional[bytes]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_base64_sample(text: str) -> Optional[bytes]:
        """Decode one ``head -c N | base64`` sample. Whitespace-joins the whole text
        first (``base64`` wraps at 76 columns), so callers hand over exactly one
        segment; anything else fails validation → None (legacy text heuristic)."""
        encoded = "".join(_strip_terminal_fence_leaks(text).split())
        if not encoded:
            # ⤷ 出口 return：返回 b''
            return b""
        if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded):
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 关键分支 try/except：捕获 (binascii.Error, ValueError)，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 base64.b64decode(encoded, validate=True)
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            # ⤷ 出口 return：返回 None
            return None

    # ┌── 函数 def _is_likely_binary_bytes()
    # │  作用：Byte-layer binary detection: 文本 iff valid UTF-8, allowing one incomplete
    # │  参数：sample: bytes
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_likely_binary_bytes(sample: bytes) -> bool:
        """Byte-layer binary detection: text iff valid UTF-8, allowing one incomplete
        multibyte sequence at the very end (artifact of the byte-boundary cut).
        NUL bytes or mid-stream invalid UTF-8 stay read-only so a read→edit→write
        round-trip never rewrites undecodable bytes as U+FFFD; a file that
        legitimately CONTAINS U+FFFD is valid UTF-8 and reads as text.

        See #80308.
        """
        if not sample:
            # ⤷ 出口 return：返回 False
            return False
        if b"\x00" in sample:
            # ⤷ 出口 return：返回 True
            return True
        # ⤷ 关键分支 try/except：捕获 UnicodeDecodeError，出错走兜底而不是崩溃。
        try:
            sample.decode("utf-8")
            # ⤷ 出口 return：返回 False
            return False
        except UnicodeDecodeError as exc:
            # UTF-8 sequences are at most 4 bytes: an error starting in the
            # last 3 bytes with a clean prefix is a boundary cut, not binary.
            if exc.start >= len(sample) - 3:
                # ⤷ 关键分支 try/except：捕获 UnicodeDecodeError，出错走兜底而不是崩溃。
                try:
                    sample[: exc.start].decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    pass
            # ⤷ 出口 return：返回 True
            return True

    # ┌── 函数 def _is_likely_binary()
    # │  作用：Legacy 文本-layer binary 检查: extension, 否则 >30% non-printable chars.
    # │  参数：self, path: str, content_sample: str = None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _is_likely_binary(self, path: str, content_sample: str = None) -> bool:
        """Legacy text-layer binary check: extension, else >30% non-printable chars."""
        if os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
            # ⤷ 出口 return：返回 True
            return True
        if content_sample:
            # Undecodable bytes arrive as U+FFFD ("printable", so the ratio misses
            # them); treat as binary so a round-trip can't write back mojibake.
            if "\ufffd" in content_sample[:1000]:
                # ⤷ 出口 return：返回 True
                return True
            non_printable = sum(1 for c in content_sample[:1000] if ord(c) < 32 and c not in '\n\r\t')
            # ⤷ 出口 return：返回 non_printable / min(len(content_sample), 1000) > 0.3
            return non_printable / min(len(content_sample), 1000) > 0.30
        # ⤷ 出口 return：返回 False
        return False

    # ┌── 函数 def _is_image()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, path: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _is_image(self, path: str) -> bool:
        # ⤷ 出口 return：返回 os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS

    # ┌── 函数 def _add_line_numbers()
    # │  作用：Prefix 每个 line 带 compact ``<n>|`` gutter, clamping long lines. 不
    # │  参数：self, content: str, start_line: int = 1
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Prefix each line with a compact ``<n>|`` gutter, clamping long lines. Not
        fixed-width: padding cost ~16% more tokens per line for no accuracy gain in
        A/B, while dropping numbers regressed line-referencing."""
        from tools.tool_output_limits import get_max_line_length
        max_line_length = get_max_line_length()
        # A trailing newline terminates the final line — it does not start a new,
        # empty one. Splitting without dropping it rendered a phantom "<N+1>|"
        # gutter line on every newline-terminated file (`cat -n` semantics).
        # Exactly ONE terminator is dropped, so a genuinely selected trailing
        # blank line in a page keeps its own number.
        if content.endswith('\n'):
            content = content[:-1]
        # ⤷ 出口 return：返回 '\n'.join((f"{i}|{(line if len(line) <= max_line_length else…
        return '\n'.join(
            f"{i}|{line if len(line) <= max_line_length else line[:max_line_length] + '... [truncated]'}"
            for i, line in enumerate(content.split('\n'), start=start_line))

    # ┌── 函数 def _expand_path()
    # │  作用：Expand ``~`` / ``~user`` via backend's shell (其主目录, 不
    # │  参数：self, path: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _expand_path(self, path: str) -> str:
        """Expand ``~`` / ``~user`` via the backend's shell (its HOME, not the
        host's). Must run BEFORE shell escaping — ~ doesn't expand in quotes."""
        if not path or not path.startswith('~'):
            # ⤷ 出口 return：返回 path
            return path
        result = self._exec("echo $HOME")
        if result.exit_code == 0 and result.stdout.strip():
            home = result.stdout.strip()
            if path == '~':
                # ⤷ 出口 return：返回 home
                return home
            if path.startswith('~/'):
                # ⤷ 出口 return：返回 home + path[1:]
                return home + path[1:]
            # ~username: validate and expand ONLY that token, so neither "~; rm -rf /"
            # nor "~user/$(malicious)" reaches the shell.
            rest = path[1:]
            slash_idx = rest.find('/')
            username = rest[:slash_idx] if slash_idx >= 0 else rest
            if username and re.fullmatch(r'[a-zA-Z0-9._-]+', username):
                expand_result = self._exec(f"echo ~{username}")
                if expand_result.exit_code == 0 and expand_result.stdout.strip():
                    # ⤷ 出口 return：返回 expand_result.stdout.strip() + path[1 + len(username):]
                    return expand_result.stdout.strip() + path[1 + len(username):]
        # ⤷ 出口 return：返回 path
        return path

    # ┌── 函数 def _escape_shell_arg()
    # │  作用：Single-quote ``arg`` 用于 shell. 在…上 Windows, native drive 路径且
    # │  参数：self, arg: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _escape_shell_arg(self, arg: str) -> str:
        """Single-quote ``arg`` for the shell. On Windows, native drive paths and
        mixed MSYS leftovers are first rewritten to the Git Bash ``/c/Users/x``
        form via the env-layer ``_bash_safe_path`` (bash eats backslashes; MSYS
        mangles drive paths), so shell file ops and the terminal ``cd`` agree."""
        from tools.environments.local import _bash_safe_path
        # ⤷ 出口 return：返回 "'" + _bash_safe_path(arg).replace("'", '\'"\'"\'') + "'"
        return "'" + _bash_safe_path(arg).replace("'", "'\"'\"'") + "'"

    # ┌── 函数 def _escape_native_tool_arg()
    # │  作用：Quote 路径用于 NATIVE Windows binary (rg, node, git ...): 那些 don't
    # │  参数：self, arg: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _escape_native_tool_arg(self, arg: str) -> str:
        """Quote a path for a NATIVE Windows binary (rg, node, git ...): those don't
        understand the MSYS ``/c/...`` form and Hermes disables MSYS argument
        conversion, so nothing translates it back (→ ``os error 3``). ``C:/Users/x``
        is accepted by every layer. Identical to ``_escape_shell_arg`` off Windows."""
        from tools.environments.local import _IS_WINDOWS, _msys_to_windows_path
        if _IS_WINDOWS and arg:
            arg = _msys_to_windows_path(arg).replace("\\", "/")
        # ⤷ 出口 return：返回 "'" + arg.replace("'", '\'"\'"\'') + "'"
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    # ┌── 函数 def _atomic_write()
    # │  作用：写入 ``内容`` atomically: 标准输入 → temp 文件在 SAME directory →
    # │  参数：self, path: str, content: str
    # │  返回：'ExecuteResult'
    # └──────────────────────────────────────────────────────────────
    def _atomic_write(self, path: str, content: str) -> "ExecuteResult":
        """Write ``content`` atomically: stdin → temp file in the SAME directory →
        ``mv -f`` (same-FS rename; cross-device ``mv`` is copy+unlink, NOT atomic).
        ``mkdir -p`` folded in. Exit 0 = swap happened; non-zero = original intact.

        Symlink targets are resolved first (replacing the link would orphan the
        target) and the temp dir recomputed from the RESOLVED target. Existing
        target: mode copied via ``stat`` (GNU ``-c%a`` / BSD ``-f%Lp``) + ``chmod``
        (``chmod --reference`` is GNU-only). New target: ``chmod "=rw"`` AFTER cat
        gives umask-default perms instead of mktemp's 0600 — not ``$(umask)``
        arithmetic (zsh parses leading-zero constants as decimal), quoted so zsh
        doesn't =word-expand. ``trap ... EXIT`` removes the temp on every failure.
        """
        q_path = self._escape_shell_arg(path)
        q_parent = self._escape_shell_arg(os.path.dirname(path) or ".")
        tmpl = self._escape_shell_arg(".hermes-tmp.XXXXXX")
        script = (
            "set -e; "
            # One shell script, fully quoted. Notes: - `mkdir -p "$d"` is folded in here so the parent
            # directory is created in the same subprocess that writes the temp file — saves one entire
            # subprocess spawn vs. a separate mkdir call. - `mktemp` lands the temp in the target's own dir
            # (-p) so `mv` is same-FS atomic; we fall back to a PID-stamped name if the backend lacks mktemp
            # (rare; busybox/macOS/Linux all ship it). - `chmod --reference` is GNU-only, so we read the
            # octal mode with `stat` (GNU `-c%a` or BSD `-f%Lp`) and `chmod` it explicitly; silent
            # best-effort — a perms-copy failure must not abort the write (the file then lands at mktemp's
            # 0600, same as pre-fix). - brand-new targets get `chmod "=rw"` — the POSIX who-less symbolic
            # form, which sets rw minus the process umask (e.g. 0644 under umask 022) instead of mktemp's
            # hardcoded 0600 (#70856). Deliberately NOT shell arithmetic on `$(umask)`: zsh (reachable via
            # _find_bash's $SHELL fallback) parses leading-zero constants as decimal and silently computes a
            # garbage mode, while `chmod "=rw"` is spec-identical in bash/dash/ash/zsh and degrades to 0600
            # (pre-fix behavior) if an exotic chmod rejects it. - `trap ... EXIT` guarantees the temp is
            # removed on every error path (cat failure, mv failure, signal) but NOT after a successful mv
            # (the temp no longer exists by then). - we `cat >` the temp, then `mv -f` it over the target.
            f"d={q_parent}; t={q_path}; "
            'if [ -L "$t" ]; then '
            'rt="$(readlink -f "$t" 2>/dev/null || realpath "$t" 2>/dev/null || true)"; '
            '[ -n "$rt" ] && { t="$rt"; d="$(dirname "$t")"; }; '
            "fi; "
            'mkdir -p "$d"; '
            'tmp="$(mktemp -p "$d" ' + tmpl + ' 2>/dev/null '
            '|| mktemp "$d/.hermes-tmp.$$.XXXXXX" 2>/dev/null '
            '|| { tmp="$d/.hermes-tmp.$$"; : > "$tmp" && echo "$tmp"; })"; '
            '[ -n "$tmp" ] || { echo "atomic write: could not create temp file" >&2; exit 1; }; '
            "trap 'rm -f \\\"$tmp\\\"' EXIT; "
            'if [ -e "$t" ]; then '
            'm="$(stat -c%a "$t" 2>/dev/null || stat -f%Lp "$t" 2>/dev/null || true)"; '
            '[ -n "$m" ] && chmod "$m" "$tmp" 2>/dev/null || true; '
            "fi; "
            'cat > "$tmp"; '
            # new file: umask-default perms instead of mktemp's 0600 (#70856). Runs AFTER cat so a
            # write-masking umask can't EACCES the stream; quoted "=rw" so zsh doesn't =word-expand it.
            'if [ ! -e "$t" ]; then chmod "=rw" "$tmp" 2>/dev/null || true; fi; '
            'mv -f "$tmp" "$t"; '
            "trap - EXIT")
        # ⤷ 出口 return：返回 self._exec(script, stdin_data=content)
        return self._exec(script, stdin_data=content)

    # ┌── 函数 def _file_has_bom()
    # │  作用：是否 在…上-disk 文件 starts 带 UTF-8 BOM. ALWAYS probes disk:。
    # │  参数：self, path: str, pre_content: Optional[str] = None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _file_has_bom(self, path: str, pre_content: Optional[str] = None) -> bool:
        """Whether the on-disk file starts with a UTF-8 BOM. ALWAYS probes disk:
        ``pre_content`` usually comes from ``read_file_raw``, which strips BOMs, so
        trusting it would silently drop the marker on rewrite. Missing → False."""
        head_result = self._head(path, 3)
        # ⤷ 出口 return：返回 head_result.exit_code == 0 and _has_bom(head_result.stdout)
        return head_result.exit_code == 0 and _has_bom(head_result.stdout)

    # ┌── 函数 def _unified_diff()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, old_content: str, new_content: str, filename: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        # ⤷ 出口 return：返回 ''.join(difflib.unified_diff(old_content.splitlines(keepends…
        return ''.join(difflib.unified_diff(
            old_content.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}", tofile=f"b/{filename}"))

    # --- READ ---------------------------------------------------------------

    # ┌── 函数 def _not_regular_error()
    # │  作用：错误用于路径那个 exists 但会 block 若读取.
    # │  参数：path: str
    # │  返回：ReadResult
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _not_regular_error(path: str) -> ReadResult:
        """Error for a path that exists but would block if read."""
        # ⤷ 出口 return：返回 ReadResult(error=f"Cannot read '{path}': not a regular file …
        return ReadResult(error=(
            f"Cannot read '{path}': not a regular file (directory, FIFO, "
            "socket, or device). Reading it could block indefinitely."))

    # ┌── 函数 def _probe_regular_file()
    # │  作用：Byte size 的 REGULAR 文件: ``(file_size, status)`` 带 status ``"ok"``,
    # │  参数：self, path: str
    # │  返回：tuple[int, str]
    # └──────────────────────────────────────────────────────────────
    def _probe_regular_file(self, path: str) -> tuple[int, str]:
        """Byte size of a REGULAR file: ``(file_size, status)`` with status ``"ok"``,
        ``"missing"``, ``"not_regular"`` or ``"bad_size"`` (unparseable ``wc``).
        ``wc -c <`` on a writer-less FIFO/socket//dev/zero blocks forever and a
        name-based blocklist can't cover a FIFO (a file TYPE at any path); ``[ -f ]``
        is a stat (symlinks followed) so it answers without touching content."""
        arg = self._escape_shell_arg(path)
        # A missing path ECHOES its sentinel: a non-zero exit with no sentinel means the shell itself did
        # not run (container still starting, removed out-of-band, transport down) — not a missing file.
        # Reporting that as "File not found" made the model trust a false negative for the whole session.
        stat_result = self._exec(
            f"if [ -f {arg} ]; then wc -c < {arg} 2>/dev/null; "
            f"elif [ -e {arg} ]; then echo {NOT_REGULAR_SENTINEL}; "
            f"else echo {MISSING_SENTINEL}; fi")
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout).strip()
        if stat_output == MISSING_SENTINEL:
            # ⤷ 出口 return：返回 (0, 'missing')
            return 0, "missing"
        if stat_output == NOT_REGULAR_SENTINEL:
            # ⤷ 出口 return：返回 (0, 'not_regular')
            return 0, "not_regular"
        if stat_result.exit_code != 0:
            # ⤷ 出口 return：返回 (0, 'env_unavailable')
            return 0, "env_unavailable"
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 (int(stat_output), 'ok')
            return int(stat_output), "ok"
        except ValueError:
            # ⤷ 出口 return：返回 (0, 'bad_size')
            return 0, "bad_size"

    # ┌── 函数 def _env_unavailable_error()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self, path: str
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _env_unavailable_error(self, path: str) -> ReadResult:
        # ⤷ 出口 return：返回 ReadResult(error=f'Terminal environment unavailable: could n…
        return ReadResult(error=(f"Terminal environment unavailable: could not stat {path} "
                                 "(the sandbox may still be starting or was removed). Retry shortly."))

    # ┌── 函数 def _detect_binary()
    # │  作用：``(is_binary, sample_bytes)`` — byte-layer detection 当 transport
    # │  参数：self, path: str
    # │  返回：tuple[bool, Optional[bytes]]
    # └──────────────────────────────────────────────────────────────
    def _detect_binary(self, path: str) -> tuple[bool, Optional[bytes]]:
        """``(is_binary, sample_bytes)`` — byte-layer detection when the transport
        allows (base64 sample), else the legacy text heuristic (sample is None)."""
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            # ⤷ 出口 return：返回 (ext_binary or self._is_likely_binary_bytes(sample_bytes), s…
            return ext_binary or self._is_likely_binary_bytes(sample_bytes), sample_bytes
        sample_output = _strip_terminal_fence_leaks(self._head(path, 1000).stdout)
        # ⤷ 出口 return：返回 (self._is_likely_binary(path, sample_output), None)
        return self._is_likely_binary(path, sample_output), None

    # UTF-16 rescue: trust a BOM first, then zero-byte PARITY (not density, so
    # mixed Latin/CJK still detects): zeros at odd indices → UTF-16 LE, at even
    # → BE; both parities or a single zero → real binary. Legacy 8-bit
    # encodings (GBK, Big5) are never guessed — a wrong silent guess is worse
    # than a clear refusal.
    # UTF-16 rescue constants (ported from MoonshotAI/kimi-code#2647, detection derived from VS Code's
    # encoding sniffer): sample the leading bytes; trust a BOM first, then a zero-byte parity heuristic —
    # zeros clustering at odd indices mean UTF-16 LE (`0xAA 0x00`), at even indices UTF-16 BE (`0x00 0xAA`).
    _UTF16_MAX_BYTES = 10 * 1024 * 1024
    _UTF16_SAMPLE_BYTES = 512

    # ┌── 函数 def _try_read_utf16()
    # │  作用：读取 ``路径`` 作为 UTF-16 transcoded UTF-8, 或 None (caller falls back
    # │  参数：self, path: str, offset: int, limit: int, file_size: int
    # │  返回：'Optional[ReadResult]'
    # └──────────────────────────────────────────────────────────────
    def _try_read_utf16(self, path: str, offset: int, limit: int,
                        file_size: int) -> "Optional[ReadResult]":
        """Read ``path`` as UTF-16 transcoded to UTF-8, or None (caller falls back
        to the binary-file error). Skips known-binary extensions and files over
        10 MiB. ``path`` must already be expanded."""
        if os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS or file_size > self._UTF16_MAX_BYTES:
            # ⤷ 出口 return：返回 None
            return None
        snippet = (
            "import sys, json, os\n"
            f"p = {path!r}\n"
            f"offset = {int(offset)}\n"
            f"limit = {int(limit)}\n"
            f"MAX = {self._UTF16_MAX_BYTES}\n"
            f"SAMPLE = {self._UTF16_SAMPLE_BYTES}\n"
            "try:\n"
            "    size = os.path.getsize(p)\n"
            "    if size > MAX:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    with open(p, 'rb') as f:\n"
            "        data = f.read()\n"
            "    sample = data[:SAMPLE]\n"
            "    enc = None\n"
            "    if sample[:2] == b'\\xfe\\xff':\n"
            "        enc = 'utf-16-be'\n"
            "    elif sample[:2] == b'\\xff\\xfe':\n"
            "        enc = 'utf-16-le'\n"
            "    else:\n"
            "        odd = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)\n"
            "        even = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)\n"
            "        if even == 0 and odd >= 2:\n"
            "            enc = 'utf-16-le'\n"
            "        elif odd == 0 and even >= 2:\n"
            "            enc = 'utf-16-be'\n"
            "    if enc is None:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    text = data.decode(enc, 'replace')\n"
            "    if text[:1] == '\\ufeff':\n"
            "        text = text[1:]\n"
            "    text = text.replace('\\r\\n', '\\n')\n"
            "    lines = text.split('\\n')\n"
            "    total = len(lines)\n"
            "    sel = lines[offset - 1: offset - 1 + limit]\n"
            "    out = {'total_lines': total, 'encoding': enc,\n"
            "           'content': '\\n'.join(sel)}\n"
            "    print('HERMES_UTF16:OK')\n"
            "    print(json.dumps(out, ensure_ascii=True))\n"
            "except Exception:\n"
            "    print('HERMES_UTF16:NO'); sys.exit(0)\n")
        result = self._run_python_snippet(snippet)
        stdout = _strip_terminal_fence_leaks(result.stdout or "")
        marker = stdout.find("HERMES_UTF16:OK")
        if result.exit_code != 0 or marker < 0:
            # ⤷ 出口 return：返回 None
            return None
        payload = stdout[marker + len("HERMES_UTF16:OK"):].strip()
        # ⤷ 关键分支 try/except：捕获 (ValueError, KeyError, TypeError)，出错走兜底而不是崩溃。
        try:
            data = json.loads(payload.split("\n", 1)[0] if "\n" in payload else payload)
            content = data["content"]
            total_lines = int(data["total_lines"])
            encoding = str(data.get("encoding", "utf-16"))
        except (ValueError, KeyError, TypeError):
            # ⤷ 出口 return：返回 None
            return None
        end_line = offset + limit - 1
        truncated = total_lines > end_line
        hint_parts = [f"Transcoded from {encoding.upper()} to UTF-8 for display. "
                      "Text edits via patch/write_file would re-encode as UTF-8."]
        if truncated:
            hint_parts.append(
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)")
        # ⤷ 出口 return：返回 ReadResult(content=self._add_line_numbers(content, offset), …
        return ReadResult(
            content=self._add_line_numbers(content, offset), total_lines=total_lines,
            file_size=file_size, truncated=truncated, hint=" ".join(hint_parts))

    # ┌── 函数 def read_file()
    # │  作用：读取文件带 pagination, binary detection, 且 line numbers.
    # │  参数：self, path: str, offset: int = 1, limit: int = 2000
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """Read a file with pagination, binary detection, and line numbers.

        ``offset`` is 1-indexed; ``limit`` is clamped by ``normalize_read_pagination``.
        One shell round-trip answers every question the read needs (existence, size,
        binary sample, page, line count, trailing newline; see ``_read_probe_cmd``).
        An unparseable reply falls back to ``_read_file_sequential`` (one probe per
        question), so an exotic shell can never do worse than before. On a local
        POSIX environment the read never touches the shell (``_read_file_native``).
        """
        path = self._expand_path(path)  # before shell escaping: ~ doesn't expand in quotes
        offset, limit = normalize_read_pagination(offset, limit)

        if self._native_read_enabled():
            # ⤷ 出口 return：返回 self._read_file_native(path, offset, limit)
            return self._read_file_native(path, offset, limit)

        # Images / known-binary extensions never inline content; the sequential
        # path stops at the probes for them, so don't stream their bytes.
        if self._is_image(path) or os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
            # ⤷ 出口 return：返回 self._read_file_sequential(path, offset, limit)
            return self._read_file_sequential(path, offset, limit)

        from tools.tool_output_limits import get_max_line_length
        line_clamp_bytes = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1
        sentinel = _new_sentinel(_READ_SENTINEL_PREFIX)
        probe = self._exec(self._read_probe_cmd(path, offset, end_line, line_clamp_bytes, sentinel))
        output = probe.stdout or ""

        if sentinel not in output:
            # Single-line replies: the path is missing or not a regular file.
            marker = _strip_terminal_fence_leaks(output).strip()
            if marker == MISSING_SENTINEL:
                return self._read_file_missing(path, offset, limit)
            if marker == NOT_REGULAR_SENTINEL:
                return self._not_regular_error(path)
            logger.debug(
                "read_file: compound probe reply for %s has no sentinel "
                "(exit %s, %d chars); falling back to sequential probes",
                path, probe.exit_code, len(output))
            return self._read_file_sequential(path, offset, limit)

        segments = _split_segments(output, sentinel)
        if probe.exit_code != 0 or len(segments) != 6:
            logger.debug(
                "read_file: compound probe for %s returned exit %s with %d "
                "segments (want 6); falling back to sequential probes",
                path, probe.exit_code, len(segments))
            return self._read_file_sequential(path, offset, limit)
        size_seg, sample_seg, page_seg, wc_seg, tail_seg, status_seg = segments

        status = _strip_terminal_fence_leaks(status_seg).split()
        # ⤷ 关键分支 try/except：捕获 (IndexError, ValueError)，出错走兜底而不是崩溃。
        try:
            sample_rc, read_rc = int(status[0]), int(status[1])
        except (IndexError, ValueError):
            logger.debug(
                "read_file: compound probe for %s has unparseable status %r; "
                "falling back to sequential probes", path, status_seg[-40:])
            return self._read_file_sequential(path, offset, limit)

        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            file_size = int(_strip_terminal_fence_leaks(size_seg).strip())
        except ValueError:
            file_size = 0

        # Byte-layer binary detection when base64 was available, else the legacy
        # text heuristic over a plain sample (one extra round-trip, shells without base64).
        sample_bytes = self._decode_base64_sample(sample_seg) if sample_rc == 0 else None
        if sample_bytes is not None:
            is_binary = self._is_likely_binary_bytes(sample_bytes)
        else:
            logger.debug(
                "read_file: no usable base64 sample for %s (base64 exit %s); "
                "paying one extra round-trip for the text heuristic", path, sample_rc)
            sample_output = _strip_terminal_fence_leaks(self._head(path, 1000).stdout)
            is_binary = self._is_likely_binary(path, sample_output)
        if is_binary:
            return self._read_binary_file(path, offset, limit, file_size, sample_bytes)

        if read_rc != 0:
            return ReadResult(error=f"Failed to read file: {_strip_terminal_fence_leaks(page_seg)}")
        read_output = _strip_terminal_fence_leaks(page_seg)
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_seg).strip())
        except ValueError:
            total_lines = 0
        tail_flag = _strip_terminal_fence_leaks(tail_seg).strip()
        file_ends_with_newline = tail_flag == "1" if tail_flag in ("0", "1") else None
        # ⤷ 出口 return：返回 self._assemble_read_result(read_output, offset=offset, end_l…
        return self._assemble_read_result(
            read_output, offset=offset, end_line=end_line, total_lines=total_lines,
            file_size=file_size, file_ends_with_newline=file_ends_with_newline)

    # ┌── 函数 def _native_read_enabled()
    # │  作用：是否 ``read_file`` 且 ``search_files`` 可以 bypass shell: 仅 POSIX + ``LocalEnvironment``。
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _native_read_enabled(self) -> bool:
        """Whether ``read_file`` and ``search_files`` may bypass the shell: only POSIX + ``LocalEnvironment``
        (file is on this host, path already native; Windows keeps the shell path since
        file_operations holds Git-Bash-style paths there). ``HERMES_NATIVE_FILE_READ=0``
        turns the fast path off."""
        flag = os.environ.get("HERMES_NATIVE_FILE_READ", "1").strip().lower()
        if flag in ("0", "false", "no", "off"):
            # ⤷ 出口 return：返回 False
            return False
        # Same "is this env the local host" test the LSP path uses; isinstance is
        # microseconds and self.env is never rebound, so nothing to memoize.
        # ⤷ 出口 return：返回 sys.platform != 'win32' and self._lsp_local_only()
        return sys.platform != "win32" and self._lsp_local_only()

    # ┌── 函数 def _read_file_native()
    # │  作用：``read_file`` 不 shell — same contract 作为 shell 路径, byte 用于
    # │  参数：self, path: str, offset: int, limit: int
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _read_file_native(self, path: str, offset: int, limit: int) -> ReadResult:
        """``read_file`` without a shell — same contract as the shell path, byte for
        byte. ``os.stat`` is the ``[ -f ]`` guard (a stat, never an open, so FIFOs and
        devices are refused before anything touches them); the first 1000 bytes drive
        the byte-layer binary check; the page is produced exactly as
        ``sed -n 'a,bp' | cut -b1-N`` prints it (each line clamped to N bytes and
        newline-terminated) then decoded with errors="replace" like the transport.
        One chunked pass counts lines and collects the page, so neither the file nor
        a pathological line is ever held whole. ``path`` is already expanded; any
        unexpected OSError hands over to the shell path."""
        import stat as _stat

        full = path if os.path.isabs(path) else os.path.join(
            getattr(self.env, "cwd", None) or self.cwd, path)
        # ⤷ 关键分支 try/except：捕获 (FileNotFoundError, NotADirectoryError)、OSError，出错走兜底而不是崩溃。
        try:
            st = os.stat(full)
        except (FileNotFoundError, NotADirectoryError):
            # ⤷ 出口 return：返回 self._read_file_missing(path, offset, limit)
            return self._read_file_missing(path, offset, limit)
        except OSError:
            return self._read_file_sequential(path, offset, limit)
        if not _stat.S_ISREG(st.st_mode):
            # ⤷ 出口 return：返回 self._not_regular_error(path)
            return self._not_regular_error(path)
        file_size = st.st_size
        if self._is_image(path):
            # ⤷ 出口 return：返回 self._image_redirect_result(file_size)
            return self._image_redirect_result(file_size)

        from tools.tool_output_limits import get_max_line_length
        clamp = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1

        page: list[bytes] = []
        total_lines = 0
        lineno = 1              # the line currently being scanned
        kept = bytearray()      # first ``clamp`` bytes of that line
        have_partial = False    # that line has bytes but no newline yet
        last_byte = b""
        # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
        try:
            with open(full, "rb") as fh:
                sample = fh.read(1000)
                ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
                if ext_binary or self._is_likely_binary_bytes(sample):
                    return self._read_binary_file(path, offset, limit, file_size, sample)
                fh.seek(0)
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    last_byte = chunk[-1:]
                    if lineno > end_line:
                        # Past the window: only the line count and trailing byte
                        # are needed, so let memchr do it instead of per-line work.
                        total_lines += chunk.count(b"\n")
                        have_partial = chunk[-1:] != b"\n"
                        continue
                    pos, n = 0, len(chunk)
                    while pos < n:
                        nl = chunk.find(b"\n", pos)
                        in_page = offset <= lineno <= end_line
                        if nl < 0:
                            if in_page and len(kept) < clamp:
                                kept += chunk[pos:pos + (clamp - len(kept))]
                            have_partial = True
                            break
                        if in_page:
                            if len(kept) < clamp:
                                kept += chunk[pos:min(nl, pos + (clamp - len(kept)))]
                            page.append(bytes(kept) + b"\n")
                        kept = bytearray()
                        have_partial = False
                        total_lines += 1
                        lineno += 1
                        pos = nl + 1
        except OSError:
            return self._read_file_sequential(path, offset, limit)
        if have_partial and offset <= lineno <= end_line:
            # ``sed`` prints a final line that lacks a newline; ``cut`` adds one.
            page.append(bytes(kept) + b"\n")

        read_output = _strip_terminal_fence_leaks(b"".join(page).decode("utf-8", errors="replace"))
        # ⤷ 出口 return：返回 self._assemble_read_result(read_output, offset=offset, end_l…
        return self._assemble_read_result(
            read_output, offset=offset, end_line=end_line, total_lines=total_lines,
            file_size=file_size,
            file_ends_with_newline=(last_byte == b"\n") if file_size else None)

    # ┌── 函数 def _image_redirect_result()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：file_size: int
    # │  返回：ReadResult
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _image_redirect_result(file_size: int) -> ReadResult:
        # ⤷ 出口 return：返回 ReadResult(is_image=True, is_binary=True, file_size=file_siz…
        return ReadResult(
            is_image=True, is_binary=True, file_size=file_size,
            hint=(
                "Image file detected. Automatically redirected to vision_analyze tool. "
                "Use vision_analyze with this file path to inspect the image contents."))

    # ┌── 函数 def _read_probe_cmd()
    # │  作用：One shell 命令 answering 每个 question ``read_file`` asks: six
    # │  参数：self, path: str, offset: int, end_line: int, line_clamp_bytes: int, sentinel: str
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _read_probe_cmd(self, path: str, offset: int, end_line: int,
                        line_clamp_bytes: int, sentinel: str) -> str:
        """One shell command answering every question ``read_file`` asks: six
        segments each closed by a ``sentinel`` line — byte size, base64 of the first
        1000 bytes, the ``sed | cut`` page, ``wc -l``, whether the last byte is a
        newline, then the base64 and page pipeline statuses. Probes run only inside
        ``[ -f ]`` (stat-not-open, like ``_probe_regular_file``) so a FIFO/device never
        reaches ``head``/``sed``. A missing path echoes ``MISSING_SENTINEL`` (a compound
        command only reports its last status). Every stage silences stderr: the local
        backend merges stderr into stdout and a stray diagnostic would land inside a
        segment. The byte clamp is ``4 * max_line_length + 1``; see ``_read_file_sequential``."""
        arg = self._escape_shell_arg(path)
        mark = f"echo {sentinel}"
        # ⤷ 出口 return：返回 f"""if [ -f {arg} ]; then wc -c < {arg} 2>/dev/null; {mark};…
        return (
            f"if [ -f {arg} ]; then "
            f"wc -c < {arg} 2>/dev/null; {mark}; "
            f"head -c 1000 {arg} 2>/dev/null | base64 2>/dev/null; __hs=$?; {mark}; "
            f"sed -n '{offset},{end_line}p' {arg} 2>/dev/null"
            f" | cut -b1-{line_clamp_bytes} 2>/dev/null; __hr=$?; {mark}; "
            f"wc -l < {arg} 2>/dev/null; {mark}; "
            f"tail -c 1 {arg} 2>/dev/null | wc -l; {mark}; "
            f'echo "$__hs $__hr"; '
            f"elif [ -e {arg} ]; then echo {NOT_REGULAR_SENTINEL}; "
            f"else echo {MISSING_SENTINEL}; fi")

    # ┌── 函数 def _read_file_missing()
    # │  作用：不-found recovery shared 由每个读取路径. Unicode-equivalent spellings
    # │  参数：self, path: str, offset: int, limit: int
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _read_file_missing(self, path: str, offset: int, limit: int) -> ReadResult:
        """Not-found recovery shared by every read path. Unicode-equivalent spellings
        (NFC/NFD, confusable spaces/quotes) render identically, so the model can never
        discover the byte mismatch by retyping — retrying is the tool's job. No
        equivalent spelling → suggest similar files."""
        variant = self._unicode_variant_match(path)
        if variant is not None:
            result = self.read_file(variant, offset=offset, limit=limit)
            note = (
                f"Note: '{path}' not found byte-for-byte; resolved to "
                f"the unicode-equivalent file '{variant}' (invisible "
                "encoding difference: NFC/NFD or special space/quote "
                "characters).")
            result.hint = f"{note} {result.hint}" if result.hint else note
            # ⤷ 出口 return：返回 result
            return result
        # ⤷ 出口 return：返回 self._suggest_similar_files(path)
        return self._suggest_similar_files(path)

    # ┌── 函数 def _read_binary_file()
    # │  作用：Binary branch shared 由每个读取路径: UTF-16 文本 (Notepad, PowerShell
    # │  参数：self, path: str, offset: int, limit: int, file_size: int, sample_bytes: Optional[bytes]
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _read_binary_file(self, path: str, offset: int, limit: int,
                          file_size: int, sample_bytes: Optional[bytes]) -> ReadResult:
        """Binary branch shared by every read path: UTF-16 text (Notepad, PowerShell
        ``>``) trips the binary guard; transcode it, else refuse with the type name.

        UTF-16 rescue (ported from MoonshotAI/kimi-code#2647): the terminal env decodes stdout as UTF-8 with
        errors="replace", so a UTF-16 text file (Windows Notepad .txt, PowerShell `>` redirects) arrives
        mangled with U+FFFD and trips the binary guard. Probe the raw bytes via the backend's Python and
        transcode to UTF-8 when a BOM or the zero-byte parity heuristic identifies UTF-16.
        """
        utf16_result = self._try_read_utf16(path, offset, limit, file_size)
        if utf16_result is not None:
            # ⤷ 出口 return：返回 utf16_result
            return utf16_result
        # ⤷ 出口 return：返回 ReadResult(is_binary=True, file_size=file_size, error=descri…
        return ReadResult(
            is_binary=True, file_size=file_size,
            error=describe_binary_file(sample_bytes, file_size))

    # ┌── 函数 def _read_file_sequential()
    # │  作用：One-probe-每次调用的读取: pre-compound 形态, 保留作为兜底用于
    # │  参数：self, path: str, offset: int, limit: int
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _read_file_sequential(self, path: str, offset: int, limit: int) -> ReadResult:
        """One-probe-per-call read: the pre-compound form, kept as fallback for
        image / known-binary extensions and unparseable compound replies. ``path`` is
        already expanded and ``offset``/``limit`` normalized."""
        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            # ⤷ 出口 return：返回 self._read_file_missing(path, offset, limit)
            return self._read_file_missing(path, offset, limit)
        if status == "not_regular":
            # ⤷ 出口 return：返回 self._not_regular_error(path)
            return self._not_regular_error(path)
        if status == "env_unavailable":
            # ⤷ 出口 return：返回 self._env_unavailable_error(path)
            return self._env_unavailable_error(path)
        if self._is_image(path):  # never inlined — redirect to the vision tool
            # ⤷ 出口 return：返回 self._image_redirect_result(file_size)
            return self._image_redirect_result(file_size)
        is_binary, sample_bytes = self._detect_binary(path)
        if is_binary:
            return self._read_binary_file(path, offset, limit, file_size, sample_bytes)

        # Clamp each line to a byte budget IN THE SHELL so a 400MB single-line file
        # never crosses the exec transport. 4*max+1 BYTES (not max+1): ``cut -b`` can
        # split a multibyte codepoint, and a tighter byte clamp would yield fewer
        # CHARS than max so the Python clamp in _add_line_numbers would never fire
        # (silent truncation). UTF-8 codepoints are ≤4 bytes, so every over-long
        # line still trips the char clamp, which also drops a boundary-split U+FFFD.
        from tools.tool_output_limits import get_max_line_length
        line_clamp_bytes = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1
        read_result = self._exec(
            f"sed -n '{offset},{end_line}p' {self._escape_shell_arg(path)}"
            f" | cut -b1-{line_clamp_bytes}")
        if read_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {read_result.stdout}")
        read_output = _strip_terminal_fence_leaks(read_result.stdout)

        wc_result = self._exec(f"wc -l < {self._escape_shell_arg(path)}")
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_result.stdout).strip())
        except ValueError:
            total_lines = 0

        # Only the page reaching the file's final line can carry the ``cut`` newline
        # artifact (see _assemble_read_result); probe the last byte just for that case.
        file_ends_with_newline: Optional[bool] = None
        if not total_lines > end_line and read_output.endswith('\n'):
            tail_result = self._exec(f"tail -c 1 {self._escape_shell_arg(path)} | wc -l")
            if tail_result.exit_code == 0:
                file_ends_with_newline = _strip_terminal_fence_leaks(tail_result.stdout).strip() != "0"
        # ⤷ 出口 return：返回 self._assemble_read_result(read_output, offset=offset, end_l…
        return self._assemble_read_result(
            read_output, offset=offset, end_line=end_line, total_lines=total_lines,
            file_size=file_size, file_ends_with_newline=file_ends_with_newline)

    # ┌── 函数 def _assemble_read_result()
    # │  作用：轮次 raw ``sed | cut`` page 进入 final ``ReadResult``. Shared 由每个
    # │  参数：self, read_output: str, offset: int, end_line: int, total_lines: int, file_size: int, file_ends_with_newline: Optional[bool]
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _assemble_read_result(self, read_output: str, *, offset: int, end_line: int,
                              total_lines: int, file_size: int,
                              file_ends_with_newline: Optional[bool]) -> ReadResult:
        """Turn a raw ``sed | cut`` page into the final ``ReadResult``. Shared by every
        read path so the BOM strip, pagination hint, ``cut`` newline-artifact fix and
        the ambiguous-silence guards never drift apart. ``file_ends_with_newline`` is
        None when the caller could not tell (artifact left alone, as before)."""
        # ``wc -l`` counts newlines, not lines: a nonempty file whose last byte
        # is not a newline holds one more line than the count (#3907). Adjust
        # here — the single choke point — so total_lines, truncation, and the
        # past-EOF guard agree on every read path (compound, sequential, native).
        if file_size > 0 and file_ends_with_newline is False:
            total_lines += 1
        if offset == 1:  # only the first chunk can carry a BOM (byte 0)
            read_output, _ = _strip_bom(read_output)
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"

        # ``cut`` always newline-terminates, so a file without a trailing newline
        # would grow a phantom empty last line; strip it when the last byte says so.
        if not truncated and read_output.endswith('\n') and file_ends_with_newline is False:
            read_output = read_output[:-1]

        # Empty content is indistinguishable from a broken tool: name the dead end.
        if file_size == 0:
            # ⤷ 出口 return：返回 ReadResult(content='', total_lines=0, file_size=0, hint='Fil…
            return ReadResult(content="", total_lines=0, file_size=0, hint="File is empty (0 bytes).")
        if offset > total_lines > 0:
            # ⤷ 出口 return：返回 ReadResult(content='', total_lines=total_lines, file_size=fi…
            return ReadResult(
                content="", total_lines=total_lines, file_size=file_size,
                hint=(
                    f"Note: offset {offset} is beyond the end of the file "
                    f"({total_lines} lines total). Retry with offset <= "
                    f"{total_lines}."))
        # ⤷ 出口 return：返回 ReadResult(content=self._add_line_numbers(read_output, offse…
        return ReadResult(
            content=self._add_line_numbers(read_output, offset), total_lines=total_lines,
            file_size=file_size, truncated=truncated, hint=hint)

    # Confusable characters seen in real filenames, collapsed after NFC.
    _CONFUSABLES = (
        ("\u202f", " "),  # narrow no-break space (macOS screenshots)
        ("\u00a0", " "),  # no-break space
        ("\u2019", "'"),  # right single quotation mark (Finder)
        ("\u2018", "'"),  # left single quotation mark
    )

    # ┌── 函数 def _unicode_variant_match()
    # │  作用：在…上-disk spelling 的文件 whose 名称是 unicode-equivalent ``路径``
    # │  参数：self, path: str
    # │  返回：Optional[str]
    # └──────────────────────────────────────────────────────────────
    def _unicode_variant_match(self, path: str) -> Optional[str]:
        """On-disk spelling of a file whose name is unicode-equivalent to ``path``
        (NFC/NFD, confusable spaces/quotes). Returns the entry only when EXACTLY one
        matches — several candidates = homoglyph collision, guessing would read the
        wrong file."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        if not filename:
            # ⤷ 出口 return：返回 None
            return None

        def _canon(name: str) -> str:
            out = unicodedata.normalize("NFC", name)
            for src, dst in self._CONFUSABLES:
                out = out.replace(src, dst)
            # ⤷ 出口 return：返回 out
            return out

        target = _canon(filename)
        ls_result = self._exec(f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null")
        if ls_result.exit_code != 0 or not ls_result.stdout.strip():
            # ⤷ 出口 return：返回 None
            return None
        candidates = [
            entry for entry in _strip_terminal_fence_leaks(ls_result.stdout).splitlines()
            if entry and entry != filename and _canon(entry) == target]
        if len(candidates) == 1:
            # ⤷ 出口 return：返回 os.path.join(dir_path, candidates[0]) if dir_path != '.' or …
            return os.path.join(dir_path, candidates[0]) if dir_path != "." or "/" in path else candidates[0]
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def _suggest_similar_files()
    # │  作用："文件不 found" 结果 listing up 5 similar 名称从 same directory.
    # │  参数：self, path: str
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def _suggest_similar_files(self, path: str) -> ReadResult:
        """"File not found" result listing up to 5 similar names from the same directory."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0].lower()
        ext = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()
        ls_result = self._exec(f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -50")
        scored: list = []  # (score, filepath) — higher is better
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for f in ls_result.stdout.strip().split('\n'):
                if not f:
                    continue
                lf = f.lower()
                score = 0
                if lf == lower_name:
                    score = 100
                elif os.path.splitext(f)[0].lower() == basename_no_ext:  # config.yml vs config.yaml
                    score = 90
                elif lf.startswith(lower_name) or lower_name.startswith(lf):
                    score = 70
                elif lower_name in lf:
                    score = 60
                elif lf in lower_name and len(lf) > 2:
                    score = 40
                elif ext and os.path.splitext(f)[1].lower() == ext:
                    common = set(lower_name) & set(lf)
                    if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                        score = 30
                # Near-miss spelling (AGENT.md -> AGENTS.md) the substring checks miss.
                if score == 0 and difflib.SequenceMatcher(None, lower_name, lf).ratio() >= 0.8:
                    score = 50
                if score > 0:
                    scored.append((score, os.path.join(dir_path, f)))
        scored.sort(key=lambda x: -x[0])
        # ⤷ 出口 return：返回 ReadResult(error=f'File not found: {path}', similar_files=[f…
        return ReadResult(error=f"File not found: {path}", similar_files=[fp for _, fp in scored[:5]])

    # ┌── 函数 def read_file_raw()
    # │  作用：Whole 文件作为 plain 字符串 (无 pagination/line numbers/clamping).
    # │  参数：self, path: str
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def read_file_raw(self, path: str) -> ReadResult:
        """Whole file as a plain string (no pagination/line numbers/clamping)."""
        path = self._expand_path(path)
        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            # ⤷ 出口 return：返回 self._suggest_similar_files(path)
            return self._suggest_similar_files(path)
        if status == "not_regular":
            # ⤷ 出口 return：返回 self._not_regular_error(path)
            return self._not_regular_error(path)
        if status == "env_unavailable":
            # ⤷ 出口 return：返回 self._env_unavailable_error(path)
            return self._env_unavailable_error(path)
        if self._is_image(path):
            # ⤷ 出口 return：返回 ReadResult(is_image=True, is_binary=True, file_size=file_siz…
            return ReadResult(is_image=True, is_binary=True, file_size=file_size)
        is_binary, sample_bytes = self._detect_binary(path)
        if is_binary:
            # ⤷ 出口 return：返回 ReadResult(is_binary=True, file_size=file_size, error=descri…
            return ReadResult(is_binary=True, file_size=file_size, error=describe_binary_file(sample_bytes, file_size))
        cat_result = self._exec(f"cat {self._escape_shell_arg(path)}")
        if cat_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {cat_result.stdout}")
        # Strip a leading BOM (a phantom U+FEFF defeats an exact first-line match);
        # write_file re-probes disk and restores it.
        raw_content, _ = _strip_bom(_strip_terminal_fence_leaks(cat_result.stdout))
        # ⤷ 出口 return：返回 ReadResult(content=raw_content, file_size=file_size)
        return ReadResult(content=raw_content, file_size=file_size)

    # ┌── 函数 def read_file_bytes()
    # │  作用：读取 binary-safe bytes (作为 base64) 从任意 shell-backed environment.
    # │  参数：self, path: str, max_bytes: Optional[int] = None
    # │  返回：ReadResult
    # └──────────────────────────────────────────────────────────────
    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read binary-safe bytes (as base64) from any shell-backed environment."""
        path = self._expand_path(path)
        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            # ⤷ 出口 return：返回 ReadResult(error=f'File not found: {path}')
            return ReadResult(error=f"File not found: {path}")
        if status == "not_regular":
            # ⤷ 出口 return：返回 self._not_regular_error(path)
            return self._not_regular_error(path)
        if status == "env_unavailable":
            # ⤷ 出口 return：返回 self._env_unavailable_error(path)
            return self._env_unavailable_error(path)
        if status == "bad_size":
            # ⤷ 出口 return：返回 ReadResult(error=f'Could not determine file size: {path}')
            return ReadResult(error=f"Could not determine file size: {path}")
        if max_bytes is not None and file_size > max_bytes:
            return ReadResult(
                file_size=file_size,
                error=f"File is too large ({file_size:,} bytes, limit is {max_bytes:,})")
        encoded = self._exec(f"base64 < {self._escape_shell_arg(path)}")
        if encoded.exit_code != 0:
            return ReadResult(error=f"Failed to read binary file: {encoded.stdout}")
        compact = "".join(_strip_terminal_fence_leaks(encoded.stdout).split())
        # ⤷ 关键分支 try/except：捕获 (ValueError, base64.binascii.Error)，出错走兜底而不是崩溃。
        try:
            base64.b64decode(compact, validate=True)
        except (ValueError, base64.binascii.Error):
            return ReadResult(error=f"Backend returned invalid binary data for: {path}")
        # ⤷ 出口 return：返回 ReadResult(base64_content=compact, file_size=file_size, is_b…
        return ReadResult(base64_content=compact, file_size=file_size, is_binary=True)

    # ┌── 函数 def delete_file()
    # │  作用：Delete single 文件 (directories 已拒绝) via backend's ``python -c``
    # │  参数：self, path: str
    # │  返回：WriteResult
    # └──────────────────────────────────────────────────────────────
    def delete_file(self, path: str) -> WriteResult:
        """Delete a single file (directories rejected) via the backend's ``python -c``
        so one code path works on local/docker/ssh AND Windows shells (no ``rm``)."""
        path = self._expand_path(path)
        denied = get_write_denied_error(path, verb="Delete")
        if denied:
            # ⤷ 出口 return：返回 WriteResult(error=denied)
            return WriteResult(error=denied)
        # Path baked in via repr() for shell-independent quoting; no
        # ``unlink(missing_ok=True)`` (a 3.7 remote interpreter lacks it).
        snippet = (
            "import shutil, pathlib, sys\n"
            f"p = pathlib.Path({path!r})\n"
            "recursive = False\n"
            "try:\n"
            "    if p.is_dir() and not p.is_symlink():\n"
            "        if recursive:\n"
            "            shutil.rmtree(p)\n"
            "        else:\n"
            "            print('is a directory: ' + str(p), file=sys.stderr); sys.exit(2)\n"
            "    else:\n"
            "        p.unlink()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr); sys.exit(1)\n")
        result = self._run_python_snippet(snippet)
        if result.exit_code != 0:
            # ⤷ 出口 return：返回 WriteResult(error=f"Failed to delete {path}: {(result.stdout…
            return WriteResult(error=f"Failed to delete {path}: {(result.stdout or '').strip() or 'unknown error'}")
        # ⤷ 出口 return：返回 WriteResult()
        return WriteResult()

    # ┌── 函数 def move_file()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, src: str, dst: str
    # │  返回：WriteResult
    # └──────────────────────────────────────────────────────────────
    def move_file(self, src: str, dst: str) -> WriteResult:
        src = self._expand_path(src)
        dst = self._expand_path(dst)
        for p in (src, dst):
            denied = get_write_denied_error(p, verb="Move")
            if denied:
                # ⤷ 出口 return：返回 WriteResult(error=denied)
                return WriteResult(error=denied)
        result = self._exec(f"mv {self._escape_shell_arg(src)} {self._escape_shell_arg(dst)}")
        if result.exit_code != 0:
            # ⤷ 出口 return：返回 WriteResult(error=f'Failed to move {src} -> {dst}: {result.s…
            return WriteResult(error=f"Failed to move {src} -> {dst}: {result.stdout}")
        # ⤷ 出口 return：返回 WriteResult()
        return WriteResult()

    # --- WRITE --------------------------------------------------------------

    # Lone surrogates OUTSIDE the surrogateescape range (U+DC80-U+DCFF round-trips
    # through the pipe; anything else can't be encoded at all).
    _LONE_SURROGATE_RE = re.compile(r"[\ud800-\udc7f\udd00-\udfff]")

    # ┌── 函数 def _reject_unencodable()
    # │  作用：Refuse 内容带 lone surrogate 在…之前任意子进程: letting 它
    # │  参数：self, path: str, content: str
    # │  返回：Optional[WriteResult]
    # └──────────────────────────────────────────────────────────────
    def _reject_unencodable(self, path: str, content: str) -> Optional[WriteResult]:
        """Refuse content with a lone surrogate BEFORE any subprocess: letting it
        reach the pipe spawns a child that hangs or truncates the target via
        empty-stdin ``cat``. A regex scan needs no encode."""
        m = self._LONE_SURROGATE_RE.search(content)
        if m:
            # ⤷ 出口 return：返回 WriteResult(error=f"Refusing to write '{path}': content cont…
            return WriteResult(error=(
                f"Refusing to write '{path}': content contains a lone "
                f"surrogate character ({m.group(0)!r}) that cannot be "
                "encoded as UTF-8. The file was NOT created or modified."))
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def _fail_closed_syntax_error()
    # │  作用：Fail-已关闭 pre-写入 gate 用于 ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/YAML/TOML):
    # │  参数：path: str, ext: str, content: str
    # │  返回：Optional[WriteResult]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _fail_closed_syntax_error(path: str, ext: str, content: str) -> Optional[WriteResult]:
        """Fail-closed pre-write gate for ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/YAML/TOML):
        a structured-format write that doesn't parse is a corrupt write, so refuse
        before any bytes touch disk. Checked against the RAW content, before the
        BOM/CRLF shims (post-shim linting would false-positive on a BOM-marked file)."""
        linter = LINTERS_INPROC.get(ext) if ext in _FAIL_CLOSED_INPROC_EXTS else None
        if linter is None:
            # ⤷ 出口 return：返回 None
            return None
        ok, err = linter(content)
        if ok or err == "__SKIP__":
            # ⤷ 出口 return：返回 None
            return None
        # ⤷ 出口 return：返回 WriteResult(error=f"Refusing to write '{path}': candidate co…
        return WriteResult(error=(
            f"Refusing to write '{path}': candidate content fails "
            f"{ext} syntax validation ({err}). The file was "
            "NOT created or modified. Fix the content and retry."))

    # ┌── 函数 def _write_probe_cmd()
    # │  作用：One shell 命令用于在…上-disk questions ``write_file`` asks. Two
    # │  参数：self, path: str, sentinel: str, body: Optional[str]
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _write_probe_cmd(self, path: str, sentinel: str, body: Optional[str]) -> str:
        """One shell command for the on-disk questions ``write_file`` asks. Two
        segments closed by a ``sentinel`` line: base64 of the first three bytes (BOM
        detection at the byte layer, same on-disk truth as ``_file_has_bom``), then
        ``body``: ``"cat"`` for the full text when pre-content is wanted, ``"sample"``
        for the 4 KB line-ending sample, or None for nothing. Gated on ``[ -f ]`` so a
        FIFO/device never reaches ``head``/``cat``; a missing path echoes ``MISSING_SENTINEL``."""
        arg = self._escape_shell_arg(path)
        if body == "cat":
            body_cmd = f"cat {arg} 2>/dev/null"
        elif body == "sample":
            body_cmd = f"head -c 4096 {arg} 2>/dev/null"
        else:
            body_cmd = ":"
        # ⤷ 出口 return：返回 f'if [ -f {arg} ]; then head -c 3 {arg} 2>/dev/null | base64…
        return (
            f"if [ -f {arg} ]; then "
            f"head -c 3 {arg} 2>/dev/null | base64 2>/dev/null; echo {sentinel}; "
            f"{body_cmd}; "
            f"else echo {MISSING_SENTINEL}; fi")

    # ┌── 函数 def _probe_write_target()
    # │  作用：``(has_bom, pre_content, original_line_ending)`` 用于 ``路径`` 在 ONE
    # │  参数：self, path: str, pre_content: Optional[str], want_pre: bool
    # │  返回：tuple[bool, Optional[str], Optional[str]]
    # └──────────────────────────────────────────────────────────────
    def _probe_write_target(self, path: str, pre_content: Optional[str], want_pre: bool,
                            ) -> tuple[bool, Optional[str], Optional[str]]:
        """``(has_bom, pre_content, original_line_ending)`` for ``path`` in ONE
        round-trip (replaces ``cat`` when pre-content is wanted, a ``head -c 4096``
        line-ending sample and a ``head -c 3`` BOM check). Semantics unchanged:
        pre-content is read only when wanted and not supplied; the line ending comes
        from pre-content when there is any, else from the sample; the BOM always comes
        from disk. An unparseable reply falls back to the separate probes."""
        if want_pre and pre_content is None:
            body_mode: Optional[str] = "cat"
        elif not pre_content:
            body_mode = "sample"
        else:
            body_mode = None

        sentinel = _new_sentinel(_WRITE_SENTINEL_PREFIX)
        probe = self._exec(self._write_probe_cmd(path, sentinel, body_mode))
        output = probe.stdout or ""
        if sentinel not in output:
            if _strip_terminal_fence_leaks(output).strip() == MISSING_SENTINEL:
                ending = _detect_line_ending(pre_content) if pre_content else None
                # ⤷ 出口 return：返回 (False, pre_content, ending)
                return False, pre_content, ending
            logger.debug(
                "write_file: pre-write probe reply for %s has no sentinel "
                "(exit %s, %d chars); falling back to sequential probes",
                path, probe.exit_code, len(output))
            # ⤷ 出口 return：返回 self._probe_write_target_sequential(path, pre_content, want_…
            return self._probe_write_target_sequential(path, pre_content, want_pre)

        segments = _split_segments(output, sentinel)
        if probe.exit_code != 0 or len(segments) != 2:
            logger.debug(
                "write_file: pre-write probe for %s returned exit %s with %d "
                "segments (want 2); falling back to sequential probes",
                path, probe.exit_code, len(segments))
            # ⤷ 出口 return：返回 self._probe_write_target_sequential(path, pre_content, want_…
            return self._probe_write_target_sequential(path, pre_content, want_pre)
        head_seg, body = segments

        head_bytes = self._decode_base64_sample(head_seg)
        if head_bytes is None:
            # No clean base64 on this shell; ask the way we used to.
            logger.debug(
                "write_file: no usable base64 head for %s; paying one extra "
                "round-trip for the BOM probe", path)
            has_bom = self._file_has_bom(path, pre_content)
        else:
            has_bom = head_bytes.startswith(_UTF8_BOM.encode("utf-8"))

        if body_mode == "cat" and body:
            pre_content = body
        if pre_content:
            ending = _detect_line_ending(pre_content)
        elif body_mode == "sample" and body:
            ending = _detect_line_ending(body)
        else:
            ending = None
        # ⤷ 出口 return：返回 (has_bom, pre_content, ending)
        return has_bom, pre_content, ending

    # ┌── 函数 def _probe_write_target_sequential()
    # │  作用：Pre-compound 形态的 ``_probe_write_target``: one exec per question.
    # │  参数：self, path: str, pre_content: Optional[str], want_pre: bool
    # │  返回：tuple[bool, Optional[str], Optional[str]]
    # └──────────────────────────────────────────────────────────────
    def _probe_write_target_sequential(self, path: str, pre_content: Optional[str], want_pre: bool,
                                       ) -> tuple[bool, Optional[str], Optional[str]]:
        """Pre-compound form of ``_probe_write_target``: one exec per question. A
        failed ``cat`` leaves pre_content None so the lint-delta and LSP consumers
        degrade gracefully."""
        if want_pre and pre_content is None:
            read_result = self._cat(path)
            if read_result.exit_code == 0 and read_result.stdout:
                pre_content = read_result.stdout
        if pre_content:
            ending = _detect_line_ending(pre_content)
        else:
            head = self._head(path, 4096)
            ending = _detect_line_ending(head.stdout) if head.exit_code == 0 and head.stdout else None
        # ⤷ 出口 return：返回 (self._file_has_bom(path, pre_content), pre_content, ending)
        return self._file_has_bom(path, pre_content), pre_content, ending

    # ┌── 函数 def _verify_written_hash()
    # │  作用：Compare 在…上-disk sha256 intended bytes: ``(verified, 错误)``.
    # │  参数：self, path: str, content_bytes: bytes
    # │  返回：tuple[Optional[bool], Optional[WriteResult]]
    # └──────────────────────────────────────────────────────────────
    def _verify_written_hash(self, path: str, content_bytes: bytes) -> tuple[Optional[bool], Optional[WriteResult]]:
        """Compare the on-disk sha256 to the intended bytes: ``(verified, error)``.
        The explicit flag saves the model a confirming re-read; a mismatch is a hard
        error. ``verified`` is None when the hash could not be taken."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            hash_result = self._exec(f"sha256sum {self._escape_shell_arg(path)} 2>/dev/null")
            if hash_result.exit_code == 0 and hash_result.stdout.strip():
                disk_sha = hash_result.stdout.strip().split()[0]
                if disk_sha != hashlib.sha256(content_bytes).hexdigest():
                    # ⤷ 出口 return：返回 (False, WriteResult(error=f'Post-write verification failed f…
                    return False, WriteResult(error=(
                        f"Post-write verification failed for {path}: on-disk "
                        "content hash differs from the intended write. The "
                        "write did not persist correctly — re-read the file "
                        "and retry."))
                # ⤷ 出口 return：返回 (True, None)
                return True, None
        except Exception:
            pass
        # ⤷ 出口 return：返回 (None, None)
        return None, None

    # ┌── 函数 def write_file()
    # │  作用：写入内容 atomically, 创建 parent directories 作为 needed.
    # │  参数：self, path: str, content: str, pre_content: Optional[str] = None
    # │  返回：WriteResult
    # └──────────────────────────────────────────────────────────────
    def write_file(self, path: str, content: str, pre_content: Optional[str] = None) -> WriteResult:
        """Write content atomically, creating parent directories as needed.

        Order: deny list → lone-surrogate refusal → fail-closed syntax gate on the
        CANDIDATE content (JSON/YAML/TOML) → one compound on-disk probe
        (pre-content when wanted, CRLF, BOM; see ``_probe_write_target``) →
        CRLF/BOM preservation → LSP baseline snapshot → atomic write (content rides
        stdin: no ARG_MAX limit) → sha256 verification → lint delta → LSP
        diagnostics when syntax is clean. ``pre_content``: pre-edit content the
        caller already has (skips the read); BOM detection always probes disk.
        """
        path = self._expand_path(path)
        denied = get_write_denied_error(path)
        if denied:
            # ⤷ 出口 return：返回 WriteResult(error=denied)
            return WriteResult(error=denied)
        refused = self._reject_unencodable(path, content)
        if refused is not None:
            # ⤷ 出口 return：返回 refused
            return refused
        ext = os.path.splitext(path)[1].lower()
        refused = self._fail_closed_syntax_error(path, ext, content)
        if refused is not None:
            # ⤷ 出口 return：返回 refused
            return refused

        # Pre-content is read only for extensions in the UNION of in-process lint and
        # LSP coverage (keeps the hot path fast for binaries).
        want_pre = ext in LINTERS_INPROC or self._lsp_handles_extension(ext)
        has_bom, pre_content, original_ending = self._probe_write_target(path, pre_content, want_pre)
        # read_file strips the BOM and models send bare-LF text, so a round-trip would
        # otherwise normalize CRLF files and drop the BOM (prepend only when absent).
        if original_ending == "\r\n":
            content = _normalize_line_endings(content, "\r\n")
        if has_bom and not _has_bom(content):
            content = _UTF8_BOM + content
        # Best-effort snapshot so the LSP tier reports only this edit's diagnostics.
        self._snapshot_lsp_baseline(path)
        # ``dirs_created`` means "parent dirs ensured" (mkdir -p is folded into
        # _atomic_write; its failure surfaces as the atomic-write error below).
        dirs_created = bool(os.path.dirname(path))
        # surrogateescape is the exact inverse of the decode that may have produced
        # this content, so these are the bytes on disk; the early rejection above
        # guarantees this cannot raise.
        content_bytes = content.encode("utf-8", "surrogateescape")
        write_result = self._atomic_write(path, content)
        if write_result.exit_code != 0:
            # ⤷ 出口 return：返回 WriteResult(error=f'Failed to write file: {write_result.stdo…
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")
        content_verified, verify_error = self._verify_written_hash(path, content_bytes)
        if verify_error is not None:
            # ⤷ 出口 return：返回 verify_error
            return verify_error

        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)
        # LSP diagnostics are a separate channel, fired only when the syntax tier is
        # clean (no point asking an LSP about a file that won't parse).
        lsp_diagnostics: Optional[str] = None
        if lint_result.success or lint_result.skipped:
            lsp_diagnostics = self._maybe_lsp_diagnostics(path, pre_content=pre_content, post_content=content) or None
        # ⤷ 出口 return：返回 WriteResult(bytes_written=len(content_bytes), dirs_created=d…
        return WriteResult(
            bytes_written=len(content_bytes), dirs_created=dirs_created, verified=content_verified,
            lint=lint_result.to_dict() if lint_result else None, lsp_diagnostics=lsp_diagnostics)

    # --- PATCH (replace mode) -----------------------------------------------

    # ┌── 函数 def _no_match_result()
    # │  作用：PatchResult 用于 failed fuzzy match. 已-applied detection first:
    # │  参数：self, path: str, content: str, old_string: str, new_string: str, match_count: int, error: Optional[str]
    # │  返回：PatchResult
    # └──────────────────────────────────────────────────────────────
    def _no_match_result(self, path: str, content: str, old_string: str,
                         new_string: str, match_count: int, error: Optional[str]) -> PatchResult:
        """PatchResult for a failed fuzzy match. Already-applied detection first: the
        most common production failure is a re-send of an edit that already landed,
        and a success-shaped no-op stops the model burning turns on re-reads.
        Otherwise attach a best-effort "Did you mean?" snippet to the error."""
        from tools.fuzzy_match import format_no_match_hint, is_already_applied
        if is_already_applied(content, old_string, new_string):
            # ⤷ 出口 return：返回 PatchResult(success=True, no_change=True, note=f'File alread…
            return PatchResult(
                success=True, no_change=True,
                note=(
                    f"File already contains the target text — the edit "
                    f"appears to be already applied to {path}. No write "
                    "performed; do not re-send this patch."))
        err_msg = error or f"Could not find match for old_string in {path}"
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
        except Exception:
            pass
        # ⤷ 出口 return：返回 PatchResult(error=err_msg)
        return PatchResult(error=err_msg)

    # ┌── 函数 def _verify_patch_persisted()
    # │  作用：Re-读取 ``路径`` 且 confirm intended bytes landed; 错误结果或 None.
    # │  参数：self, path: str, new_content: str
    # │  返回：Optional[PatchResult]
    # └──────────────────────────────────────────────────────────────
    def _verify_patch_persisted(self, path: str, new_content: str) -> Optional[PatchResult]:
        """Re-read ``path`` and confirm the intended bytes landed; error result or None.
        Catches silent persistence failures (FS oddities, races, truncated pipe).
        Line endings are normalized first (Windows text-mode ``open()`` writes LF as
        CRLF) and the re-read's BOM stripped (``new_content`` is the BOM-less
        string we matched against)."""
        verify_result = self._cat(path)
        if verify_result.exit_code != 0:
            # ⤷ 出口 return：返回 PatchResult(error=f'Post-write verification failed: could no…
            return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
        bomless, _ = _strip_bom(verify_result.stdout)
        on_disk = bomless.replace("\r\n", "\n").replace("\r", "\n")
        intended = new_content.replace("\r\n", "\n").replace("\r", "\n")
        if on_disk != intended:
            # ⤷ 出口 return：返回 PatchResult(error=f'Post-write verification failed for {path…
            return PatchResult(error=(
                f"Post-write verification failed for {path}: on-disk content "
                f"differs from intended write "
                f"(wrote {len(intended)} chars, read back "
                f"{len(on_disk)} chars after normalizing line endings). "
                "The patch did not persist. Re-read the file and try again."))
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def patch_replace()
    # │  作用：Replace 文本在 文件 using fuzzy matching (``old_string`` 必须是
    # │  参数：self, path: str, old_string: str, new_string: str, replace_all: bool = False
    # │  返回：PatchResult
    # └──────────────────────────────────────────────────────────────
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching (``old_string`` must be
        unique unless ``replace_all``). Returns a PatchResult with diff + lint."""
        path = self._expand_path(path)
        denied = get_write_denied_error(path)
        if denied:
            # ⤷ 出口 return：返回 PatchResult(error=denied)
            return PatchResult(error=denied)
        read_result = self._cat(path)
        if read_result.exit_code != 0:
            # ⤷ 出口 return：返回 PatchResult(error=f'Failed to read file: {path}')
            return PatchResult(error=f"Failed to read file: {path}")
        # Match and diff on BOM-stripped content (a phantom U+FEFF defeats an exact
        # first-line match); the raw read becomes write_file's pre_content.
        raw_content = read_result.stdout
        content, _ = _strip_bom(raw_content)

        from tools.fuzzy_match import fuzzy_find_and_replace
        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all)
        if error or match_count == 0:
            # ⤷ 出口 return：返回 self._no_match_result(path, content, old_string, new_string,…
            return self._no_match_result(path, content, old_string, new_string, match_count, error)
        # Models send bare-LF old/new strings; normalize the substituted region to
        # the file's ending so CRLF files stay consistent.
        file_ending = _detect_line_ending(content)
        if file_ending:
            new_content = _normalize_line_endings(new_content, file_ending)
        write_result = self.write_file(path, new_content, pre_content=raw_content)
        if write_result.error:
            # ⤷ 出口 return：返回 PatchResult(error=f'Failed to write changes: {write_result.e…
            return PatchResult(error=f"Failed to write changes: {write_result.error}")
        verify_error = self._verify_patch_persisted(path, new_content)
        if verify_error is not None:
            # ⤷ 出口 return：返回 verify_error
            return verify_error
        lint_result = self._check_lint_delta(path, pre_content=content, post_content=new_content)
        # ⤷ 出口 return：返回 PatchResult(success=True, diff=self._unified_diff(content, n…
        return PatchResult(
            success=True, diff=self._unified_diff(content, new_content, path), files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None,
            # From the internal write_file call, whose baseline was the pre-patch content.
            lsp_diagnostics=write_result.lsp_diagnostics)

    # ┌── 函数 def patch_v4a()
    # │  作用：Apply V4A format patch (``*** Begin Patch`` / ``*** 更新文件:`` /
    # │  参数：self, patch_content: str
    # │  返回：PatchResult
    # └──────────────────────────────────────────────────────────────
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch (``*** Begin Patch`` / ``*** Update File:`` /
        ``@@ hint @@`` hunks / ``*** End Patch``)."""
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            # ⤷ 出口 return：返回 PatchResult(error=f'Failed to parse patch: {parse_error}')
            return PatchResult(error=f"Failed to parse patch: {parse_error}")
        # ⤷ 出口 return：返回 apply_v4a_operations(operations, self)
        return apply_v4a_operations(operations, self)

    # --- SEARCH -------------------------------------------------------------

    # ┌── 函数 def search()
    # │  作用：Search 用于内容 (regex, ``target="内容"``) 或文件 (glob,
    # │  参数：self, pattern: str, path: str = '.', target: str = 'content', file_glob: Optional[str] = None, limit: int = 50, offset: int = 0, output_mode: str = 'content'  …共 10 个
    # │  返回：SearchResult
    # └──────────────────────────────────────────────────────────────
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0,
               order: str = "discovery") -> SearchResult:
        """Search for content (regex, ``target="content"``) or files (glob,
        ``target="files"``). ``output_mode``: "content", "files_only" or "count";
        ``context``: lines of context around matches; ``order``: file-search
        ordering — fast "discovery" or exact "modified" time."""
        offset, limit = normalize_search_pagination(offset, limit)
        if target == "files" and order not in {"discovery", "modified"}:
            # ⤷ 出口 return：返回 SearchResult(error=f"Invalid file search order {order!r}; ex…
            return SearchResult(
                error=(f"Invalid file search order {order!r}; expected "
                       "'discovery' or 'modified'."))
        path = self._expand_path(path)
        exists_probe = self._path_exists_probe(path)
        if "exists" not in exists_probe and "not_found" not in exists_probe:
            # ⤷ 出口 return：返回 SearchResult(error=f'Terminal environment unavailable: could…
            return SearchResult(error=(f"Terminal environment unavailable: could not stat {path} "
                                       "(the sandbox may still be starting or was removed). Retry shortly."))
        if "not_found" in exists_probe:
            # Models often pass several paths in one string: search the parts that exist.
            multi = self._try_multi_path_search(
                pattern, path, target, file_glob, limit, offset, output_mode, context, order)
            if multi is not None:
                # ⤷ 出口 return：返回 multi
                return multi
            # ⤷ 出口 return：返回 self._path_not_found_result(path)
            return self._path_not_found_result(path)
        result = self._dispatch_search(pattern, path, target, file_glob, limit, offset,
                                       output_mode, context, order)
        exclusions = self._macos_search_exclusions(path)
        if exclusions and not result.error:
            skipped = ", ".join(item.split("/")[-1] for item in exclusions)
            result.warning = (
                "Skipped macOS protected folders during broad search to avoid "
                f"an unattended privacy prompt: {skipped}. Search a protected "
                "folder directly when access is intentional.")
        # ⤷ 出口 return：返回 result
        return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from typing import ClassVar  # noqa: F401,E402
from typing import List  # noqa: F401,E402
from agent.file_safety import build_write_denied_paths  # noqa: F401,E402
from agent.file_safety import build_write_denied_prefixes  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import posixpath  # noqa: F401,E402
import threading  # noqa: F401,E402

MAX_LINES = 2000

MAX_LINE_LENGTH = 2000

WRITE_DENIED_PATHS = build_write_denied_paths(_HOME)

WRITE_DENIED_PREFIXES = build_write_denied_prefixes(_HOME)


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_READ_LIMIT': ('tools.file_operations_common', 'DEFAULT_READ_LIMIT'),
    'DEFAULT_READ_OFFSET': ('tools.file_operations_common', 'DEFAULT_READ_OFFSET'),
    'DEFAULT_SEARCH_LIMIT': ('tools.file_operations_common', 'DEFAULT_SEARCH_LIMIT'),
    'DEFAULT_SEARCH_OFFSET': ('tools.file_operations_common', 'DEFAULT_SEARCH_OFFSET'),
    'LINTERS': ('tools.file_operations_lint', 'LINTERS'),
    'LintResult': ('tools.file_operations_common', 'LintResult'),
    'MAX_FILE_SIZE': ('tools.transcription_common', 'MAX_FILE_SIZE'),
    'SEARCH_PRUNE_DIR_NAMES': ('agent.search_policy', 'SEARCH_PRUNE_DIR_NAMES'),
    'SearchMatch': ('tools.file_operations_common', 'SearchMatch'),
    'build_write_denied_paths': ('agent.file_safety', 'build_write_denied_paths'),
    'build_write_denied_prefixes': ('agent.file_safety', 'build_write_denied_prefixes'),
    'tool_interrupt': ('tools', 'interrupt'),
}


# ┌── 函数 def __getattr__()
# │  作用：魔术方法 __getattr__（Python 在特定场景自动调用，见下方说明）
# │  参数：name
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
# 💡【本项目惯用法】模块级 __getattr__：当外部访问本模块里尚未导入的名字时才会触发，
#    这里用来做「懒加载」——真正用到时才 import，避免启动慢和循环导入。
#    所以你看到 `from xxx import YYY` 在本文件里搜不到定义，是正常的。
def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    # ⤷ 出口 return：返回 getattr(importlib.import_module(target[0]), target[1])
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
