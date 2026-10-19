"""CLI presentation -- spinner, kawaii faces, tool preview formatting.

Pure display functions with no AIAgent dependency; used for CLI feedback.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：agent/display.py（约 1096 行）
# 地位：命令行展示层——转圈动画、工具调用预览、可爱表情提示。
#
# 它没有 AIAgent 依赖，是纯展示函数，所以最容易读，适合拿来练手。
#
# 主要内容：
#   - class KawaiiSpinner：终端转圈动画（start / update_text / stop，还支持 with 语法）
#     实现要点：单独线程 + 光标控制 + 回车符(CR)回到行首覆盖上一行
#   - build_tool_preview / build_tool_label / build_status_phrase：
#     把工具调用压缩成一行人类可读的预览，比如 `terminal: pip install xxx`
#   - summarize_shell_command()：把复杂的 shell 命令抽取出「主干」做展示
#   - redact_tool_args_for_display()：展示前先把密码之类的敏感信息打码
#   - extract_edit_diff / render_edit_diff_with_delta：在终端里渲染文件改动的 diff
#   - _get_cute_tool_message()：按工具类型返回一句卖萌的完成提示（kawaii faces）
#
# 值得学：capture_local_edit_snapshot() —— 编辑前先快照，编辑后算 diff，这样可以实时显示改动。
# ==========================================================================
#

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from utils import safe_json_loads
from agent.redact import redact_sensitive_text
from agent.tool_result_classification import file_mutation_result_landed

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"
_MAX_INLINE_DIFF_FILES = 6
_MAX_INLINE_DIFF_LINES = 80


# ┌── 函数 def _display_url()
# │  作用：Extract 展示-仅 URL 不 assuming 模型参数类型.
# │  参数：value: Any
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _display_url(value: Any) -> str:
    """Extract a display-only URL without assuming model argument types."""
    value = (value.get("url") or value.get("href")) if isinstance(value, dict) else value
    # ⤷ 出口 return：返回 value.strip() if isinstance(value, str) else ''
    return value.strip() if isinstance(value, str) else ""


# ┌── 函数 def _http_url()
# │  作用：``candidate`` 当它 解析作为 absolute http(s) URL, 否则 None.
# │  参数：candidate: str
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _http_url(candidate: str) -> str | None:
    """``candidate`` when it parses as an absolute http(s) URL, else None."""
    # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 candidate if parsed.scheme.lower() in {'http', 'https'} and …
    return candidate if parsed.scheme.lower() in {"http", "https"} and parsed.netloc else None


# ┌── 函数 def _hex_rgb()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：h: str
# │  返回：tuple[int, int, int]
# └──────────────────────────────────────────────────────────────
def _hex_rgb(h: str) -> tuple[int, int, int]:
    # ⤷ 出口 return：返回 (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


# ┌── 函数 def _get_skin()
# │  作用：Active skin 配置, 或 None 当 unavailable (lazy 导入 avoids cycles).
# │  参数：无
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _get_skin():
    """Active skin config, or None when unavailable (lazy import avoids cycles)."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from hermes_cli.skin_engine import get_active_skin
        # ⤷ 出口 return：返回 get_active_skin()
        return get_active_skin()
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# Diff colors resolve lazily from the skin engine (light/dark aware) and are
# cached after the first resolution.
_diff_colors_cached: dict[str, str] | None = None


# Foreground diff colors: key -> (skin color key, dark-terminal fallback RGB).
_DIFF_FG = {
    "dim": ("banner_dim", (150, 150, 150)), "file": ("session_label", (180, 160, 255)),
    "hunk": ("session_border", (120, 120, 140)),
}
# Background diff colors (white text on a dark tint of the skin's ui_error/ui_ok):
# key -> (skin key, default hex, dominant channel index, dark-terminal fallback).
_DIFF_BG = {
    "minus": ("ui_error", "#ef5350", 0, "\033[38;2;255;255;255;48;2;120;20;20m"),
    "plus": ("ui_ok", "#4caf50", 1, "\033[38;2;255;255;255;48;2;20;90;20m"),
}


# ┌── 函数 def _fg()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：r: int, g: int, b: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _fg(r: int, g: int, b: int) -> str:
    # ⤷ 出口 return：返回 f'\x1b[38;2;{r};{g};{b}m'
    return f"\033[38;2;{r};{g};{b}m"


# ┌── 函数 def _tinted_bg()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：rgb: tuple[int, int, int], dominant: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _tinted_bg(rgb: tuple[int, int, int], dominant: int) -> str:
    r, g, b = (max(v // 2, 20) if i == dominant else max(v // 4, 10) for i, v in enumerate(rgb))
    # ⤷ 出口 return：返回 f'\x1b[38;2;255;255;255;48;2;{r};{g};{b}m'
    return f"\033[38;2;255;255;255;48;2;{r};{g};{b}m"


# ┌── 函数 def _diff_ansi()
# │  作用：返回 ANSI escapes 用于 diff 展示, resolved 从 active skin.。
# │  参数：无
# │  返回：dict[str, str]
# └──────────────────────────────────────────────────────────────
def _diff_ansi() -> dict[str, str]:
    """Return ANSI escapes for diff display, resolved from the active skin."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _diff_colors_cached
    if _diff_colors_cached is not None:
        # ⤷ 出口 return：返回 _diff_colors_cached
        return _diff_colors_cached
    colors = {k: _fg(*rgb) for k, (_, rgb) in _DIFF_FG.items()} | {k: v[3] for k, v in _DIFF_BG.items()}
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        skin = _get_skin()
        for key, (skin_key, fallback) in _DIFF_FG.items():
            h = skin.get_color(skin_key, "")
            colors[key] = _fg(*(_hex_rgb(h) if h and len(h) == 7 and h[0] == "#" else fallback))
        bg_hex = {key: skin.get_color(skin_key, default) for key, (skin_key, default, _, _) in _DIFF_BG.items()}
        for key, (_, _, dominant, _) in _DIFF_BG.items():
            h = bg_hex[key]
            if h and len(h) == 7:
                colors[key] = _tinted_bg(_hex_rgb(h), dominant)
    except Exception:
        pass
    _diff_colors_cached = colors
    # ⤷ 出口 return：返回 colors
    return colors


# ┌── 类 class LocalEditSnapshot
# │  说明：Pre-工具 filesystem snapshot used render diffs locally 在…之后写入.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass
class LocalEditSnapshot:
    """Pre-tool filesystem snapshot used to render diffs locally after writes."""
    paths: list[Path] = field(default_factory=list)
    before: dict[str, str | None] = field(default_factory=dict)


# Configurable tool preview length; set once at startup from display.tool_preview_length.
_tool_preview_max_len: int = 0  # 0 = unlimited
_friendly_tool_labels: bool = True


# ┌── 函数 def set_tool_preview_max_len()
# │  作用：设置 global max length 用于工具 call previews. 0 = 无限制.
# │  参数：n: int
# │  返回：None
# └──────────────────────────────────────────────────────────────
def set_tool_preview_max_len(n: int) -> None:
    """Set the global max length for tool call previews. 0 = no limit."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _tool_preview_max_len
    _tool_preview_max_len = max(int(n), 0) if n else 0


# ┌── 函数 def get_tool_preview_max_len()
# │  作用：返回 configured max preview length (0 = unlimited).。
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def get_tool_preview_max_len() -> int:
    """Return the configured max preview length (0 = unlimited)."""
    # ⤷ 出口 return：返回 _tool_preview_max_len
    return _tool_preview_max_len


# ┌── 函数 def set_friendly_tool_labels()
# │  作用：Toggle friendly human-phrased 工具 labels (展示.friendly_tool_labels).
# │  参数：enabled: bool
# │  返回：None
# └──────────────────────────────────────────────────────────────
def set_friendly_tool_labels(enabled: bool) -> None:
    """Toggle friendly human-phrased tool labels (display.friendly_tool_labels)."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _friendly_tool_labels
    _friendly_tool_labels = bool(enabled)


# ┌── 函数 def get_skin_tool_prefix()
# │  作用：获取工具输出 prefix character 从 active skin.
# │  参数：无
# │  返回：str
# └──────────────────────────────────────────────────────────────
def get_skin_tool_prefix() -> str:
    """Get tool output prefix character from active skin."""
    skin = _get_skin()
    # ⤷ 出口 return：返回 skin.tool_prefix if skin else '┊'
    return skin.tool_prefix if skin else "┊"


# ┌── 函数 def get_tool_emoji()
# │  作用：展示 emoji 用于工具: skin ``tool_emojis`` 覆盖值, 然后 registry, 然后 *默认值*.
# │  参数：tool_name: str, default: str = '⚡'
# │  返回：str
# └──────────────────────────────────────────────────────────────
def get_tool_emoji(tool_name: str, default: str = "⚡") -> str:
    """Display emoji for a tool: skin ``tool_emojis`` override, then registry, then *default*."""
    skin = _get_skin()
    override = skin.tool_emojis.get(tool_name) if skin and skin.tool_emojis else None
    if override:
        # ⤷ 出口 return：返回 override
        return override
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.registry import registry
        # ⤷ 出口 return：返回 registry.get_emoji(tool_name, default='') or default
        return registry.get_emoji(tool_name, default="") or default
    except Exception:
        # ⤷ 出口 return：返回 default
        return default


# ── Tool preview (one-line summary of a tool call's primary argument) ─────

# ┌── 函数 def _oneline()
# │  作用：把空白字符（含换行）压缩为单个空格。
# │  参数：text: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _oneline(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    # ⤷ 出口 return：返回 ' '.join(text.split())
    return " ".join(text.split())


# ┌── 函数 def _tail_trunc()
# │  作用：Tail-truncate ``限制`` chars 带 ``...`` (0/None = unlimited). 结果 never
# │  参数：text: str, limit: int | None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _tail_trunc(text: str, limit: int | None) -> str:
    """Tail-truncate to ``limit`` chars with ``...`` (0/None = unlimited). The result never
    exceeds ``limit``: for 1-3 the ellipsis itself is clipped (``text[:limit - 3]`` would go
    negative and hand back almost the whole string, #9439)."""
    if not limit or limit <= 0 or len(text) <= limit:
        # ⤷ 出口 return：返回 text
        return text
    # ⤷ 出口 return：返回 '.' * limit if limit <= 3 else text[:limit - 3] + '...'
    return "." * limit if limit <= 3 else text[:limit - 3] + "..."


# ┌── 函数 def _clip()
# │  作用：``文本[:n]`` plus ``...`` 当 longer (used inside quoted previews).
# │  参数：text: str, n: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _clip(text: str, n: int) -> str:
    """``text[:n]`` plus ``...`` when longer (used inside quoted previews)."""
    # ⤷ 出口 return：返回 f"{text[:n]}{('...' if len(text) > n else '')}"
    return f"{text[:n]}{'...' if len(text) > n else ''}"


# ┌── 类 class ToolPreview
# │  说明：compact 工具 preview plus presentation facts lost truncation.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ToolPreview:
    """A compact tool preview plus presentation facts lost to truncation."""

    text: str
    truncated: bool = False
    url: str | None = None


# ── Shell command summarisation ──────────────────────────────────────────

_SHELL_SILENT_HEADS = {"cd", "pushd", "popd", "export", "set", "unset", "source", ".", "true", "false", ":"}
_SHELL_PIPE_TAIL_HEADS = {"head", "tail", "wc", "sort", "uniq"}


# ┌── 函数 def _shell_basename()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：head: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _shell_basename(head: str) -> str:
    # ⤷ 出口 return：返回 head.rsplit('/', 1)[-1] if head else ''
    return head.rsplit("/", 1)[-1] if head else ""


# ┌── 函数 def _scan_quoted()
# │  作用：Yield ``(index, char, inside_quotes)``; quote 关闭 unless backslash-escaped.
# │  参数：text: str
# │  返回：Iterator[tuple[int, str, bool]]
# └──────────────────────────────────────────────────────────────
def _scan_quoted(text: str) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(index, char, inside_quotes)``; a quote closes unless backslash-escaped."""
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield i, ch, True
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield i, ch, True
        else:
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield i, ch, False


# ┌── 函数 def _split_shell_words()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：segment: str
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _split_shell_words(segment: str) -> list[str]:
    parts: list[list[str]] = [[]]
    for _, ch, quoted in _scan_quoted(segment):
        if not quoted and ch.isspace():
            parts.append([])
        else:
            parts[-1].append(ch)
    # ⤷ 出口 return：返回 [''.join(p) for p in parts if p]
    return ["".join(p) for p in parts if p]


# ┌── 函数 def _strip_shell_pipe_tail()
# │  作用：丢弃 trailing ``| head/tail/wc/sort/uniq ...`` pipeline 阶段.
# │  参数：segment: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _strip_shell_pipe_tail(segment: str) -> str:
    """Drop a trailing ``| head/tail/wc/sort/uniq ...`` pipeline stage."""
    words = _split_shell_words(segment)
    for i, word in enumerate(words):
        if word == "|" and _shell_basename(words[i + 1] if i + 1 < len(words) else "") in _SHELL_PIPE_TAIL_HEADS:
            words = words[:i]
            break
    # ⤷ 出口 return：返回 ' '.join(words).strip()
    return " ".join(words).strip()


# ┌── 函数 def _split_shell_compound()
# │  作用：Split 在…上 unquoted ``&&`` / ``||`` / ``;`` / newline, 丢弃 pipe tails per segment.
# │  参数：command: str
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _split_shell_compound(command: str) -> list[str]:
    """Split on unquoted ``&&`` / ``||`` / ``;`` / newline, dropping pipe tails per segment."""
    raw: list[list[str]] = [[]]
    skip = False
    for i, ch, quoted in _scan_quoted(command):
        if skip:
            skip = False
        elif not quoted and (command.startswith("&&", i) or command.startswith("||", i)):
            raw.append([])
            skip = True
        elif not quoted and ch in {";", "\n"}:
            raw.append([])
        else:
            raw[-1].append(ch)
    segments = (_strip_shell_pipe_tail("".join(buf).strip()) for buf in raw)
    # ⤷ 出口 return：返回 [s for s in segments if s]
    return [s for s in segments if s]


# ┌── 函数 def _shell_head_word()
# │  作用：命令名称的 segment, skipping leading ``VAR=值`` assignments.
# │  参数：segment: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _shell_head_word(segment: str) -> str:
    """Command name of a segment, skipping leading ``VAR=value`` assignments."""
    words = _split_shell_words(segment)
    while words and re.match(r"^[A-Za-z_]\w*=", words[0]):
        words.pop(0)
    # ⤷ 出口 return：返回 _shell_basename(words[0] if words else '')
    return _shell_basename(words[0] if words else "")


# ┌── 函数 def _clean_shell_segment()
# │  作用：丢弃 redirections (``> 文件``, ``2>&1``) 从 segment.
# │  参数：segment: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _clean_shell_segment(segment: str) -> str:
    """Drop redirections (``> file``, ``2>&1``) from a segment."""
    words = _split_shell_words(segment)
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if re.match(r"^\d*(?:>>?|<)$", word):
            i += 2  # operator + target
        elif re.match(r"^\d*(?:>&|<&)\d+$", word):
            i += 1
        else:
            out.append(word)
            i += 1
    # ⤷ 出口 return：返回 ' '.join(out).strip()
    return " ".join(out).strip()


# ┌── 函数 def _is_shell_boundary_echo()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：segment: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_shell_boundary_echo(segment: str) -> bool:
    words = _split_shell_words(segment)
    if _shell_basename(words[0] if words else "") != "echo":
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 出口 return：返回 bool(re.search('-{2,}|_exit=|(?:^|\\s|=)\\$[?{]|PIPESTATUS',…
    return bool(re.search(r"-{2,}|_exit=|(?:^|\s|=)\$[?{]|PIPESTATUS", " ".join(words[1:])))


# ┌── 函数 def summarize_shell_command()
# │  作用：Compact shell 包装器/plumbing 用于展示当 preserving raw 命令 elsewhere.
# │  参数：command: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def summarize_shell_command(command: str) -> str:
    """Compact shell wrapper/plumbing for display while preserving raw command elsewhere."""
    original = _oneline(command)
    if not original:
        # ⤷ 出口 return：返回 ''
        return ""
    segments = _split_shell_compound(original)
    if len(segments) <= 1:
        # ⤷ 出口 return：返回 _clean_shell_segment(segments[0] if segments else original) …
        return _clean_shell_segment(segments[0] if segments else original) or original
    core: list[str] = []
    for segment in segments:
        cleaned = _clean_shell_segment(segment)
        if cleaned and _shell_head_word(cleaned) not in _SHELL_SILENT_HEADS and not _is_shell_boundary_echo(cleaned):
            core.append(cleaned)
    if not core:
        # ⤷ 出口 return：返回 original
        return original
    count = len(core) - 1
    # ⤷ 出口 return：返回 core[0] if not count else f"{core[0]} + {count} {('command' …
    return core[0] if not count else f"{core[0]} + {count} {'command' if count == 1 else 'commands'}"


# ┌── 函数 def _read_file_line_label()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _read_file_line_label(args: dict) -> str:
    offset, limit = args.get("offset"), args.get("limit")
    if not isinstance(offset, int) or offset <= 0:
        # ⤷ 出口 return：返回 ''
        return ""
    # ⤷ 出口 return：返回 f'L{offset}-{offset + limit - 1}' if isinstance(limit, int) …
    return f"L{offset}-{offset + limit - 1}" if isinstance(limit, int) and limit > 1 else f"L{offset}"


# ┌── 函数 def redact_browser_typed_text_for_display()
# │  作用：Replace 每个 occurrence 的 secret-looking browser_type 值带 其 redacted 形态.
# │  参数：value: Any, typed_text: Any
# │  返回：Any
# └──────────────────────────────────────────────────────────────
def redact_browser_typed_text_for_display(value: Any, typed_text: Any) -> Any:
    """Replace every occurrence of a secret-looking browser_type value with its redacted form.

    Backends echo the attempted input in error strings/metadata, so it is swapped before
    reaching logs, callbacks, the model, or chat history. Forced regardless of
    ``security.redact_secrets``: a leaked typed credential is a security boundary.
    """
    needle = "" if typed_text is None else str(typed_text)
    redacted = redact_sensitive_text(needle, force=True) if needle else needle
    if redacted == needle:
        # ⤷ 出口 return：返回 value
        return value
    if isinstance(value, str):
        # ⤷ 出口 return：返回 value.replace(needle, redacted)
        return value.replace(needle, redacted)
    if isinstance(value, dict):
        # ⤷ 出口 return：返回 {key: redact_browser_typed_text_for_display(item, typed_text…
        return {key: redact_browser_typed_text_for_display(item, typed_text) for key, item in value.items()}
    if isinstance(value, list):
        # ⤷ 出口 return：返回 [redact_browser_typed_text_for_display(item, typed_text) for…
        return [redact_browser_typed_text_for_display(item, typed_text) for item in value]
    if isinstance(value, tuple):
        # ⤷ 出口 return：返回 tuple((redact_browser_typed_text_for_display(item, typed_tex…
        return tuple(redact_browser_typed_text_for_display(item, typed_text) for item in value)
    # ⤷ 出口 return：返回 value
    return value


# ┌── 函数 def redact_tool_args_for_display()
# │  作用：返回 复制的 工具 args safe 用于日志/progress UI (masks ``browser_type`` secrets).。
# │  参数：tool_name: str, args: dict | None
# │  返回：dict | None
# └──────────────────────────────────────────────────────────────
def redact_tool_args_for_display(tool_name: str, args: dict | None) -> dict | None:
    """Return a copy of tool args safe for logs/progress UI (masks ``browser_type`` secrets)."""
    if not isinstance(args, dict):
        # ⤷ 出口 return：返回 args
        return args
    if tool_name == "browser_type" and isinstance(args.get("text"), str):
        # ⤷ 出口 return：返回 {**args, 'text': redact_sensitive_text(args['text'], force=T…
        return {**args, "text": redact_sensitive_text(args["text"], force=True)}
    # ⤷ 出口 return：返回 args
    return args


# ┌── 函数 def _delegate_task_goals()
# │  作用：One truncated goal 字符串 per 字典 task (``?`` 当 missing).
# │  参数：tasks: Any, per_goal_len: int
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _delegate_task_goals(tasks: Any, *, per_goal_len: int) -> list[str]:
    """One truncated goal string per dict task (``?`` when missing)."""
    if not isinstance(tasks, list):
        # ⤷ 出口 return：返回 []
        return []
    raw_goals = (task.get("goal") for task in tasks if isinstance(task, dict))
    # ⤷ 出口 return：返回 [_tail_trunc(('?' if g is None else _oneline(str(g))) or '?'…
    return [_tail_trunc(("?" if g is None else _oneline(str(g))) or "?", per_goal_len) for g in raw_goals]


# ┌── 函数 def _browser_exec_step_label()
# │  作用：User-friendly 步骤 label 从 browser_exec code's leading comment.
# │  参数：args: dict, max_chars: int = 80
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _browser_exec_step_label(args: dict, max_chars: int = 80) -> str | None:
    """User-friendly step label from browser_exec code's leading comment."""
    first = str(args.get("code", "") or "").strip().split("\n", 1)[0].strip()
    label = first.lstrip("#").strip() if first.startswith("#") else ""
    if not label:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 label[:max_chars - 1] + '…' if len(label) > max_chars else l…
    return label[: max_chars - 1] + "…" if len(label) > max_chars else label


_PRIMARY_ARGS = {
    "terminal": "command", "web_search": "query", "web_extract": "urls", "read_file": "path",
    "write_file": "path", "patch": "path", "search_files": "pattern", "browser_navigate": "url",
    "browser_click": "ref", "browser_type": "text", "image_generate": "prompt", "text_to_speech": "text",
    "vision_analyze": "question", "skill_view": "name", "skills_list": "category", "cronjob_manage": "action",
    "execute_code": "code", "browser_exec": "code", "delegate_task": "goal", "clarify": "question",
    "skill_manage": "name",
}
_FALLBACK_PREVIEW_KEYS = ("query", "text", "command", "path", "name", "prompt", "code", "goal")


# ┌── 函数 def _delegate_action_preview()
# │  作用：Shared ``列表/steer/stop <id>`` preview 用于 delegate_task, 或 None 用于 spawn calls.
# │  参数：args: dict
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _delegate_action_preview(args: dict) -> str | None:
    """Shared ``list/steer/stop <id>`` preview for delegate_task, or None for spawn calls."""
    action = str(args.get("action") or "").strip().lower()
    if action in ("list", "steer", "stop"):
        # ⤷ 出口 return：返回 f"{action} {str(args.get('subagent_id') or '').strip()}".str…
        return f"{action} {str(args.get('subagent_id') or '').strip()}".strip()
    # ⤷ 出口 return：返回 None
    return None


# ┌── 函数 def _preview_browser_exec()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, max_len: int
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _preview_browser_exec(args: dict, max_len: int) -> str | None:
    label = _browser_exec_step_label(args)
    if label is not None:
        # ⤷ 出口 return：返回 _tail_trunc(label, max_len)
        return _tail_trunc(label, max_len)
    # ⤷ 出口 return：返回 _tail_trunc(_oneline(str(args.get('code', '') or '')), max_l…
    return _tail_trunc(_oneline(str(args.get("code", "") or "")), max_len) or None


# ┌── 函数 def _preview_delegate_task()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, max_len: int
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _preview_delegate_task(args: dict, max_len: int) -> str | None:
    action_preview = _delegate_action_preview(args)
    tasks = args.get("tasks")
    if action_preview is not None:
        # ⤷ 出口 return：返回 _tail_trunc(action_preview, max_len)
        return _tail_trunc(action_preview, max_len)
    if tasks and isinstance(tasks, list):
        goals = _delegate_task_goals(tasks, per_goal_len=40)
        preview = f"{len(goals)} tasks: " + " | ".join(goals) if goals else f"{len(tasks)} parallel tasks"
        # ⤷ 出口 return：返回 _tail_trunc(preview, max_len)
        return _tail_trunc(preview, max_len)
    goal = args.get("goal", "")
    # ⤷ 出口 return：返回 None if goal is None else _tail_trunc(_oneline(str(goal)), m…
    return None if goal is None else _tail_trunc(_oneline(str(goal)), max_len) or None


# ┌── 函数 def _preview_process_manage()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, _max_len: int
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _preview_process_manage(args: dict, _max_len: int) -> str | None:
    action, sid, data, timeout_val = (args.get(k) for k in ("action", "session_id", "data", "timeout"))
    parts = [str(action) if action else "", str(sid)[:16] if sid else "",
             f'"{_oneline(str(data)[:20])}"' if data else "", f"{timeout_val}s" if timeout_val and action == "wait" else ""]
    # ⤷ 出口 return：返回 ' '.join((p for p in parts if p)) or None
    return " ".join(p for p in parts if p) or None


# ┌── 函数 def _preview_todo_list()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, _max_len: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _preview_todo_list(args: dict, _max_len: int) -> str:
    todos_arg = args.get("todos")
    verb = "updating" if args.get("merge", False) else "planning"
    # ⤷ 出口 return：返回 'reading task list' if todos_arg is None else f'{verb} {len(…
    return "reading task list" if todos_arg is None else f"{verb} {len(todos_arg)} task(s)"


# ┌── 函数 def _preview_shell()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：key: str
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _preview_shell(key: str):
    def _build(args: dict, max_len: int) -> str | None:
        command = args.get(key)
        # ⤷ 出口 return：返回 None if command is None else _tail_trunc(summarize_shell_com…
        return None if command is None else _tail_trunc(summarize_shell_command(str(command)), max_len) or None
    # ⤷ 出口 return：返回 _build
    return _build


# ┌── 函数 def _preview_read_file()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, max_len: int
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _preview_read_file(args: dict, max_len: int) -> str | None:
    path = args.get("path") or args.get("file") or args.get("filepath")
    label = (Path(str(path).replace("\\", "/")).name or str(path)) if path is not None else None
    # ⤷ 出口 return：返回 None if label is None else _tail_trunc(f'{label} {_read_file…
    return None if label is None else _tail_trunc(f"{label} {_read_file_line_label(args)}".strip(), max_len) or None


# ┌── 函数 def _preview_memory()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, _max_len: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _preview_memory(args: dict, _max_len: int) -> str:
    action, target = args.get("action", ""), args.get("target", "")
    if action == "add":
        # ⤷ 出口 return：返回 f'''+{target}: "{_clip(_oneline(args.get('content', '')), 25…
        return f"+{target}: \"{_clip(_oneline(args.get('content', '')), 25)}\""
    if action in ("replace", "remove"):
        old = _oneline(args.get("old_text") or "") or "<missing old_text>"
        # ⤷ 出口 return：返回 f'''{('~' if action == 'replace' else '-')}{target}: "{old[:…
        return f"{'~' if action == 'replace' else '-'}{target}: \"{old[:20]}\""
    # ⤷ 出口 return：返回 action
    return action


# ┌── 函数 def _preview_send_message()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, _max_len: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _preview_send_message(args: dict, _max_len: int) -> str:
    # ⤷ 出口 return：返回 f'''to {args.get('target', '?')}: "{_tail_trunc(_oneline(arg…
    return f"to {args.get('target', '?')}: \"{_tail_trunc(_oneline(args.get('message', '')), 20)}\""


# ┌── 函数 def _preview_skill_view()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args: dict, max_len: int
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _preview_skill_view(args: dict, max_len: int) -> str | None:
    name = _oneline(str(args.get("name") or ""))
    file_path = args.get("file_path")
    label = (f"{name} → {_oneline(str(file_path))}" if name else _oneline(str(file_path))) if file_path else name
    # ⤷ 出口 return：返回 _tail_trunc(label, max_len) or None
    return _tail_trunc(label, max_len) or None


# Tool-specific preview builders: f(args, max_len) -> preview. Tools not listed
# fall through to the primary-argument lookup in build_tool_preview.
_PREVIEW_BUILDERS = {
    "browser_exec": _preview_browser_exec, "delegate_task": _preview_delegate_task,
    "process_manage": _preview_process_manage, "todo_list": _preview_todo_list,
    "terminal": _preview_shell("command"), "execute_code": _preview_shell("code"),
    "read_file": _preview_read_file, "memory": _preview_memory, "send_message": _preview_send_message,
    "skill_view": _preview_skill_view,
    "session_search": lambda args, _m: f"recall: \"{_clip(_oneline(args.get('query', '')), 25)}\"",
}


# ┌── 函数 def build_tool_preview()
# │  作用：构建 short preview 的工具 call's primary 参数用于展示.
# │  参数：tool_name: str, args: dict, max_len: int | None = None
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def build_tool_preview(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Build a short preview of a tool call's primary argument for display.

    *max_len* ``None`` defers to the global ``_tool_preview_max_len``; ``0`` means unlimited.
    """
    if max_len is None:
        max_len = _tool_preview_max_len
    if not args:
        # ⤷ 出口 return：返回 None
        return None
    args = redact_tool_args_for_display(tool_name, args) or args
    builder = _PREVIEW_BUILDERS.get(tool_name)
    if builder is not None:
        # ⤷ 出口 return：返回 builder(args, max_len)
        return builder(args, max_len)
    key = _PRIMARY_ARGS.get(tool_name) or next((k for k in _FALLBACK_PREVIEW_KEYS if k in args), None)
    if not key or key not in args:
        # ⤷ 出口 return：返回 None
        return None
    value = args[key]
    preview = _oneline(str((value[0] if value else "") if isinstance(value, list) else value))
    # ⤷ 出口 return：返回 _tail_trunc(preview, max_len) if preview else None
    return _tail_trunc(preview, max_len) if preview else None


# ┌── 函数 def prepare_tool_preview()
# │  作用：Compact preview plus explicit truncation/URL facts ( uncapped preview 是
# │  参数：tool_name: str, args: dict | None, fallback: str, max_len: int
# │  返回：ToolPreview
# └──────────────────────────────────────────────────────────────
def prepare_tool_preview(tool_name: str, args: dict | None, *, fallback: str, max_len: int) -> ToolPreview:
    """Compact preview plus explicit truncation/URL facts (the uncapped preview is
    rebuilt from the arguments so an upstream display cap cannot drop its link target)."""
    full_text = build_tool_preview(tool_name, args, max_len=0) or fallback
    text = _tail_trunc(full_text, max_len)
    truncated = text != full_text
    url = _http_url(_display_url(full_text)) if truncated else None
    # ⤷ 出口 return：返回 ToolPreview(text=text, truncated=truncated, url=url)
    return ToolPreview(text=text, truncated=truncated, url=url)


# ── Friendly tool labels: "web_search <q>" -> "Searching the web for <q>" ──
# Curated built-ins only — we know each core tool's semantics so the verb is fixed,
# not computed; custom/plugin/MCP tools have no entry and fall back to the raw preview.

_TOOL_VERBS: dict[str, str] = {
    "web_search": "Searching the web", "web_extract": "Reading",
    "browser_navigate": "Browsing", "browser_click": "Clicking", "browser_type": "Typing",
    "read_file": "Reading", "write_file": "Writing", "patch": "Editing", "search_files": "Searching files",
    "terminal": "Running", "execute_code": "Running code",
    "image_generate": "Generating image", "video_generate": "Generating video",
    "text_to_speech": "Generating speech", "vision_analyze": "Looking at the image",
    "session_search": "Searching past sessions",
    "skill_view": "Reading skill", "skills_list": "Listing skills", "skill_manage": "Updating skill",
    "delegate_task": "Delegating", "cronjob_manage": "Scheduling", "clarify": "Asking",
    "memory": "Updating memory", "todo_list": "Updating tasks",
}
# Verbs that read better without the argument preview appended.
_TOOL_VERBS_NO_PREVIEW: frozenset[str] = frozenset({"skills_list", "session_search"})
# Verbs joined to the preview with " for " (search-style phrasing).
_TOOL_VERBS_FOR_CONNECTOR: frozenset[str] = frozenset({"web_search", "search_files"})

# ┌── 函数 def get_tool_verb()
# │  作用：Friendly verb 用于已构建-在工具, 或 None (labels disabled / 无 curated verb);
# │  参数：tool_name: str
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def get_tool_verb(tool_name: str) -> str | None:
    """Friendly verb for a built-in tool, or None (labels disabled / no curated verb);
    callers compose ``f"{verb}{tool_verb_connector(tool)}{preview}"`` themselves."""
    # ⤷ 出口 return：返回 _TOOL_VERBS.get(tool_name) if _friendly_tool_labels else Non…
    return _TOOL_VERBS.get(tool_name) if _friendly_tool_labels else None


# ┌── 函数 def tool_verb_connector()
# │  作用：返回 connector between verb 且其 preview (" 用于 " 或 " ").。
# │  参数：tool_name: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def tool_verb_connector(tool_name: str) -> str:
    """Return the connector between a verb and its preview (" for " or " ")."""
    # ⤷ 出口 return：返回 ' for ' if tool_name in _TOOL_VERBS_FOR_CONNECTOR else ' '
    return " for " if tool_name in _TOOL_VERBS_FOR_CONNECTOR else " "


# ┌── 函数 def verb_drops_preview()
# │  作用：是否 verb 应当 render alone, 不参数 preview.。
# │  参数：tool_name: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def verb_drops_preview(tool_name: str) -> bool:
    """Whether the verb should render alone, without the argument preview."""
    # ⤷ 出口 return：返回 tool_name in _TOOL_VERBS_NO_PREVIEW
    return tool_name in _TOOL_VERBS_NO_PREVIEW


# ┌── 函数 def build_status_phrase()
# │  作用：Lowercase "是 <verb> <preview>…" phrase following 机器人's 展示名称 (Slack setStatus).
# │  参数：tool_name: str, args: dict | None, max_len: int = 49
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def build_status_phrase(tool_name: str, args: dict | None, max_len: int = 49) -> str | None:
    """Lowercase "is <verb> <preview>…" phrase following the bot's display name (Slack setStatus).

    ``args=None`` gives a verb-only phrase (``display.live_status: verb`` keeps previews out
    of shared channels). None for ``_thinking`` or disabled labels so callers use their
    static default. Default ``max_len`` stays under Slack's ~50-char status truncation.
    """
    if not tool_name or tool_name == "_thinking" or not _friendly_tool_labels:
        # ⤷ 出口 return：返回 None
        return None
    verb = _TOOL_VERBS.get(tool_name)
    phrase = f"is {verb[0].lower()}{verb[1:]}" if verb else f"is using {tool_name}"
    with_preview = args and verb and tool_name not in _TOOL_VERBS_NO_PREVIEW
    preview = build_tool_preview(tool_name, args, max_len=None) if with_preview else None
    if preview:  # previews can contain newlines (terminal commands); keep the first line
        phrase = f"{phrase}{tool_verb_connector(tool_name)}{preview.splitlines()[0].strip()}"
    # ⤷ 出口 return：返回 phrase[:max_len - 2].rstrip() + '…' if len(phrase) > max_len…
    return phrase[: max_len - 2].rstrip() + "…" if len(phrase) > max_len - 1 else phrase + "…"


# ┌── 函数 def build_tool_label()
# │  作用：Human-phrased label ("Searching web 用于 ...") 用于 curated 已构建-ins; other
# │  参数：tool_name: str, args: dict, max_len: int | None = None
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def build_tool_label(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Human-phrased label ("Searching the web for ...") for curated built-ins; other
    tools (or labels disabled) get the raw preview, so it is a drop-in for build_tool_preview."""
    verb = get_tool_verb(tool_name)
    if verb and tool_name in _TOOL_VERBS_NO_PREVIEW:
        # ⤷ 出口 return：返回 verb
        return verb
    preview = build_tool_preview(tool_name, args, max_len=max_len)
    if not verb:
        # ⤷ 出口 return：返回 preview
        return preview
    # ⤷ 出口 return：返回 f'{verb}{tool_verb_connector(tool_name)}{preview}' if previe…
    return f"{verb}{tool_verb_connector(tool_name)}{preview}" if preview else verb


# ── Inline diff previews for write actions ────────────────────────────────

# ┌── 函数 def _resolved_path()
# │  作用：Resolve possibly-relative filesystem 路径 against current cwd.
# │  参数：path: str
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _resolved_path(path: str) -> Path:
    """Resolve a possibly-relative filesystem path against the current cwd."""
    # ⤷ 出口 return：返回 Path.cwd() / Path(os.path.expanduser(path))
    return Path.cwd() / Path(os.path.expanduser(path))  # a `/` with an absolute rhs keeps the rhs


# ┌── 函数 def _snapshot_text()
# │  作用：返回 UTF-8 文件内容, 或 None 用于 missing/unreadable 文件.。
# │  参数：path: Path
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _snapshot_text(path: Path) -> str | None:
    """Return UTF-8 file content, or None for missing/unreadable files."""
    # ⤷ 关键分支 try/except：捕获 (UnicodeDecodeError, OSError)，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 path.read_text(encoding='utf-8')
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):  # FileNotFoundError/IsADirectoryError are OSErrors
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _display_diff_path()
# │  作用：Prefer cwd-relative 路径在 diffs 当 available.
# │  参数：path: Path
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _display_diff_path(path: Path) -> str:
    """Prefer cwd-relative paths in diffs when available."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 str(path.resolve().relative_to(Path.cwd().resolve()))
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        # ⤷ 出口 return：返回 str(path)
        return str(path)


# ┌── 函数 def _resolve_skill_manage_paths()
# │  作用：Resolve skill_manage 写入 targets filesystem 路径.
# │  参数：args: dict
# │  返回：list[Path]
# └──────────────────────────────────────────────────────────────
def _resolve_skill_manage_paths(args: dict) -> list[Path]:
    """Resolve skill_manage write targets to filesystem paths."""
    action = args.get("action")
    name = args.get("name")
    if not action or not name:
        # ⤷ 出口 return：返回 []
        return []
    from tools.skill_manager_tool import _find_skill, _resolve_skill_dir

    if action == "create":
        # ⤷ 出口 return：返回 [_resolve_skill_dir(name, args.get('category')) / 'SKILL.md'…
        return [_resolve_skill_dir(name, args.get("category")) / "SKILL.md"]
    existing = _find_skill(name)
    if not existing:
        # ⤷ 出口 return：返回 []
        return []
    skill_dir = Path(existing["path"])
    file_path = args.get("file_path")
    if action == "delete":
        # ⤷ 出口 return：返回 [path for path in sorted(skill_dir.rglob('*')) if path.is_fi…
        return [path for path in sorted(skill_dir.rglob("*")) if path.is_file()]
    if file_path and action in {"edit", "patch", "write_file", "remove_file"}:
        # ⤷ 出口 return：返回 [skill_dir / file_path]
        return [skill_dir / file_path]
    # ⤷ 出口 return：返回 [skill_dir / 'SKILL.md'] if action in {'edit', 'patch'} else…
    return [skill_dir / "SKILL.md"] if action in {"edit", "patch"} else []


# ┌── 函数 def _resolve_local_edit_paths()
# │  作用：Resolve local filesystem targets 用于写入-capable 工具.
# │  参数：tool_name: str, function_args: dict | None
# │  返回：list[Path]
# └──────────────────────────────────────────────────────────────
def _resolve_local_edit_paths(tool_name: str, function_args: dict | None) -> list[Path]:
    """Resolve local filesystem targets for write-capable tools."""
    if not isinstance(function_args, dict):
        # ⤷ 出口 return：返回 []
        return []
    if tool_name == "skill_manage":
        # ⤷ 出口 return：返回 _resolve_skill_manage_paths(function_args)
        return _resolve_skill_manage_paths(function_args)
    path = function_args.get("path") if tool_name in {"write_file", "patch"} else None
    # ⤷ 出口 return：返回 [_resolved_path(path)] if path else []
    return [_resolved_path(path)] if path else []


# ┌── 函数 def capture_local_edit_snapshot()
# │  作用：Capture 在…之前-状态用于 local 写入 previews.
# │  参数：tool_name: str, function_args: dict | None
# │  返回：LocalEditSnapshot | None
# └──────────────────────────────────────────────────────────────
def capture_local_edit_snapshot(tool_name: str, function_args: dict | None) -> LocalEditSnapshot | None:
    """Capture before-state for local write previews."""
    paths = _resolve_local_edit_paths(tool_name, function_args)
    if not paths:
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 LocalEditSnapshot(paths=paths, before={str(path): _snapshot_…
    return LocalEditSnapshot(paths=paths, before={str(path): _snapshot_text(path) for path in paths})


# ┌── 函数 def _result_succeeded()
# │  作用：Conservatively 检测 whether 工具结果 represents success.
# │  参数：result: str | None
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _result_succeeded(result: str | None) -> bool:
    """Conservatively detect whether a tool result represents success."""
    data = safe_json_loads(result) if result else None
    if not isinstance(data, dict) or data.get("error"):
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 出口 return：返回 bool(data.get('success')) if 'success' in data else True
    return bool(data.get("success")) if "success" in data else True


# ┌── 函数 def _diff_from_snapshot()
# │  作用：Generate unified diff 文本从 stored 在…之前-状态且 current 文件.
# │  参数：snapshot: LocalEditSnapshot | None
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def _diff_from_snapshot(snapshot: LocalEditSnapshot | None) -> str | None:
    """Generate unified diff text from a stored before-state and current files."""
    if not snapshot:
        # ⤷ 出口 return：返回 None
        return None
    chunks: list[str] = []
    for path in snapshot.paths:
        before, after = snapshot.before.get(str(path)), _snapshot_text(path)
        if before == after:
            continue
        display_path = _display_diff_path(path)
        diff = "".join(unified_diff(
            (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
            fromfile=f"a/{display_path}", tofile=f"b/{display_path}",
        ))
        if diff:
            chunks.append(diff if diff.endswith("\n") else diff + "\n")
    # ⤷ 出口 return：返回 ''.join(chunks) or None
    return "".join(chunks) or None


# ┌── 函数 def extract_edit_diff()
# │  作用：Extract unified diff 从文件-edit 工具结果.
# │  参数：tool_name: str, result: str | None, function_args: dict | None = ..., snapshot: LocalEditSnapshot | None = ...
# │  返回：str | None
# └──────────────────────────────────────────────────────────────
def extract_edit_diff(
    tool_name: str, result: str | None, *,
    function_args: dict | None = None, snapshot: LocalEditSnapshot | None = None,
) -> str | None:
    """Extract a unified diff from a file-edit tool result."""
    if tool_name == "patch" and result:
        data = safe_json_loads(result)
        diff = data.get("diff") if isinstance(data, dict) else None
        if isinstance(diff, str) and diff.strip():
            # ⤷ 出口 return：返回 diff
            return diff
    if tool_name not in {"write_file", "patch", "skill_manage"} or not _result_succeeded(result):
        # ⤷ 出口 return：返回 None
        return None
    # ⤷ 出口 return：返回 _diff_from_snapshot(snapshot)
    return _diff_from_snapshot(snapshot)


# ┌── 函数 def _emit_inline_diff()
# │  作用：Emit rendered diff 文本 through CLI's prompt_toolkit-safe printer.
# │  参数：diff_text: str, print_fn
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _emit_inline_diff(diff_text: str, print_fn) -> bool:
    """Emit rendered diff text through the CLI's prompt_toolkit-safe printer."""
    if print_fn is None or not diff_text:
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        for line in ["  ┊ review diff", *diff_text.rstrip("\n").splitlines()]:
            print_fn(line)
        # ⤷ 出口 return：返回 True
        return True
    except Exception:
        # ⤷ 出口 return：返回 False
        return False


# Unified-diff line prefix -> diff color key (checked in order; "--- "/"+++ " handled first).
_DIFF_LINE_COLORS = (("@@", "hunk"), ("-", "minus"), ("+", "plus"), (" ", "dim"))


# ┌── 函数 def _render_inline_unified_diff()
# │  作用：Render unified diff lines 在 Hermes' inline transcript style.
# │  参数：diff: str
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _render_inline_unified_diff(diff: str) -> list[str]:
    """Render unified diff lines in Hermes' inline transcript style."""
    rendered: list[str] = []
    from_file = to_file = None
    for raw_line in diff.splitlines():
        if raw_line.startswith("--- "):
            from_file = raw_line[4:].strip()
            continue
        if raw_line.startswith("+++ "):
            to_file = raw_line[4:].strip()
            if from_file or to_file:
                rendered.append(f"{_diff_ansi()['file']}{from_file or 'a/?'} → {to_file or 'b/?'}{_ANSI_RESET}")
            continue
        color = next((c for prefix, c in _DIFF_LINE_COLORS if raw_line.startswith(prefix)), None)
        if color:
            rendered.append(f"{_diff_ansi()[color]}{raw_line}{_ANSI_RESET}")
        elif raw_line:
            rendered.append(raw_line)
    # ⤷ 出口 return：返回 rendered
    return rendered


# ┌── 函数 def _split_unified_diff_sections()
# │  作用：Split unified diff 进入 per-文件 sections.
# │  参数：diff: str
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _split_unified_diff_sections(diff: str) -> list[str]:
    """Split a unified diff into per-file sections."""
    sections: list[list[str]] = [[]]
    for line in diff.splitlines():
        if line.startswith("--- ") and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    # ⤷ 出口 return：返回 ['\n'.join(section) for section in sections if section]
    return ["\n".join(section) for section in sections if section]


# ┌── 函数 def _summarize_rendered_diff_sections()
# │  作用：Render diff sections 当 capping 文件 count 且 total line count.
# │  参数：diff: str, max_files: int = ..., max_lines: int = ...
# │  返回：list[str]
# └──────────────────────────────────────────────────────────────
def _summarize_rendered_diff_sections(
    diff: str, *, max_files: int = _MAX_INLINE_DIFF_FILES, max_lines: int = _MAX_INLINE_DIFF_LINES,
) -> list[str]:
    """Render diff sections while capping file count and total line count."""
    sections = _split_unified_diff_sections(diff)
    rendered: list[str] = []
    omitted_files = omitted_lines = 0
    for idx, section in enumerate(sections):
        section_lines = _render_inline_unified_diff(section)
        remaining_budget = max_lines - len(rendered)
        if idx >= max_files or remaining_budget <= 0:
            omitted_files += 1
            omitted_lines += len(section_lines)
            continue
        if len(section_lines) <= remaining_budget:
            rendered.extend(section_lines)
            continue
        rendered.extend(section_lines[:remaining_budget])
        omitted_lines += len(section_lines) - remaining_budget
        omitted_files += 1 + max(0, len(sections) - idx - 1)
        for leftover in sections[idx + 1:]:
            omitted_lines += len(_render_inline_unified_diff(leftover))
        break
    if omitted_files or omitted_lines:
        summary = f"… omitted {omitted_lines} diff line(s)"
        if omitted_files:
            summary += f" across {omitted_files} additional file(s)/section(s)"
        rendered.append(f"{_diff_ansi()['hunk']}{summary}{_ANSI_RESET}")
    # ⤷ 出口 return：返回 rendered
    return rendered


# ┌── 函数 def render_edit_diff_with_delta()
# │  作用：Render edit diff inline 不 taking over 终端 UI.
# │  参数：tool_name: str, result: str | None, function_args: dict | None = ..., snapshot: LocalEditSnapshot | None = ..., print_fn = ...
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def render_edit_diff_with_delta(
    tool_name: str, result: str | None, *,
    function_args: dict | None = None, snapshot: LocalEditSnapshot | None = None, print_fn=None,
) -> bool:
    """Render an edit diff inline without taking over the terminal UI."""
    diff = extract_edit_diff(tool_name, result, function_args=function_args, snapshot=snapshot)
    if not diff:
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        rendered_lines = _summarize_rendered_diff_sections(diff)
    except Exception as exc:
        logger.debug("Could not render inline diff: %s", exc)
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 出口 return：返回 _emit_inline_diff('\n'.join(rendered_lines), print_fn)
    return _emit_inline_diff("\n".join(rendered_lines), print_fn)


# ── KawaiiSpinner ─────────────────────────────────────────────────────────

# ┌── 类 class KawaiiSpinner
# │  说明：Animated spinner 带 kawaii faces 用于 CLI feedback 在…期间工具 execution.
# │  成员：共 13 个方法；对外可用 4 个；含魔术方法 __init__, __enter__, __exit__
# └──────────────────────────────────────────────────────────────
class KawaiiSpinner:
    """Animated spinner with kawaii faces for CLI feedback during tool execution."""

    SPINNERS = {
        'dots': list('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'), 'bounce': list('⠁⠂⠄⡀⢀⠠⠐⠈'), 'grow': list('▁▂▃▄▅▆▇█▇▆▅▄▃▂'),
        'arrows': list('←↖↑↗→↘↓↙'), 'star': list('✶✷✸✹✺✹✸✷'), 'moon': list('🌑🌒🌓🌔🌕🌖🌗🌘'),
        'pulse': list('◜◠◝◞◡◟'), 'brain': list('🧠💭💡✨💫🌟💡💭'), 'sparkle': list('⁺˚*✧✦✧*˚'),
    }
    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ", "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]
    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)", "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]
    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating", "deliberating", "mulling",
        "reflecting", "processing", "reasoning", "analyzing", "computing", "synthesizing", "formulating",
        "brainstorming",
    ]

    # ┌── 函数 def _skin_spinner_list()
    # │  作用：当 absent/empty. 时返回 active skin's ``spinner[键]`` 列表, 或 *兜底*。
    # │  参数：key: str, fallback: list
    # │  返回：list
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _skin_spinner_list(key: str, fallback: list) -> list:
        """Return the active skin's ``spinner[key]`` list, or *fallback* when absent/empty."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            skin = _get_skin()
            # ⤷ 出口 return：返回 (skin.spinner.get(key, []) if skin else None) or fallback
            return (skin.spinner.get(key, []) if skin else None) or fallback
        except Exception:
            # ⤷ 出口 return：返回 fallback
            return fallback

    get_waiting_faces = classmethod(lambda cls: cls._skin_spinner_list("waiting_faces", cls.KAWAII_WAITING))
    get_thinking_faces = classmethod(lambda cls: cls._skin_spinner_list("thinking_faces", cls.KAWAII_THINKING))
    get_thinking_verbs = classmethod(lambda cls: cls._skin_spinner_list("thinking_verbs", cls.THINKING_VERBS))

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, message: str = '', spinner_type: str = 'dots', print_fn = None
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __init__(self, message: str = "", spinner_type: str = 'dots', print_fn=None):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = self.start_time = None
        self.frame_idx = self.last_line_len = 0
        self._print_fn = print_fn  # when set, bypasses self._out so silenced agents stay silent
        self._out = sys.stdout  # captured NOW, before any child redirect_stdout(devnull) replaces it

    # ┌── 函数 def _write()
    # │  作用：写入 via print_fn 当 supplied, 否则标准输出 captured 在 creation.
    # │  参数：self, text: str, end: str = '\n', flush: bool = False
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _write(self, text: str, end: str = '\n', flush: bool = False):
        """Write via print_fn when supplied, else to the stdout captured at creation."""
        if self._print_fn is not None:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                self._print_fn(text)
            except Exception:
                pass
            return
        # ⤷ 关键分支 try/except：捕获 (ValueError, OSError)，出错走兜底而不是崩溃。
        try:
            self._out.write(text + end)
            if flush:
                self._out.flush()
        except (ValueError, OSError):
            pass

    # ┌── 函数 def _is_tty()
    # │  作用：检查若 输出是 real 终端, safe against 已关闭 streams.
    # │  参数：self
    # │  返回：bool
    # │  【语法】@property 把方法伪装成属性：外面写 obj.x 而不是 obj.x()。
    # └──────────────────────────────────────────────────────────────
    @property
    def _is_tty(self) -> bool:
        """Check if output is a real terminal, safe against closed streams."""
        # ⤷ 关键分支 try/except：捕获 (ValueError, OSError)，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 hasattr(self._out, 'isatty') and self._out.isatty()
            return hasattr(self._out, 'isatty') and self._out.isatty()
        except (ValueError, OSError):
            # ⤷ 出口 return：返回 False
            return False

    # ┌── 函数 def _is_patch_stdout_proxy()
    # │  作用：当 prompt_toolkit's StdoutProxy: 它注入 newlines around 每个冲刷从而 时返回 True。
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _is_patch_stdout_proxy(self) -> bool:
        """True for prompt_toolkit's StdoutProxy: it injects newlines around each flush so the
        \\r overwrite never lands, and the CLI drives its own TUI spinner widget in that mode."""
        # ⤷ 关键分支 try/except：捕获 ImportError，出错走兜底而不是崩溃。
        try:
            from prompt_toolkit.patch_stdout import StdoutProxy
            # ⤷ 出口 return：返回 isinstance(self._out, StdoutProxy)
            return isinstance(self._out, StdoutProxy)
        except ImportError:
            # ⤷ 出口 return：返回 False
            return False

    # ┌── 函数 def _animate()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _animate(self):
        tty = self._is_tty
        # Non-TTY (Docker, systemd, pipe): log once instead of spamming frames.
        if not tty:
            self._write(f"  [tool] {self.message}", flush=True)
        # Under patch_stdout the \r animation would overdraw the TUI status bar.
        if not tty or self._is_patch_stdout_proxy():
            while self.running:
                time.sleep(0.5 if not tty else 0.1)
            return
        skin = _get_skin()
        wings = skin.get_spinner_wings() if skin else []
        while self.running:
            if os.getenv("HERMES_SPINNER_PAUSE"):
                time.sleep(0.1)
                continue
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = f"({time.time() - self.start_time:.1f}s)"
            left, right = wings[self.frame_idx % len(wings)] if wings else ("", "")
            line = f"  {left} {frame} {self.message} {right} {elapsed}" if wings else f"  {frame} {self.message} {elapsed}"
            self._write(f"\r{line}{' ' * max(self.last_line_len - len(line), 0)}", end='', flush=True)
            self.last_line_len = len(line)
            self.frame_idx += 1
            time.sleep(0.12)

    # ┌── 函数 def start()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def start(self):
        if self.running:
            return
        self.running, self.start_time = True, time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    # ┌── 函数 def update_text()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, new_message: str
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def update_text(self, new_message: str):
        self.message = new_message

    # ┌── 函数 def _clear_line_blanks()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：self
    # │  返回：str
    # └──────────────────────────────────────────────────────────────
    def _clear_line_blanks(self) -> str:
        # ⤷ 出口 return：返回 ' ' * max(self.last_line_len + 5, 40)
        return ' ' * max(self.last_line_len + 5, 40)  # spaces, not \033[K: garbles under patch_stdout

    # ┌── 函数 def print_above()
    # │  作用：打印 line above spinner (next tick redraws 它 below). Works inside
    # │  参数：self, text: str
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def print_above(self, text: str):
        """Print a line above the spinner (next tick redraws it below). Works inside
        redirect_stdout(devnull) because _write targets the stdout captured at creation."""
        self._write(f"\r{self._clear_line_blanks()}\r  {text}" if self.running else f"  {text}", flush=True)

    # ┌── 函数 def stop()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self, final_message: str = None
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        is_tty = self._is_tty
        if is_tty:
            self._write(f"\r{self._clear_line_blanks()}\r", end='', flush=True)
        if final_message:
            elapsed = f" ({time.time() - self.start_time:.1f}s)" if self.start_time else ""
            self._write(f"  {final_message}" if is_tty else f"  [done] {final_message}{elapsed}", flush=True)

    # ┌── 函数 def __enter__()
    # │  作用：魔术方法 __enter__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        # ⤷ 出口 return：返回 self
        return self

    # ┌── 函数 def __exit__()
    # │  作用：魔术方法 __exit__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self, *exc
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __exit__(self, *exc):
        self.stop()
        # ⤷ 出口 return：返回 False
        return False


# ── Cute tool message (completion line that replaces the spinner) ─────────

_ERROR_SUFFIX_MAX_LEN = 48


# ┌── 函数 def _trim_error()
# │  作用：Shrink 错误消息用于 inline 展示 (long '文件不 found' 路径 -> filename).
# │  参数：msg: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _trim_error(msg: str) -> str:
    """Shrink an error message for inline display (long 'File not found' paths -> filename)."""
    msg = msg.strip()
    if "File not found:" in msg:
        tail = msg.partition("File not found:")[2].strip()
        if "/" in tail:
            msg = f"File not found: {tail.rsplit('/', 1)[-1]}"
    # ⤷ 出口 return：返回 _tail_trunc(msg, _ERROR_SUFFIX_MAX_LEN)
    return _tail_trunc(msg, _ERROR_SUFFIX_MAX_LEN)


# ┌── 函数 def _detect_tool_failure()
# │  作用：返回 ``(is_failure, suffix)`` 用于工具结果, e.g. ``(True, " [exit 1]")``.。
# │  参数：tool_name: str, result: str | None
# │  返回：tuple[bool, str]
# └──────────────────────────────────────────────────────────────
def _detect_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Return ``(is_failure, suffix)`` for a tool result, e.g. ``(True, " [exit 1]")``."""
    if result is None or file_mutation_result_landed(tool_name, result):
        # ⤷ 出口 return：返回 (False, '')
        return False, ""
    data = safe_json_loads(result)

    # Terminal: non-zero exit code is the canonical failure signal.
    if tool_name == "terminal":
        exit_code = data.get("exit_code") if isinstance(data, dict) else None
        if exit_code is None or exit_code == 0:
            # ⤷ 出口 return：返回 (False, '')
            return False, ""
        err_msg = data.get("error")
        # ⤷ 出口 return：返回 (True, f' [{_trim_error(str(err_msg))}]' if err_msg else f' …
        return True, f" [{_trim_error(str(err_msg))}]" if err_msg else f" [exit {exit_code}]"

    if isinstance(data, dict):
        failed = data.get("success") is False
        # Memory: distinguish "store full" from real errors.
        if tool_name == "memory" and failed and "exceed the limit" in data.get("error", ""):
            # ⤷ 出口 return：返回 (True, ' [full]')
            return True, " [full]"
        err = data.get("error") or data.get("message")
        if err and (failed or "error" in data):
            return True, f" [{_trim_error(str(err))}]"
    # Multimodal results (dicts) are successes; failures arrive as JSON-encoded strings.
    if isinstance(result, str) and (
        '"error"' in result[:500].lower() or '"failed"' in result[:500].lower() or result.startswith("Error")
    ):
        # ⤷ 出口 return：返回 (True, ' [error]')
        return True, " [error]"
    # ⤷ 出口 return：返回 (False, '')
    return False, ""


# ┌── 函数 def _domain()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：url: str
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _domain(url: str) -> str:
    # ⤷ 出口 return：返回 url.replace('https://', '').replace('http://', '').split('/'…
    return url.replace("https://", "").replace("http://", "").split("/")[0]


# ┌── 函数 def _cute_trunc()
# │  作用：按配置的预览上限做尾部截断（0 表示不限）。
# │  参数：s
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_trunc(s) -> str:
    """Tail-truncate to the configured preview cap (0 = unlimited)."""
    # ⤷ 出口 return：返回 _tail_trunc(str(s), _tool_preview_max_len)
    return _tail_trunc(str(s), _tool_preview_max_len)


# ┌── 函数 def _cute_path()
# │  作用：Head-truncate 路径 configured preview cap, 保留 filename end.
# │  参数：p
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_path(p) -> str:
    """Head-truncate a path to the configured preview cap, keeping the filename end."""
    p = str(p)
    limit = _tool_preview_max_len
    if not limit or len(p) <= limit:
        # ⤷ 出口 return：返回 p
        return p
    # ⤷ 出口 return：返回 '.' * limit if limit <= 3 else '...' + p[-(limit - 3):]
    return "." * limit if limit <= 3 else "..." + p[-(limit - 3):]


# ┌── 函数 def _cute_web_extract()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_web_extract(a: dict, _r) -> str:
    urls = a.get("urls", [])
    url = _display_url(urls[0] if isinstance(urls, list) else urls) if urls else ""
    if not url:
        # ⤷ 出口 return：返回 '┊ 📄 fetch     pages'
        return "┊ 📄 fetch     pages"
    extra = f" +{len(urls)-1}" if isinstance(urls, list) and len(urls) > 1 else ""
    # ⤷ 出口 return：返回 f'┊ 📄 fetch     {_cute_trunc(_domain(url))}{extra}'
    return f"┊ 📄 fetch     {_cute_trunc(_domain(url))}{extra}"


# ┌── 函数 def _cute_todo_list()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, result
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_todo_list(a: dict, result) -> str:
    todos_arg = a.get("todos")
    total = done = 0
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        summary = (safe_json_loads(result) or {}).get("summary", {}) if result else {}
        total, done = summary.get("total", 0), summary.get("completed", 0)
    except Exception:
        pass
    if todos_arg is None:
        detail = f"{done}/{total} task(s)" if total > 0 else "reading tasks"
    elif a.get("merge", False):
        detail = f"update {done}/{total} ✓" if total > 0 and done > 0 else f"update {len(todos_arg)} task(s)"
    else:
        detail = f"{done}/{total} task(s)" if total > 0 and done > 0 else f"{len(todos_arg)} task(s)"
    # ⤷ 出口 return：返回 f'┊ 📋 plan      {detail}'
    return f"┊ 📋 plan      {detail}"


# ┌── 函数 def _cute_memory()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_memory(a: dict, _r) -> str:
    action, target = a.get("action", "?"), a.get("target", "")
    if action == "add":
        # ⤷ 出口 return：返回 f'''┊ 🧠 memory    +{target}: "{_cute_trunc(a.get('content', …
        return f"┊ 🧠 memory    +{target}: \"{_cute_trunc(a.get('content', ''))}\""
    if action in ("replace", "remove"):
        old = a.get("old_text") or "<missing old_text>"
        # ⤷ 出口 return：返回 f'''┊ 🧠 memory    {('~' if action == 'replace' else '-')}{ta…
        return f"┊ 🧠 memory    {'~' if action == 'replace' else '-'}{target}: \"{_cute_trunc(old)}\""
    # ⤷ 出口 return：返回 f'┊ 🧠 memory    {action}'
    return f"┊ 🧠 memory    {action}"


# ┌── 函数 def _cute_skill_view()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_skill_view(a: dict, _r) -> str:
    label, file_path = a.get("name", ""), a.get("file_path")
    label = (f"{label} → {file_path}" if label else str(file_path)) if file_path else label
    # ⤷ 出口 return：返回 f'┊ 📚 skill     {_cute_trunc(label)}'
    return f"┊ 📚 skill     {_cute_trunc(label)}"


# ┌── 函数 def _cute_cronjob()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_cronjob(a: dict, _r) -> str:
    action = a.get("action", "?")
    if action == "create":
        skills = a.get("skills") or ([a.get("skill")] if a.get("skill") else [])
        label = a.get("name") or (skills[0] if skills else None) or a.get("prompt", "task")
        # ⤷ 出口 return：返回 f'┊ ⏰ cron      create {_cute_trunc(label)}'
        return f"┊ ⏰ cron      create {_cute_trunc(label)}"
    # ⤷ 出口 return：返回 '┊ ⏰ cron      listing' if action == 'list' else f"┊ ⏰ cron …
    return "┊ ⏰ cron      listing" if action == "list" else f"┊ ⏰ cron      {action} {a.get('job_id', '')}"


# ┌── 函数 def _cute_execute_code()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_execute_code(a: dict, _r) -> str:
    code = a.get("code", "").strip()
    # ⤷ 出口 return：返回 f"┊ 🐍 exec      {_cute_trunc(code.split(chr(10))[0] if code …
    return f"┊ 🐍 exec      {_cute_trunc(code.split(chr(10))[0] if code else '')}"


# ┌── 函数 def _cute_browser_exec()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_browser_exec(a: dict, _r) -> str:
    # Leading `# …` comment becomes the step label; code stays collapsed behind the preview cap.
    label = _browser_exec_step_label(a)
    # ⤷ 出口 return：返回 f"┊ 🌐 browser   {(_cute_trunc(_oneline(str(a.get('code', '')…
    return f"┊ 🌐 browser   {_cute_trunc(_oneline(str(a.get('code', '') or ''))) if label is None else label}"


# ┌── 函数 def _cute_delegate()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_delegate(a: dict, _r) -> str:
    action_preview = _delegate_action_preview(a)
    tasks = a.get("tasks")
    if action_preview is not None:
        # ⤷ 出口 return：返回 f'┊ 🔀 delegate  {_cute_trunc(action_preview)}'
        return f"┊ 🔀 delegate  {_cute_trunc(action_preview)}"
    if tasks and isinstance(tasks, list):
        goals = _delegate_task_goals(tasks, per_goal_len=30)
        # ⤷ 出口 return：返回 f"┊ 🔀 delegate  {len(goals) or len(tasks)}x: {_cute_trunc(' …
        return f"┊ 🔀 delegate  {len(goals) or len(tasks)}x: {_cute_trunc(' | '.join(goals) if goals else 'parallel')}"
    # ⤷ 出口 return：返回 f"┊ 🔀 delegate  {_cute_trunc(a.get('goal', ''))}"
    return f"┊ 🔀 delegate  {_cute_trunc(a.get('goal', ''))}"


# ┌── 函数 def _cute_process_manage()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：a: dict, _r
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _cute_process_manage(a: dict, _r) -> str:
    action, sid = a.get("action", "?"), a.get("session_id", "")[:12]
    # ⤷ 出口 return：返回 f"┊ ⚙️  proc      {('ls processes' if action == 'list' else …
    return f"┊ ⚙️  proc      {'ls processes' if action == 'list' else f'{action} {sid}'}"


_SCROLL_ARROWS = {"down": "↓", "up": "↑", "right": "→", "left": "←"}

# Completion-line renderers: tool -> f(args, result) -> "┊ {emoji} {verb:9} {detail}" (duration appended by caller).
_CUTE_LINES = {
    "web_search": lambda a, r: f"┊ 🔍 search    {_cute_trunc(a.get('query', ''))}",
    "web_extract": _cute_web_extract,
    "terminal": lambda a, r: f"┊ 💻 $         {_cute_trunc(build_tool_preview('terminal', a) or a.get('command', ''))}",
    "process_manage": _cute_process_manage,
    "read_file": lambda a, r: f"┊ 📖 read      {_cute_trunc(build_tool_preview('read_file', a) or a.get('path', ''))}",
    "write_file": lambda a, r: f"┊ ✍️  write     {_cute_path(a.get('path', ''))}",
    "patch": lambda a, r: f"┊ 🔧 patch     {_cute_path(a.get('path', ''))}",
    "search_files": lambda a, r: f"┊ 🔎 {'find' if a.get('target', 'content') == 'files' else 'grep':9} {_cute_trunc(a.get('pattern', ''))}",
    "browser_navigate": lambda a, r: f"┊ 🌐 navigate  {_cute_trunc(_domain(a.get('url', '')))}",
    "browser_snapshot": lambda a, r: f"┊ 📸 snapshot  {'full' if a.get('full') else 'compact'}",
    "browser_click": lambda a, r: f"┊ 👆 click     {a.get('ref', '?')}",
    "browser_type": lambda a, r: f"┊ ⌨️  type      \"{_cute_trunc(a.get('text', ''))}\"",
    "browser_scroll": lambda a, r: f"┊ {_SCROLL_ARROWS.get(a.get('direction', 'down'), '↓')}  scroll    {a.get('direction', 'down')}",
    "browser_back": lambda a, r: "┊ ◀️  back    ",
    "browser_press": lambda a, r: f"┊ ⌨️  press     {a.get('key', '?')}",
    "browser_get_images": lambda a, r: "┊ 🖼️  images    extracting",
    "browser_vision": lambda a, r: "┊ 👁️  vision    analyzing page",
    "todo_list": _cute_todo_list,
    "session_search": lambda a, r: f"┊ 🔍 recall    \"{_cute_trunc(a.get('query', ''))}\"",
    "memory": _cute_memory,
    "skills_list": lambda a, r: f"┊ 📚 skills    list {a.get('category', 'all')}",
    "skill_view": _cute_skill_view,
    "image_generate": lambda a, r: f"┊ 🎨 create    {_cute_trunc(a.get('prompt', ''))}",
    "text_to_speech": lambda a, r: f"┊ 🔊 speak     {_cute_trunc(a.get('text', ''))}",
    "vision_analyze": lambda a, r: f"┊ 👁️  vision    {_cute_trunc(a.get('question', ''))}",
    "send_message": lambda a, r: f"┊ 📨 send      {a.get('target', '?')}: \"{_cute_trunc(a.get('message', ''))}\"",
    "cronjob_manage": _cute_cronjob,
    "execute_code": _cute_execute_code,
    "browser_exec": _cute_browser_exec,
    "delegate_task": _cute_delegate,
}


# ┌── 函数 def _get_cute_tool_message()
# │  作用：工具 completion line 用于 CLI quiet 模式: ``| {emoji} {verb:9} {detail} {duration}``, plus
# │  参数：tool_name: str, args: dict, duration: float, result: str | None = None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _get_cute_tool_message(tool_name: str, args: dict, duration: float, result: str | None = None) -> str:
    """Tool completion line for CLI quiet mode: ``| {emoji} {verb:9} {detail}  {duration}``, plus a
    failure suffix from :func:`_detect_tool_failure`; the leading ``┊`` becomes the skin's tool prefix."""
    args = redact_tool_args_for_display(tool_name, args) or args
    is_failure, failure_suffix = _detect_tool_failure(tool_name, result)
    render = _CUTE_LINES.get(tool_name)
    body = render(args, result) if render else f"┊ ⚡ {tool_name[:9]:9} {_cute_trunc(build_tool_preview(tool_name, args) or '')}"
    line = f"{body}  {duration:.1f}s".replace("┊", get_skin_tool_prefix(), 1)
    # ⤷ 出口 return：返回 f'{line}{failure_suffix}' if is_failure else line
    return f"{line}{failure_suffix}" if is_failure else line


# ┌── 函数 def get_cute_tool_message()
# │  作用：Render completion label 不 letting cosmetic failures escape.
# │  参数：tool_name: str, args: dict, duration: float, result: str | None = None
# │  返回：str
# └──────────────────────────────────────────────────────────────
def get_cute_tool_message(tool_name: str, args: dict, duration: float, result: str | None = None) -> str:
    """Render a completion label without letting cosmetic failures escape."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 _get_cute_tool_message(tool_name, args, duration, result=res…
        return _get_cute_tool_message(tool_name, args, duration, result=result)
    except Exception as exc:  # noqa: BLE001 — display must never abort a turn
        logger.debug("Tool completion label failed for %s: %s", tool_name, exc)
        safe_name = tool_name[:9] if isinstance(tool_name, str) and tool_name else "tool"
        safe_duration = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "done"
        # ⤷ 出口 return：返回 f'┊ ⚡ {safe_name:9} completed  {safe_duration}'
        return f"┊ ⚡ {safe_name:9} completed  {safe_duration}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

# ┌── 函数 def get_friendly_tool_labels()
# │  作用：返回 是否 friendly 工具 labels 是 enabled.。。
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def get_friendly_tool_labels() -> bool:
    """Return whether friendly tool labels are enabled."""
    # ⤷ 出口 return：返回 _friendly_tool_labels
    return _friendly_tool_labels
# ---- END PLUGIN-COMPAT ----
