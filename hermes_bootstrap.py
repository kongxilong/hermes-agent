"""Windows UTF-8 bootstrap for Hermes entry points (no-op on POSIX).

Windows binds stdio to the console code page (cp1252), so ``print("café")`` raises
``UnicodeEncodeError``, and Python children inherit the same default unless
``PYTHONUTF8``/``PYTHONIOENCODING`` are set. Import this module first in every entry
point (``hermes``, ``hermes-agent``, ``hermes-acp``, ``gateway.run``, ``batch_runner``,
``cron/scheduler``). It does NOT re-exec with ``-X utf8``: ``open()`` in the current
process still needs an explicit ``encoding="utf-8"`` (ruff ``PLW1514``). POSIX is left
alone deliberately — users' ``LANG``/``LC_*`` choices are respected.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：hermes_bootstrap.py（约 119 行，很短，建议完整读完）
# 地位：启动引导小工具，被所有入口脚本最先 import。
#
# 解决什么问题：
#   Windows 把标准输入输出绑定到系统代码页（cp1252），所以 print 一个非 ASCII 字符（比如 café、中文）
#   会直接抛 UnicodeEncodeError。本文件就是在 Windows 上强制切到 UTF-8 模式；在 macOS/Linux 上是空操作。
#
# 四个函数：
#   apply_windows_utf8_bootstrap()   设置 PYTHONUTF8 / PYTHONIOENCODING 等环境变量
#   suppress_platform_ver_console()  关掉 Windows 上那条多余的 platform 版本输出
#   harden_import_path(src_root)     整理 sys.path，防止导入到错误的同名模块
#   activate_durable_lazy_target()   启用懒加载目标
#
# 给新手的启发：这是「跨平台兼容层」的典型写法——把平台差异集中在一个小文件里，其他代码假装它不存在。
# ==========================================================================
#

from __future__ import annotations

import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False


# ┌── 函数 def apply_windows_utf8_bootstrap()
# │  作用：Apply Windows UTF-8 bootstrap 一旦; True 仅当 它是 applied 这个 call.
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def apply_windows_utf8_bootstrap() -> bool:
    """Apply the Windows UTF-8 bootstrap once; True only when it was applied this call."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _bootstrap_applied

    if not _IS_WINDOWS or _bootstrap_applied:
        # ⤷ 出口 return：返回 False
        return False

    # setdefault() so a user can opt out with PYTHONUTF8=0 / PYTHONIOENCODING=...
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # os.environ changes don't rebind streams bound at interpreter startup, so
    # reconfigure them in-process. errors="replace" keeps a non-UTF-8 legacy
    # pipe on stdin from crashing us (U+FFFD instead of an exception).
    # Non-TextIOWrapper streams (BytesIO in tests, embedded hosts) have no
    # reconfigure(): skip — the env-var fix for children is the bigger win.
    for stream_name in ("stdout", "stderr", "stdin"):
        reconfigure = getattr(getattr(sys, stream_name, None), "reconfigure", None)
        if reconfigure is None:
            continue
        # ⤷ 关键分支 try/except：捕获 (OSError, ValueError)，出错走兜底而不是崩溃。
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass  # closed, or replaced with something non-reconfigurable

    _bootstrap_applied = True
    # ⤷ 出口 return：返回 True
    return True


# ┌── 函数 def suppress_platform_ver_console()
# │  作用：Stub ``平台._syscmd_ver`` 在…上 Windows — decode-crash + console-flash 护栏.
# │  参数：无
# │  返回：None
# └──────────────────────────────────────────────────────────────
def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows — decode-crash + console-flash guard.

    ``platform.win32_ver()`` (reached via ``platform.platform()``, which the OpenAI SDK
    calls) shells out ``cmd /c ver`` with ``shell=True`` and no ``CREATE_NO_WINDOW``: a
    windowless parent (pythonw gateway, slash/kanban workers) flashes a console per call,
    and Python 3.11.0/3.11.1 (no ``encoding="locale"`` fix) strict-utf-8-decodes the OEM
    code page output under PEP 540 mode and raises (#69413). Returning the inputs makes
    ``win32_ver()`` fall back to ``sys.getwindowsversion()`` — same data, no subprocess.
    Mirrors ``hermes_cli._subprocess_compat.suppress_platform_ver_console`` for callers
    that never import ``hermes_cli.main``; double application is harmless.
    """
    if not _IS_WINDOWS:
        return
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                # ⤷ 出口 return：返回 (system, release, version)
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        pass  # hardening only — never break an entry point


# ┌── 函数 def harden_import_path()
# │  作用：Stop 包在 current directory 从 shadowing Hermes 模块.
# │  参数：src_root: str | None = None
# │  返回：None
# └──────────────────────────────────────────────────────────────
def harden_import_path(src_root: str | None = None) -> None:
    """Stop a package in the current directory from shadowing Hermes modules.

    Hermes ships top-level modules with common names (``utils``, ``proxy``, ``ui``); a
    project with its own ``utils/`` launched from its directory would win the import.
    The cwd reaches ``sys.path`` as ``""``/``"."`` (script/``-m`` launches) AND as an
    absolute path (venv activation, PYTHONPATH), so both are handled: relative forms are
    dropped and the Hermes root is *relocated* to the front, not merely inserted when
    absent. ``src_root`` defaults to this module's directory (the repo root for every
    shipped entry point), so no spawner env var is required.
    """
    root = src_root or os.environ.get("HERMES_PYTHON_SRC_ROOT") or os.path.dirname(
        os.path.abspath(__file__)
    )

    sys.path[:] = [p for p in sys.path if p not in ("", ".")]

    root_abs = os.path.abspath(root)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != root_abs]
    sys.path.insert(0, root)


# ┌── 函数 def activate_durable_lazy_target()
# │  作用：Put durable lazy-安装 dir (``HERMES_LAZY_INSTALL_TARGET``) 在…上 ``sys.路径``.
# │  参数：无
# │  返回：None
# └──────────────────────────────────────────────────────────────
def activate_durable_lazy_target() -> None:
    """Put the durable lazy-install dir (``HERMES_LAZY_INSTALL_TARGET``) on ``sys.path``.

    Immutable Docker images seal the venv and redirect lazy installs to the data volume;
    packages installed there on a previous run must be importable before any backend
    imports its SDK. Appends to the END of ``sys.path`` so the core venv always wins name
    collisions (see ``tools.lazy_deps``). Never raises; unset target is a no-op.
    """
    if not os.environ.get("HERMES_LAZY_INSTALL_TARGET", "").strip():
        return
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools import lazy_deps
        lazy_deps.activate_durable_lazy_target()
    except Exception:
        pass  # a failed activation just leaves the backend reporting itself unavailable


# Apply on import — entry points only need ``import hermes_bootstrap`` first.
apply_windows_utf8_bootstrap()
suppress_platform_ver_console()
activate_durable_lazy_target()
