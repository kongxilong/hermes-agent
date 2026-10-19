"""Process Registry -- in-memory registry for background processes spawned via
terminal(background=true): rolling 200KB output buffer, poll/log/wait/kill, JSON
checkpoint for crash recovery, session-scoped tracking for gateway reset protection.
Nothing runs on the host unless TERMINAL_ENV=local; other backends run in their sandbox.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：tools/process_registry.py（约 2337 行）
# 地位：后台进程的注册表（terminal(background=true) 启动的进程都记在这里）。
#
# 能力：滚动的 200KB 输出缓冲、查看输出(poll/log)、等待(wait)、杀死(kill)、
#   JSON 检查点（崩溃后可恢复）、按会话隔离追踪。
#
# 核心类：
#   class ProcessSession（@dataclass）  一个后台进程会话：append_output / mark_exited
#   class ProcessRegistry                注册表本体（73 个方法）
#
# 有意思的实现细节：
#   - systemd 集成：_systemd_scope_argv / _systemd_run_user_scope_available
#     （在 Linux 上用 systemd-run 把子进程放进独立的 scope，方便统一管理和清理）
#   - 主机进程存活判断：_is_host_pid_alive / _safe_host_start_time / _host_pid_is_ours
#     光看 pid 不够，还要确认这个 pid 是不是真的是我们的进程（pid 可能被复用）
#   - 输出模式监听：_check_watch_patterns（后台进程输出出现某 pattern 时通知）
#   - 结果脱敏：_redact_process_result
#
# 提示：只有 TERMINAL_ENV=local 时才会在宿主机上真跑；其他后端跑在各自沙箱里。
# ==========================================================================
#

import codecs
from contextlib import suppress
import json
import logging
import os
import platform
import shlex
import signal
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"
# systemd transient scopes exist only on Linux; gate every scope-path branch on this
# (not merely "not Windows") so macOS and other POSIX platforms never touch systemd.
# See #70716.
_IS_LINUX = platform.system() == "Linux"
from tools.environments.local import _find_shell, _resolve_safe_cwd, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, NamedTuple, Optional

from hermes_cli.config import get_hermes_home

from tools.process_registry_notifications import format_process_notification
from tools.process_registry_checkpoint import ProcessCheckpointMixin
from tools.process_registry_results import load_completed_results, save_completed_result

logger = logging.getLogger(__name__)

# Crash-recovery checkpoint (gateway only)
CHECKPOINT_PATH = get_hermes_home() / "processes.json"
_CHECKPOINT_PATH_AT_IMPORT = CHECKPOINT_PATH


# ┌── 函数 def _checkpoint_path()
# │  作用：Active profile's checkpoint file at call time: the patched ``CHECKPOINT_PATH`` when a test
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _checkpoint_path() -> Path:
    """Active profile's checkpoint file at call time: the patched ``CHECKPOINT_PATH`` when a test
    changed it, else live profile-scoped HERMES_HOME — the multiplexed gateway serves every
    profile from one process, so the import-time constant would pin every profile's process
    checkpoint to the launch home."""
    # ⤷ 出口 return：返回 CHECKPOINT_PATH if CHECKPOINT_PATH != _CHECKPOINT_PATH_AT_IM…
    return CHECKPOINT_PATH if CHECKPOINT_PATH != _CHECKPOINT_PATH_AT_IMPORT else get_hermes_home() / "processes.json"

MAX_OUTPUT_CHARS = 200_000      # rolling output buffer
FINISHED_TTL_SECONDS = 1800     # keep finished processes 30 minutes
MAX_PROCESSES = 64              # max tracked processes (LRU pruning)

# Watch-pattern rate limiting, PER SESSION: one watch-match notification per
# WATCH_MIN_INTERVAL_SECONDS; a match inside the cooldown is dropped and counts as one
# strike per window; WATCH_STRIKE_LIMIT consecutive strike windows permanently disable
# watching and fall back to notify_on_complete semantics.
WATCH_MIN_INTERVAL_SECONDS = 15
WATCH_STRIKE_LIMIT = 3
# Lifetime cap, independent of strikes: a pattern recurring just above the cooldown never
# strikes yet forces a full-context agent turn each time; watch_patterns is "ONLY for
# rare one-shot signals", so after this many deliveries fall back to notify_on_complete.
# MAX_ACTIVE_PROCESS_AGE = 86400  # 24h default — see session_reset.bg_process_max_age_hours (#29177)
# A process whose pattern recurs at a cadence just above WATCH_MIN_INTERVAL_SECONDS (e.g. a service
# restarted repeatedly over a day) never trips the consecutive-strike limit, since each match lands in its
# own clean cooldown window, yet still forces a full-context agent turn every single time (#93513).
# watch_patterns is documented as "ONLY for rare one-shot mid-process signals", so once a session has
# delivered this many matches over its whole life we disable it and fall back to notify_on_complete, same as
# the strike-limit path.
WATCH_LIFETIME_MAX_HITS = 8
# Global circuit breaker across all sessions so concurrent siblings can't collectively
# flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


# --- systemd cgroup isolation for gateway-spawned local executors ------------------
# Under a systemd gateway with MemoryMax, local background commands inherit the gateway's
# cgroup, so a memory-heavy executor can get the ENTIRE gateway killed by systemd-oomd;
# ``systemd-run --user --scope`` gives the worker its own transient cgroup. Usability is
# probed once (binary present but user D-Bus absent in system services/containers).
# A memory-heavy executor (Codex, tests, Node) can push the whole cgroup past MemoryMax and trigger
# systemd-oomd to kill the ENTIRE gateway — taking down the messaging control plane and silently losing the
# active turn. We probe *once* whether ``systemd-run --user --scope`` is actually usable (the binary can
# exist on the PATH while the user D-Bus session is unavailable — common for system services and
# containers), and cache the result for the process lifetime. See #70716.
_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


# ┌── 函数 def _worker_memory_max_bytes()
# │  作用：Finite per-worker cgroup limit that can never widen host risk.
# │  参数：无
# │  返回：int
# └──────────────────────────────────────────────────────────────
def _worker_memory_max_bytes() -> int:
    """Finite per-worker cgroup limit that can never widen host risk.
    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is honored only when it *tightens* the safe
    bound (min of the gateway's cgroup-v2 ``memory.max`` and half of physical RAM,
    capped at 4 GiB), so an oversized override cannot exceed the enclosing slice.

    The proposed local-memory-guard environment override is honored when it tightens the safe bound, so this
    isolation composes with PR #57121 instead of inventing a second knob.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    candidates: List[int] = []
    with suppress(OSError, ValueError):
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        v2 = next((ln for ln in lines if ln.startswith("0::")), None)
        if v2 is not None:
            relative = v2.partition("::")[2].lstrip("/")
            raw_limit = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(encoding="utf-8").strip()
            if raw_limit.isdigit() and int(raw_limit) >= _MIN_WORKER_MEMORY_MAX_BYTES:
                candidates.append(int(raw_limit))
    with suppress(OSError, ValueError, TypeError):
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(min(_WORKER_MEMORY_MAX_CAP_BYTES, max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2)))
    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    # ⤷ 出口 return：返回 min(override_bound, safe_bound) if override_bound else safe_…
    return min(override_bound, safe_bound) if override_bound else safe_bound


# ┌── 函数 def _systemd_scope_argv()
# │  作用：``systemd-run --user --scope`` argv shared by the probe and real spawns.
# │  参数：binary: str, unit_name: str, *argv
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def _systemd_scope_argv(binary: str, unit_name: str, *argv: str) -> List[str]:
    """``systemd-run --user --scope`` argv shared by the probe and real spawns.
    ``--collect`` self-cleans the scope after exit; ``--unit`` names it for systemctl.
    No ``OOMPolicy=``: transient scopes reject it on systemd <253 (#102486)."""
    # ⤷ 出口 return：返回 [binary, '--user', '--scope', '--quiet', '--unit', unit_name…
    return [
        binary, "--user", "--scope", "--quiet", "--unit", unit_name, "--collect",
        "--property", "MemoryAccounting=yes",
        "--property", f"MemoryMax={_worker_memory_max_bytes()}",
        "--", *argv,
    ]


# ┌── 函数 def _default_user_runtime_dir()
# │  作用：``/run/user/<uid>``; a function so tests can point it at a temp dir with a real socket.
# │  参数：无
# │  返回：Path
# └──────────────────────────────────────────────────────────────
def _default_user_runtime_dir() -> Path:
    """``/run/user/<uid>``; a function so tests can point it at a temp dir with a real socket."""
    # ⤷ 出口 return：返回 Path(f'/run/user/{os.getuid()}')
    return Path(f"/run/user/{os.getuid()}")  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env


# ┌── 函数 def _secure_user_runtime_dir()
# │  作用：Accept only an absolute, owned, non-writable real directory.
# │  参数：path: Path
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _secure_user_runtime_dir(path: Path) -> bool:
    """Accept only an absolute, owned, non-writable real directory."""
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        metadata = path.lstat()
        # ⤷ 出口 return：返回 path.is_absolute() and stat.S_ISDIR(metadata.st_mode) and (m…
        return (
            path.is_absolute()
            and stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env
            and metadata.st_mode & 0o022 == 0
        )
    except OSError:
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def systemd_user_bus_env()
# │  作用：Build an environment that can reach this user's lingering systemd manager.
# │  参数：base_env: Optional[Dict[str, str]] = None
# │  返回：Dict[str, str]
# └──────────────────────────────────────────────────────────────
def systemd_user_bus_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build an environment that can reach this user's lingering systemd manager.

    System-level gateway units run as an unprivileged ``User=`` but normally do
    not inherit login-session variables.  When the conventional runtime
    directory is owned by this uid and its bus exists, derive the two standard
    variables.  Derived fresh on every call rather than adopted once at boot:
    linger may be enabled after the gateway started (existing installs), so
    the bus can appear later and the probe's failure TTL must be able to
    recover (#104893).
    The returned copy is passed explicitly to the probe and every scoped spawn;
    ``os.environ`` is left unchanged.
    """
    env = dict(os.environ if base_env is None else base_env)
    if not _IS_LINUX:
        # ⤷ 出口 return：返回 env
        return env
    configured = env.get("XDG_RUNTIME_DIR")
    if configured and _secure_user_runtime_dir(Path(configured)):
        runtime_dir = Path(configured)
    else:
        runtime_dir = _default_user_runtime_dir()
        if not _secure_user_runtime_dir(runtime_dir):
            # ⤷ 出口 return：返回 env
            return env

    bus_path = runtime_dir / "bus"
    # ⤷ 关键分支 try/except：捕获 OSError，出错走兜底而不是崩溃。
    try:
        bus_metadata = bus_path.lstat()
    except OSError:
        # ⤷ 出口 return：返回 env
        return env
    if not stat.S_ISSOCK(bus_metadata.st_mode) or bus_metadata.st_uid != os.getuid():  # windows-footgun: ok — behind the _IS_LINUX gate above
        # ⤷ 出口 return：返回 env
        return env

    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"
    # ⤷ 出口 return：返回 env
    return env


# ┌── 函数 def _systemd_scope_cached()
# │  作用：Cached probe verdict, or None when a (re)probe is due. True is permanent; False
# │  参数：无
# │  返回：Optional[bool]
# └──────────────────────────────────────────────────────────────
def _systemd_scope_cached() -> Optional[bool]:
    """Cached probe verdict, or None when a (re)probe is due. True is permanent; False
    expires after ``_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS`` so a D-Bus blip isn't sticky."""
    if _SYSTEMD_SCOPE_AVAILABLE is True:
        # ⤷ 出口 return：返回 True
        return True
    stale = time.monotonic() - _SYSTEMD_SCOPE_PROBED_AT >= _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    # ⤷ 出口 return：返回 None if _SYSTEMD_SCOPE_AVAILABLE is None or stale else False
    return None if _SYSTEMD_SCOPE_AVAILABLE is None or stale else False


# ┌── 函数 def _systemd_run_user_scope_available()
# │  作用：True if ``systemd-run --user --scope`` can create a cgroup.
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _systemd_run_user_scope_available() -> bool:
    """True if ``systemd-run --user --scope`` can create a cgroup.
    ``shutil.which`` alone is insufficient: system services and containers may lack
    the user D-Bus bus even with the binary on PATH (every spawn would fail with
    ``Failed to connect to user bus``), so a cheap probe is run and cached.

    Use ``/bin/sh -c 'exit 0'``: NixOS provides ``/bin/sh`` but not ``/bin/true``
    (#105365), regardless of the gateway service's PATH."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    verdict = _systemd_scope_cached()
    if verdict is not None:
        # ⤷ 出口 return：返回 verdict
        return verdict
    # Double-checked locking: a concurrent first-use spawn must not observe a temporary
    # False mid-probe, or it would launch back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        verdict = _systemd_scope_cached()
        if verdict is not None:
            # ⤷ 出口 return：返回 verdict
            return verdict
        available = False
        if _IS_LINUX:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # Unique unit avoids collisions; the timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        _systemd_scope_argv(binary, probe_unit, "/bin/sh", "-c", "exit 0"),
                        capture_output=True,
                        timeout=3,
                        env=systemd_user_bus_env(),
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode, (result.stderr or b"").decode("utf-8", "replace").strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)
        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        # ⤷ 出口 return：返回 available
        return available


# ┌── 函数 def _is_supervised_gateway_process()
# │  作用：Whether this process is the live, supervised Hermes gateway itself.
# │  参数：无
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _is_supervised_gateway_process() -> bool:
    """Whether this process is the live, supervised Hermes gateway itself.
    Supervisor markers and ``_HERMES_GATEWAY`` are inherited by every descendant (and
    importing ``gateway.run`` sets the latter), so also require ownership of the live
    gateway PID file — scopes are for the gateway, not terminal children or CLIs."""
    if os.environ.get("_HERMES_GATEWAY") != "1":
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        # ⤷ 出口 return：返回 is_gateway_supervisor_process() and get_running_pid(cleanup_…
        return is_gateway_supervisor_process() and get_running_pid(cleanup_stale=False) == os.getpid()
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def _build_systemd_scope_argv()
# │  作用：Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation with its own
# │  参数：shell_argv: List[str], unit_suffix: str
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def _build_systemd_scope_argv(shell_argv: List[str], unit_suffix: str) -> List[str]:
    """Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation with its own
    memory accounting, so an OOM in the worker cannot kill the gateway cgroup.

    ``--collect`` makes the transient scope self-clean after exit; ``--unit`` gives it a recognisable name
    for ``systemctl --user status`` / journalctl. See #70716.
    """
    import shutil

    binary = shutil.which("systemd-run")
    if binary is None:
        # Caller should have probed availability; never pass None into Popen anyway.
        # ⤷ 出口 return：返回 shell_argv
        return shell_argv
    # ⤷ 出口 return：返回 _systemd_scope_argv(binary, f'hermes-worker-{unit_suffix}', …
    return _systemd_scope_argv(binary, f"hermes-worker-{unit_suffix}", *shell_argv)


_scope_degraded_warned = False


# ┌── 函数 def _warn_scope_degraded_once()
# │  作用：Warn once per process: the condition is host-level and the probe verdict
# │  参数：detail: str
# │  返回：None
# └──────────────────────────────────────────────────────────────
def _warn_scope_degraded_once(detail: str) -> None:
    """Warn once per process: the condition is host-level and the probe verdict
    is cached, so this would otherwise fire on every cron dispatch."""
    # ⤷ global：声明接下来赋值的是模块级全局变量，而不是新建局部变量。
    global _scope_degraded_warned
    if _scope_degraded_warned:
        return
    _scope_degraded_warned = True
    logger.warning(
        "managed gateway: %s; cron children are dispatched as direct external subprocesses "
        "without restart-safe cgroup isolation (killed if the gateway restarts mid-job). "
        "Set cron.require_restart_safe_scope=true in config.yaml to fail closed instead.",
        detail,
    )


# ┌── 类 class GatewayChildDispatch（继承自：NamedTuple）
# │  说明：How a managed-gateway child is launched.
# │  成员：共 0 个方法；全部是下划线开头的内部方法
# └──────────────────────────────────────────────────────────────
class GatewayChildDispatch(NamedTuple):
    """How a managed-gateway child is launched.

    ``in_process``: not a managed systemd gateway, ``argv is command``, the caller
    keeps its in-process path.  ``scoped``: ``argv`` is the systemd-run wrapper.
    ``degraded``: no user scope could be created; ``argv`` is the direct command but
    the caller MUST still launch it as an external subprocess — the distinct mode
    exists so this case can never collapse into ``in_process`` and recreate the
    restart interruption #101940 closed.
    """

    mode: Literal["in_process", "scoped", "degraded"]
    argv: List[str]


# ┌── 函数 def restart_safe_gateway_child_argv()
# │  作用：Place a managed-systemd gateway child outside the gateway cgroup.
# │  参数：command: List[str], unit_suffix: str, require_restart_safe_scope: bool
# │  返回：GatewayChildDispatch
# └──────────────────────────────────────────────────────────────
def restart_safe_gateway_child_argv(
    command: List[str], *, unit_suffix: str, require_restart_safe_scope: bool,
) -> GatewayChildDispatch:
    """Place a managed-systemd gateway child outside the gateway cgroup.

    A systemd-supervised gateway restart kills every process in the service
    cgroup, so children that must survive it run in a transient user scope.
    Hosts with no user systemd session (containers, LXCs without linger) cannot
    create one; hard-failing there is a silent cron outage, so callers state the
    policy: ``require_restart_safe_scope=True`` raises (kanban's long-lived
    workers), ``False`` degrades to a direct external subprocess with a
    once-per-process warning (cron, behind ``cron.require_restart_safe_scope``).
    """
    if not _IS_LINUX:
        # ⤷ 出口 return：返回 GatewayChildDispatch('in_process', command)
        return GatewayChildDispatch("in_process", command)
    if not _is_supervised_gateway_process() or not os.environ.get("INVOCATION_ID"):
        # ⤷ 出口 return：返回 GatewayChildDispatch('in_process', command)
        return GatewayChildDispatch("in_process", command)

    def _degrade(detail: str) -> GatewayChildDispatch:
        if require_restart_safe_scope:
            # Stored as the cron execution's error and shown on the job row: name the remedy.
            raise RuntimeError(f"cannot create restart-safe systemd scope for gateway child: {detail}")
        _warn_scope_degraded_once(detail)
        # ⤷ 出口 return：返回 GatewayChildDispatch('degraded', command)
        return GatewayChildDispatch("degraded", command)

    if not _systemd_run_user_scope_available():
        # ⤷ 出口 return：返回 _degrade(f'systemd-run --user --scope is unavailable (usuall…
        return _degrade(
            "systemd-run --user --scope is unavailable (usually no reachable user D-Bus session at "
            f"/run/user/{os.getuid()}/bus). On a system-level service install, run "  # windows-footgun: ok — behind the _IS_LINUX return above
            "`sudo loginctl enable-linger <gateway-user>` and restart the gateway."
        )
    scoped = _build_systemd_scope_argv(command, unit_suffix=unit_suffix)
    if scoped == command:
        # ⤷ 出口 return：返回 _degrade('systemd-run disappeared after the availability pro…
        return _degrade("systemd-run disappeared after the availability probe")
    # ⤷ 出口 return：返回 GatewayChildDispatch('scoped', scoped)
    return GatewayChildDispatch("scoped", scoped)


# ┌── 函数 def _stop_systemd_unit()
# │  作用：Stop a transient systemd user scope by unit name.
# │  参数：unit_name: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def _stop_systemd_unit(unit_name: str) -> bool:
    """Stop a transient systemd user scope by unit name.
    Reaps the *entire* cgroup — catching double-forked descendants reparented to init
    inside the scope that survive a plain PID signal (SIGTERM all, SIGKILL after
    ``TimeoutStopSec``). True if stopped or already gone; False if ``systemctl`` is
    unavailable or the stop failed.

    See #70716.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        # ⤷ 出口 return：返回 False
        return False
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        result = subprocess.run(
            [binary, "--user", "stop", unit_name],
            capture_output=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(),
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            if any(marker in stderr.lower() for marker in ("not loaded", "not found", "does not exist")):
                # ⤷ 出口 return：返回 True
                return True
            logger.debug("systemctl --user stop %s exited %d: %s", unit_name, result.returncode, stderr)
            # ⤷ 出口 return：返回 False
            return False
        # ⤷ 出口 return：返回 True
        return True
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        # ⤷ 出口 return：返回 False
        return False


# ┌── 函数 def format_uptime_short()
# │  作用：（源码未写 docstring，可从下方参数与返回值推断）
# │  参数：seconds: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def format_uptime_short(seconds: int) -> str:
    s = max(0, int(seconds))
    if s < 60:
        # ⤷ 出口 return：返回 f'{s}s'
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        # ⤷ 出口 return：返回 f'{mins}m {secs}s'
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    # ⤷ 出口 return：返回 f'{hours}h {mins}m'
    return f"{hours}h {mins}m"


# ┌── 函数 def _not_found()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：session_id: str
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _not_found(session_id: str) -> dict:
    # ⤷ 出口 return：返回 {'status': 'not_found', 'error': f'No process with ID {sessi…
    return {"status": "not_found", "error": f"No process with ID {session_id}"}


# ┌── 函数 def _output_tail()
# │  作用：Last *n* chars of the session output with ANSI sequences stripped.
# │  参数：session: 'ProcessSession', n: int
# │  返回：str
# └──────────────────────────────────────────────────────────────
def _output_tail(session: "ProcessSession", n: int) -> str:
    """Last *n* chars of the session output with ANSI sequences stripped."""
    from tools.ansi_strip import strip_ansi

    # ⤷ 出口 return：返回 strip_ansi(session.output_buffer[-n:])
    return strip_ansi(session.output_buffer[-n:])


# ┌── 类 class ProcessSession
# │  说明：A tracked background process with output buffering.
# │  成员：共 2 个方法；对外可用 2 个
# └──────────────────────────────────────────────────────────────
@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # "proc_xxxxxxxxxxxx"
    command: str
    task_id: str = ""                           # Task/sandbox isolation key (CONTAINER key,
                                                # may be collapsed by _resolve_container_task_id)
    owner_task_id: str = ""                     # RAW spawning task id ("sa-..."); ownership
                                                # checks must use this, not task_id
    session_key: str = ""                       # Gateway session key (reset protection)
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Environment object (sandbox spawns)
    cwd: Optional[str] = None
    started_at: float = 0.0                     # time.time() of spawn
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False
    exit_code: Optional[int] = None             # None while running
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling tail (last max_output_chars)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # Recovered from checkpoint (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    handoff_note: str = ""                      # why a subagent handed this process to its parent (rides the notice)
    # Watcher/notification routing (persisted for crash recovery)
    # systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    # (#70716)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the spawning conversation; lets the gateway drop completions whose
    # session was closed at a user boundary (/new) instead of injecting into the NEW one.
    parent_session_id: str = ""
    notify_on_complete: bool = False            # Queue agent notification on exit
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Rate-limit window state (see WATCH_*). A strike is a WINDOW with drops, not a drop.
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (use_pty=True)

    # ┌── 函数 def append_output()
    # │  作用：Append to the rolling output buffer under the session lock, keeping the tail.
    # │  参数：self, text: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def append_output(self, text: str) -> None:
        """Append to the rolling output buffer under the session lock, keeping the tail."""
        with self._lock:
            self.output_buffer += text
            if len(self.output_buffer) > self.max_output_chars:
                self.output_buffer = self.output_buffer[-self.max_output_chars:]

    # ┌── 函数 def mark_exited()
    # │  作用：Record an exit. A kill that raced the observer already recorded its own
    # │  参数：self, exit_code, reason: str = 'exited', source: str = ''
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def mark_exited(self, exit_code, reason: str = "exited", source: str = "") -> None:
        """Record an exit. A kill that raced the observer already recorded its own
        exit_code/reason; never overwrite it."""
        self.exited = True
        if self.completion_reason != "killed":
            self.exit_code = exit_code
            self.completion_reason = reason
            if source:
                self.termination_source = source


# Watcher routing fields, in event-dict key order (``watcher_<key>`` on the session).
_WATCHER_ROUTE_KEYS = ("platform", "chat_id", "user_id", "user_name", "thread_id", "message_id")
# Session fields persisted verbatim in the crash-recovery checkpoint (plus
# ``session_id``; ``command`` is redacted and ``owner_task_id`` defaulted on write).
_CHECKPOINT_FIELDS = (
    "command", "pid", "pid_scope", "host_start_time", "systemd_unit", "cwd",
    "started_at", "task_id", "owner_task_id", "session_key",
    *(f"watcher_{k}" for k in _WATCHER_ROUTE_KEYS), "watcher_interval",
    "parent_session_id", "notify_on_complete", "watch_patterns")
_CHECKPOINT_DEFAULTS = {
    f.name: ([] if f.name == "watch_patterns" else f.default)
    for f in ProcessSession.__dataclass_fields__.values()
    if f.name in _CHECKPOINT_FIELDS
}


# ┌── 类 class ProcessRegistry（继承自：ProcessCheckpointMixin）
# │  说明：In-memory registry of running and finished background processes.
# │  【语法】Mixin 是一种「插件式父类」：本身不能单独用，靠多继承把一组方法拼进主类。看到 xxxMixin 就当成「这批方法从别的文件来」。
# │  成员：共 73 个方法；对外可用 27 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class ProcessRegistry(ProcessCheckpointMixin):
    """In-memory registry of running and finished background processes.
    Thread-safe: accessed from executor threads (terminal_tool, process handlers),
    the gateway asyncio loop (watchers, reset checks) and the cleanup thread."""

    _SHELL_NOISE_SUBSTRINGS = (
        "no job control in this shell", "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device")

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def __init__(self):
        self._running: Dict[str, ProcessSession] = {}
        self._finished: Dict[str, ProcessSession] = {}
        self._lock = threading.Lock()
        # Side-channel for check_interval watchers (gateway reads after agent run)
        self.pending_watchers: List[Dict[str, Any]] = []
        # Unified queue for all background events (distinguished by "type"); the CLI
        # process_loop and the gateway drain it after each agent turn to trigger new turns.
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()
        # Rehydrate durable delegation completions once, at registry startup.
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.async_delegation import restore_undelivered_completions
            restore_undelivered_completions(self.completion_queue)
        except Exception as exc:
            logger.warning("Could not restore async delegation completions: %s", exc)
        # Completions the agent already consumed via wait()/read_log() (output in
        # hand): drain loops AND gateway/tui watchers skip them.
        self._completion_consumed: set = set()
        # Sessions merely *observed* exited via poll(). poll() is read-only and must NOT
        # mark consumed (a status check would suppress the watcher's autonomous delivery
        # turn), but the CLI has the poll result inline in the same turn, so
        # drain_notifications() skips these to avoid a duplicate [SYSTEM: ...];
        # gateway/tui watchers deliberately ignore this set.
        # See #8228.
        self._poll_observed: set = set()
        # Global watch-match circuit breaker across all sessions.
        self._global_watch_lock = threading.Lock()
        self._global_watch_window_start = self._global_watch_tripped_until = 0.0
        self._global_watch_window_hits = self._global_watch_suppressed_during_trip = 0
        # Driver-installed sinks (desktop gateway): on_output(session, chunk) streams
        # live output from reader threads; on_close(session_or_none, process_id) drops
        # a read-only terminal tab without killing the process.
        self.on_output = None
        self.on_close = None

    # ┌── 函数 def _clean_shell_noise()
    # │  作用：Strip shell startup warnings from the beginning of output.
    # │  参数：text: str
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """Strip shell startup warnings from the beginning of output."""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        # ⤷ 出口 return：返回 '\n'.join(lines)
        return "\n".join(lines)

    # ┌── 函数 def _emit_output()
    # │  作用：Forward a chunk to the live-output sink; called from reader threads, never raises.
    # │  参数：self, session: ProcessSession, chunk: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _emit_output(self, session: ProcessSession, chunk: str) -> None:
        """Forward a chunk to the live-output sink; called from reader threads, never raises."""
        sink = self.on_output
        if sink is None or not chunk:
            return
        with suppress(Exception):
            sink(session, chunk)

    # ┌── 函数 def _check_watch_patterns()
    # │  作用：Scan a freshly-read chunk for watch patterns and queue notifications.
    # │  参数：self, session: ProcessSession, new_text: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan a freshly-read chunk for watch patterns and queue notifications.
        Per-session rate limiting (see WATCH_* constants): one match per cooldown
        window, a match inside the window is one strike, WATCH_STRIKE_LIMIT consecutive
        strikes or WATCH_LIFETIME_MAX_HITS total deliveries disable watching and
        promote the session to notify_on_complete."""
        if not session.watch_patterns or session._watch_disabled:
            return
        # Late chunks after the reader declared exit are post-exit noise; dropping them
        # avoids stale notifications minutes after the process ended.
        if session.exited:
            return
        hits = [  # (first matching pattern, line) — one match per line
            (next(p for p in session.watch_patterns if p in line), line.rstrip())
            for line in new_text.splitlines() if any(p in line for p in session.watch_patterns)]
        if not hits:
            return
        matched_pattern = hits[0][0]
        matched_lines = [line for _, line in hits]
        now = time.time()
        with session._lock:
            if session._watch_cooldown_until and now < session._watch_cooldown_until:
                # Inside the cooldown: drop, count one strike per window, disable +
                # promote once the strike limit is hit.
                session._watch_suppressed += len(matched_lines)
                if session._watch_strike_candidate:
                    return
                session._watch_strike_candidate = True
                session._watch_consecutive_strikes += 1
                if session._watch_consecutive_strikes < WATCH_STRIKE_LIMIT:
                    return
                session._watch_disabled = True
                # Promote so the agent still gets exactly one notification on exit,
                # plus exactly one summary so it sees why things went quiet.
                session.notify_on_complete = True
                self._emit_watch_disabled(
                    session, session._watch_suppressed,
                    f"{WATCH_STRIKE_LIMIT} consecutive rate-limit windows triggered "
                    f"(min spacing {WATCH_MIN_INTERVAL_SECONDS}s). ")
                return
            # Cooldown expired. A prior window with no drops resets the
            # consecutive-strike counter (healthy cadence again).
            if session._watch_cooldown_until and not session._watch_strike_candidate:
                session._watch_consecutive_strikes = 0
            session._watch_strike_candidate = False
            # Emit and start a new cooldown window.
            session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
            session._watch_hits += 1
            suppressed = session._watch_suppressed
            session._watch_suppressed = 0
            # Lifetime cap: this match is still delivered, but no further ones.
            lifetime_exhausted = session._watch_hits >= WATCH_LIFETIME_MAX_HITS
            if lifetime_exhausted:
                session._watch_disabled = True
                session.notify_on_complete = True
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"
        if self._global_watch_admit(now):
            notification = {
                **self._watch_event_base(session),
                "type": "watch_match",
                "pattern": matched_pattern,
                "output": output,
                "suppressed": suppressed,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)
        # Even when the breaker drops the final match, still explain the silence.
        if lifetime_exhausted:
            self._emit_watch_disabled(
                session, 0, f"reached the lifetime cap of {WATCH_LIFETIME_MAX_HITS} delivered matches. ",
            )

    # ┌── 函数 def _emit_watch_disabled()
    # │  作用：Queue the one-shot watch_disabled summary (strike-limit or lifetime-cap path).
    # │  参数：self, session: ProcessSession, suppressed: int, why: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _emit_watch_disabled(self, session: ProcessSession, suppressed: int, why: str) -> None:
        """Queue the one-shot watch_disabled summary (strike-limit or lifetime-cap path)."""
        self.completion_queue.put({
            **self._watch_event_base(session),
            "type": "watch_disabled",
            "suppressed": suppressed,
            "message": (
                f"Watch patterns disabled for process {session.id} — {why}"
                f"Falling back to notify_on_complete semantics; you'll get "
                f"exactly one notification when the process exits."),
        })

    # ┌── 函数 def _watch_event_base()
    # │  作用：Session identity + watcher routing fields shared by every watch event.
    # │  参数：session: ProcessSession
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _watch_event_base(session: ProcessSession) -> dict:
        """Session identity + watcher routing fields shared by every watch event."""
        # ⤷ 出口 return：返回 {'session_id': session.id, 'session_key': session.session_ke…
        return {
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
        }

    # ┌── 函数 def _global_watch_event()
    # │  作用：Unaddressed (all-sessions) watch breaker event.
    # │  参数：type_: str, message: str, **extra
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _global_watch_event(type_: str, message: str, **extra) -> dict:
        """Unaddressed (all-sessions) watch breaker event."""
        # ⤷ 出口 return：返回 {'session_id': '', 'session_key': '', 'command': '', 'type':…
        return {
            "session_id": "", "session_key": "", "command": "", "type": type_, **extra,
            "message": message,
            "platform": "", "chat_id": "", "user_id": "", "user_name": "", "thread_id": "",
        }

    # ┌── 函数 def _global_watch_admit()
    # │  作用：True if this watch_match may pass the global breaker.
    # │  参数：self, now: float
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _global_watch_admit(self, now: float) -> bool:
        """True if this watch_match may pass the global breaker.
        In cooldown: drop and count. Otherwise slide the rolling window; exceeding
        the cap trips the breaker for WATCH_GLOBAL_COOLDOWN_SECONDS with ONE
        "tripped" summary, and the cooldown's end emits ONE "released" summary."""
        events = []  # summary events, queued outside the lock
        with self._global_watch_lock:
            # Handle cooldown expiry first so we can emit the release summary.
            if self._global_watch_tripped_until and now >= self._global_watch_tripped_until:
                suppressed = self._global_watch_suppressed_during_trip
                self._global_watch_tripped_until = 0.0
                self._global_watch_suppressed_during_trip = 0
                self._global_watch_window_start, self._global_watch_window_hits = now, 0
                if suppressed > 0:
                    events.append(self._global_watch_event(
                        "watch_overflow_released",
                        f"Watch-pattern notifications resumed. "
                        f"{suppressed} match event(s) were suppressed during the flood.",
                        suppressed=suppressed))
            if self._global_watch_tripped_until and now < self._global_watch_tripped_until:
                # Still in cooldown — drop and count.
                self._global_watch_suppressed_during_trip += 1
                admit = False
            else:
                if now - self._global_watch_window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
                    self._global_watch_window_start, self._global_watch_window_hits = now, 0
                admit = self._global_watch_window_hits < WATCH_GLOBAL_MAX_PER_WINDOW
                if admit:
                    self._global_watch_window_hits += 1
                else:
                    self._global_watch_tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
                    self._global_watch_suppressed_during_trip += 1
                    events.append(self._global_watch_event(
                        "watch_overflow_tripped",
                        f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                        f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s across all processes. "
                        f"Suppressing further watch_match events for "
                        f"{WATCH_GLOBAL_COOLDOWN_SECONDS}s."))
        for msg in events:
            self.completion_queue.put(msg)
        # ⤷ 出口 return：返回 admit
        return admit

    # ┌── 函数 def _is_host_pid_alive()
    # │  作用：Best-effort liveness check for host-visible PIDs.
    # │  参数：pid: Optional[int]
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_host_pid_alive(pid: Optional[int]) -> bool:
        """Best-effort liveness check for host-visible PIDs."""
        if not pid:
            # ⤷ 出口 return：返回 False
            return False
        # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484) — use the
        # cross-platform existence check.
        from gateway.status import _pid_exists
        # ⤷ 出口 return：返回 _pid_exists(pid)
        return _pid_exists(pid)

    # ┌── 函数 def _safe_host_start_time()
    # │  作用：Kernel start ticks for a host PID, or None when unavailable.
    # │  参数：pid: Optional[int]
    # │  返回：Optional[int]
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_host_start_time(pid: Optional[int]) -> Optional[int]:
        """Kernel start ticks for a host PID, or None when unavailable."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from gateway.status import get_process_start_time
            # ⤷ 出口 return：返回 get_process_start_time(pid) if pid else None
            return get_process_start_time(pid) if pid else None
        except Exception:
            # ⤷ 出口 return：返回 None
            return None

    # ┌── 函数 def _host_pid_is_ours()
    # │  作用：True only if ``pid`` is alive AND still the process we spawned.
    # │  参数：cls, pid: Optional[int], expected_start: Optional[int]
    # │  返回：bool
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def _host_pid_is_ours(cls, pid: Optional[int], expected_start: Optional[int]) -> bool:
        """True only if ``pid`` is alive AND still the process we spawned.
        The kernel recycles PIDs, so a stored number can later name an unrelated
        process (seen in the wild: a browser's session leader tree-killed). The kernel
        start time captured at spawn must match the live one; with no baseline
        (legacy checkpoints, no ``/proc``) degrade to a bare liveness check."""
        # ⤷ 出口 return：返回 cls._is_host_pid_alive(pid) and (expected_start is None or c…
        return cls._is_host_pid_alive(pid) and (
            expected_start is None or cls._safe_host_start_time(pid) == expected_start)

    # ┌── 函数 def _refresh_detached_session()
    # │  作用：Update recovered host-PID sessions when the underlying process has exited.
    # │  参数：self, session: Optional[ProcessSession]
    # │  返回：Optional[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def _refresh_detached_session(self, session: Optional[ProcessSession]) -> Optional[ProcessSession]:
        """Update recovered host-PID sessions when the underlying process has exited."""
        if session is None or session.exited or not session.detached or session.pid_scope != "host":
            # ⤷ 出口 return：返回 session
            return session
        # A recycled PID (alive but not ours) counts as "our process exited" so a
        # later kill() can never tree-kill the stranger.
        if self._host_pid_is_ours(session.pid, session.host_start_time):
            # ⤷ 出口 return：返回 session
            return session
        with session._lock:
            if session.exited:
                # ⤷ 出口 return：返回 session
                return session
            # No waitable handle survives recovery, so the real exit code is unknown.
            session.exited, session.exit_code = True, None
        self._move_to_finished(session)
        # ⤷ 出口 return：返回 session
        return session

    # ┌── 函数 def _proc_alive()
    # │  作用：True if a psutil.Process is running and not a zombie (already dead, just unreaped).
    # │  参数：proc
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie (already dead, just unreaped)."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            import psutil
            # ⤷ 出口 return：返回 proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            # ⤷ 出口 return：返回 False
            return False

    # ┌── 函数 def _config_value()
    # │  作用：``config.yaml`` value for ``section.key``, else the DEFAULT_CONFIG value.
    # │  参数：section: str, key: str, fallback
    # │  返回：（未标注类型）
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _config_value(section: str, key: str, fallback):
        """``config.yaml`` value for ``section.key``, else the DEFAULT_CONFIG value.
        Raises if config is unreadable; callers wrap with their own hard fallback so
        registry code paths never crash on a broken config file."""
        from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), section, key)
        # ⤷ 出口 return：返回 DEFAULT_CONFIG[section][key] if val is None else val
        return DEFAULT_CONFIG[section][key] if val is None else val

    # ┌── 函数 def _config_seconds()
    # │  作用：``terminal.<key>`` as a non-negative float (0 disables); *fallback* if unreadable.
    # │  参数：key: str, fallback: float
    # │  返回：float
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _config_seconds(key: str, fallback: float) -> float:
        """``terminal.<key>`` as a non-negative float (0 disables); *fallback* if unreadable."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 max(float(ProcessRegistry._config_value('terminal', key, fal…
            return max(float(ProcessRegistry._config_value("terminal", key, fallback)), 0.0)
        except Exception:
            # ⤷ 出口 return：返回 fallback
            return fallback

    # ┌── 函数 def _daemon_term_grace_seconds()
    # │  作用：Grace (s) between SIGTERM and escalated SIGKILL; 0 disables escalation.
    # │  参数：无
    # │  返回：float
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _daemon_term_grace_seconds() -> float:
        """Grace (s) between SIGTERM and escalated SIGKILL; 0 disables escalation."""
        # ⤷ 出口 return：返回 ProcessRegistry._config_seconds('daemon_term_grace_seconds',…
        return ProcessRegistry._config_seconds("daemon_term_grace_seconds", 2.0)

    # ┌── 函数 def _terminate_host_pid()
    # │  作用：Terminate a host-visible PID and its descendants.
    # │  参数：cls, pid: int, expected_start: Optional[int] = None
    # │  返回：None
    # │  【语法】@classmethod 第一个参数是 cls（类本身）而不是 self（实例）。
    # └──────────────────────────────────────────────────────────────
    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.
        ``expected_start`` (kernel start time at spawn) is re-validated first: a mismatch
        or dead PID means the number was recycled onto a stranger and we refuse to touch
        it — a leaked orphan beats tree-killing someone's browser. POSIX: psutil SIGTERMs
        children before the parent (so trees aren't reparented to init and survive), then
        SIGKILLs survivors after ``terminal.daemon_term_grace_seconds``. Windows:
        ``taskkill /T /F`` (psutil's stale PPID links miss orphans there); ``os.kill``
        is the fallback."""
        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid)
            return

        def _sigterm_quietly():
            with suppress(OSError, ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
        if _IS_WINDOWS:
            # ⤷ 关键分支 try/except：捕获 (FileNotFoundError, subprocess.TimeoutExpired, OSError)，出错走兜底而不是崩溃。
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=10, creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                _sigterm_quietly()
            return
        import psutil
        gone = (psutil.NoSuchProcess, psutil.AccessDenied, OSError)
        # ⤷ 关键分支 try/except：捕获 psutil.NoSuchProcess、(OSError, PermissionError)，出错走兜底而不是崩溃。
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            _sigterm_quietly()
            return
        # Snapshot the whole tree (children before parent) and SIGTERM each.
        # ⤷ 关键分支 try/except：捕获 gone，出错走兜底而不是崩溃。
        try:
            targets = parent.children(recursive=True)
        except gone:
            targets = []
        targets.append(parent)
        for proc in targets:
            with suppress(gone):
                proc.terminate()
        # Escalate to SIGKILL for anything that ignored SIGTERM within the grace window.
        # ``psutil.wait_procs``' gone/alive partition is deliberately NOT trusted: it
        # reaps via ``Process.wait()`` and mis-partitions across zombie transitions in a
        # parent/child tree, leaving survivors un-killed. Re-probing every target is
        # deterministic.
        grace = cls._daemon_term_grace_seconds()
        if grace <= 0:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(cls._proc_alive(_p) for _p in targets):
            time.sleep(0.05)
        for proc in targets:
            with suppress(gone):
                if cls._proc_alive(proc):
                    proc.kill()  # SIGKILL on POSIX
                    logger.info("Escalated to SIGKILL for pid %d (ignored SIGTERM within %.1fs grace)", proc.pid, grace)

    # ----- Spawn -----

    # ┌── 函数 def _new_session()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：command, task_id, owner_task_id, session_key, cwd, **extra
    # │  返回：ProcessSession
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _new_session(command, task_id, owner_task_id, session_key, cwd, **extra) -> ProcessSession:
        from gateway.session_context import get_session_env

        # ⤷ 出口 return：返回 ProcessSession(id=f'proc_{uuid.uuid4().hex[:12]}', command=c…
        return ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}", command=command, task_id=task_id,
            owner_task_id=owner_task_id or task_id, session_key=session_key, cwd=cwd,
            parent_session_id=get_session_env("HERMES_SESSION_ID", ""),
            started_at=time.time(), **extra)

    # ┌── 函数 def _env_temp_dir()
    # │  作用：Return the writable sandbox temp dir for env-backed background tasks.
    # │  参数：env: Any
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _env_temp_dir(env: Any) -> str:
        """Return the writable sandbox temp dir for env-backed background tasks."""
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                temp_dir = get_temp_dir()
                if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                    # ⤷ 出口 return：返回 temp_dir.rstrip('/') or '/'
                    return temp_dir.rstrip("/") or "/"
            except Exception as exc:
                logger.debug("Could not resolve environment temp dir: %s", exc)
        # ⤷ 出口 return：返回 '/tmp'
        return "/tmp"

    # ┌── 函数 def _scope_argv()
    # │  作用：Login-shell argv for *safe_command* (parity with LocalEnvironment: rc files
    # │  参数：self, session: ProcessSession, safe_command: str, unit_suffix: str, label: str
    # │  返回：List[str]
    # └──────────────────────────────────────────────────────────────
    def _scope_argv(self, session: ProcessSession, safe_command: str, unit_suffix: str, label: str) -> List[str]:
        """Login-shell argv for *safe_command* (parity with LocalEnvironment: rc files
        sourced, user tools on PATH), wrapped in a transient systemd scope when we are
        the supervised gateway (own cgroup: an OOM kills only the worker, not the
        gateway and its messaging control plane)."""
        argv = [_find_shell(), "-lic", f"set +m; {safe_command}"]
        # This applies to both pipe mode and the PTY path above. See #70716.
        in_supervised_gateway = _IS_LINUX and _is_supervised_gateway_process()
        if in_supervised_gateway and _systemd_run_user_scope_available():
            session.systemd_unit = f"hermes-worker-{unit_suffix}.scope"
            # ⤷ 出口 return：返回 _build_systemd_scope_argv(argv, unit_suffix=unit_suffix)
            return _build_systemd_scope_argv(argv, unit_suffix=unit_suffix)
        if in_supervised_gateway:
            # Under a supervisor but no private cgroup: a worker OOM can still take
            # the whole gateway down.
            logger.debug(
                "%s background executor not isolated in a systemd scope "
                "(systemd-run --user unavailable); worker shares the gateway cgroup.", label)
        # ⤷ 出口 return：返回 argv
        return argv

    # ┌── 函数 def _spawn_env()
    # │  作用：Sanitized child env; PYTHONUNBUFFERED so tqdm/datasets-style buffering
    # │  参数：env_vars: dict
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _spawn_env(env_vars: dict) -> dict:
        """Sanitized child env; PYTHONUNBUFFERED so tqdm/datasets-style buffering
        doesn't hide progress from process(action="poll")."""
        env = _sanitize_subprocess_env(os.environ, env_vars)
        env["PYTHONUNBUFFERED"] = "1"
        # ⤷ 出口 return：返回 env
        return env

    # ┌── 函数 def _track_started()
    # │  作用：Register before the reader can publish completion, even for an exited child.
    # │  参数：self, session: ProcessSession, reader_target, reader_name: str, extra_args = ()
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _track_started(self, session: ProcessSession, reader_target, reader_name: str, extra_args=()) -> None:
        """Register before the reader can publish completion, even for an exited child."""
        from contextvars import copy_context

        # Reader completion must retain the producer's multiplex profile scope.
        reader = threading.Thread(target=copy_context().run, args=(reader_target, session, *extra_args),
                                  daemon=True, name=reader_name)
        session._reader_thread = reader
        with self._lock:
            self._prune_if_needed()
            # Completion takes this lock too. Starting here also leaves no
            # ghost entry if the interpreter cannot start another thread.
            reader.start()
            self._running[session.id] = session
        self._write_checkpoint()

    # ┌── 函数 def _spawn_local_pty()
    # │  作用：PTY spawn for interactive CLI tools (Codex, Claude Code, REPLs).
    # │  参数：self, session: ProcessSession, safe_command: str, env_vars: dict
    # │  返回：ProcessSession
    # └──────────────────────────────────────────────────────────────
    def _spawn_local_pty(self, session: ProcessSession, safe_command: str, env_vars: dict) -> ProcessSession:
        """PTY spawn for interactive CLI tools (Codex, Claude Code, REPLs).
        Raises ImportError when no PTY backend is installed and re-raises any spawn
        failure; ``spawn_local`` falls back to pipe mode in both cases."""
        if _IS_WINDOWS:
            from winpty import PtyProcess as _PtyProcessCls
        else:
            from ptyprocess import PtyProcess as _PtyProcessCls
        pty_argv = self._scope_argv(session, safe_command, session.id, "PTY")
        pty_env = self._spawn_env(env_vars)
        if session.systemd_unit:
            pty_env = systemd_user_bus_env(pty_env)
        # A PTY is a real TTY, so pager-happy tools (git log/diff, man) WILL page and
        # hang waiting for `q` — default them to cat, honoring any pager the user set.
        pty_env.setdefault("GIT_PAGER", "cat")
        pty_env.setdefault("PAGER", "cat")
        pty_proc = _PtyProcessCls.spawn(pty_argv, cwd=session.cwd, env=pty_env, dimensions=(30, 120))
        session.pid = pty_proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        session._pty = pty_proc
        self._track_started(session, self._pty_reader_loop, f"proc-pty-reader-{session.id}")
        # ⤷ 出口 return：返回 session
        return session

    # ┌── 函数 def spawn_local()
    # │  作用：Spawn a background process locally (TERMINAL_ENV=local; other backends use
    # │  参数：self, command: str, cwd: str = None, task_id: str = '', session_key: str = '', env_vars: dict = None, use_pty: bool = False, owner_task_id: str = ''
    # │  返回：ProcessSession
    # └──────────────────────────────────────────────────────────────
    def spawn_local(
        self, command: str, cwd: str = None, task_id: str = "", session_key: str = "",
        env_vars: dict = None, use_pty: bool = False, owner_task_id: str = "") -> ProcessSession:
        """Spawn a background process locally (TERMINAL_ENV=local; other backends use
        spawn_via_env()). ``use_pty`` requests a pseudo-terminal via ptyprocess/pywinpty
        for interactive CLIs, falling back to a plain pipe when unavailable or failing."""
        # Bash parses ``A && B &`` as ``(A && B) &`` — a subshell that holds our stdout
        # pipe open forever when B is a long-running server. The rewriter turns it into
        # ``A && { B & }``. Lazy import: terminal_tool imports this module.
        # Guard against the `A && B &` subshell-wait trap (issue #68915).
        from tools.terminal_tool_sudo import _rewrite_compound_background as _rewrite_bg

        safe_command = _rewrite_bg(command)
        session = self._new_session(command, task_id, owner_task_id, session_key, _resolve_safe_cwd(cwd or os.getcwd()))
        pty_scope_attempted = False
        if use_pty:
            # ⤷ 关键分支 try/except：捕获 ImportError、Exception，出错走兜底而不是崩溃。
            try:
                # ⤷ 出口 return：返回 self._spawn_local_pty(session, safe_command, env_vars)
                return self._spawn_local_pty(session, safe_command, env_vars)
            except ImportError:
                logger.warning("ptyprocess not installed, falling back to pipe mode")
            except Exception as e:
                logger.warning("PTY spawn failed (%s), falling back to pipe mode", e)
                if session.systemd_unit:
                    pty_scope_attempted = True
                    if not _stop_systemd_unit(session.systemd_unit):
                        # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
                        raise RuntimeError(
                            "PTY scope could not be reaped; refusing pipe fallback "
                            "to avoid duplicate command execution"
                        ) from e
                    session.systemd_unit = ""
        # Pipe path (non-PTY or PTY fallback).
        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}
        unit_suffix = f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
        spawn_argv = self._scope_argv(session, safe_command, unit_suffix, "Local")
        spawn_env = self._spawn_env(env_vars)
        if session.systemd_unit:
            spawn_env = systemd_user_bus_env(spawn_env)
        # start_new_session is REQUIRED with systemd-run --scope too: the scope does not
        # give the worker a new session, so from an interactive TUI the worker would
        # share the foreground process group and background spawns would stop the whole
        # session (observed as dead TUIs in state T). Cgroup isolation is unaffected —
        # the scope attaches to the invoked process, not the spawning session.
        proc = subprocess.Popen(
            spawn_argv, text=True, cwd=session.cwd, env=spawn_env, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, **_popen_kwargs)
        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            self._track_started(session, self._reader_loop, f"proc-reader-{session.id}")
        except Exception:
            self._reap_untracked(session, proc)
            # ⤷ 关键分支 raise：主动抛出异常，交给上层处理。
            raise
        # ⤷ 出口 return：返回 session
        return session

    # ┌── 函数 def _reap_untracked()
    # │  作用：Post-Popen setup failed: kill the orphaned subprocess (and any setsid
    # │  参数：self, session: ProcessSession, proc: subprocess.Popen
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _reap_untracked(self, session: ProcessSession, proc: subprocess.Popen) -> None:
        """Post-Popen setup failed: kill the orphaned subprocess (and any setsid
        descendants) so nothing leaks untracked."""
        with suppress(Exception):
            if session.systemd_unit:
                # Scope teardown is the authoritative cleanup for the worker cgroup
                # (never killpg here); the wrapper PID is terminated as fallback.
                _stop_systemd_unit(session.systemd_unit)
                # The worker runs in its own systemd scope and, since the #70716 session-isolation fix, its
                # own session. Stop the scope (kills every process in the worker cgroup), then terminate the
                # systemd-run wrapper PID as fallback.
                self._terminate_host_pid(proc.pid, session.host_start_time)
            elif not _IS_WINDOWS:
                # ⤷ 关键分支 try/except：捕获 (ProcessLookupError, PermissionError, OSError)，出错走兜底而不是崩溃。
                try:
                    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
                    os.killpg(os.getpgid(proc.pid), kill_signal)  # windows-footgun: ok - guarded by _IS_WINDOWS above
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
            else:
                proc.kill()
        with suppress(Exception):
            proc.wait(timeout=5)

    # ┌── 函数 def adopt_local()
    # │  作用：Take over a still-running foreground Popen as a tracked background session
    # │  参数：self, proc: subprocess.Popen, command: str, cwd: Optional[str], task_id: str = ..., session_key: str = ..., owner_task_id: str = ..., output_so_far: str = ...  …共 9 个
    # │  返回：ProcessSession
    # └──────────────────────────────────────────────────────────────
    def adopt_local(
        self, proc: subprocess.Popen, *, command: str, cwd: Optional[str], task_id: str = "",
        session_key: str = "", owner_task_id: str = "", output_so_far: str = "",
        notify_on_complete: bool = True) -> ProcessSession:
        """Take over a still-running foreground Popen as a tracked background session
        (yield-to-background: the user sent a message while the command was running).
        The caller has stopped its own drain thread; the registry's reader continues from
        the pipe's current position and ``output_so_far`` seeds the buffer so nothing
        already captured is lost."""
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd)
        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        session.notify_on_complete = notify_on_complete
        if output_so_far:
            session.append_output(output_so_far)
        self._track_started(session, self._reader_loop, f"proc-reader-{session.id}")
        # ⤷ 出口 return：返回 session
        return session

    # ┌── 函数 def spawn_via_env()
    # │  作用：Spawn a background process inside a non-local backend's sandbox.
    # │  参数：self, env: Any, command: str, cwd: str = None, task_id: str = '', session_key: str = '', timeout: int = 10, owner_task_id: str = ''
    # │  返回：ProcessSession
    # └──────────────────────────────────────────────────────────────
    def spawn_via_env(
        self, env: Any, command: str, cwd: str = None, task_id: str = "", session_key: str = "",
        timeout: int = 10, owner_task_id: str = "") -> ProcessSession:
        """Spawn a background process inside a non-local backend's sandbox.
        The command is wrapped to capture its in-sandbox PID and redirect output to a
        log file that later execute() calls poll. No live pipe or stdin, but it runs in
        the correct sandbox context."""
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd, env_ref=env, pid_scope="sandbox")
        temp_dir = self._env_temp_dir(env)
        log_path, pid_path, exit_path = (f"{temp_dir}/hermes_bg_{session.id}.{ext}" for ext in ("log", "pid", "exit"))
        q = shlex.quote
        bg_command = (
            f"mkdir -p {q(temp_dir)} && "
            f"( nohup bash -lc {q(command)} > {q(log_path)} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {q(exit_path)} ) & "
            f"echo $! > {q(pid_path)} && cat {q(pid_path)}")
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            result = env.execute(bg_command, timeout=timeout, rewrite_compound_background=False)
            output = result.get("output", "").strip()
            session.pid = next((int(ln) for ln in map(str.strip, output.splitlines()) if ln.isdigit()), None)
            # No PID from the wrapper (syntax error, broken redirect): a failed launch,
            # not a fake running session.
            if session.pid is None:
                session.mark_exited(int(result.get("returncode", -1)) or -1, "failed_start", "failed_start")
                session.output_buffer = output
        except Exception as e:
            session.mark_exited(-1, "failed_start", "failed_start")
            session.output_buffer = f"Failed to start: {e}"
        if session.exited:
            with self._lock:
                self._prune_if_needed()
        else:
            self._track_started(
                session, self._env_poller_loop, f"proc-poller-{session.id}", (env, log_path, pid_path, exit_path))
        # ⤷ 出口 return：返回 session
        return session

    # ----- Reader / Poller Threads -----

    # ┌── 函数 def _reader_loop()
    # │  作用：Background thread: read stdout from a local Popen process.
    # │  参数：self, session: ProcessSession
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _reader_loop(self, session: ProcessSession):
        """Background thread: read stdout from a local Popen process.
        ``buffer.read1(4096)`` not ``TextIOWrapper.read(4096)``: on pipes the latter
        blocks until EOF, landing "live" output in one burst at exit. Orphaned-pipe
        guard: a backgrounded grandchild (``node server.js &``) inherits our pipe's write
        end so EOF never arrives while it lives, which would park this thread and never
        fire ``notify_on_complete``; on POSIX we ``select()`` and stop draining shortly
        after the direct child exits (mirrors ``environments/base.py::_wait_for_process``).
        Windows pipes lack select(), so the lazy ``_reconcile_local_exit`` is the net.

        Windows pipes don't support select(); the blocking path is kept there and the lazy reconcile in
        poll()/wait() remains the safety net. See #68915, #8340.
        """
        first_chunk = True
        # A split multibyte UTF-8 char would become U+FFFD with stateless decoding; the
        # incremental decoder holds the partial sequence until the rest arrives.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # Incremental decoder: raw pipe reads can split a multibyte UTF-8 character across two read1()
        # chunks. A stateless per-chunk ``bytes.decode(errors="replace")`` turns both halves into U+FFFD
        # mojibake. The incremental decoder holds the partial sequence until the continuation bytes arrive —
        # same treatment the foreground path already has in
        # ``tools/environments/base.py::_wait_for_process``. (Ported from openclaw/openclaw#112325.)
        def _append_chunk(chunk: str):
            # ⤷ nonlocal：声明赋值的是外层函数作用域的变量（闭包常见）。
            nonlocal first_chunk
            if first_chunk:
                chunk = self._clean_shell_noise(chunk)
                first_chunk = False
            self._ingest_output(session, chunk)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            proc = session.process
            if proc is None or proc.stdout is None:
                return
            stdout = proc.stdout
            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)

            def _read_once():
                """One 4 KiB read: decoded text ('' for a partial multibyte tail), None at EOF."""
                if raw_read is None:  # mocked/alternate streams without a raw buffer: less "live"
                    # ⤷ 出口 return：返回 stdout.read(4096) or None
                    return stdout.read(4096) or None
                raw = raw_read(4096)
                # ⤷ 出口 return：返回 decoder.decode(raw) if raw else None
                return decoder.decode(raw) if raw else None
            # select() needs a real OS fd; mocked streams (tests, adapters) may lack
            # fileno() and use the blocking read instead.
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                fd = stdout.fileno() if raw_read is not None and not _IS_WINDOWS else None
            except Exception:
                fd = None
            if not (isinstance(fd, int) and fd >= 0):
                fd = None
            if fd is not None:
                import select as _select
            idle_after_exit = 0
            while True:
                if fd is not None:
                    # ⤷ 关键分支 try/except：捕获 (ValueError, OSError)，出错走兜底而不是崩溃。
                    try:
                        ready, _, _ = _select.select([fd], [], [], 0.2)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if not ready:
                        # Direct child gone and pipe idle ~200ms: a few more cycles for a
                        # buffered tail, then stop rather than wait forever on an orphaned
                        # grandchild's pipe.
                        if proc.poll() is not None:
                            # See #68915.
                            idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
                        continue
                chunk = _read_once()
                if chunk is None:
                    break  # true EOF — all writers closed
                if chunk:
                    _append_chunk(chunk)
                idle_after_exit = 0
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            self._finish_reader(
                session, decoder, _append_chunk, "Process",
                lambda: session.process.wait(timeout=5), lambda: session.process.returncode)

    # ┌── 函数 def _finish_reader()
    # │  作用：Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
    # │  参数：self, session, decoder, append, label, wait, exit_code
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _finish_reader(self, session, decoder, append, label, wait, exit_code) -> None:
        """Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
        one U+FFFD instead of vanishing), reap the child (no zombies), record the exit."""
        with suppress(Exception):
            tail = decoder.decode(b"", final=True)
            if tail:
                append(tail)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            wait()
        except Exception as e:
            logger.debug("%s wait timed out or failed: %s", label, e)
        self._finish_exited(session, exit_code())

    # ┌── 函数 def _log_delta_command()
    # │  作用：Shell command that reads only the log bytes written since ``offset``
    # │  参数：quoted_log_path: str, offset: int
    # │  返回：str
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _log_delta_command(quoted_log_path: str, offset: int) -> str:
        """Shell command that reads only the log bytes written since ``offset``
        (``cat``-ing the whole file every poll re-sends all output over docker/SSH).

        Prints one header line ``"<size> <offset>"`` then the bytes in [offset, size).
        The size is read first and the tail cut at that same size, so a growing file
        never sends a byte twice; a file that shrank was rotated/truncated, so the
        offset drops to 0 and the reader starts over. The window end is pulled back
        to a UTF-8 character boundary (the backend decodes each ``execute()`` result
        on its own, so a straddling multibyte char would become U+FFFD and break watch
        patterns at the seam): up to 3 trailing continuation bytes are held for the
        next poll and the header reports the trimmed size."""
        # ⤷ 出口 return：返回 f"""O={offset}; S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/…
        return (
            f"O={offset}; "
            f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
            f"S=${{S:-0}}; "
            f'if [ "$S" -lt "$O" ]; then O=0; fi; '
            # Scan back up to 3 continuation bytes (octal 200-277) to the lead byte; if
            # the lead's declared length (3xx=2, 34x-35x=3, 36x-37x=4) exceeds the bytes
            # present, trim to before it. Complete sequences and ASCII tails untouched.
            f'N=0; P=$S; while [ "$P" -gt "$O" ] && [ "$N" -lt 3 ]; do '
            f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
            f'case "$B" in 2[0-7][0-7]) P=$((P-1)); N=$((N+1));; *) break;; esac; done; '
            f'if [ "$N" -gt 0 ] || [ "$P" -eq "$S" ]; then '
            f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
            f'case "$B" in 3[0-3][0-7]) L=2;; 3[4-5][0-7]) L=3;; 3[6-7][0-7]) L=4;; *) L=1;; esac; '
            f'if [ "$L" -gt $((N+1)) ]; then S=$((P-1)); fi; fi; '
            f'echo "$S $O"; '
            f'if [ "$S" -gt "$O" ]; then '
            f"tail -c +$((O+1)) {quoted_log_path} 2>/dev/null | head -c $((S-O)); fi"
        )

    # ┌── 函数 def _env_poller_loop()
    # │  作用：Background thread: poll a sandbox log file for non-local backends.
    # │  参数：self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _env_poller_loop(self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str):
        """Background thread: poll a sandbox log file for non-local backends."""
        q = shlex.quote
        # Byte offset already read from the log (bytes, not chars: the shell counts bytes).
        prev_output_bytes = 0
        while not session.exited:
            time.sleep(2)
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                # Read only the bytes written since the last poll.
                raw = env.execute(self._log_delta_command(q(log_path), prev_output_bytes),
                                  timeout=10).get("output", "")
                header, _, delta = raw.partition("\n")
                # ⤷ 关键分支 try/except：捕获 ValueError，出错走兜底而不是崩溃。
                try:
                    size_str, offset_str = header.split()
                    new_size = int(size_str)
                    used_offset = int(offset_str)
                except ValueError:
                    # No usable header (command failed, shell missing a tool): skip this
                    # poll rather than act on a half-read value.
                    new_size = None
                    used_offset = None
                    delta = ""
                if new_size is not None:
                    if used_offset < prev_output_bytes:
                        # Log rotated/truncated: what we hold no longer lines up. Restart.
                        with session._lock:
                            session.output_buffer = ""
                    prev_output_bytes = new_size
                if delta:
                    with session._lock:
                        session.output_buffer += delta
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
                    self._check_watch_patterns(session, delta)
                    self._emit_output(session, delta)

                check = env.execute(
                    f"kill -0 \"$(cat {q(pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?", timeout=5)
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    # Exited -- read the exit code captured by the wrapper shell.
                    exit_str = env.execute(f"cat {q(exit_path)} 2>/dev/null", timeout=5).get("output", "").strip()
                    # ⤷ 关键分支 try/except：捕获 (ValueError, IndexError)，出错走兜底而不是崩溃。
                    try:
                        exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        exit_code = -1
                    session.exit_code = exit_code  # unlike mark_exited, a raced kill still takes this code
                    self._finish_exited(session, exit_code)
                    return
            except Exception:
                # Environment might be gone (sandbox reaped, etc.)
                session.exited, session.exit_code = True, -1
                session.completion_reason, session.termination_source = "lost", "backend_lost"
                self._move_to_finished(session)
                return

    # ┌── 函数 def _pty_reader_loop()
    # │  作用：Background thread: read output from a PTY process.
    # │  参数：self, session: ProcessSession
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _pty_reader_loop(self, session: ProcessSession):
        """Background thread: read output from a PTY process."""
        pty = session._pty
        # Same split-multibyte handling as _reader_loop.
        # PTY reads can split a multibyte UTF-8 character across chunks just like pipe reads — hold partial
        # sequences until the rest arrives. (Ported from openclaw/openclaw#112325.)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # Programs in a PTY can block waiting for replies to device-status / window-size /
        # cursor-position / DEC private-mode queries. Answer the bounded set and strip the
        # queries from captured output. POSIX only: Windows ConPTY is a real console host that
        # answers itself (and pywinpty yields str chunks, not bytes).
        responder = None
        if not _IS_WINDOWS:
            from tools.pty_query_responder import PtyQueryResponder
            responder = PtyQueryResponder(rows=30, cols=120)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            while pty.isalive():
                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes; pywinpty returns str
                        if responder is not None and isinstance(chunk, bytes):
                            chunk, replies = responder.process(chunk)
                            if replies:
                                # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
                                try:
                                    pty.write(replies)
                                except Exception:
                                    logger.debug(
                                        "PTY query response write failed",
                                        exc_info=True,
                                    )
                        text = chunk if isinstance(chunk, str) else decoder.decode(chunk)
                        if text:
                            self._ingest_output(session, text)
                except Exception:  # EOFError included
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)
        if responder is not None:
            # A query prefix split across the final reads is plain output after all.
            tail = decoder.decode(responder.flush())
            if tail:
                self._ingest_output(session, tail)
        self._finish_reader(
            session, decoder, lambda t: self._ingest_output(session, t), "PTY",
            pty.wait, lambda: pty.exitstatus if hasattr(pty, 'exitstatus') else -1)

    # ┌── 函数 def _ingest_output()
    # │  作用：Buffer a freshly-read chunk, then scan watch patterns and stream it live.
    # │  参数：self, session: ProcessSession, text: str
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _ingest_output(self, session: ProcessSession, text: str) -> None:
        """Buffer a freshly-read chunk, then scan watch patterns and stream it live."""
        session.append_output(text)
        self._check_watch_patterns(session, text)
        self._emit_output(session, text)

    # ┌── 函数 def _finish_exited()
    # │  作用：Mark a reader-observed exit (a raced kill keeps its own code/reason) and finish.
    # │  参数：self, session: ProcessSession, exit_code
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _finish_exited(self, session: ProcessSession, exit_code) -> None:
        """Mark a reader-observed exit (a raced kill keeps its own code/reason) and finish."""
        session.mark_exited(exit_code)
        self._move_to_finished(session)

    # ┌── 函数 def _move_to_finished()
    # │  作用：Move a session from running to finished.
    # │  参数：self, session: ProcessSession
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _move_to_finished(self, session: ProcessSession):
        """Move a session from running to finished.
        Idempotent: kill_process() and the reader thread can both call this; only
        the FIRST move enqueues the completion notification, so no duplicates."""
        with self._lock:
            was_running = session.id in self._running
            if was_running:
                # Keep the session tracked until its result is durable. A finite
                # parent must not observe completion and exit during this write.
                save_completed_result(session)
                self._running.pop(session.id)
            self._finished[session.id] = session
        # Release the retained Popen/PTY handles now: otherwise every
        # finished-but-unpruned session keeps its stdout pipe (or PTY master)
        # FD open until FINISHED_TTL_SECONDS elapses, and heavy background
        # churn can exhaust the gateway's FD limit. On the reader-thread path
        # the pipe is already at EOF; on the kill/reconcile paths the reader
        # may still be draining — its next read raises on the closed stream
        # and the loop exits, dropping at most the unread tail of a process
        # that was just killed. poll()/wait()/read_log() serve from the
        # buffered ``output_buffer``, never from the pipe.
        self._release_finished_handles(session)
        self._write_checkpoint()
        if was_running and session.notify_on_complete:
            notification = {
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "task_id": session.task_id,
                "owner_task_id": session.owner_task_id or session.task_id,
                "command": session.command,
                **({"handoff_note": session.handoff_note} if session.handoff_note else {}),
                **self._exit_fields(session),
                "output": _output_tail(session, 2000),
                # Stable producer identity across checkpoint recovery (unlike a
                # consumer-observed completion timestamp).
                "started_at": session.started_at,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)
        session._completion_event.set()

    # ┌── 函数 def _exit_fields()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：session: ProcessSession
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _exit_fields(session: ProcessSession) -> dict:
        # ⤷ 出口 return：返回 {'exit_code': session.exit_code, 'completion_reason': sessio…
        return {
            "exit_code": session.exit_code,
            "completion_reason": session.completion_reason,
            "termination_source": session.termination_source,
        }

    # ┌── 函数 def _release_finished_handles()
    # │  作用：Close a finished session's OS handles (Popen pipes / PTY master).
    # │  参数：self, session: ProcessSession
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _release_finished_handles(self, session: ProcessSession):
        """Close a finished session's OS handles (Popen pipes / PTY master).

        Best-effort and idempotent: the session may have no local Popen (env
        backends, detached recovery), or the handles may already be closed by
        the reader loop / kill path. Closing a Popen's stream objects does not
        kill anything — the child has already exited — it only releases the
        parent's pipe FDs, which is exactly the retained-resource leak.
        """
        proc = session.process
        if proc is not None:
            for stream in (proc.stdout, proc.stderr, proc.stdin):
                if stream is not None:
                    with suppress(OSError, ValueError):  # a stdin flush can hit EPIPE
                        stream.close()
        if session._pty is not None:
            # ptyprocess/pywinpty close() is idempotent (``closed`` flag) and
            # closes the master fd exactly once; it raises only if the child
            # ignores SIGKILL, which we don't want to surface on the finish path.
            with suppress(Exception):
                session._pty.close()

    # ----- Query Methods -----

    # ┌── 函数 def is_completion_consumed()
    # │  作用：Check if a completion notification was already consumed via wait/log.
    # │  参数：self, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/log."""
        # ⤷ 出口 return：返回 session_id in self._completion_consumed
        return session_id in self._completion_consumed

    # ┌── 函数 def is_session_waiting()
    # │  作用：Whether a goal loop (``hermes_cli.goals`` wait barrier) should stay parked on
    # │  参数：self, session_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def is_session_waiting(self, session_id: str) -> bool:
        """Whether a goal loop (``hermes_cli.goals`` wait barrier) should stay parked on
        this session: still running AND, with ``watch_patterns``, none matched yet (a
        long-lived watcher unblocks on its trigger, not on exit). Unknown/exited/
        already-fired sessions return False so a stale barrier can never wedge the loop."""
        with self._lock:
            session = (self._running.get(session_id) or self._finished.get(session_id)) if session_id else None
        if session is None:
            # ⤷ 出口 return：返回 False
            return False
        with suppress(Exception):
            self._refresh_detached_session(session)
        # ⤷ 出口 return：返回 not session.exited and (not (session.watch_patterns and (not…
        return not session.exited and not (
            session.watch_patterns and not session._watch_disabled and session._watch_hits > 0)

    # ┌── 函数 def wait_for_pending_completions()
    # │  作用：Bounded linger for ``notify_on_complete`` background processes at one-shot exit.
    # │  参数：self, task_id: Optional[str] = None, timeout: float | None = ..., poll_interval: float = ...
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def wait_for_pending_completions(
        self, task_id: Optional[str] = None, *, timeout: float | None = None, poll_interval: float = 1.0,
    ) -> dict:
        """Bounded linger for ``notify_on_complete`` background processes at one-shot exit.
        A one-shot CLI run (``hermes -q/-Q/-z``) exits when its turn ends; a background
        process it spawned still holds a stdout pipe owned by the dying parent and dies of
        SIGPIPE seconds later (Bot Mode handoff replies were the visible casualty). Only
        ``notify_on_complete`` processes carry a completion contract — servers/daemons/
        watchers aren't the parent's to wait for. ``task_id=None`` waits on every tracked
        process; ``timeout=None`` reads ``terminal.oneshot_completion_wait_seconds`` (``<= 0``
        disables). Each pass re-reconciles child state so an orphaned-pipe exit can't wedge
        the linger. Returns ``{"waited", "completed", "timed_out"}`` id lists.

        Bot Mode handoff REPLIES are the visible casualty (#90879): a recipient invoked as ``hermes -p <bot>
        chat -Q --query-file ...`` dispatches its reply via ``message_agent`` / ``bot_relay`` exactly this
        way, then exits, and the reply process is destroyed ~3s later. The sender waits forever for a reply
        that was already killed.
        See #17327.
        """
        if timeout is None:
            timeout = self._oneshot_completion_wait_seconds()
        result: dict = {"waited": [], "completed": [], "timed_out": []}
        with self._lock:
            # `_finished` too: `_move_to_finished` pops a session from `_running` and enqueues its completion
            # only after releasing handles and writing the checkpoint. A parent whose turn ends inside that
            # window would otherwise see nothing pending, drain nothing and exit without the follow-up turn.
            pending = [
                s for store in (self._running, self._finished) for s in store.values()
                if s.notify_on_complete and not s._completion_event.is_set() and (task_id is None or s.task_id == task_id)
            ]
        if not pending or timeout <= 0:
            # ⤷ 出口 return：返回 result
            return result
        result["waited"] = [s.id for s in pending]
        logger.info(
            "One-shot exit lingering (bounded %ss) for %d notify_on_complete "
            "background process(es): %s",
            timeout, len(pending), ", ".join(s.id for s in pending))
        deadline = time.monotonic() + max(float(timeout), 0.0)
        interval = max(float(poll_interval), 0.05)
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
        except Exception:
            _is_interrupted = lambda: False  # noqa: E731
        interrupted = False
        for session in pending:
            # ⤷ 关键分支 try/except：捕获 KeyboardInterrupt，出错走兜底而不是崩溃。
            try:
                while not session._completion_event.is_set():
                    if interrupted or _is_interrupted():
                        interrupted = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # Reconcile first so orphaned-pipe and detached exits fire the event.
                    with suppress(Exception):
                        # Reconcile first: catches direct-child exits whose reader is blocked on a pipe held
                        # open by a descendant (#17327) and detached/env sessions, so the event actually
                        # fires.
                        # Reconcile against real child state before reading session.exited. Guards against
                        # orphaned-pipe reader hangs (issue #17327).
                        # Reconcile against real child state — guards against orphaned- pipe reader hangs
                        # where the reader is blocked but the direct child has already exited (issue
                        # #17327).
                        self._reconcile_local_exit(session)
                        self._refresh_detached_session(session)
                    if session._completion_event.is_set():
                        break
                    session._completion_event.wait(min(remaining, interval))
            except KeyboardInterrupt:
                # Stop waiting, but never let the interrupt skip the caller's durable
                # teardown (session flush, end_session) that follows.
                interrupted = True
            result["completed" if session._completion_event.is_set() else "timed_out"].append(session.id)
        if result["timed_out"]:
            logger.warning(
                "One-shot exit linger timed out after %ss with %d background "
                "process(es) still running: %s — they may be killed when this "
                "process exits.",
                timeout, len(result["timed_out"]), ", ".join(result["timed_out"]))
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def _oneshot_completion_wait_seconds()
    # │  作用：Linger (s) for one-shot exits with pending notify_on_complete processes; 0 disables.
    # │  参数：无
    # │  返回：float
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _oneshot_completion_wait_seconds() -> float:
        """Linger (s) for one-shot exits with pending notify_on_complete processes; 0 disables."""
        # ⤷ 出口 return：返回 ProcessRegistry._config_seconds('oneshot_completion_wait_sec…
        return ProcessRegistry._config_seconds("oneshot_completion_wait_seconds", 600.0)

    # ┌── 函数 def _drain_should_skip()
    # │  作用：Skip a completion the CLI agent already has this turn — consumed via wait/log
    # │  参数：self, session_id: str, skip_poll_observed: bool = ...
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _drain_should_skip(self, session_id: str, *, skip_poll_observed: bool = True) -> bool:
        """Skip a completion the CLI agent already has this turn — consumed via wait/log
        or observed inline via poll(). Gateway/tui watchers check only
        ``is_completion_consumed`` so a read-only poll never suppresses their turn.

        Skips when the agent has either truly consumed the output (wait/log → ``_completion_consumed``) or
        observed the exit inline via poll() (``_poll_observed``). In both cases the CLI agent already has
        the result this turn, so injecting a [SYSTEM: ...] completion would be a duplicate (#8228).
        """
        # ⤷ 出口 return：返回 session_id in self._completion_consumed or (skip_poll_observ…
        return session_id in self._completion_consumed or (skip_poll_observed and session_id in self._poll_observed)

    # ┌── 函数 def _surface_child_process_notifications()
    # │  作用：``delegation.surface_child_process_notifications``; False on any config
    # │  参数：无
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _surface_child_process_notifications() -> bool:
        """``delegation.surface_child_process_notifications``; False on any config
        error — never crash the drain loop."""
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            # ⤷ 出口 return：返回 bool(ProcessRegistry._config_value('delegation', 'surface_ch…
            return bool(ProcessRegistry._config_value("delegation", "surface_child_process_notifications", False))
        except Exception:
            # ⤷ 出口 return：返回 False
            return False

    # ┌── 函数 def _owns_event()
    # │  作用：Routing verdict for one drained event (see drain_notifications); False = requeue.
    # │  参数：evt: dict, session_key: str, owns_event, is_async_delegation: bool
    # │  返回：bool
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _owns_event(evt: dict, session_key: str, owns_event, is_async_delegation: bool) -> bool:
        """Routing verdict for one drained event (see drain_notifications); False = requeue."""
        evt_session_key = str(evt.get("session_key") or "")
        requires_positive_proof = is_async_delegation or bool(evt_session_key or evt.get("origin_ui_session_id"))
        if owns_event is not None and requires_positive_proof:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                # ⤷ 出口 return：返回 bool(owns_event(evt))
                return bool(owns_event(evt))
            except Exception:
                # ⤷ 出口 return：返回 False
                return False  # fail closed — never leak on a broken check
        if session_key and requires_positive_proof:
            # ⤷ 出口 return：返回 evt_session_key == session_key
            return evt_session_key == session_key
        # Restored payloads from a previous process: an unfiltered drain cannot prove
        # ownership, so leave them for the owner.
        # ⤷ 出口 return：返回 not (is_async_delegation and evt.get('restored'))
        return not (is_async_delegation and evt.get("restored"))

    # ┌── 函数 def drain_notifications()
    # │  作用：Pop all pending events and return ``(raw_event, formatted_text)`` pairs.
    # │  参数：self, session_key: str = '', owns_event = None, skip_poll_observed: bool = ...
    # │  返回：'list[tuple[dict, str]]'
    # └──────────────────────────────────────────────────────────────
    def drain_notifications(
        self, session_key: str = "", owns_event=None, *, skip_poll_observed: bool = True,
    ) -> "list[tuple[dict, str]]":
        """Pop all pending events and return ``(raw_event, formatted_text)`` pairs.
        Skips completions per ``_drain_should_skip`` (gateway/TUI pass
        ``skip_poll_observed=False``). Routing (``_owns_event``): async-delegation events
        always need ownership proof, ordinary events once they carry ``session_key`` or
        ``origin_ui_session_id``; ``owns_event(evt)`` (strongest; the TUI passes a
        compression-chain-aware check) consumes ONLY on True, ``session_key`` uses plain
        equality; non-owned events are re-queued for their owner. No filter consumes
        everything (legacy single-session) except restored delegation payloads (fail-closed)."""
        results: "list[tuple[dict, str]]" = []
        requeue: "list[dict]" = []
        # delegation.surface_child_process_notifications, read at most once per drain
        # and only when an sa- event shows up.
        surface_child: "bool | None" = None
        while not self.completion_queue.empty():
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                evt = self.completion_queue.get_nowait()
            except Exception:
                break
            is_async_delegation = evt.get("type") == "async_delegation"
            if not self._owns_event(evt, session_key, owns_event, is_async_delegation):
                requeue.append(evt)
                continue
            # Routing happened first so a foreign session cannot drop the owner's
            # event via its own consumed/observed state.
            _evt_sid = evt.get("session_id", "")
            if evt.get("type") == "completion" and self._drain_should_skip(
                _evt_sid, skip_poll_observed=skip_poll_observed):
                continue
            # Subagent-owned process notifications are suppressed by default — the
            # child's delegation result is the deliverable. Judge ownership on
            # owner_task_id (RAW spawning id; task_id is the container key, collapsed
            # by _resolve_container_task_id). Dropped, NOT requeued: children never
            # drain, so a requeue would pin the event forever. 'async_delegation'
            # is the result itself and is NEVER suppressed.
            _evt_task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "")
            if not is_async_delegation and _evt_task_id.startswith("sa-"):
                if surface_child is None:
                    surface_child = self._surface_child_process_notifications()
                if not surface_child:
                    logger.debug(
                        "Suppressed subagent-owned process notification "
                        "(delegation.surface_child_process_notifications=false): "
                        "type=%s session_id=%s task_id=%s",
                        evt.get("type", "completion"), _evt_sid, _evt_task_id)
                    continue
            if text := format_process_notification(evt):
                results.append((evt, text))
        for evt in requeue:
            self.completion_queue.put(evt)
        # ⤷ 出口 return：返回 results
        return results

    # Minimum suffix chars for prefix resolution; "p"/"proc_1" are too collision-prone.
    _MIN_PREFIX_CHARS = 4

    # ┌── 函数 def get()
    # │  作用：Session by full ID or unique prefix (``proc_4dae`` / bare ``4dae``, like git
    # │  参数：self, session_id: str
    # │  返回：Optional[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Session by full ID or unique prefix (``proc_4dae`` / bare ``4dae``, like git
        short hashes); ambiguous or too-short prefixes resolve to None, never a guess."""
        if not isinstance(session_id, str) or not session_id:
            # ⤷ 出口 return：返回 None
            return None
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            session = load_completed_results(session_id).get(session_id)
        # ⤷ 出口 return：返回 self._refresh_detached_session(session if session is not Non…
        return self._refresh_detached_session(session if session is not None else self._resolve_prefix(session_id))

    # ┌── 函数 def _resolve_prefix()
    # │  作用：Resolve a unique session-ID prefix (a bare hex tail is normalized to
    # │  参数：self, session_id: str
    # │  返回：Optional[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def _resolve_prefix(self, session_id: str) -> Optional[ProcessSession]:
        """Resolve a unique session-ID prefix (a bare hex tail is normalized to
        ``proc_<tail>``); :meth:`get` tries exact first."""
        query = session_id.strip() if isinstance(session_id, str) else ""
        if not query:
            # ⤷ 出口 return：返回 None
            return None
        if not query.startswith("proc_"):
            query = f"proc_{query}"
        if len(query) - len("proc_") < self._MIN_PREFIX_CHARS:
            # ⤷ 出口 return：返回 None
            return None
        matches = load_completed_results(query)
        with self._lock:
            matches.update({
                sid: s for store in (self._running, self._finished)
                for sid, s in store.items() if sid.startswith(query)
            })
        # ⤷ 出口 return：返回 next(iter(matches.values())) if len(matches) == 1 else None
        return next(iter(matches.values())) if len(matches) == 1 else None

    # ┌── 函数 def _reconcile_local_exit()
    # │  作用：Reconcile ``session.exited`` against the real child state.
    # │  参数：self, session: 'ProcessSession'
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def _reconcile_local_exit(self, session: "ProcessSession") -> None:
        """Reconcile ``session.exited`` against the real child state.
        The reader flips ``exited`` only at EOF; when the direct child has exited but a
        descendant (e.g. a daemon from ``hermes update``) holds the pipe open, poll()
        would report "running" forever. If ``Popen.poll()`` has an exit code, drain
        readable bytes non-blocking and flip ``exited``. No-op for env/PTY, exited and
        detached sessions.

        The reader thread (`_reader_loop`) sets `session.exited = True` only in its `finally` block, which
        runs when `stdout.read()` returns EOF. If the direct `Popen` child has exited but a descendant
        process (e.g. a daemon spawned by `hermes update` restarting the gateway) is still holding the
        stdout pipe open, the reader blocks forever and poll() keeps returning "running" indefinitely (issue
        #17327 — 74 polls over 7 minutes on Feishu).
        """
        if session is None or session.exited:
            return
        proc = getattr(session, "process", None)
        if proc is None:
            return
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            rc = proc.poll()
        except Exception:
            return
        if rc is None:
            return  # Direct child still running — reader block is legitimate.
        # Best-effort non-blocking drain of whatever the reader hasn't consumed.
        stdout = getattr(proc, "stdout", None)
        if stdout is not None and not _IS_WINDOWS:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                import fcntl
                fd = stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
                try:
                    with suppress(BlockingIOError, OSError, ValueError):
                        chunk = stdout.read()
                        if chunk:
                            session.append_output(chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace"))
                finally:
                    with suppress(Exception):
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
            except Exception as e:
                logger.debug("Non-blocking drain failed for %s: %s", session.id, e)
        with session._lock:
            session.mark_exited(rc)
        logger.info(
            "Reconciled session %s: direct child exited with code %s but reader "
            "was still blocked (orphaned pipe). Flipped to exited.",
            session.id, rc)
        self._move_to_finished(session)

    # ┌── 函数 def _status_head()
    # │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
    # │  参数：session: ProcessSession
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _status_head(session: ProcessSession) -> dict:
        # ⤷ 出口 return：返回 {'session_id': session.id, 'command': session.command, 'stat…
        return {"session_id": session.id, "command": session.command, "status": "exited" if session.exited else "running"}

    # ┌── 函数 def poll()
    # │  作用：Check status and get new output for a background process.
    # │  参数：self, session_id: str
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def poll(self, session_id: str) -> dict:
        """Check status and get new output for a background process."""
        session = self.get(session_id)
        if session is None:
            # ⤷ 出口 return：返回 _not_found(session_id)
            return _not_found(session_id)
        self._reconcile_local_exit(session)  # orphaned-pipe reader guard
        with session._lock:
            output_preview = _output_tail(session, 1000)
        result = {
            **self._status_head(session), "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at), "output_preview": output_preview}
        if session.exited:
            result.update(self._exit_fields(session))
            # Read-only: record in _poll_observed (CLI inline dedup) but NOT in
            # _completion_consumed, or a status check would suppress the watcher's
            # autonomous delivery turn. See __init__.
            self._poll_observed.add(session_id)
        if session.detached:
            result.update(detached=True, note="Process recovered after restart -- output history unavailable")
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def read_log()
    # │  作用：Read the full output log with optional pagination by lines.
    # │  参数：self, session_id: str, offset: int | None = None, limit: int = 200
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def read_log(self, session_id: str, offset: int | None = None, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            # ⤷ 出口 return：返回 _not_found(session_id)
            return _not_found(session_id)
        with session._lock:
            full_output = strip_ansi(session.output_buffer)
        lines = full_output.splitlines()
        total_lines = len(lines)
        # offset=None -> last N lines; an explicit offset=0 means the HEAD (don't
        # conflate the two via falsiness).
        # An explicit offset=0 means "start from the first line" — previously it was conflated with the
        # default and silently returned the TAIL instead of the head (same falsy-coercion class as the
        # wait() timeout guard; salvaged from PR #60004, credit @isheng-eqi).
        if offset is None and limit > 0:
            selected = lines[-limit:]
            observed_completion_output = bool(selected) or total_lines == 0
        else:
            offset = offset or 0
            selected = lines[offset:offset + limit]
            stop = slice(offset, offset + limit).indices(total_lines)[1]
            observed_completion_output = total_lines == 0 or (bool(selected) and stop == total_lines)
        result = {
            **self._status_head(session), "output": "\n".join(selected),
            "total_lines": total_lines, "showing": f"{len(selected)} lines"}
        if session.exited and observed_completion_output:
            self._completion_consumed.add(session_id)
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def wait()
    # │  作用：Block until the process exits, the timeout elapses, the user interrupts, or a
    # │  参数：self, session_id: str, timeout: int = None
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def wait(self, session_id: str, timeout: int = None) -> dict:
        """Block until the process exits, the timeout elapses, the user interrupts, or a
        mid-turn user message (steer/redirect → ``request_yield``) releases the wait.
        ``timeout`` defaults to (and is clamped by) TERMINAL_TIMEOUT. Returns a dict
        with status exited|timeout|interrupted|not_found|error and an output snapshot."""
        from tools.interrupt import consume_yield as _consume_yield, is_interrupted as _is_interrupted

        # ⤷ 关键分支 try/except：捕获 (ValueError, TypeError)，出错走兜底而不是崩溃。
        try:
            max_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            max_timeout = 180
        # The schema says minimum=1 but not every caller enforces it; timeout=0 is
        # falsy and would silently fall through to the default wait.
        if timeout is not None and timeout <= 0:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': f'timeout must be positive (got…
            return {"status": "error", "error": f"timeout must be positive (got {timeout})"}
        timeout_note = None
        effective_timeout = timeout or max_timeout
        if timeout and timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = f"Requested wait of {timeout}s was clamped to configured limit of {max_timeout}s"
        session = self.get(session_id)
        if session is None:
            # ⤷ 出口 return：返回 _not_found(session_id)
            return _not_found(session_id)
        deadline = time.monotonic() + effective_timeout
        while time.monotonic() < deadline:
            session = self._refresh_detached_session(session)
            if session is None:
                # ⤷ 出口 return：返回 _not_found(session_id)
                return _not_found(session_id)
            self._reconcile_local_exit(session)  # orphaned-pipe reader guard
            result = None
            if session.exited:
                self._completion_consumed.add(session_id)
                result = self._exit_snapshot(session, "exited")
            elif _is_interrupted():
                result = {
                    "status": "interrupted", "command": session.command, "output": _output_tail(session, 1000),
                    "note": "User sent a new message -- wait interrupted"}
            elif _consume_yield(threading.current_thread().ident):
                # A steer/redirect landed mid-turn: redirect() asks tool workers to YIELD so
                # the user's message is delivered instead of parked behind this wait. The
                # process is untouched and still notify-tracked; the model should read the
                # steer text and respond, not re-issue the wait (kimi-code#3697 class).
                result = {
                    "status": "interrupted", "command": session.command, "output": _output_tail(session, 1000),
                    "process_running": True,
                    "note": ("User sent a new message -- wait released; the process is still "
                             "running and you will be notified on exit. Respond to the user now.")}
            if result is not None:
                if timeout_note:
                    result["timeout_note"] = timeout_note
                # ⤷ 出口 return：返回 result
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            session._completion_event.wait(timeout=min(1.0, remaining))
        result = {
            "status": "timeout", "command": session.command, "output": _output_tail(session, 1000),
            # Not a failure — models re-issued identical waits after misreading this as an error.
            "process_running": True}
        base_note = (
            f"Wait window of {effective_timeout}s elapsed — the process is still running. This is not an error.")
        if session.started_at:
            base_note += f" Uptime: {int(time.time() - session.started_at)}s."
        base_note += (
            " notify_on_complete is set: you will be notified on exit — do more work instead of waiting again."
            if session.notify_on_complete else
            " Poll again later or use terminal(background=true, "
            "notify_on_complete=true) next time for automatic notification.")
        result["timeout_note"] = f"{timeout_note}. {base_note}" if timeout_note else base_note
        # ⤷ 出口 return：返回 result
        return result

    # ┌── 函数 def _exit_snapshot()
    # │  作用：Result dict for an exited session: exit metadata + last 2000 chars of output.
    # │  参数：session: ProcessSession, status: str
    # │  返回：dict
    # │  【语法】@staticmethod 表示这个方法不依赖实例（没有 self），相当于挂到类上的普通函数。
    # └──────────────────────────────────────────────────────────────
    @staticmethod
    def _exit_snapshot(session: ProcessSession, status: str) -> dict:
        """Result dict for an exited session: exit metadata + last 2000 chars of output."""
        # ⤷ 出口 return：返回 {'status': status, 'command': session.command, **ProcessRegi…
        return {
            "status": status, "command": session.command,
            **ProcessRegistry._exit_fields(session), "output": _output_tail(session, 2000)}

    # ┌── 函数 def kill_process()
    # │  作用：Kill a background process and return its output snapshot.
    # │  参数：self, session_id: str, source: str = ..., consume_output: bool = ...
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def kill_process(
        self, session_id: str, *, source: str = "process.kill", consume_output: bool = True,
    ) -> dict:
        """Kill a background process and return its output snapshot.
        ``consume_output`` is true for explicit tool/RPC kills (the caller sees the
        output). Bulk cleanup passes false so it doesn't suppress an autonomous
        completion notification — except abandoned-turn reaping (``kill_started_since``),
        which passes true so a killed abandoned process can't revive stopped work."""
        session = self.get(session_id)
        if session is None:
            # ⤷ 出口 return：返回 _not_found(session_id)
            return _not_found(session_id)
        if session.exited:
            # A double-forked descendant may still be alive in the systemd scope even
            # though the main process exited — stop the scope to reap survivors.
            # See #70716.
            # If the worker was spawned in its own systemd scope (#70716), stop the entire unit to reap any
            # double-forked descendants that were reparented inside the scope and survived the PID signal
            # above (reviewer gap #2). ``systemctl --user stop`` sends SIGTERM to every process in the
            # cgroup and escalates to SIGKILL after TimeoutStopSec. This is additive — the PID-based kill
            # above already handled the main process; this catches stragglers.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            with session._lock:
                result = self._exit_snapshot(session, "already_exited")
            # Only suppress the autonomous turn after its output is present in
            # the explicit kill result, matching wait/log consumption.
            if consume_output:
                self._completion_consumed.add(session_id)
            # ⤷ 出口 return：返回 result
            return result
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            early = self._signal_kill(session, session_id, consume_output)
            if early is not None:
                # ⤷ 出口 return：返回 early
                return early
            # Additive to the PID kill: stopping the scope reaps double-forked
            # descendants reparented inside the cgroup.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            # Capture output, mark consumed, THEN expose ``exited`` to watcher tasks —
            # closes the delayed-notification race without losing the transcript.
            with session._lock:
                output = _output_tail(session, 2000)
                if consume_output:
                    self._completion_consumed.add(session_id)
                session.exited = True
                session.exit_code = -15  # SIGTERM
                session.completion_reason = "killed"
                session.termination_source = source
            self._move_to_finished(session)
            self._write_checkpoint()
            # ⤷ 出口 return：返回 {'status': 'killed', 'session_id': session.id, 'completion_r…
            return {
                "status": "killed", "session_id": session.id, "completion_reason": session.completion_reason,
                "termination_source": session.termination_source, "output": output}
        except Exception as e:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': str(e)}
            return {"status": "error", "error": str(e)}

    # ┌── 函数 def _signal_kill()
    # │  作用：Deliver the kill via PTY, local Popen tree, sandbox exec or recovered host
    # │  参数：self, session: ProcessSession, session_id: str, consume_output: bool
    # │  返回：Optional[dict]
    # └──────────────────────────────────────────────────────────────
    def _signal_kill(self, session: ProcessSession, session_id: str, consume_output: bool) -> Optional[dict]:
        """Deliver the kill via PTY, local Popen tree, sandbox exec or recovered host
        PID. Returns a final result dict when the kill cannot proceed (recycled/dead
        recovered PID, or no runtime handle), else None."""
        if session._pty:
            # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
            try:
                session._pty.terminate(force=True)
            except Exception:
                if session.pid:
                    os.kill(session.pid, signal.SIGTERM)
        elif session.process:
            # Tree kill: on Windows Popen.terminate() only kills the shell wrapper and
            # leaves Git Bash descendants behind.
            self._terminate_host_pid(session.process.pid, session.host_start_time)
        elif session.env_ref and session.pid:
            session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
        elif session.detached and session.pid_scope == "host" and session.pid:
            # Identity check, not bare liveness: a gone/recycled PID means our
            # process exited — never tree-kill the stranger. Still stop an owned
            # scope: a daemonized descendant may survive the wrapper PID.
            # If this recovered session also carries an owned systemd scope, stop that scope before
            # returning: a daemonized descendant may still be alive there even though the wrapper PID exited
            # or was recycled across the gateway restart (#70716, teknium1 review).
            if not self._host_pid_is_ours(session.pid, session.host_start_time):
                if session.systemd_unit:
                    _stop_systemd_unit(session.systemd_unit)
                with session._lock:
                    session.exited = True
                    session.exit_code = None
                    output = _output_tail(session, 2000)
                if consume_output:
                    self._completion_consumed.add(session_id)
                self._move_to_finished(session)
                # ⤷ 出口 return：返回 {'status': 'already_exited', 'exit_code': session.exit_code,…
                return {"status": "already_exited", "exit_code": session.exit_code, "output": output}
            self._terminate_host_pid(session.pid, session.host_start_time)
        else:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': 'Recovered process cannot be ki…
            return {
                # Reject non-positive timeouts — the schema declares minimum=1, but not every caller
                # enforces schemas before dispatch. timeout=0 is falsy, so without this guard it silently
                # fell through (`0 or max_timeout`) to the DEFAULT wait instead of erroring. Salvaged from
                # PR #60004 (credit @isheng-eqi).
                "status": "error",
                "error": "Recovered process cannot be killed after restart because "
                         "its original runtime handle is no longer available",
            }
        # ⤷ 出口 return：返回 None
        return None

    # ┌── 函数 def _stdin_op()
    # │  作用：Run a stdin operation on a running session — ``pty_op(pty)`` under PTY mode,
    # │  参数：self, session_id: str, pty_op, pipe_op, ok: dict
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def _stdin_op(self, session_id: str, pty_op, pipe_op, ok: dict) -> dict:
        """Run a stdin operation on a running session — ``pty_op(pty)`` under PTY mode,
        else ``pipe_op(stdin)`` on the Popen pipe — and return *ok* on success."""
        session = self.get(session_id)
        if session is None:
            # ⤷ 出口 return：返回 _not_found(session_id)
            return _not_found(session_id)
        if session.exited:
            # ⤷ 出口 return：返回 {'status': 'already_exited', 'error': 'Process has already f…
            return {"status": "already_exited", "error": "Process has already finished"}
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            if session._pty:
                pty_op(session._pty)
            elif not session.process or not session.process.stdin:
                # ⤷ 出口 return：返回 {'status': 'error', 'error': 'Process stdin not available (n…
                return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
            else:
                pipe_op(session.process.stdin)
            # ⤷ 出口 return：返回 ok
            return ok
        except Exception as e:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': str(e)}
            return {"status": "error", "error": str(e)}

    # ┌── 函数 def write_stdin()
    # │  作用：Send raw data to a running process's stdin (no newline appended).
    # │  参数：self, session_id: str, data: str
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def write_stdin(self, session_id: str, data: str) -> dict:
        """Send raw data to a running process's stdin (no newline appended)."""

        def via_pty(pty):
            # pywinpty expects str on Windows; ptyprocess expects bytes on POSIX.
            if _IS_WINDOWS:
                pty.write(data.decode("utf-8") if isinstance(data, bytes) else str(data))
            else:
                # surrogateescape: a PTY is a byte stream — round-trip the original
                # bytes instead of crashing on surrogate content.
                pty.write(data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data)

        def via_pipe(stdin):
            stdin.write(data)
            stdin.flush()
        # ⤷ 出口 return：返回 self._stdin_op(session_id, via_pty, via_pipe, {'status': 'ok…
        return self._stdin_op(session_id, via_pty, via_pipe, {"status": "ok", "bytes_written": len(data)})

    # ┌── 函数 def submit_stdin()
    # │  作用：Send data + newline to stdin (like pressing Enter).
    # │  参数：self, session_id: str, data: str = ''
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """Send data + newline to stdin (like pressing Enter).
        On a Windows PTY, Enter is a carriage return: ConPTY treats ``\\r`` as
        end-of-line and a bare ``\\n`` through pywinpty is NOT a line terminator — the
        child's blocking line read (``readline()``, Go ``bufio.Scanner``) never returns
        and the process hangs looking healthy. ``\\r\\n`` gives it both; POSIX keeps ``\\n``."""
        session = self.get(session_id)
        # ⤷ 出口 return：返回 self.write_stdin(session_id, data + ('\r\n' if _IS_WINDOWS a…
        return self.write_stdin(session_id, data + ("\r\n" if _IS_WINDOWS and session and session._pty else "\n"))

    # ┌── 函数 def request_close_terminal()
    # │  作用：Ask the desktop GUI to close this process's read-only terminal tab. Does NOT
    # │  参数：self, session_id: str
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def request_close_terminal(self, session_id: str) -> dict:
        """Ask the desktop GUI to close this process's read-only terminal tab. Does NOT
        kill the process — output keeps buffering and the tab can be reopened from the
        status stack. Errors when no UI close sink is wired."""
        if self.on_close is None:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': 'close_terminal is only availab…
            return {"status": "error", "error": "close_terminal is only available in the Hermes desktop app."}
        # The session may already be finished (or pruned) — the tab can still
        # linger and be closed, so a missing session is not an error here.
        # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
        try:
            self.on_close(self.get(session_id), session_id)
        except Exception as e:
            # ⤷ 出口 return：返回 {'status': 'error', 'error': str(e)}
            return {"status": "error", "error": str(e)}
        # ⤷ 出口 return：返回 {'status': 'ok', 'closed': session_id, 'note': 'Closed the r…
        return {
            "status": "ok", "closed": session_id,
            "note": "Closed the read-only terminal tab. The process was not killed; "
                    "its output remains available and the user can reopen the tab "
                    "from the status stack."}

    # ┌── 函数 def close_stdin()
    # │  作用：Close a running process's stdin / send EOF without killing the process.
    # │  参数：self, session_id: str
    # │  返回：dict
    # └──────────────────────────────────────────────────────────────
    def close_stdin(self, session_id: str) -> dict:
        """Close a running process's stdin / send EOF without killing the process."""
        session = self.get(session_id)
        msg = "EOF sent" if session is not None and session._pty else "stdin closed"
        # ⤷ 出口 return：返回 self._stdin_op(session_id, lambda pty: pty.sendeof(), lambda…
        return self._stdin_op(
            session_id, lambda pty: pty.sendeof(), lambda stdin: stdin.close(), {"status": "ok", "message": msg})

    # ┌── 函数 def count_running()
    # │  作用：O(1) running count for status-bar polling; dict ``len()`` is atomic, no lock.
    # │  参数：self
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def count_running(self) -> int:
        """O(1) running count for status-bar polling; dict ``len()`` is atomic, no lock."""
        # ⤷ 出口 return：返回 len(self._running)
        return len(self._running)

    # ┌── 函数 def list_sessions()
    # │  作用：Running and recently-finished processes for ``task_id`` and/or ``session_key``;
    # │  参数：self, task_id: str = None, session_key: str = None, include_retained: bool = ...
    # │  返回：list
    # └──────────────────────────────────────────────────────────────
    def list_sessions(self, task_id: str = None, session_key: str = None, *, include_retained: bool = False) -> list:
        """Running and recently-finished processes for ``task_id`` and/or ``session_key``;
        cross-task entries sharing the gateway session (a forgotten preview server
        blocking session reset) are flagged ``"session_scoped": true``.

        When ``task_id`` is given, processes for that task are included. When ``session_key`` is also given,
        session-scoped background processes (``background: true``) registered under that gateway session are
        surfaced too, even if they belong to a different task — so the agent can discover a forgotten
        preview server that is blocking session reset (#29177).
        """
        # Only an explicit tool query reads historical receipts. Status bars and
        # gateway liveness scans call this frequently and need the live registry.
        sessions = load_completed_results() if include_retained else {}
        with self._lock:
            sessions.update(self._finished)
            sessions.update(self._running)
        all_sessions = [self._refresh_detached_session(s) for s in sessions.values()]
        if task_id or session_key:
            all_sessions = [
                s for s in all_sessions
                if (task_id and s.task_id == task_id) or (session_key and s.session_key == session_key)
            ]
        result = []
        for s in all_sessions:
            # List-only refreshes must observe child exit even while descendants
            # keep the capture pipe open; retain the existing completion owner.
            self._reconcile_local_exit(s)
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "owner_task_id": s.owner_task_id or s.task_id,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            # Flag processes surfaced only because they share the gateway session (not the current task) —
            # these are the long-lived background processes a user may have forgotten about (#29177).
            if task_id and session_key and s.task_id != task_id and s.session_key == session_key:
                entry["session_scoped"] = True
            # Trigger metadata for goal-loop judges (a watcher may never exit).
            if s.watch_patterns and not s._watch_disabled:
                entry.update(watch_patterns=list(s.watch_patterns), watch_hit=s._watch_hits > 0)
            if s.notify_on_complete:
                entry["notify_on_complete"] = True
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            result.append(entry)
        # ⤷ 出口 return：返回 result
        return result

    # ----- Session/Task Queries (for gateway integration) -----

    # ┌── 函数 def _any_running()
    # │  作用：True if any still-running session satisfies *predicate*, after refreshing
    # │  参数：self, predicate
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def _any_running(self, predicate) -> bool:
        """True if any still-running session satisfies *predicate*, after refreshing
        detached sessions so a finished-but-unreaped process reads as inactive."""
        with self._lock:
            sessions = list(self._running.values())
        for session in sessions:
            self._refresh_detached_session(session)
        with self._lock:
            # ⤷ 出口 return：返回 any((not s.exited and predicate(s) for s in self._running.va…
            return any(not s.exited and predicate(s) for s in self._running.values())

    # ┌── 函数 def has_active_processes()
    # │  作用：Whether any process for ``task_id`` is still running.
    # │  参数：self, task_id: str
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_active_processes(self, task_id: str) -> bool:
        """Whether any process for ``task_id`` is still running."""
        # ⤷ 出口 return：返回 self._any_running(lambda s: s.task_id == task_id)
        return self._any_running(lambda s: s.task_id == task_id)

    # ┌── 函数 def running_owned_by()
    # │  作用：Running processes whose RAW spawning owner is ``owner_task_id``.
    # │  参数：self, owner_task_id: str
    # │  返回：List[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def running_owned_by(self, owner_task_id: str) -> List[ProcessSession]:
        """Running processes whose RAW spawning owner is ``owner_task_id``."""
        with self._lock:
            # ⤷ 出口 return：返回 [s for s in self._running.values() if s.owner_task_id == own…
            return [s for s in self._running.values() if s.owner_task_id == owner_task_id and not s.exited]

    # ┌── 函数 def unread_completions_owned_by()
    # │  作用：Exited ``notify_on_complete`` processes of ``owner_task_id`` whose result nobody read (no wait/log/poll).
    # │  参数：self, owner_task_id: str
    # │  返回：List[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def unread_completions_owned_by(self, owner_task_id: str) -> List[ProcessSession]:
        """Exited ``notify_on_complete`` processes of ``owner_task_id`` whose result nobody read (no wait/log/poll).
        A child's completion notice is suppressed in the parent, so an unread exit is otherwise lost silently."""
        with self._lock:
            # ⤷ 出口 return：返回 [s for s in self._finished.values() if s.owner_task_id == ow…
            return [s for s in self._finished.values()
                    if s.owner_task_id == owner_task_id and s.notify_on_complete
                    and s.id not in self._completion_consumed and s.id not in self._poll_observed]

    # ┌── 函数 def transfer_ownership()
    # │  作用：Move a RUNNING process from one owner to another under the registry lock. Ownership is the ``owner_task_id``
    # │  参数：self, session_id: str, from_owner: str, to_owner: str, to_task_id: str, to_session_key: str, note: str = ...
    # │  返回：Optional[ProcessSession]
    # └──────────────────────────────────────────────────────────────
    def transfer_ownership(self, session_id: str, *, from_owner: str, to_owner: str, to_task_id: str,
                           to_session_key: str, note: str = "") -> Optional[ProcessSession]:
        """Move a RUNNING process from one owner to another under the registry lock. Ownership is the ``owner_task_id``
        field: completion notices are stamped from it at exit time and teardown kills by it, so flipping it here is the
        whole transfer. Returns the session, or None when it is unknown, already exited, or not owned by ``from_owner``
        (the caller must not report a transfer that did not happen)."""
        session = self.get(session_id)
        with self._lock:
            if session is None or session.exited or session.owner_task_id != from_owner:
                # ⤷ 出口 return：返回 None
                return None
            session.owner_task_id = to_owner
            session.task_id = to_task_id
            session.session_key = to_session_key
            session.handoff_note = note
            # ⤷ 出口 return：返回 session
            return session

    # ┌── 函数 def has_active_for_session()
    # │  作用：Active processes for a gateway session key. Processes older than
    # │  参数：self, session_key: str, max_active_age: Optional[float] = None
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_active_for_session(self, session_key: str, max_active_age: Optional[float] = None) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten ``http.server``
        can't freeze session idle/daily reset forever; ``None`` keeps legacy behaviour
        (any running process blocks)."""
        now = time.time()
        # ⤷ 出口 return：返回 self._any_running(lambda s: s.session_key == session_key and…
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age))

    # ┌── 函数 def has_any_active()
    # │  作用：Whether ANY background process is running — scale-to-zero must not
    # │  参数：self
    # │  返回：bool
    # └──────────────────────────────────────────────────────────────
    def has_any_active(self) -> bool:
        """Whether ANY background process is running — scale-to-zero must not
        suspend a gateway with live background work or the process is lost."""
        # ⤷ 出口 return：返回 self._any_running(lambda s: True)
        return self._any_running(lambda s: True)

    # ┌── 函数 def snapshot_running_ids()
    # │  作用：Running IDs owned by ``task_id`` — a turn-boundary marker: on timeout
    # │  参数：self, task_id: str
    # │  返回：frozenset[str]
    # └──────────────────────────────────────────────────────────────
    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        """Running IDs owned by ``task_id`` — a turn-boundary marker: on timeout
        only processes absent from the starting snapshot belong to the abandoned
        turn; older ones intentionally span turns and must survive."""
        with self._lock:
            # ⤷ 出口 return：返回 frozenset((s.id for s in self._running.values() if s.task_id…
            return frozenset(s.id for s in self._running.values() if s.task_id == task_id and not s.exited)

    # ┌── 函数 def kill_started_since()
    # │  作用：Kill ``task_id`` processes created after ``baseline_ids``. Output is
    # │  参数：self, task_id: str, baseline_ids, source: str
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def kill_started_since(self, task_id: str, baseline_ids, *, source: str) -> int:
        """Kill ``task_id`` processes created after ``baseline_ids``. Output is
        consumed so an abandoned turn can't enqueue a follow-up reviving work the
        timeout deliberately stopped."""
        # ⤷ 出口 return：返回 self.kill_all(task_id, exclude_ids=frozenset(baseline_ids or…
        return self.kill_all(task_id, exclude_ids=frozenset(baseline_ids or ()), source=source, consume_output=True)

    # ┌── 函数 def kill_all()
    # │  作用：Kill all running processes, optionally filtered by task_id. Returns count killed.
    # │  参数：self, task_id: Optional[str] = None, exclude_ids: frozenset = ..., source: str = ..., consume_output: bool = ...
    # │  返回：int
    # └──────────────────────────────────────────────────────────────
    def kill_all(
        self, task_id: Optional[str] = None, *, exclude_ids: frozenset = frozenset(),
        source: str = "kill_all", consume_output: bool = False) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and s.id not in exclude_ids and not s.exited
            ]
        # ⤷ 出口 return：返回 sum((self.kill_process(s.id, source=source, consume_output=c…
        return sum(
            self.kill_process(s.id, source=source, consume_output=consume_output).get("status")
            in {"killed", "already_exited"}
            for s in targets)

    # ----- Cleanup / Pruning -----

    # ┌── 函数 def _prune_if_needed()
    # │  作用：Drop expired finished sessions, then the oldest survivor while over
    # │  参数：self
    # │  返回：（未标注类型）
    # └──────────────────────────────────────────────────────────────
    def _prune_if_needed(self):
        """Drop expired finished sessions, then the oldest survivor while over
        MAX_PROCESSES. Must hold _lock."""
        now = time.time()
        expired = [sid for sid, s in self._finished.items() if (now - s.started_at) > FINISHED_TTL_SECONDS]
        over_cap = len(self._running) + len(self._finished) - len(expired) >= MAX_PROCESSES
        if over_cap and (survivors := [sid for sid in self._finished if sid not in expired]):
            expired.append(min(survivors, key=lambda sid: self._finished[sid].started_at))
        for sid in expired:
            # Belt-and-suspenders handle release: sessions normally arrive in
            # _finished via _move_to_finished(), which already released their
            # Popen/PTY handles — but any session inserted into _finished
            # directly (defensive paths, historical checkpoints) would
            # otherwise carry its OS handles to the grave unreleased. The
            # release is idempotent, so double-closing is safe.
            self._release_finished_handles(self._finished[sid])
            del self._finished[sid]
        # Belt-and-suspenders against module-lifetime growth: forget consumed /
        # poll-observed marks for any session no longer tracked at all.
        tracked = self._running.keys() | self._finished.keys()
        self._completion_consumed &= tracked
        self._poll_observed &= tracked



process_registry = ProcessRegistry()


# --- the "process_manage" tool schema + handler -----------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process_manage",
    # The enum names the verbs; the description keeps only non-obvious semantics
    # (write-vs-submit is the one real trap: a lone \n on a Windows PTY is not Enter).
    # See #95681.
    "description": (
        "Poll, wait on, or kill background terminal processes (from "
        "terminal(background=true)). "
        "Completed results remain retrievable by session_id when resuming their owning conversation "
        "(up to 7 days, newest 64 results per profile; rolling output tail). "
        "poll: status + new output. log: full output, paged. wait: block "
        "until exit or timeout (partial output on timeout). write vs "
        "submit: submit appends Enter — use it to answer prompts; write "
        "sends raw bytes, no newline. close: EOF stdin. kill: terminate. "
        "handoff (subagents only): transfer a running process you started to your parent agent, which then "
        "receives its completion; `data` = one sentence on its purpose. Subagent-owned processes are otherwise "
        "killed when the subagent finishes and their notifications never reach the parent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close", "handoff"]
            },
            "session_id": {
                "type": "string",
                "description": "From terminal background output; any unique prefix works ('4dae' for proc_4dae56ca81f6). Required except for 'list'."
            },
            "data": {
                "type": "string",
                "description": "Stdin text for write/submit; purpose sentence for handoff."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds for 'wait'.",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "Log line offset (default: last 200)."
            },
            "limit": {
                "type": "integer",
                "description": "Max log lines.",
                "minimum": 1
            }
        },
        "required": ["action"]
    }
}


# ┌── 函数 def _redact_process_result()
# │  作用：Redact secrets from background-process output before it reaches the model,
# │  参数：result: dict
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _redact_process_result(result: dict) -> dict:
    """Redact secrets from background-process output before it reaches the model,
    session.db and CLI, mirroring the foreground ``terminal`` redaction so the two
    surfaces can't diverge. Respects ``security.redact_secrets``; ``redact_terminal_output``
    picks ``code_file`` from the recorded command. The command itself is redacted too.

    The command string itself is also redacted in case it carried an inline credential. See #43025.
    """
    if not isinstance(result, dict):
        # ⤷ 出口 return：返回 result
        return result
    from agent.redact import redact_sensitive_text, redact_terminal_output

    command = result.get("command") or ""
    for key in ("output", "output_preview"):
        if isinstance(value := result.get(key), str) and value:
            result[key] = redact_terminal_output(value, command)
    if isinstance(command, str) and command:
        result["command"] = redact_sensitive_text(command, code_file=True)
    # ⤷ 出口 return：返回 result
    return result


# ┌── 函数 def _list_processes()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：task_id
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _list_processes(task_id) -> dict:
    # Also surface session-scoped background processes (e.g. a forgotten preview
    # server): they share the gateway session_key and can block session reset.
    session_key = ""
    with suppress(Exception):
        # See #29177.
        from tools.approval_context import get_current_session_key
        session_key = get_current_session_key(default="") or ""
    # ⤷ 出口 return：返回 {'processes': [_redact_process_result(p) for p in process_re…
    return {"processes": [
        _redact_process_result(p)
        for p in process_registry.list_sessions(
            task_id=task_id, session_key=session_key or None, include_retained=True)]}


# action -> (handler(session_id, args) -> dict, redact output?). Output-bearing
# actions are redacted; stdin actions return only status.
_SESSION_ACTIONS = {
    "poll": (lambda sid, a: process_registry.poll(sid), True),
    "log": (lambda sid, a: process_registry.read_log(sid, offset=a.get("offset"), limit=a.get("limit", 200)), True),
    "wait": (lambda sid, a: process_registry.wait(sid, timeout=a.get("timeout")), True),
    "kill": (lambda sid, a: process_registry.kill_process(sid), True),
    "write": (lambda sid, a: process_registry.write_stdin(sid, str(a.get("data", ""))), False),
    "submit": (lambda sid, a: process_registry.submit_stdin(sid, str(a.get("data", ""))), False),
    "close": (lambda sid, a: process_registry.close_stdin(sid), False),
}


# ┌── 函数 def _handoff_process()
# │  作用：Subagent-only: transfer a running background process to the parent agent so its completion is delivered THERE
# │  参数：session_id: str, args: dict, task_id: Optional[str]
# │  返回：dict
# └──────────────────────────────────────────────────────────────
def _handoff_process(session_id: str, args: dict, task_id: Optional[str]) -> dict:
    """Subagent-only: transfer a running background process to the parent agent so its completion is delivered THERE
    (child-owned process notices are suppressed and child teardown kills what it owns). Validated against the live spawn
    tree: the caller must be a registered child and must own the process; anything else is an error, never a silent
    no-op, so a PID mentioned in prose can't masquerade as a transfer."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    from tools.terminal_tool import _resolve_container_task_id
    with _active_subagents_lock:
        record = _active_subagents.get(str(task_id or ""))
    child = record.get("agent") if record else None
    parent_ref = getattr(child, "_delegate_parent_ref", None)
    parent = parent_ref() if callable(parent_ref) else None
    if parent is None:
        # ⤷ 出口 return：返回 {'error': 'handoff is only available to a running subagent w…
        return {"error": "handoff is only available to a running subagent with a live parent; you are not one."}
    parent_owner = str(getattr(parent, "_current_task_id", "") or getattr(parent, "session_id", "") or "")
    if not parent_owner:
        # ⤷ 出口 return：返回 {'error': "parent has no process owner id yet; retry after t…
        return {"error": "parent has no process owner id yet; retry after the parent's turn has started."}
    handed = getattr(child, "_handed_off_processes", None)
    if handed is None:
        handed = child._handed_off_processes = []
    if len(handed) >= _MAX_HANDOFFS_PER_CHILD:
        # ⤷ 出口 return：返回 {'error': f'handoff cap reached ({_MAX_HANDOFFS_PER_CHILD} p…
        return {"error": f"handoff cap reached ({_MAX_HANDOFFS_PER_CHILD} per subagent); wait on or kill the rest yourself."}
    note = str(args.get("data") or "").strip()
    if not note:
        # ⤷ 出口 return：返回 {'error': 'handoff requires `data`: one sentence saying what…
        return {"error": "handoff requires `data`: one sentence saying what the process is for and what the parent should do with its result."}
    session = process_registry.transfer_ownership(
        session_id, from_owner=str(task_id or ""), to_owner=parent_owner,
        to_task_id=_resolve_container_task_id(parent_owner),
        to_session_key=str(getattr(parent, "session_id", "") or ""), note=note)
    if session is None:
        # ⤷ 出口 return：返回 {'error': f'cannot hand off {session_id}: not a running proc…
        return {"error": f"cannot hand off {session_id}: not a running process you own (already exited? read its result "
                         "with poll/log and report it instead)."}
    handed.append({"session_id": session.id, "command": session.command, "note": note})
    # ⤷ 出口 return：返回 {'status': 'handed_off', 'session_id': session.id, 'command'…
    return {"status": "handed_off", "session_id": session.id, "command": session.command,
            "note": "Your parent now owns this process and will receive its completion; you will not. Mention the handoff "
                    "in your final answer."}


_MAX_HANDOFFS_PER_CHILD = 3


# ┌── 函数 def _handle_process()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：args, **kw
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _handle_process(args, **kw):
    action = args.get("action", "")
    # Coerce to string — some models send session_id as an integer
    session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""
    if action == "list":
        # ⤷ 出口 return：返回 json.dumps(_list_processes(kw.get('task_id')), ensure_ascii=…
        return json.dumps(_list_processes(kw.get("task_id")), ensure_ascii=False)
    if action == "handoff":
        if not session_id:
            # ⤷ 出口 return：返回 tool_error('session_id is required for handoff')
            return tool_error("session_id is required for handoff")
        # ⤷ 出口 return：返回 json.dumps(_handoff_process(session_id, args, kw.get('task_i…
        return json.dumps(_handoff_process(session_id, args, kw.get("task_id")), ensure_ascii=False)
    if action in _SESSION_ACTIONS:
        if not session_id:
            # ⤷ 出口 return：返回 tool_error(f'session_id is required for {action}')
            return tool_error(f"session_id is required for {action}")
        handler, redact = _SESSION_ACTIONS[action]
        result = handler(session_id, args)
        # ⤷ 出口 return：返回 json.dumps(_redact_process_result(result) if redact else res…
        return json.dumps(_redact_process_result(result) if redact else result, ensure_ascii=False)
    # ⤷ 出口 return：返回 tool_error(f'Unknown process action: {action}. Use: list, po…
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close, handoff")


registry.register(
    name="process_manage",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

MAX_ACTIVE_PROCESS_AGE = 86400  # 24h default — see session_reset.bg_process_max_age_hours (#29177)
# ---- END PLUGIN-COMPAT ----
