"""Ownership leases for replaceable runtime registrations.

The coordinator models registration *generations*, not just value identity: the same provider
singleton may be registered again after an older ownership generation was unloaded.
"""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：registration_lifecycle.py（约 95 行，很短，建议完整读）
# 地位：可被替换的运行时注册项的「所有权租约」管理。
#
# 解决什么问题：
#   某个全局对象（比如一个 provider 单例）可以被反复注册/卸载。如果旧的所有者还没清理干净，
#   新的就注册上来，就会出现「两个版本同时在跑」的混乱。
#
# 核心抽象：
#   ReplacementLease（租约）：谁注册，谁拿到一张租约，负责 dispose（释放）。
#   ReplacementCoordinator（协调器）：用「代（generation）」来区分，而不是只看值是否相同。
#     transaction() 提供一个事务上下文，acquire() 拿租约，dispose() 释放。
#
# 给新手的启发：这是「资源生命周期管理」的教科书式小例子，比直接写全局变量靠谱得多。
# ==========================================================================
#

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


# ┌── 函数 def same_registration()
# │  作用：Compare opaque registry snapshots using identity only (element-wise for tuples).
# │  参数：left: Any, right: Any
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def same_registration(left: Any, right: Any) -> bool:
    """Compare opaque registry snapshots using identity only (element-wise for tuples)."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        # ⤷ 出口 return：返回 len(left) == len(right) and all((same_registration(a, b) for…
        return len(left) == len(right) and all(same_registration(a, b) for a, b in zip(left, right))
    # ⤷ 出口 return：返回 left is right
    return left is right


# ┌── 类 class ReplacementLease
# │  说明：One ownership generation in a replaceable registry slot.
# │  成员：共 1 个方法；对外可用 1 个
# └──────────────────────────────────────────────────────────────
@dataclass
class ReplacementLease:
    """One ownership generation in a replaceable registry slot."""

    coordinator: "ReplacementCoordinator"
    slot: Hashable
    current: Any
    previous: Any
    restore: Callable[[Any], bool]
    finalize: Callable[[], None] | None = None
    predecessor: "ReplacementLease | None" = None
    active: bool = field(default=True, init=False)

    # ┌── 函数 def dispose()
    # │  作用：（源码未写 docstring，可从下方参数与返回值推断）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def dispose(self) -> None:
        self.coordinator.dispose(self)


# ┌── 类 class ReplacementCoordinator
# │  说明：Link and remove registration generations in arbitrary unload order.
# │  成员：共 4 个方法；对外可用 3 个；含魔术方法 __init__
# └──────────────────────────────────────────────────────────────
class ReplacementCoordinator:
    """Link and remove registration generations in arbitrary unload order."""

    # ┌── 函数 def __init__()
    # │  作用：魔术方法 __init__（Python 在特定场景自动调用，见下方说明）
    # │  参数：self
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._active: dict[Hashable, list[ReplacementLease]] = {}
        self._lock = threading.RLock()

    # ┌── 函数 def transaction()
    # │  作用：Serialize a registry snapshot/write/acquire with lease disposal.
    # │  参数：self
    # │  返回：（未标注类型）
    # │  【语法】@contextmanager 把「有 yield 的函数」变成上下文管理器，调用方可以用 `with xxx() as v:` 自动做进入/清理。yield 之前是准备，yield 之后是收尾。
    # └──────────────────────────────────────────────────────────────
    @contextmanager
    def transaction(self):
        """Serialize a registry snapshot/write/acquire with lease disposal."""
        with self._lock:
            # ⤷ 关键分支 yield：产出值并暂停，调用方用 for / next 逐个取（生成器）。
            yield

    # ┌── 函数 def acquire()
    # │  作用：Attach a new live generation to the matching active predecessor.
    # │  参数：self, slot: Hashable, current: Any, previous: Any, restore: Callable[[Any], bool], finalize: Callable[[], None] | None = ...
    # │  返回：ReplacementLease
    # └──────────────────────────────────────────────────────────────
    def acquire(self, slot: Hashable, *, current: Any, previous: Any, restore: Callable[[Any], bool],
                finalize: Callable[[], None] | None = None) -> ReplacementLease:
        """Attach a new live generation to the matching active predecessor."""
        with self._lock:
            leases = self._active.setdefault(slot, [])
            predecessor = next((c for c in reversed(leases) if c.active and same_registration(c.current, previous)), None)
            lease = ReplacementLease(self, slot, current, previous, restore, finalize, predecessor)
            leases.append(lease)
            # ⤷ 出口 return：返回 lease
            return lease

    # ┌── 函数 def dispose()
    # │  作用：Remove *lease*, restoring the nearest still-live predecessor.
    # │  参数：self, lease: ReplacementLease
    # │  返回：None
    # └──────────────────────────────────────────────────────────────
    def dispose(self, lease: ReplacementLease) -> None:
        """Remove *lease*, restoring the nearest still-live predecessor.

        ``restore`` -> ``finalize`` -> slot pruning each run even when an earlier step raises.
        """
        with self._lock:
            if not lease.active:
                return
            leases = self._active.get(lease.slot, [])
            latest = next((c for c in reversed(leases) if c.active), None)
            lease.active = False
            # An older generation can share the exact object identity of a newer one; registry-level
            # CAS cannot tell them apart, so only the latest live generation may mutate the slot.
            # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
            try:
                # ⤷ 关键分支 try/except：捕获 ，出错走兜底而不是崩溃。
                try:
                    if latest is lease:
                        # Restore the nearest still-live predecessor, else the last dead generation's previous.
                        replacement, predecessor = lease.previous, lease.predecessor
                        while predecessor is not None and not predecessor.active:
                            replacement, predecessor = predecessor.previous, predecessor.predecessor
                        lease.restore(predecessor.current if predecessor is not None else replacement)
                finally:
                    if lease.finalize is not None:
                        lease.finalize()
            finally:
                live = [item for item in leases if item.active]
                if live:
                    self._active[lease.slot] = live
                elif leases:
                    self._active.pop(lease.slot, None)


replacement_coordinator = ReplacementCoordinator()
