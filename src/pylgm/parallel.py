"""Thread-pool fan-out for independent conditional fits, and BLAS thread control.

Splu and BLAS both release the GIL, so a batch of otherwise-independent fits
(a finite-difference gradient's forward/backward points, an INLA grid) can run
concurrently on threads without the pickling cost or restrictions of
processes -- the compiled families involved are not picklable (local
closures). Threads only help once BLAS itself is pinned to one thread per
worker, or the workers oversubscribe the machine's cores fighting each other.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from threadpoolctl import threadpool_limits


def validate_workers(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_blas_threads(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("blas_threads must be None or a positive integer")
    return value


def blas_limit(num_workers: int, blas_threads: int | None):
    """A context manager pinning BLAS threads for the duration of a fit/search.

    ``num_workers == 1 and blas_threads is None`` leaves BLAS alone -- exactly
    today's behaviour, no threadpoolctl call at all. Otherwise BLAS is capped
    at ``blas_threads`` (or 1, when fanning fits out across workers: each
    thread gets its own BLAS calls, and letting each of them also spawn BLAS's
    own threads oversubscribes the machine).
    """
    if num_workers == 1 and blas_threads is None:
        return nullcontext()
    return threadpool_limits(limits=blas_threads if blas_threads is not None else 1, user_api="blas")


def map_ordered(function: Callable, items: Iterable, num_workers: int) -> list:
    """``[function(i) for i in items]``, run concurrently when it pays off.

    Order is always preserved. Serial when ``num_workers == 1`` or there is at
    most one item (a pool would add overhead for no parallelism); otherwise a
    fresh ``ThreadPoolExecutor`` per call -- simple, and the fits dominate the
    cost anyway.
    """
    items = list(items)
    if num_workers == 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(num_workers, len(items))) as pool:
        return list(pool.map(function, items))
