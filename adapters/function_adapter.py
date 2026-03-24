from typing import Any, Callable

from ..core import ProfileResult, RuntimeProfiler


def profile_function(
    profiler: RuntimeProfiler, target_fn: Callable[..., Any], *args, **kwargs
) -> ProfileResult:
    return profiler.run(target_fn, *args, **kwargs)
