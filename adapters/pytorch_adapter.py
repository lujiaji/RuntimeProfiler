from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional

import torch.nn as nn

from ..core import RuntimeProfiler


def _module_name(module: nn.Module) -> str:
    return module.__class__.__name__


@contextmanager
def trace_module_forward(
    profiler: RuntimeProfiler,
    module: nn.Module,
    include_types: Optional[Iterable[str]] = None,
    max_modules: int = 0,
) -> Generator[None, None, None]:
    include_types = set(include_types or [])
    handles: List = []
    open_traces: List = []

    def _pre_hook(mod, _inputs):
        if include_types and _module_name(mod) not in include_types:
            open_traces.append(None)
            return
        ctx = profiler.trace(f"forward:{_module_name(mod)}")
        ctx.__enter__()
        open_traces.append(ctx)

    def _post_hook(_mod, _inputs, _outputs):
        if not open_traces:
            return
        ctx = open_traces.pop()
        if ctx is not None:
            ctx.__exit__(None, None, None)

    registered = 0
    for sub_module in module.modules():
        if sub_module is module:
            continue
        if max_modules > 0 and registered >= max_modules:
            break
        handles.append(sub_module.register_forward_pre_hook(_pre_hook))
        handles.append(sub_module.register_forward_hook(_post_hook))
        registered += 1

    try:
        yield
    finally:
        for h in handles:
            h.remove()
