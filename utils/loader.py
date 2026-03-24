import importlib
from typing import Callable


def load_callable_from_module(module_name: str, fn_name: str) -> Callable:
    module = importlib.import_module(module_name)
    target = getattr(module, fn_name, None)
    if target is None:
        raise AttributeError(f"Function `{fn_name}` not found in module `{module_name}`")
    if not callable(target):
        raise TypeError(f"`{module_name}.{fn_name}` is not callable")
    return target
