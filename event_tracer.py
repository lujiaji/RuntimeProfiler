import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Dict, Generator, List, Optional


@dataclass
class TraceEvent:
    name: str
    start_ns: int
    end_ns: int
    depth: int
    tid: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventTracer:
    def __init__(self) -> None:
        self._events: List[TraceEvent] = []
        self._events_lock = threading.Lock()
        self._local = threading.local()

    def _get_stack(self) -> List[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @contextmanager
    def trace(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[None, None, None]:
        stack = self._get_stack()
        depth = len(stack)
        stack.append(name)
        start_ns = time.perf_counter_ns()
        tid = threading.get_ident()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            stack.pop()
            event = TraceEvent(
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                depth=depth,
                tid=tid,
                metadata=metadata,
            )
            with self._events_lock:
                self._events.append(event)

    def trace_fn(self, name: Optional[str] = None):
        def _decorator(func):
            event_name = name or func.__qualname__

            @wraps(func)
            def _wrapper(*args, **kwargs):
                with self.trace(event_name):
                    return func(*args, **kwargs)

            return _wrapper

        return _decorator

    def get_events(self) -> List[TraceEvent]:
        with self._events_lock:
            return list(self._events)

    def clear(self) -> None:
        with self._events_lock:
            self._events.clear()
