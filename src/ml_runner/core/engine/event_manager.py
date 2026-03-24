from collections import defaultdict
from enum import Enum
from typing import Union, Callable
import inspect


class EventManager:
    """
    Decoupled event backbone. Two ways to register listeners:

      1. @listen("event") + event_manager.register(instance)
         Automatically subscribes all decorated methods on the object.

      2. event_manager.subscribe(callable, "event")
         Explicitly subscribe any callable — useful for lambdas / plain functions.
    """
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def clear(self):
        """Remove all subscribed listeners from this EventManager instance."""
        self._listeners.clear()

    def subscribe(self, listener: Callable, event: Union[Enum, str]):
        """
        Explicitly subscribe a plain callable to an event.
        Use this for functions/lambdas that cannot carry a @listen decorator.
        """
        event_name = event.value if isinstance(event, Enum) else event

        if listener not in self._listeners[event_name]:
            self._listeners[event_name].append(listener)

    def emit(self, event: Union[Enum, str], **kwargs):
        """
        Emit an event, dispatching kwargs to all registered listeners.
        Only passes the kwargs each listener actually accepts.
        """
        event_name = event.value if isinstance(event, Enum) else event

        for listener in self._listeners.get(event_name, []):
            try:
                sig = inspect.signature(listener)
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    listener(**kwargs)
                else:
                    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    listener(**filtered)
            except Exception:
                listener(**kwargs)
