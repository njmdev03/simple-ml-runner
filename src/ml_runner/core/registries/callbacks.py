from typing import Callable, Dict, List
import inspect
from enum import Enum

# Registry mapping unbound function -> list of event identifiers (str or Enum)
_CALLBACK_REGISTRY: Dict[Callable, List[object]] = {}


def Callback(event_name: object):
    """Decorator to register a function or method (unbound) as a callback for an event.

    The event_name may be a string or an Enum member. Registered functions are stored
    by their underlying function object; use `attach(instance, event_manager)` to bind
    and subscribe decorated methods on an instance to a specific EventManager.
    """
    def decorator(func: Callable):
        if func not in _CALLBACK_REGISTRY:
            _CALLBACK_REGISTRY[func] = []
        _CALLBACK_REGISTRY[func].append(event_name)
        return func

    return decorator


def attach(instance: object, event_manager) -> None:
    """Bind decorated methods on `instance` and subscribe them to `event_manager`.

    For each method on the instance whose underlying function is registered in the
    callback registry, subscribe the bound method to the EventManager.
    Uses `event_manager.subscribe()` when available and falls back to the
    internal listener list to remain compatible with current EventManager.
    """
    for _, method in inspect.getmembers(instance, predicate=inspect.ismethod):
        event_names = _CALLBACK_REGISTRY.get(method.__func__)
        if event_names:
            for event_name in event_names:
                # Normalize Enum members to their string value so EventManager.emit() finds them
                key = event_name.value if isinstance(event_name, Enum) else event_name
                try:
                    event_manager.subscribe(method, key)
                except Exception:
                    # Fallback to internal listener append if subscribe is not present
                    listeners = getattr(event_manager, "_listeners", None)
                    if listeners is not None:
                        if method not in listeners.get(key, []):
                            listeners.setdefault(key, []).append(method)


def get_registered_events() -> List[object]:
    """Return a list of all event names that have callbacks registered (may contain duplicates)."""
    events = []
    for evs in _CALLBACK_REGISTRY.values():
        events.extend(evs)
    return events


def clear_registry() -> None:
    """Clear the global callback registry. Useful for tests/batch resets."""
    _CALLBACK_REGISTRY.clear()
