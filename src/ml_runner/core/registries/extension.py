from .base import BaseRegistry


class ExtensionRegistry(BaseRegistry):
    _registry = {}


def Extension(*names: str):
    return ExtensionRegistry.register(*names)


# def Extension(name: str):
#     def decorator(ext_cls):
#         key = name.lower()
#         ExtensionRegistry._registry[key] = ext_cls
#         ext_cls._extension_name = key

#         # Find config classes inside extension
#         configs = {}

#         for attr_name in dir(ext_cls):
#             attr = getattr(ext_cls, attr_name)

#             # if getattr(attr, "_extension_config", False):
#             namespace = getattr(attr, "_config_namespace", None)
#             cfg_key = namespace or key
#             configs[cfg_key] = attr

#         ext_cls._config_classes = configs

#         return ext_cls
#     return decorator


# def Config(namespace: str | None = None):
#     def decorator(cls):
#         # cls._extension_config = True
#         cls._config_namespace = namespace
#         return cls

#     return decorator
