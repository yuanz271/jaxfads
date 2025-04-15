import functools

from jax import random as jrnd


class SingletonMeta(type):
    _instances = {}  # class variable

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# class Registry:
#     def __init__(self):
#         self._registry = {}

#     def register(self, name=None):

#         def decorate(klass):
#             nonlocal name
#             if name is None:
#                 name = klass.__name__
#             if name in self._registry:
#                 warnings.warn(f"{name} exists")
#             else:    
#                 self._registry[name] = klass
#             return klass
        
#         return decorate

#     def get_class(self, name):
#         klass = self._registry[name]
#         return klass


def newkey(func):

    @functools.wraps(func)
    def wrapper(*args, key, **kwargs):
        key, subkey = jrnd.split(key)
        return subkey, func(*args, key=key, **kwargs)
    
    return wrapper
