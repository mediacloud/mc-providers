import warnings

from typing import Callable, ParamSpec, TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

def deprecated(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
    # This is a decorator which can be used to mark functions as deprecated. It will result in a
    # warning being emitted when the function is used.
    def new_func(*args:Param.args, **kwargs:Param.kwargs) -> RetType:
        warnings.warn(f"Call to deprecated function {func.__name__}.", category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func
