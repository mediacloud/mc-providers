from typing import Callable, Iterable, Optional

from .provider import ContentProvider

from typing import ParamSpec, TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


class CachingManager():
    """
    Use this to set caching functionality on the providers library for whatever system your using
    something like:
    `CachingManager.cache_function = your_cache_function`
    your_cache_function should have a signature like (function_to_cache, cache_prefix, args, kwargs) -> tuple[Any, bool]
    (second item in tuple indicatin if it was cached or not already)
    """
    cache_function = None

    # typing from https://stackoverflow.com/questions/47060133/python-3-type-hinting-for-decorator
    @classmethod
    def cache(cls, custom_prefix_for_key: Optional[str] = None, kwargs_to_ignore: Iterable[str] = []) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
        """
        @param custom_prefix_for_key: if specified, will be used in place of function name for cache_key generation
        """
        def decorator(fn: Callable[Param, RetType]) -> Callable[Param, RetType]:
            # WISH: detect if 'fn' is being declared as a method to a ContentProvider!!
            def wrapper(*args:Param.args, **kwargs:Param.kwargs) -> RetType:
                # blindly assume that everything decorated is a ContentProvider method!
                inst = args[0]
                assert isinstance(inst, ContentProvider)
                # check caching enabled and wasn't disabled when Provider instantiated
                if cls.cache_function is not None and inst._caching > 0:
                    cache_prefix = custom_prefix_for_key or fn.__name__
                    # remove any kwargs already processed
                    # and included in positional args (ie; query string)
                    if kwargs_to_ignore and kwargs:
                        kwargs = kwargs.copy()
                        for kw in kwargs_to_ignore:
                            if kw in kwargs:
                                kwargs.pop(kw)
                    results, was_cached = cls.cache_function(fn, cache_prefix, *args, **kwargs)
                    return results
                else:
                    return fn(*args, **kwargs)
            return wrapper
        return decorator
