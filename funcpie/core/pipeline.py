"""Functional pipeline and composition utilities for FuncPie."""

from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pipe(value: object, *funcs: Callable[..., object]) -> object:
    """Pass a value through a sequence of functions left-to-right.

    Example:
        >>> pipe(5, lambda x: x + 1, lambda x: x * 2)
        12
    """
    res: object = value
    for fn in funcs:
        res = fn(res)
    return res


def compose(*funcs: Callable[..., object]) -> Callable[..., object]:
    """Compose multiple functions right-to-left.

    compose(f, g, h)(x) == f(g(h(x)))

    Example:
        >>> double = lambda x: x * 2
        >>> inc = lambda x: x + 1
        >>> f = compose(double, inc)
        >>> f(5)  # (5 + 1) * 2 = 12
        12
    """
    if not funcs:
        return lambda x: x

    def composed(*args: object, **kwargs: object) -> object:
        res: object = funcs[-1](*args, **kwargs)
        for fn in reversed(funcs[:-1]):
            res = fn(res)
        return res

    return composed


class Pipeable:
    """Wrapper that enables Unix-style pipeline syntax using the `|` operator.

    Example:
        >>> add_one = Pipeable(lambda x: x + 1)
        >>> double = Pipeable(lambda x: x * 2)
        >>> 5 | add_one | double
        12
    """

    def __init__(self, fn: Callable[..., object]):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.fn(*args, **kwargs)

    def __ror__(self, other: object) -> object:
        return self.fn(other)

    def __or__(self, other: Callable[..., object]) -> Pipeable:
        if isinstance(other, Pipeable):
            return Pipeable(lambda *args, **kwargs: other(self(*args, **kwargs)))
        return Pipeable(lambda *args, **kwargs: other(self(*args, **kwargs)))


def pipeable(fn: Callable[..., object]) -> Pipeable:
    """Decorator to make a function Pipeable with `|`.

    Example:
        >>> @pipeable
        ... def square(x):
        ...     return x ** 2
        >>> 4 | square
        16
    """
    return Pipeable(fn)


class CurriedCallable:
    """Callable wrapper returned by curry allowing recursive partial application."""

    def __call__(self, *args: object, **kwargs: object) -> CurriedCallable:
        ...


def curry(fn: Callable[..., object]) -> CurriedCallable:
    """Curry a function so it can be called with arguments one at a time.

    Example:
        >>> @curry
        ... def add(a, b):
        ...     return a + b
        >>> add5 = add(5)
        >>> add5(10)
        15
    """
    sig = inspect.signature(fn)
    required_args = [
        p
        for p in sig.parameters.values()
        if p.default == inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    min_args = len(required_args)

    @functools.wraps(fn)
    def curried(*args: object, **kwargs: object) -> object:
        if len(args) >= min_args:
            return fn(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(
            *(args + more_args), **{**kwargs, **more_kwargs}
        )

    return curried  # type: ignore[return-value]


def tap(action: Callable[[T], object]) -> Callable[[T], T]:
    """Execute a side-effect function on value and return the value unmodified.

    Useful in pipelines for debugging or logging.
    """

    def _tap(val: T) -> T:
        action(val)
        return val

    return _tap
