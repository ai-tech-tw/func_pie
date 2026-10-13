"""Basic arithmetic operations and calculation utilities."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

Number = Union[int, float]


class Calculation:
    """Arithmetic calculation engine supporting basic and extended operations.

    Backward-compatible with the original FuncPie Calculation class,
    with added support for optional chained accumulator calculations.
    """

    def __init__(self, initial: Optional[Number] = None):
        self.exec_code: List[str] = ["+", "-", "*", "/", "**", "//", "%"]
        self._current_value: Optional[float] = float(initial) if initial is not None else None
        self.exec: Dict[str, Any] = {}

    @property
    def value(self) -> Optional[float]:
        """Current accumulated value if using fluent chaining."""
        return self._current_value

    @overload
    def add(self, num: Number, add: None = None) -> Calculation: ...
    @overload
    def add(self, num: Number, add: Number) -> Number: ...
    def add(self, num: Number, add: Optional[Number] = None) -> Union[Number, Calculation]:
        """Add two numbers, or add to the accumulated value if chained."""
        if add is None and self._current_value is not None:
            self._current_value += float(num)
            return self
        if add is None:
            raise TypeError("Calculation.add() missing required argument: 'add'")
        res = num + add
        self._current_value = float(res)
        return res

    @overload
    def sub(self, num: Number, sub: None = None) -> Calculation: ...
    @overload
    def sub(self, num: Number, sub: Number) -> Number: ...
    def sub(self, num: Number, sub: Optional[Number] = None) -> Union[Number, Calculation]:
        """Subtract sub from num, or subtract from accumulated value."""
        if sub is None and self._current_value is not None:
            self._current_value -= float(num)
            return self
        if sub is None:
            raise TypeError("Calculation.sub() missing required argument: 'sub'")
        res = num - sub
        self._current_value = float(res)
        return res

    @overload
    def mul(self, num: Number, mul: None = None) -> Calculation: ...
    @overload
    def mul(self, num: Number, mul: Number) -> Number: ...
    def mul(self, num: Number, mul: Optional[Number] = None) -> Union[Number, Calculation]:
        """Multiply two numbers, or multiply accumulated value."""
        if mul is None and self._current_value is not None:
            self._current_value *= float(num)
            return self
        if mul is None:
            raise TypeError("Calculation.mul() missing required argument: 'mul'")
        res = num * mul
        self._current_value = float(res)
        return res

    @overload
    def div(self, num: Number, div: None = None) -> Calculation: ...
    @overload
    def div(self, num: Number, div: Number) -> float: ...
    def div(self, num: Number, div: Optional[Number] = None) -> Union[float, Calculation]:
        """Divide num by div, or divide accumulated value.

        Raises:
            ZeroDivisionError: If the divisor is zero.
        """
        divisor = div if div is not None else num
        if divisor == 0:
            raise ZeroDivisionError("division by zero is undefined")

        if div is None and self._current_value is not None:
            self._current_value /= float(divisor)
            return self
        if div is None:
            raise TypeError("Calculation.div() missing required argument: 'div'")
        res = num / div
        self._current_value = float(res)
        return res

    @overload
    def pow(self, num: Number, exp: None = None) -> Calculation: ...
    @overload
    def pow(self, num: Number, exp: Number) -> Number: ...
    def pow(self, num: Number, exp: Optional[Number] = None) -> Union[Number, Calculation]:
        """Raise num to exp, or raise accumulated value."""
        if exp is None and self._current_value is not None:
            self._current_value = float(self._current_value ** num)
            return self
        if exp is None:
            raise TypeError("Calculation.pow() missing required argument: 'exp'")
        res = num ** exp
        self._current_value = float(res)
        return res

    @overload
    def mod(self, num: Number, mod: None = None) -> Calculation: ...
    @overload
    def mod(self, num: Number, mod: Number) -> Number: ...
    def mod(self, num: Number, mod: Optional[Number] = None) -> Union[Number, Calculation]:
        """Modulo operation."""
        modulo = mod if mod is not None else num
        if modulo == 0:
            raise ZeroDivisionError("modulo by zero is undefined")

        if mod is None and self._current_value is not None:
            self._current_value %= float(modulo)
            return self
        if mod is None:
            raise TypeError("Calculation.mod() missing required argument: 'mod'")
        res = num % mod
        self._current_value = float(res)
        return res

    @overload
    def floordiv(self, num: Number, div: None = None) -> Calculation: ...
    @overload
    def floordiv(self, num: Number, div: Number) -> Number: ...
    def floordiv(self, num: Number, div: Optional[Number] = None) -> Union[Number, Calculation]:
        """Integer floor division."""
        divisor = div if div is not None else num
        if divisor == 0:
            raise ZeroDivisionError("integer division by zero is undefined")

        if div is None and self._current_value is not None:
            self._current_value //= float(divisor)
            return self
        if div is None:
            raise TypeError("Calculation.floordiv() missing required argument: 'div'")
        res = num // div
        self._current_value = float(res)
        return res

    def usage(self) -> Tuple[Dict[str, Any], List[str]]:
        """Return dictionary mapping operator strings to methods and the list of codes."""
        self.exec = {
            self.exec_code[0]: self.add,
            self.exec_code[1]: self.sub,
            self.exec_code[2]: self.mul,
            self.exec_code[3]: self.div,
            self.exec_code[4]: self.pow,
            self.exec_code[5]: self.floordiv,
            self.exec_code[6]: self.mod,
        }
        return self.exec, self.exec_code


# Standalone functional arithmetic utilities
def add(a: Number, b: Number) -> Number:
    """Return a + b."""
    return a + b


def sub(a: Number, b: Number) -> Number:
    """Return a - b."""
    return a - b


def mul(a: Number, b: Number) -> Number:
    """Return a * b."""
    return a * b


def div(a: Number, b: Number) -> float:
    """Return a / b with zero-division validation."""
    if b == 0:
        raise ZeroDivisionError("division by zero is undefined")
    return a / b


def power(base: Number, exp: Number) -> Number:
    """Return base raised to power exp."""
    return base ** exp


def clamp(val: Number, min_val: Number, max_val: Number) -> Number:
    """Clamp a value within [min_val, max_val]."""
    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")
    return max(min_val, min(max_val, val))


def sign(x: Number) -> int:
    """Return sign of x: -1 if x < 0, 0 if x == 0, 1 if x > 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def is_close(a: Number, b: Number, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Check if two numbers are approximately equal."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def lerp(a: Number, b: Number, t: float) -> float:
    """Linear interpolation between a and b with factor t in [0, 1]."""
    return a + (b - a) * t


def remap(
    val: Number,
    in_min: Number,
    in_max: Number,
    out_min: Number,
    out_max: Number,
) -> float:
    """Linearly map val from [in_min, in_max] range to [out_min, out_max] range."""
    if in_min == in_max:
        raise ValueError("in_min cannot equal in_max")
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
