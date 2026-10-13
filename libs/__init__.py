"""Backward compatibility package for legacy libs imports."""

import sys
import funcpie

sys.modules["libs"] = sys.modules.get("libs", sys.modules[__name__])
