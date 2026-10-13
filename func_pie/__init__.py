"""func_pie compatibility alias for funcpie."""

import sys
from funcpie import *
from funcpie import __all__, __author__, __license__, __version__, core, math, physics

# Forward module reference
sys.modules["func_pie"] = sys.modules["funcpie"]
