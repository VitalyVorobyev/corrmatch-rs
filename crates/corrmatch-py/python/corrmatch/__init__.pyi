"""Type stubs for the corrmatch package.

The recommended entry point for type-safe usage. ``CompileConfig`` and
``MatchConfig`` here are the Python-native frozen dataclasses from
:mod:`corrmatch.config`. ``Template``, ``CompiledTemplate``, ``Matcher``,
and ``Match`` are thin Python shim classes from :mod:`corrmatch._wrappers`.
"""

from numpy.typing import NDArray
import numpy as np

from ._corrmatch import (
    rotate_u8_bilinear_masked as rotate_u8_bilinear_masked,
    __version__ as __version__,
)
from ._wrappers import (
    Match as Match,
    Template as Template,
    CompiledTemplate as CompiledTemplate,
    Matcher as Matcher,
)
from .config import CompileConfig as CompileConfig, MatchConfig as MatchConfig

def match_template(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    metric: str = ...,
    rotation: str = ...,
    parallel: bool = ...,
) -> Match: ...

__all__: list[str]
