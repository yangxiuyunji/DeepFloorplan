"""Fengshui utilities for floorplan analysis.

This package currently provides:

* ``analyze_missing_corners`` – detect missing corners using the
  *Luo Shu* nine-palace principles.
* ``analyze_eightstars`` – evaluate room orientations with the
  *BaZhai* (Eight Mansions) method.
* ``general_remedies`` – common remedies for *Luo Shu* missing corners.
* ``general_guidelines`` – generic layout tips for the *BaZhai* method.
"""

from .luoshu_missing_corner import analyze_missing_corners, general_remedies
from .bazhai_eightstars import analyze_eightstars, general_guidelines

__all__ = [
    "analyze_missing_corners",
    "analyze_eightstars",
    "general_remedies",
    "general_guidelines",
]
