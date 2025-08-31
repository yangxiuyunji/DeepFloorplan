"""Fengshui utilities for floorplan analysis.

This package currently provides:

* ``analyze_missing_corners`` – detect missing corners using the
  *Luo Shu* nine-palace principles.
* ``analyze_eightstars`` – evaluate room orientations with the
  *BaZhai* (Eight Mansions) method.
"""

from .luoshu_missing_corner import analyze_missing_corners
from .bazhai_eightstars import analyze_eightstars

__all__ = ["analyze_missing_corners", "analyze_eightstars"]
