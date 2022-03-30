from dataclasses import dataclass
from typing import Tuple


@dataclass
class Color:
    bgr: Tuple[int, int, int]
    h_range: Tuple[int, int]
