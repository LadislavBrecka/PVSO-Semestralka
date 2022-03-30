import enum
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Color:
    bgr: Tuple[int, int, int]
    h_range: Tuple[int, int]


class Colors(enum.Enum):
    RED = Color((0, 0, 255), (150, 180))
    GREEN = Color((0, 255, 0), (50, 80))
    YELLOW = Color((0, 255, 255), (20, 40))
    INVALID = None

