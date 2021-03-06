import enum
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Color:
    bgr: Tuple[int, int, int]
    h_range: Tuple[int, int]
    s_range: Tuple[int, int]
    v_range: Tuple[int, int]
    mark_bgr: Tuple[int, int, int]


class Colors(enum.Enum):
    RED = Color((0, 0, 255),
                (150, 180),
                (150, 255),
                (150, 255),
                (0, 255, 0)
                )
    GREEN = Color((0, 255, 0),
                  (50, 80),
                  (100, 255),
                  (100, 255),
                  (0, 0, 255)
                  )
    YELLOW = Color((0, 255, 255),
                   (20, 40),
                   (100, 255),
                   (100, 255),
                   (0, 0, 255)
                   )
    BLUE = Color((255, 0, 0),
                 (100, 130),
                 (100, 255),
                 (100, 255),
                 (0, 0, 255)
                 )
    INVALID = None


class CmdColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

