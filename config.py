from dataclasses import dataclass
import numpy as np

# colors for detecting
from color import Color

RED    = Color((0, 0, 255), (150, 180))
GREEN  = Color((0, 255, 0), (50, 80))
YELLOW = Color((0, 255, 255), (20, 40))

# sigma for canny detection
SIGMA = 0.33

# image setting
width = 640
height = 480



