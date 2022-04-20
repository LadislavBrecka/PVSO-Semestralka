import numpy as np


def dist(point1, point2):
    p1 = np.array(point1[0:2])
    p2 = np.array(point2[0:2])
    return np.linalg.norm(p1 - p2)