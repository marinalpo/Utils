import numpy as np

def get_intersections(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(np.int32(boxA[0]), np.int32(boxB[0]))
    yA = max(np.int32(boxA[1]), np.int32(boxB[1]))
    xB = min(np.int32(boxA[2]), np.int32(boxB[2]))
    yB = min(np.int32(boxA[3]), np.int32(boxB[3]))
    return xA, yA, xB, yB


def compute_area(xA, yA, xB, yB):
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea