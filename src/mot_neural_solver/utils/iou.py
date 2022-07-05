import numpy as np
import time
import pycocotools.mask as rletools
def iou(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4)
        boxB: numpy array of bounding boxes with size (M, 4)

    Returns:
        numpy array of size (N,M), where the (i, j) element is the IoU between the ith box in boxA and jth box in boxB.

    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    x11, y11, x12, y12 = np.split(boxA, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangles
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangles
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou

def iou_pairs(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4).
        boxB: numpy array of bounding boxes with size (N, 4)

    Returns:
        numpy array of size (N,), where the ith element is the IoU between the ith box in boxA and boxB.

    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / (boxAArea + boxBArea - interArea).astype(float)

    return iou


def iou_mask(maskA, maskB):
    """
    Args:
        maskA: numpy array of bounding boxes with size (N, ) containing Rs
        maskB: numpy array of bounding boxes with size (M, ) containing Rs
        (see https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py for Rs format)

    Returns:
        numpy array of size (N,M), where the (i, j) element is the IoU between the ith mask in maskA and jth mask in maskB.
    """
    rle_A = maskA.tolist()
    rle_B = maskB.tolist()
    iscrowd = np.array([False]*len(rle_B))

    return rletools.iou(rle_A, rle_B, iscrowd)
