import numpy as np
import torch


def ensure_unique_masks(frame_masks):
    """
    Ensures that there are not overlapping pixels in the masks of a frame. This is a MOTS challenge submission
    requirement

    frame_masks: np.array (N, H, W)
    returns: np.array (N, H, W)
    """
    safe_masks = np.zeros_like(frame_masks)

    # Get the maximum mask ix
    i = np.argmax(frame_masks, axis=0)

    # Create indices to access max elements
    h, w = np.indices((frame_masks.shape[1], frame_masks.shape[2]))
    ix = (i, h, w)

    # Copy max elements
    safe_masks[ix] = frame_masks[ix]

    return safe_masks
