import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, ball
from skimage.filters import threshold_triangle

# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, generate_ball


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


def create_brain_mask(data):
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    brain_mask = None
    for c in range(0, data.shape[0]):
        # Threshold with triangle method. Otsu usually works well also but triangler might be better for PET
        thresh = threshold_triangle(data[c])
        brain_mask_c = data[c] > thresh
        # Clean
        brain_mask_c = binary_opening(brain_mask_c, ball(3))
        # Retrieve largest region
        brain_mask_c = remove_all_but_largest_component(brain_mask_c)
        if brain_mask is None:
            brain_mask = brain_mask_c
        else:
            brain_mask |= brain_mask_c
    return brain_mask


def crop_to_brain(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_brain_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox

