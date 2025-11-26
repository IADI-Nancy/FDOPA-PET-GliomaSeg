import os
import six
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold
from batchgenerators.utilities.file_and_folder_operations import save_json


def get_images_path(path, data_info):
    """
    Return a list of patient directory paths found under a given root directory, excluding
    patients flagged as having segmentation problems in the provided data_info.

    Parameters
    ----------
    path : str
        Filesystem path to the directory that contains patient subdirectories.
    data_info : pandas.DataFrame or mapping-like
        Metadata indexed by patient identifier (typically the folder name). If this
        object contains a column named 'Problem_segmentation', rows with value 1 in
        that column will be considered excluded. If the column is absent, no patients
        will be excluded.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple (all_paths, excluded_patients):
        - all_paths: list of absolute paths (strings) to immediate subdirectories of
          `path` that exist on disk and are not listed in `excluded_patients`.
        - excluded_patients: list of patient identifiers (strings taken from data_info.index)
          that were excluded because 'Problem_segmentation' == 1. Returns an empty list
          if no exclusions apply.
    """
    if 'Problem_segmentation' in data_info:
        # First exclude patients with segmentation problem
        excluded_patients = data_info.index[data_info['Problem_segmentation'] == 1].tolist()
        print('Number of patients excluded: %d' % len(excluded_patients))
    else:
        excluded_patients = []
    all_paths = []
    try:
        patient_list = [_ for _ in os.listdir(path) if os.path.isdir(os.path.join(path, _))]
        for patient in patient_list:
            if os.path.isdir(os.path.join(path, patient)):
                if patient not in excluded_patients:
                    all_paths.append(os.path.join(path, patient))
        return all_paths, excluded_patients
    except Exception as e:
        raise Exception('Error reading paths: %s' % e)


def get_image_plane(img, plane):
    """
    Reorient a 3D image to a specified anatomical plane.

    Parameters
    ----------
    img : SimpleITK.Image
        A 3-dimensional SimpleITK image.
    plane : str
        Target plane for the output orientation. Supported values:
          - 'Axial'   : flip the third axis (z) only.
          - 'Sagittal': flip the third axis, then permute axes to (y, z, x).
          - any other value (commonly 'Coronal'): flip the third axis, then
            permute axes to (x, z, y).

    Returns
    -------
    img: SimpleITK.Image
        The input image reoriented to the requested plane.
    """
    if plane == 'Axial':
        img = img[:, :, ::-1]
    elif plane == 'Sagittal':
        img = sitk.PermuteAxes(img[:, :, ::-1], (1, 2, 0))
    else:
        img = sitk.PermuteAxes(img[:, :, ::-1], (0, 2, 1))
    return img


def get_whole_brain(path):
    """
    Return the skull-stripped whole-brain mask as a SimpleITK Image if present.

    Parameters
    ----------
        path : str or os.PathLike 
            Path to the directory expected to contain 'Static_PET_skullstripped_mask.nii.gz'.

    Returns
    -------
        brain_mask: SimpleITK.Image or None 
            A SimpleITK Image loaded from 'Static_PET_skullstripped_mask.nii.gz'
            when the file exists; otherwise None.
    """
    skullstrip_mask_path = os.path.join(path, 'Static_PET_skullstripped_mask.nii.gz')
    if os.path.exists(skullstrip_mask_path):
        brain_mask = sitk.ReadImage(skullstrip_mask_path)
    else:
        brain_mask = None
    return brain_mask


def center_brain(image, mask, path):
    """
    Center the input image (and optionally its mask) on the brain centroid found from a skull-stripped mask.

    This function attempts to obtain a skull-stripped whole-brain mask using get_whole_brain(path). If a brain mask
    is found, the brain centroid is computed in physical space using SimpleITK's LabelShapeStatisticsImageFilter.
    A translation that moves the image center to the brain centroid is created and applied to the input image using
    sitk.Resample. If a mask is provided, the same transform is applied to the mask using label-appropriate
    interpolation. If no brain mask is found, the input image and mask are returned unchanged and an identity transform
    is returned.

    Parameters
    ----------
    image : SimpleITK.Image
        The input image to be centered. Coordinates and size of this image are used to compute the image center in
        physical space.
    mask : SimpleITK.Image or None
        An optional label image (mask) corresponding to `image`. If provided and a skull-stripped brain mask exists,
        the mask will be resampled with the same translation transform applied to `image`. If None, only `image` is
        processed.
    path : str or PathLike
        Path used by get_whole_brain(...) to locate a skull-stripped brain mask for the case. If get_whole_brain(path)
        returns None, no centering is performed.

    Returns
    -------
    centered_image : SimpleITK.Image
        The resampled (translated) image when a brain mask is found; otherwise the original `image`.
    centered_mask : SimpleITK.Image or None
        The resampled mask if `mask` was provided and a brain mask exists; otherwise None if `mask` was None, or the
        original `mask` if no brain mask was found.
    transform : SimpleITK.Transform
        The translation transform applied to the image/mask (SimpleITK.TranslationTransform with the computed
        offset when centering was performed). If no brain mask was found, an identity SimpleITK.Transform() is returned.
    """
    # Center brain only if skullstrip mask exist otherwise centering is bad due to soft tissues included in brain mask with Otsu like methods
    brain_mask = get_whole_brain(path)
    if brain_mask is not None:
        shape_filter = sitk.LabelShapeStatisticsImageFilter()
        shape_filter.Execute(brain_mask)
        brain_center = shape_filter.GetCentroid(1)
        image_center = image.TransformContinuousIndexToPhysicalPoint([sz/2 for sz in image.GetSize()])
        translation = [brain_center[i] - image_center[i] for i in range(len(brain_center))]
        transform = sitk.TranslationTransform(3)
        transform.SetOffset(translation)
        centered_image = apply_func_vector_image(image, sitk.Resample, transform, sitk.sitkLinear, 0.0)
        if mask is not None:
            centered_mask = apply_func_vector_image(mask, sitk.Resample, transform, sitk.sitkLabelLinear, 0.0)
        else:
            centered_mask = None
    else:
        print('Brain not centered due to missing skullstrip mask')
        centered_image = image
        centered_mask = mask
        transform = sitk.Transform()
    return centered_image, centered_mask, transform


def get_TEP_mask(path, n_labels, skull_stripping=False):
    """
    Create a vector label mask for a PET study stored under a case directory.

    This function loads ROI segmentation files from a directory structure and
    composes a SimpleITK vector image that can be used as a multi-channel
    label mask for training or evaluation. It supports two modes:
    - n_labels == 2: returns a 3-channel mask with background, brain and tumor
        channels (suitable for softmax with separate brain and tumor classes).
    - n_labels == 1: returns a single-channel mask for tumor presence only.

    Files expected under the provided path:
    - <path>/ROI/Brain.nii.gz               (binary brain mask, used when n_labels == 2)
    - <path>/ROI/Tumor_TBR1,6_seg.nii.gz   (binary tumor mask)
    If skull_stripping is True, the function also expects:
    - <path>/Static_PET_skullstripped_mask.nii.gz  (binary mask applied to all channels)

    Parameters
    ----------
    path : str
            Filesystem path to the case directory (the function will look for a subdirectory
            named "ROI" containing the ROI images).
    n_labels : int
            Number of label channels requested. Supported values:
            - 2: compose [background, brain, tumor] channels (3-channel vector image).
            - 1: compose [tumor] channel (single-channel image).
    skull_stripping : bool, optional
            If True, apply the skull-stripping mask (<path>/Static_PET_skullstripped_mask.nii.gz)
            to the composed mask using sitk.Mask (default: False).

    Returns
    -------
    mask : SimpleITK.Image
            A SimpleITK vector image containing the composed mask channels. For n_labels == 2
            the channel order is [background, brain, tumor]. For n_labels == 1 the image
            contains a single channel representing tumor presence.
    label_names : dict
            A mapping from human-readable label names to integer class indices. For example:
            - n_labels == 2 -> {'background': 0, 'brain': 1, 'tumor': 2}
            - n_labels == 1 -> {'background': 0, 'tumor': 1}
    """
    ROI_dir = os.path.join(path, 'ROI')
    if n_labels == 2:
        # Softmax activation with tumor + brain
        brain = sitk.ReadImage(os.path.join(ROI_dir, 'Brain.nii.gz')) > 0
        tumor = sitk.ReadImage(os.path.join(ROI_dir, 'Tumor_TBR1,6_seg.nii.gz')) > 0
        background = sitk.And(brain == 0, tumor == 0)
        mask = sitk.Compose([background, brain, tumor])
        label_names = {'background': 0, 'brain': 1, 'tumor': 2}
    elif n_labels == 1:
        tumor = sitk.ReadImage(os.path.join(ROI_dir, 'Tumor_TBR1,6_seg.nii.gz')) > 0
        mask = sitk.Compose([tumor])
        label_names = {'background': 0, 'tumor': 1}
    else:
        raise ValueError('Got n_labels = %d. Only 1, 2 and 3 implemented.' % n_labels)
    if skull_stripping:
        skullstrip_mask = sitk.ReadImage(os.path.join(path, 'Static_PET_skullstripped_mask.nii.gz'))
        mask = apply_func_vector_image(mask, sitk.Mask, skullstrip_mask)
    return mask, label_names

def convert_mask_nnUNet_format(mask, n_labels):
    """
    Convert a one-hot (multi-component) SimpleITK mask into the scalar label format
    expected by nnU-Net.

    Parameters
    ----------
    mask : SimpleITK.Image
        A SimpleITK VectorImage where each component (vector index) is a binary mask
        for one class. Each component should contain values {0, 1} indicating absence/
        presence of that class at each voxel.
    n_labels : int
        Number of label channels present in `mask`. If n_labels != 1 the function
        composes a single-channel uint8 image where each voxel's value equals the
        index of the component that is set. If n_labels == 1 the function simply
        extracts and returns the first component as a scalar image.

    Returns
    -------
    final_mask: SimpleITK.Image
        A scalar SimpleITK image (sitk.sitkUInt8 when composing multiple channels)
        whose voxel values encode class labels by channel index. The returned image
        preserves the spatial metadata (origin, spacing, direction) of the input mask.
    """
    # Convert mask from one hot encoded to required format
    if n_labels != 1:
        final_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        final_mask.CopyInformation(mask)
        for i in range(mask.GetNumberOfComponentsPerPixel()):
            mask_i = sitk.VectorIndexSelectionCast(mask, i)
            if np.any(sitk.GetArrayFromImage(sitk.And(mask_i, final_mask)) != 0):
                raise ValueError('Intersection between masks detected and not handled by this code.')
            final_mask += mask_i * i
    else:
        final_mask = sitk.VectorIndexSelectionCast(mask, 0)
    return final_mask


def get_TEP_data(path, skull_stripping=False):
    """
    Load static PET image data from a dataset directory, optionally applying a skull-stripping mask.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the directory containing the PET files. Expected filenames in the directory:
        - "Static_PET.nii.gz" (unmasked image)
        - "Static_PET_skullstripped.nii.gz" (masked image, required if skull_stripping is True)
        - "Static_PET_skullstripped_mask.nii.gz" (binary mask, required if skull_stripping is True)
    skull_stripping : bool, optional
        If True, the function will read the skull-stripped image and its corresponding mask and
        return the masked image (voxels outside the mask set to zero). If False (default),
        the unmasked "Static_PET.nii.gz" image is returned.

    Returns
    -------
    SimpleITK.Image
        A SimpleITK Image object. If skull_stripping is True, the returned image is the result
        of sitk.Mask(skullstripped_image, skullstrip_mask). Otherwise, it is the image read from
        "Static_PET.nii.gz".
    """
    if skull_stripping == True:
        # Ensure that values are equal to 0 outside of skullstrip mask. Sometimes we add small values
        skullstrip_mask = sitk.ReadImage(os.path.join(path, 'Static_PET_skullstripped_mask.nii.gz'))
        skullstripped_image = sitk.ReadImage(os.path.join(path, 'Static_PET_skullstripped.nii.gz'))
        return sitk.Mask(skullstripped_image, skullstrip_mask)
    else:
        return sitk.ReadImage(os.path.join(path, 'Static_PET.nii.gz'))


def get_TEP_data_mask(path, plane, n_labels, ref_spacing, ref_size,
                      skull_stripping=False, brain_centering=True):
    """
    Prepare and return a PET image and its corresponding segmentation mask resampled and padded/cropped
    to a reference spacing and size.

    This function:
    - Loads a PET image from the provided path.
    - Optionally loads an associated ROI/label image if a 'ROI' subfolder exists.
    - Computes the spacing and padding required to match the provided reference spacing and size.
    - Ensures the image has a channel dimension.
    - Optionally recenters the image and mask on the brain.
    - Resamples, pads and/or crops the image and mask to match the requested plane, spacing and size.
    - If a label named 'brain' is present, enforces that the brain label is present on the recommended
        minimum number of slices (n_slices=6).

    Parameters
    ----------
    path : str
            Filesystem path to the subject directory containing the PET image and optionally an 'ROI'
            subdirectory with segmentation masks.
    plane : str or int
            Target anatomical plane used for resampling/padding/cropping. Typical values are strings such
            as 'axial', 'coronal', 'sagittal' (or an integer code if the downstream resampling utility
            expects one). This value is forwarded to resample_pad_crop_image.
    n_labels : int
            Expected number of labels in the ROI (used when loading/validating the mask).
    ref_spacing : float
            Target isotropic voxel spacing (mm) to resample the image to.
    ref_size : sequence of int (length 3)
            Target image size (voxels) in x, y, z after padding/cropping to match ref_spacing.
    skull_stripping : bool, optional (default=False)
            If True, request/skull-strip the input PET image when loading (passed to get_TEP_data and
            get_TEP_mask).
    brain_centering : bool, optional (default=True)
            If True, call center_brain to center image and mask on the brain before resampling. Note:
            any spatial transform returned by center_brain is not returned by this function.

    Returns
    -------
    image : SimpleITK.Image
            The PET image after ensuring a channel dimension and after resampling/padding/cropping to the
            reference spacing/size. The image is ready for downstream processing or model input.
    mask : SimpleITK.Image or None
            The segmentation/label image aligned with `image`. If no ROI was present under `path`, this
            will be None. When present, label values correspond to those produced by get_TEP_mask; if a
            label named 'brain' exists, the mask will have been post-processed to ensure a minimum number
            of brain slices (recommended n_slices=6).
    """
    image = get_TEP_data(path, skull_stripping=skull_stripping)
    if os.path.exists(os.path.join(path, 'ROI')):
        mask, label_names = get_TEP_mask(path, n_labels, skull_stripping=skull_stripping)
    else:
        mask, label_names = None, []
    # Compute the aimed spacing and padding to match with target spacing and size
    im_padding = get_padding(image, ref_spacing, ref_size)
    # To always return image with channels (compatibility with older code)
    image = sitk.Compose([image])
    if brain_centering:
        image, mask, transform = center_brain(image, mask, path)
    # Resample PET image to match ref image
    image, mask = resample_pad_crop_image(image, mask, plane, ref_spacing, im_padding)
    if 'brain' in label_names:
        # Ensure brain mask is present on the same number of slices (6 = recommandation in 10.1186/s13550-017-0295-y)
        mask = apply_func_vector_image(mask, check_brain_n_slices, apply_on_index=label_names['brain'], n_slices=6)
    return image, mask


def IBSI_resampling(image=None, mask=None, **kwargs):
    """
    Resample an image and/or mask to a specified spacing while aligning the centers of the original
    and resampled grids according to the IBSI (Image Biomarker Standardisation Initiative) approach.
    https://github.com/Radiomics/pyradiomics/issues/498
    This function supports:
    - Resampling of intensity images (including vector images) to sitk.sitkFloat32 output.
    - Resampling of label/mask images using label-aware linear interpolation (sitk.sitkLabelLinear)
        to produce a sitk.sitkUInt8 output. Multi-label masks are handled by label-linear strategy
        (i.e. interpolating each label as a binary mask and choosing the label with maximum value).
    - Optional rounding of intensity values to a requested number of decimal places.
    - Passing the interpolator as either a SimpleITK interpolator constant or its string name.
    - Preserving direction and computing an output origin so that the centers of the input and
        resampled grids are aligned. If any entry in the requested spacing is zero, the input spacing
        for that axis is used (no resampling along that axis).
    Parameters
    ----------
    image : SimpleITK.Image or None
            The input intensity image to be resampled. If None, only the mask (if provided) is resampled.
            Vector images (multi-component pixels) are supported and resampled per-component.
    mask : SimpleITK.Image or None
            The input label/mask image to be resampled. If None, only the intensity image (if provided) is
            resampled.
    **kwargs :
            resampledPixelSpacing : float or sequence of float
                    Desired output spacing per axis (e.g., [sx, sy, sz]). Entries equal to 0 indicate that the
                    spacing from the input image/mask should be used for that axis.
            grayValuePrecision : int or None, optional
                    If provided, round intensity image voxel values to this many decimal places after
                    resampling. If None, no rounding is done. Rounding is applied per channel for vector images.
            interpolator : str or SimpleITK interpolator constant, optional
                    Interpolator used for intensity image resampling (default: sitk.sitkLinear). If a string is
                    provided, the function will attempt to resolve it as an attribute of the SimpleITK module
                    (e.g., "sitkLinear" or "sitkNearestNeighbor"); on failure the linear interpolator is used.
    Returns
    -------
    res_im : SimpleITK.Image or None
            The resampled intensity image cast to sitk.sitkFloat32 (or composed float vector image).
            None if the input `image` was None.
    res_ma : SimpleITK.Image or None
            The resampled mask image with pixel type sitk.sitkUInt8 (or composed uint8 vector image).
            None if the input `mask` was None.
    """
    # 
    # resample image to new spacing, align centers of both resampling grids.
    if image is None and mask is None:
        raise ValueError('image and mask can not both be None')
    spacing = kwargs.get('resampledPixelSpacing')
    grayValuePrecision = kwargs.get('grayValuePrecision')
    interpolator = kwargs.get('interpolator', sitk.sitkLinear)

    try:
        if isinstance(interpolator, six.string_types):
            interpolator = getattr(sitk, interpolator)
    except Exception:
        interpolator = sitk.sitkLinear

    if image is None:
        im_spacing = np.array(mask.GetSpacing(), dtype='float')
        im_size = np.array(mask.GetSize(), dtype='float')
        direction = mask.GetDirection()

        spacing = np.where(np.array(spacing) == 0, im_spacing, spacing)

        spacingRatio = im_spacing / spacing
        newSize = np.ceil(im_size * spacingRatio)

        new_origin = tuple(np.array(mask.GetOrigin()) + 0.5 *
                        ((im_size - 1) * im_spacing - (newSize - 1) * spacing))
    else: 
        im_spacing = np.array(image.GetSpacing(), dtype='float')
        im_size = np.array(image.GetSize(), dtype='float')
        direction = image.GetDirection()

        spacing = np.where(np.array(spacing) == 0, im_spacing, spacing)

        spacingRatio = im_spacing / spacing
        newSize = np.ceil(im_size * spacingRatio)

        new_origin = tuple(np.array(image.GetOrigin()) + 0.5 *
                        ((im_size - 1) * im_spacing - (newSize - 1) * spacing))

    rif = sitk.ResampleImageFilter()
    rif.SetOutputOrigin(new_origin)
    rif.SetSize(np.array(newSize, dtype='int').tolist())
    rif.SetOutputDirection(direction)
    rif.SetOutputSpacing(spacing)

    if image is not None:
        rif.SetOutputPixelType(sitk.sitkFloat32)
        rif.SetInterpolator(interpolator)
        if image.GetPixelID() == sitk.sitkVectorFloat32 or image.GetPixelID() == sitk.sitkVectorFloat64:
            res_im_list = []
            for i in range(image.GetNumberOfComponentsPerPixel()):
                res_im_i = sitk.VectorIndexSelectionCast(image, i)
                res_im_i = rif.Execute(sitk.Cast(res_im_i, sitk.sitkFloat32))
                res_im_list.append(res_im_i)
            res_im = sitk.Compose(res_im_list)
        else:
            res_im = rif.Execute(sitk.Cast(image, sitk.sitkFloat32))

        # Round to n decimals (0 = to nearest integer)
        if grayValuePrecision is not None:
            if image.GetPixelID() == sitk.sitkVectorFloat32 or image.GetPixelID() == sitk.sitkVectorFloat64:
                res_im_list = []
                for i in range(image.GetNumberOfComponentsPerPixel()):
                    res_im_i = sitk.VectorIndexSelectionCast(image, i)
                    im_arr = sitk.GetArrayFromImage(res_im_i)
                    im_arr = np.round(im_arr, grayValuePrecision)
                    round_im = sitk.GetImageFromArray(im_arr)
                    round_im.CopyInformation(res_im_i)
                    res_im_list.append(round_im)
                res_im = sitk.Compose(res_im_list)
            else:
                im_arr = sitk.GetArrayFromImage(res_im)
                im_arr = np.round(im_arr, grayValuePrecision)
                round_im = sitk.GetImageFromArray(im_arr)
                round_im.CopyInformation(res_im)
                res_im = round_im
    else:
        res_im = None

    if mask is not None:
        rif.SetOutputPixelType(sitk.sitkUInt8)
        # Linear interpolator with 0.5 threshold only works for binary image and not multi label image
        # But linear interpolation produce smoother mask than nearest neighbor
        # Workaround sitk.sitkLabelLinear https://insight-journal.org/browse/publication/950
        # Interpolate each label independently as if it were a binary mask and return the label with the maximum interpolated value
        rif.SetInterpolator(sitk.sitkLabelLinear)
        if mask.GetPixelID() == sitk.sitkVectorUInt8:
            res_ma_list = []
            for i in range(mask.GetNumberOfComponentsPerPixel()):
                res_ma_i = sitk.VectorIndexSelectionCast(mask, i)
                res_ma_i = rif.Execute(res_ma_i)
                res_ma_list.append(res_ma_i)
            res_ma = sitk.Compose(res_ma_list)
        else:
            res_ma = rif.Execute(mask)
    else:
        res_ma = None
    
    return res_im, res_ma
    

def get_padding(target_image, ref_spacing=None, ref_size=None):
    """
    Compute per-axis padding (before, after) required to match a target image, after
    resampling, to a given reference spacing and reference size.

    Parameters
    ----------
    target_image : SimpleITK.Image
        The image to inspect and (internally) resample. This function calls
        IBSI_resampling(target_image, resampledPixelSpacing=ref_spacing) so the
        object must be compatible with that helper (typically a SimpleITK Image).
    ref_spacing : sequence of float, optional
        Desired voxel spacing to which `target_image` will be resampled before
        computing padding. If None (or if `ref_size` is None) no padding is
        computed and the function returns None.
    ref_size : sequence of int, optional
        Desired image size (number of voxels) for each axis after resampling. Must
        have the same number of elements as the image dimensionality.

    Returns
    -------
    im_padding: list[tuple[int, int]] or None
        A list of (pad_before, pad_after) integer pairs, one per image axis, that
        indicate how many voxels to add before and after the image to reach
        `ref_size`. If the size difference along an axis is odd, the extra voxel is
        added to the "after" side (i.e. pad_after = pad_before + 1). If a returned
        padding value is negative, it indicates that the resampled image is larger
        than `ref_size` along that axis (i.e. cropping would be required). If
        either `ref_spacing` or `ref_size` is None the function returns None.
    """
    if ref_spacing is not None and ref_size is not None:
        modif_resampled, _ = IBSI_resampling(target_image, resampledPixelSpacing=ref_spacing)
        size_difference = np.asarray(ref_size) - np.asarray(modif_resampled.GetSize())
        im_padding = [(_//2, _//2 + 1) if _ % 2 == 1 else (_//2, _//2) for _ in size_difference]
    else:
        im_padding = None
    return im_padding


def apply_func_vector_image(vector_image, func, *func_args, apply_on_index=None, **func_kwargs):
    """
    Apply a scalar-image function to selected channels of a multi-component (vector) SimpleITK image.

    Parameters
    ----------
    vector_image : SimpleITK.Image
        A multi-component (vector) image. The number of components is obtained
        via vector_image.GetNumberOfComponentsPerPixel().
    func : callable
        A function that accepts a single-component SimpleITK.Image as its first
        argument and returns a processed single-component SimpleITK.Image. Additional
        positional and keyword arguments passed to this wrapper are forwarded to
        func.
    *func_args
        Additional positional arguments to pass to func.
    apply_on_index : None, int, or iterable of int, optional
        Specifies which component indices the function should be applied to.
        - If None (default), func is applied to all components [0 .. C-1].
        - If an int, that single component index is used.
        - If an iterable of ints, those indices are used.
        Indices are zero-based. Indices outside the valid range will raise an error
        from SimpleITK when selecting components.
    **func_kwargs
        Additional keyword arguments to pass to func.

    Returns
    -------
    vector_image: SimpleITK.Image
        A new multi-component (vector) image produced by composing each component
        after optionally applying func to the requested components. Components not
        listed in apply_on_index are kept unchanged.
    """
    if apply_on_index is None:
        apply_on_index = list(range(vector_image.GetNumberOfComponentsPerPixel()))
    else:
        if isinstance(apply_on_index, int):
            apply_on_index = [apply_on_index]
    img_list = []
    for i in range(vector_image.GetNumberOfComponentsPerPixel()):
        img_i = sitk.VectorIndexSelectionCast(vector_image, i)
        if i in apply_on_index:
            img_i = func(img_i, *func_args, **func_kwargs)
        img_list.append(img_i)
    vector_image = sitk.Compose(img_list)
    return vector_image


def resample_pad_crop_image(image, mask, plane, im_spacing, im_padding):
    """
    Resample, pad/crop, and extract a specific plane from an image and its corresponding mask.

    This function performs three optional operations on a SimpleITK image and mask pair, in order:
    1. Resampling: if im_spacing is provided, resamples image and mask to the specified spacing.
    2. Padding/Cropping: if im_padding is provided, applies per-dimension padding or cropping.
        - im_padding should be an iterable of (pad_start, pad_end) pairs, one pair per image dimension.
        - A positive value indicates padding (ConstantPad with value 0).
        - A negative value indicates cropping (Crop by the absolute value).
        - Padding and cropping are computed separately for start and end of each dimension.
        - Padding/cropping is applied to image and mask independently; vector images/masks use
          apply_func_vector_image to apply the operation per component.
    3. Plane extraction: extracts a specific plane/orientation from the (possibly resampled
        and padded/cropped) image and mask using get_image_plane. Vector images/masks are
        handled with apply_func_vector_image.

    Parameters
    ----------
    image : SimpleITK.Image or None
         The input image. May be a scalar image or a vector image (e.g., multi-channel).
         If None, image processing steps are skipped and None is returned for the image.
    mask : SimpleITK.Image or None
         The corresponding mask. May be a scalar mask or a vector mask (e.g., one-hot channels).
         If None, mask processing steps are skipped and None is returned for the mask.
    plane : object
         Plane/orientation identifier passed to get_image_plane. The exact accepted values
         depend on the implementation of get_image_plane (e.g., axis index, string token).
    im_spacing : sequence or None
         Target spacing for resampling. If not None, IBSI_resampling is called with
         resampledPixelSpacing=im_spacing and linear interpolation for the image.
         If None, no resampling is performed.
    im_padding : sequence of (start, end) pairs or None
         Per-dimension padding/cropping specification. Example for a 3D image:
         [(p0_start, p0_end), (p1_start, p1_end), (p2_start, p2_end)].
         Positive values indicate padding, negative values indicate cropping.
         If None, no padding or cropping is performed.

    Returns
    -------
    image: SimpleITK.Image or None
         The processed image after optional resampling, padding/cropping, and plane extraction.
         None if the input `image` was None.
    mask: SimpleITK.Image or None
         The processed mask after optional resampling, padding/cropping, and plane extraction.
         None if the input `mask` was None.
    """
    if im_spacing is not None:
        image, mask = IBSI_resampling(image, mask, resampledPixelSpacing=im_spacing, interpolator=sitk.sitkLinear)
    if im_padding is not None:
        padding_start = [int(_[0]) if _[0] > 0 else 0 for _ in im_padding]
        padding_end = [int(_[1]) if _[1] > 0 else 0 for _ in im_padding]
        to_pad = any(_ != 0 for _ in padding_start) or any(_ != 0 for _ in padding_end)
        cropping_start = [int(abs(_[0])) if _[0] < 0 else 0 for _ in im_padding]
        cropping_end = [int(abs(_[1])) if _[1] < 0 else 0 for _ in im_padding]
        to_crop = any(_ != 0 for _ in cropping_start) or any(_ != 0 for _ in cropping_end)
        if image is not None:
            if image.GetPixelID() in [sitk.sitkVectorFloat32, sitk.sitkVectorFloat64]:
                if to_pad:
                    image = apply_func_vector_image(image, sitk.ConstantPad, padding_start, padding_end, 0.)
                if to_crop:
                    image = apply_func_vector_image(image, sitk.Crop, cropping_start, cropping_end)
            else:
                if to_pad:
                    image = sitk.ConstantPad(image, padding_start, padding_end, 0.)
                if to_crop:
                    image = sitk.Crop(image, cropping_start, cropping_end)
        if mask is not None:
            if mask.GetPixelID() == sitk.sitkVectorUInt8:
                if to_pad:
                    mask = apply_func_vector_image(mask, sitk.ConstantPad, padding_start, padding_end, 0.)
                if to_crop:
                    mask = apply_func_vector_image(mask, sitk.Crop, cropping_start, cropping_end)
            else:
                if to_pad:
                    mask = sitk.ConstantPad(mask, padding_start, padding_end, 0.)
                if to_crop:
                    mask = sitk.Crop(mask, cropping_start, cropping_end)
    if image is not None:
        if image.GetPixelID() in [sitk.sitkVectorFloat32, sitk.sitkVectorFloat64]:
            image = apply_func_vector_image(image, get_image_plane, plane)
        else:
            image = get_image_plane(image, plane)
    if mask is not None:
        if mask.GetPixelID() == sitk.sitkVectorUInt8:
            mask = apply_func_vector_image(mask, get_image_plane, plane)
        else:
            mask = get_image_plane(mask, plane)
    return image, mask


def check_brain_n_slices(brain_mask, n_slices=6):
    """
    Adjust the number of axial (z) slices in a 3D binary brain mask to a target count.

    This function performs the following steps:
    - Computes label statistics for the provided mask and returns the original mask
        unchanged if no foreground label 1 is present.
    - Keeps only the largest connected component (relabels components and selects
        label 1).
    - Computes the bounding box of the largest component and measures its extent
        along the z (axial) axis (assumes z is the third axis, index 2).
    - If the current z-extent is larger than the requested n_slices, removes slices
        from the two ends of the bounding box as evenly as possible (setting those
        slices to zero).
    - If the current z-extent is smaller than n_slices, pads by copying the
        outermost existing slices outward (copying the lowest and/or highest slice
        of the bounding box) to reach the target size.
    - Returns the resulting binary mask.

    Parameters
    ----------
    brain_mask : SimpleITK.Image
            3D binary brain mask where foreground voxels have value 1. The function
            expects the z (axial) axis to be the third axis (index 2) when using
            numpy-style indexing. The implementation uses SimpleITK filters to compute
            connected components and bounding boxes, so a SimpleITK.Image is supported;
            NumPy arrays may work if they are compatible with the functions called.
    n_slices : int, optional
            Target number of slices along the z axis for the brain region (default: 6).

    Returns
    -------
    brain_mask : SimpleITK.Image
            A 3D binary mask (same type as the input when possible) where only the
            largest connected component remains and the axial extent of that component
            has been adjusted to n_slices by removing or adding slices as described.
    """
    label_shape = sitk.LabelShapeStatisticsImageFilter()
    label_shape.Execute(brain_mask)
    if not 1 in label_shape.GetLabels():
        return brain_mask
    # If several components not connected keep the biggest (relabel store the biggest element as label 1)
    cc = sitk.ConnectedComponent(brain_mask)
    relabeled_cc = sitk.RelabelComponent(cc)
    brain_mask = relabeled_cc == 1
    # Get size in z
    bbox = label_shape.GetBoundingBox(1)
    current_n_slices = bbox[-1]
    slice_diff = current_n_slices - n_slices
    if slice_diff > 0:
        # We set to 0 slice outside range
        n_slices_to_remove = (slice_diff//2, slice_diff//2 + 1) if slice_diff % 2 == 1 else (slice_diff//2, slice_diff//2)
        slices_idx_to_remove = [bbox[2] + _ for _ in range(n_slices_to_remove[0])] + [bbox[2] + (bbox[-1] - 1) - _ for _ in range(n_slices_to_remove[1])]
        for idx in slices_idx_to_remove:
            brain_mask[:, :, idx] = 0
    elif slice_diff < 0:
        # We copy lowest/uppest slice
        # TODO : eventually instead of copying we can apply a translation / affine transform so that ROI follow the natural shape of the brain
        n_slices_to_add = (abs(slice_diff)//2 + 1, abs(slice_diff)//2) if abs(slice_diff) % 2 == 1 else (abs(slice_diff)//2, abs(slice_diff)//2)
        lowest_slice = bbox[2]
        for n in range(n_slices_to_add[0]):
            brain_mask[:, :, (lowest_slice - 1) - n] = brain_mask[:, :, lowest_slice]
        uppest_slice = bbox[2] + (bbox[-1] - 1)
        for n in range(n_slices_to_add[1]):
            brain_mask[:, :, (uppest_slice + 1) + n] = brain_mask[:, :, uppest_slice]
    return brain_mask


def get_dataset_fullname(dataset_number=None, dataset_name=None):
    """
    Return the full dataset folder name that matches a given task number or dataset name substring.

    This function looks for a single directory inside the nnUNet raw data directory (taken
    from the environment variable 'nnUNet_raw') whose name contains the provided substring.
    The substring is derived from either a numeric task identifier (zero-padded to three
    digits) or from an explicit dataset name string. If both inputs are provided, dataset_name
    takes precedence.

    Parameters
    ----------
    dataset_number : int or str, optional
        Numeric task identifier (e.g. 1 or "1"). If provided, it will be converted to a
        string and zero-padded to a width of three characters (e.g. 1 -> "001") before
        searching. If the string form of dataset_number has more than three digits, a
        ValueError is raised.
    dataset_name : str, optional
        Explicit name or substring to search for in the dataset directory names. If provided,
        this value overrides dataset_number.

    Returns
    -------
    full_dataset_name: str
        The single matching dataset directory name found inside the directory specified by
        the 'nnUNet_raw' environment variable.
    """
    path_raw_data = os.environ.get('nnUNet_raw')
    if dataset_number is not None:
        if len(str(dataset_number)) > 3:
            raise ValueError('Task number has more than three digits.')
        else:
            dataset_substring = str(dataset_number).zfill(3)
    if dataset_name is not None:
        dataset_substring = dataset_name
    full_dataset_name = [dataset for dataset in os.listdir(path_raw_data) if dataset_substring in dataset]
    if len(full_dataset_name) == 0:
        raise ValueError('No matching name for the given inputs')
    elif len(full_dataset_name) > 1:
        raise ValueError('More than one matching name for the given inputs')
    return full_dataset_name[0]


def stratify(link_patients, data_info, n_splits=5, random_state=42, shuffle=True, verbose=False, save_path=None):
    """
    Stratify patient IDs for cross-validation using stratified K-fold on combined Device and Recurrence labels.

    The function builds stratified train/validation splits of patients based on a concatenated
    stratification key created from the 'Device' and 'Recurrence' fields in `data_info`. Returned
    IDs are formatted with the prefix "PETseg_" and correspond to values from the 'nnUNet ID'
    column of `link_patients`.

    Parameters
    ----------
    link_patients : pandas.DataFrame
        DataFrame containing at least the columns:
          - 'nnUNet ID' : nnUNet identifiers used to produce the returned train/val lists
          - 'Original ID' : original identifiers used to align `data_info` via .loc for stratification
    data_info : pandas.DataFrame or dict-like
        Table-like object indexed by the same identifiers found in link_patients['Original ID'].
        Must contain the columns/keys 'Device' and 'Recurrence' whose string representations
        will be concatenated to form the stratification label.
    n_splits : int, optional (default=5)
        Number of folds for StratifiedKFold.
    random_state : int or RandomState instance, optional (default=42)
        Random seed or RandomState for reproducible shuffling in StratifiedKFold.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples before splitting in StratifiedKFold.
    verbose : bool, optional (default=False)
        If True, print the Original IDs assigned to each train and validation split during construction.
    save_path : str or pathlib.Path or None, optional (default=None)
        If provided, the resulting list of split dictionaries will be written to this path using
        a helper function `save_json`. Each element of the saved list has the form:
          {'train': [<PETseg_nnUNetID>, ...], 'val': [<PETseg_nnUNetID>, ...]}

    Returns
    -------
    splits: list of dict
        A list of length `n_splits`. Each element is a dict with keys:
          - 'train': list of strings, train identifiers prefixed with "PETseg_"
          - 'val'  : list of strings, validation identifiers prefixed with "PETseg_"
    """
    splits = []
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    concat_status = data_info['Device'].astype(str) + data_info['Recurrence'].astype(str)
    concat_status = concat_status.loc[link_patients['Original ID']]
    for train_index, validation_index in skf.split((link_patients["nnUNet ID"].to_numpy()), concat_status):
        splits.append({'train': ["PETseg_%s" % _ for _ in link_patients['nnUNet ID'].to_numpy()[train_index]], 
                       'val': ["PETseg_%s" % _ for _ in link_patients['nnUNet ID'].to_numpy()[validation_index]]})
        if verbose:
            print("train:", link_patients['Original ID'].to_numpy()[train_index], 
                  "val:", link_patients['Original ID'].to_numpy()[validation_index])
    if save_path is not None:
        save_json(splits, save_path, sort_keys=False)
    return splits