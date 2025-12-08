import os
import argparse
import subprocess
import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import pandas as pd
from ..utils.data_utils import IBSI_resampling
from datetime import datetime
from natsort import natsorted
from warnings import warn


def segmentation_post_treatment(segmented_binary_image, kernel_size=[10, 10, 10], method='Closing', use_image_spacing=True, select_biggest=False):
    """
    Post-process a binary segmentation image with morphological operations, optional Gaussian smoothing,
    and optional largest-component selection.

    Parameters
    ----------
    segmented_binary_image : SimpleITK.Image
        Binary segmentation image (non-zero foreground). The function expects a SimpleITK image.
    kernel_size : float, int, or sequence of numbers, optional
        Size parameter expressed in millimeters by default. If a single float/int is provided it will be
        replicated for every image dimension. When use_image_spacing is True (default) the values are
        converted to voxel units by dividing by the image spacing and rounding to integers before use.
        For morphological operations the values are interpreted as structuring element radii/size (in voxels).
        For smoothing ('Smoothing_all_axis' / 'Smoothing_z') the values are passed to SimpleITK.DiscreteGaussian
        (interpreted as variance(s) per axis by that filter).
    method : {'Closing', 'ClosingReconstruction', 'Opening', 'OpeningReconstruction',
              'Smoothing_all_axis', 'Smoothing_z'} or None, optional
        Post-processing operation to apply:
        - 'Closing' : sitk.BinaryMorphologicalClosing with a ball structuring element.
        - 'ClosingReconstruction' : sitk.BinaryClosingByReconstruction.
        - 'Opening' : sitk.BinaryMorphologicalOpening.
        - 'OpeningReconstruction' : sitk.BinaryOpeningByReconstruction.
        - 'Smoothing_all_axis' : apply sitk.DiscreteGaussian to the whole image (cast to Float32),
                               then threshold at 0.5 to restore a binary image.
        - 'Smoothing_z' : apply sitk.DiscreteGaussian only along the last image axis (zero variance on other axes),
                          then threshold at 0.5.
        - None : skip morphological/smoothing operations.
    use_image_spacing : bool, optional
        If True (default) kernel_size values are interpreted in physical millimeters and converted to voxel
        units using the image spacing. If False kernel_size is treated as already in voxel units.
    select_biggest : bool, optional
        If True, after other processing only the largest connected component (by voxel count) is kept.
        The background label (0) is ignored when computing component sizes.

    Returns
    -------
    segmented_binary_image: SimpleITK.Image
        A binary SimpleITK image after the requested post-processing. The returned image keeps image
        metadata (origin/spacing/direction) as maintained by the SimpleITK operations.
    """
    if isinstance(kernel_size, (float, int)):
        kernel_size = [kernel_size for _ in range(segmented_binary_image.GetDimension())]
    if use_image_spacing:
        kernel_size = np.round(kernel_size / np.array(segmented_binary_image.GetSpacing()))
    if method is not None:
        if method == 'Closing':
            segmented_binary_image = sitk.BinaryMorphologicalClosing(segmented_binary_image, [int(_) for _ in kernel_size], sitk.sitkBall)
        elif method == 'ClosingReconstruction':
            segmented_binary_image = sitk.BinaryClosingByReconstruction(segmented_binary_image, [int(_) for _ in kernel_size], sitk.sitkBall)
        elif method == 'Opening':
            segmented_binary_image = sitk.BinaryMorphologicalOpening(segmented_binary_image, [int(_) for _ in kernel_size], sitk.sitkBall)
        elif method == 'OpeningReconstruction':
            segmented_binary_image = sitk.BinaryOpeningByReconstruction(segmented_binary_image, [int(_) for _ in kernel_size], sitk.sitkBall)
        elif method == 'Smoothing_all_axis':
            # in this case kernel_size must be the variance of the gaussian kernel, take your disposition
            smoothed_image = sitk.DiscreteGaussian(sitk.Cast(segmented_binary_image, sitk.sitkFloat32), kernel_size, useImageSpacing=False)
            segmented_binary_image = smoothed_image > 0.5
        elif method == 'Smoothing_z':
            # in this case kernel_size must be the variance of the gaussian kernel, take your disposition
            smoothed_image = sitk.DiscreteGaussian(sitk.Cast(segmented_binary_image, sitk.sitkFloat32), 
                                                   [0] * (len(kernel_size) - 1) + [kernel_size[-1]], useImageSpacing=False)
            segmented_binary_image = smoothed_image > 0.5
        else:
            raise ValueError
    if select_biggest:
        # Only keep biggest part
        cc = sitk.ConnectedComponent(segmented_binary_image)
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(segmented_binary_image, cc)
        label_size = {label_stats.GetCount(_): _ for _ in label_stats.GetLabels() if _ != 0}
        if label_size:
            segmented_binary_image = cc == label_size[max(label_size.keys())]
    return segmented_binary_image


def absolute_threshold_segmentation(image, box, threshold):
    """
    Segment an image within a provided mask by applying an absolute intensity threshold.

    Parameters
    ----------
    image : SimpleITK.Image
        The input image to be segmented.
    box : SimpleITK.Image
        A mask image (same size/geometry as `image`) that defines the region of interest.
        The mask will be cast to the pixel type of `image` before multiplying, so it
        should typically be a binary mask (0 for outside, 1 for inside).
    threshold : numbers.Real
        The intensity threshold. Voxels with intensity greater than or equal to this
        value (after masking) are considered foreground.

    Returns
    -------
    segmented_binary_image: SimpleITK.Image
        A binary SimpleITK image representing the segmentation result. Voxels inside
        the boxed region with intensity >= threshold are set to True (or 1); voxels
        outside the boxed region or below the threshold are False (or 0).
    """
    boxed_image = image * sitk.Cast(box, image.GetPixelID())
    segmented_binary_image = boxed_image >= threshold
    return segmented_binary_image


def tumor_TBR_segmentation(image, box, ref_region, threshold_factor=1.6):
    """
    Segment a tumor in a PET image by thresholding based on target-to-background ratio (TBR).

    This function computes a global threshold as threshold_factor times the mean uptake
    within a provided reference region (label value 1 in ref_region) and then calls
    absolute_threshold_segmentation(...) to produce a binary segmentation within the
    provided bounding box.

    Parameters
    ----------
    image : SimpleITK.Image
        The PET image (grayscale) to be segmented.
    box : SimpleITK.Image
        Region-of-interest specification forwarded to absolute_threshold_segmentation.
        The exact expected format is the same as accepted by absolute_threshold_segmentation
        in this codebase (e.g., a tuple of indices/sizes or a set of slices). This function
        does not modify the box; it is used to restrict where the thresholding is applied.
    ref_region : SimpleITK.Image
        Binary or label image used as the reference/background region for estimating
        mean uptake. The function expects the reference region to contain voxels labeled
        with the integer 1 (i.e., label value 1 is used by sitk.LabelStatisticsImageFilter).
    threshold_factor : float, optional
        Multiplier applied to the mean uptake in ref_region to compute the segmentation
        threshold. Default is 1.6.

    Returns
    -------
    seg: SimpleITK.Image
        A binary segmentation image (same spatial metadata as the input image) produced
        by absolute_threshold_segmentation using the computed threshold.
    threshold : float
        The numeric intensity threshold used to generate seg (threshold_factor *
        mean(ref_region voxels with label 1)).
    """
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(image, ref_region)
    threshold = threshold_factor * label_stats.GetMean(1)
    seg = absolute_threshold_segmentation(image, box, threshold)
    return seg, threshold


def check_missing_instances(series_path_list):
    """
    Check for missing DICOM instances (slice numbers) within a series.

    This function reads the DICOM files specified by series_path_list, extracts each
    file's InstanceNumber, and returns any missing instance numbers in the contiguous
    range between the minimum and maximum InstanceNumber found.

    Args:
        series_path_list (Sequence[str] | Sequence[pathlib.Path]):
            Iterable of file paths (strings or Path objects) pointing to DICOM files
            that belong to the same series. Files may be in any order.

    Returns:
        instance_missing: List[int]:
            A list of integer instance numbers that are missing inside the detected
            range [min(InstanceNumber), max(InstanceNumber)]. The list is empty when
            no missing instances are found.
    """
    ds_list = [dicom.read_file(_) for _ in series_path_list]
    instance_numbers = [int(ds.InstanceNumber) for ds in ds_list]
    # With this method we can't detect if first/last slice is missing. Not that important since we can't correct it
    instance_missing = [_ for _ in range(min(instance_numbers), max(instance_numbers) + 1) if not _ in instance_numbers]
    if instance_missing:
        warn('Missing slice detected inside the volume: %s' % instance_missing)
    return instance_missing


def manually_construct_image(series_path_list):
    """
    Construct a SimpleITK Image volume from a list of DICOM file paths, preserving
    image geometry and interpolating any missing slices.

    Parameters
    ----------
    series_path_list : Sequence[str]
        Iterable of file paths pointing to DICOM slice files that belong to the
        same series. Each file is expected to contain the following DICOM tags:
        - InstanceNumber (used for ordering)
        - PixelSpacing (2 values: row, column)
        - ImagePositionPatient (at least 3 values; z coordinate used to compute z spacing)
        - ImageOrientationPatient (6 values: direction cosines for X and Y)
        - Rows, Columns, NumberOfSlices (Rows and Columns are used for size;
          NumberOfSlices is used as the z size if present)

    Returns
    -------
    new_image: sitk.Image
        A SimpleITK Image with dimensions (Rows, Columns, NumberOfSlices) where
        the spacing, origin and direction are set from the DICOM metadata:
        - spacing = (PixelSpacing[0], PixelSpacing[1], computed_z_spacing)
        - origin = ImagePositionPatient of the reference slice (the slice that
          corresponds to the first InstanceNumber in the provided files)
        - direction = tuple(ImageOrientationPatient[0:3] + computed_z_direction)
        Slices present in the input are copied into their corresponding z positions
        (z index determined by InstanceNumber after normalizing by the minimum
        InstanceNumber). Missing slices are reconstructed by a simple linear
        interpolation strategy.
    """
    ds_list = [dicom.read_file(_) for _ in series_path_list]
    instance_dic = {int(ds.InstanceNumber): series_path_list[i] for i, ds in enumerate(ds_list)}
    first_key = list(instance_dic.keys())[0]
    ref_image = sitk.ReadImage(instance_dic[first_key])

    x_y_spacing = tuple(float(_) for _ in ds_list[0].PixelSpacing)
    z_spacing = abs(float(ds_list[0].ImagePositionPatient[2]) - float(ds_list[1].ImagePositionPatient[2]))
    image_size = (ds_list[0].Rows, ds_list[0].Columns, ds_list[0].NumberOfSlices)
    x_y_orientation = [float(_) for _ in ds_list[0].ImageOrientationPatient]
    z_orientation = np.cross(x_y_orientation[:len(x_y_orientation) // 2], x_y_orientation[len(x_y_orientation) // 2:])

    new_image = sitk.Image(image_size, ref_image.GetPixelID())
    new_image.SetSpacing(x_y_spacing + (z_spacing,))
    new_image.SetOrigin(ds_list[series_path_list.index(instance_dic[first_key])].ImagePositionPatient)
    new_image.SetDirection(tuple(x_y_orientation) + tuple(z_orientation))

    # First fill with slices that are not missing
    for i, index in enumerate(range(min(instance_dic.keys()), max(instance_dic.keys()) + 1)):
        if index in instance_dic:
            new_image[:, :, i] = sitk.ReadImage(instance_dic[index])
    # Then take care of missing slices
    for i, index in enumerate(range(min(instance_dic.keys()), max(instance_dic.keys()) + 1)):
        if index not in instance_dic:
            # We extract a volume with 2 x z_spacing then interpolate back to true spacing to get the interpolated values at missing slices
            if i % 2 == 1:
                # Previous and next slice are even numbers and will be kept when taking only one slice out of two
                new_image_twice_spacing = new_image[:, :, ::2]
            else:
                # Previous and next slice are odd numbers and will not be kept when taking only one slice out of two. So we take one slice out of 2 starting from the second slice (index 1)
                new_image_twice_spacing = new_image[:, :, 1::2]
            new_image_res = sitk.Resample(new_image_twice_spacing, new_image, sitk.Transform(), sitk.sitkLinear)
            new_image[:, :, i] = new_image_res[:, :, i]
    return new_image


def get_decay_correction_time(ds):
    """
    Extract and return the decay correction timestamp from a pydicom Dataset.

    This function looks for the DecayCorrectionDateTime tag (0x018,0x9701) in the
    provided DICOM dataset and, if present, parses it into a pandas.Timestamp.
    If that tag is absent, the function falls back to using SeriesDate and
    SeriesTime to construct the timestamp.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        A DICOM dataset expected to contain either:
          - DecayCorrectionDateTime: string in the form 'YYYYMMDDHHMMSS' (may
            include fractional seconds after a '.'), or
          - SeriesDate: string in the form 'YYYYMMDD' and
            SeriesTime: string in the form 'HHMMSS' (may include fractional
            seconds after a '.').
        The function inspects the presence of the DecayCorrectionDateTime tag
        using the tag tuple (0x018, 0x9701).

    Returns
    -------
    decay_correction_date_time: pandas.Timestamp
        A timezone-naive pandas.Timestamp representing the decay correction
        datetime. The conversion uses a day-first interpretation (DD/MM/YYYY)
        to match the original formatting used before calling pandas.to_datetime.
    """
    if (0x018, 0x9701) in ds:
        decay_correction_date_time = ds.DecayCorrectionDateTime.split('.')[0]
        decay_correction_date_time = datetime.strptime(decay_correction_date_time, '%Y%m%d%H%M%S')
        decay_correction_date_time = pd.to_datetime('%s' % decay_correction_date_time.strftime('%d/%m/%Y %H:%M:%S'), dayfirst=True)
    else:
        # Series date and time
        series_date = datetime.strptime(ds.SeriesDate, '%Y%m%d')
        # Series time in s from midnigth
        series_time = datetime.strptime(ds.SeriesTime.split('.')[0], '%H%M%S')
        decay_correction_date_time = pd.to_datetime('%s %s' % (series_date.strftime('%d/%m/%Y'),
                                                               series_time.strftime('%H:%M:%S')), dayfirst=True)
    return decay_correction_date_time


def get_injection_time(ds):
    """
    Return the radiopharmaceutical injection datetime parsed from a DICOM dataset.

    This function extracts and parses the radiopharmaceutical injection time from the
    provided DICOM dataset `ds`. It supports two input encodings commonly found in
    PET DICOMs:

    - If the RadiopharmaceuticalInformationSequence item contains a combined
        RadiopharmaceuticalStartDateTime (format "YYYYMMDDHHMMSS[.fraction]"), that
        value is parsed (fractional seconds are discarded).
    - Otherwise, the function falls back to using SeriesDate ("YYYYMMDD") together
        with RadiopharmaceuticalStartTime ("HHMMSS[.fraction]") from the same sequence
        item to build the full datetime.

    The returned value is a pandas.Timestamp created with pd.to_datetime and uses
    day-first parsing for the intermediate string formatting performed by this
    function.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset-like
            DICOM dataset expected to contain `RadiopharmaceuticalInformationSequence`
            (index 0) and either `RadiopharmaceuticalStartDateTime` or both
            `RadiopharmaceuticalStartTime` and `SeriesDate`. Typically this is a
            pydicom FileDataset.

    Returns
    -------
    injection_date_time: pandas.Timestamp
            A timezone-naive timestamp representing the injection datetime (seconds
            precision; fractional seconds removed).
    """
    # Injection time in s from midnight
    if (0x018, 0x1078) in ds.RadiopharmaceuticalInformationSequence[0]:
        injection_date_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime.split('.')[0]
        injection_date_time = datetime.strptime(injection_date_time, '%Y%m%d%H%M%S')
        injection_date_time = pd.to_datetime('%s' % injection_date_time.strftime('%d/%m/%Y %H:%M:%S'), dayfirst=True)
    else:
        injection_date = datetime.strptime(ds.SeriesDate, '%Y%m%d')
        injection_time = datetime.strptime(
            ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime.split('.')[0], '%H%M%S')
        injection_date_time = pd.to_datetime('%s %s' % (injection_date.strftime('%d/%m/%Y'),
                                                        injection_time.strftime('%H:%M:%S')), dayfirst=True)
    return injection_date_time


def bqml_to_SUV(current_image, ds):
    """
    Convert a PET image from BQML (Bq/mL) to SUV (body-weight normalized).

    Parameters
    ----------
    current_image : SimpleITK.Image
        PET image as a SimpleITK Image containing activity concentration in Bq/mL.
    ds : pydicom.dataset.Dataset
        DICOM dataset that must contain the following attributes/values:
          - Units == 'BQML'
          - CorrectedImage contain both 'ATTN' and 'DECY'
          - DecayCorrection == 'START'
          - RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife (seconds)
          - RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose (Bq)
          - PatientWeight (kg)
        Additionally, get_decay_correction_time(ds) and get_injection_time(ds) must be available and return datetimes.

    Returns
    -------
    current_image: SimpleITK.Image
        The same SimpleITK Image object as current_image, scaled to SUVbw (unitless).
        The operation is performed in-place and the modified image is returned.
    """
    # Check everything is ok before correction
    if getattr(ds, "Units", None) != 'BQML':
        raise ValueError("Unsupported Units=%s; expected 'BQML'." % getattr(ds, "Units", None))
    corrected = set(getattr(ds, "CorrectedImage", []))
    if not {"ATTN", "DECY"}.issubset(corrected):
        raise ValueError("CorrectedImage must contain ATTN and DECY; got %s" % sorted(corrected))
    if getattr(ds, "DecayCorrection", None) != 'START':
        raise ValueError("DecayCorrection must be 'START'; got %s" % getattr(ds, "DecayCorrection", None))
    decay_correction_date_time = get_decay_correction_time(ds)
    patient_weight = 1000 * float(ds.PatientWeight)  # Weight in g
    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)  # Half life in s
    activity_injected = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)  # Activity inj in Bq for PET
    injection_date_time = get_injection_time(ds)
    Lambda = np.log(2) / half_life
    SUV_decay_time = (decay_correction_date_time - injection_date_time).total_seconds()
    decayed_dose = activity_injected * np.exp(-Lambda * SUV_decay_time)
    SUVbw_scale_factor = patient_weight / decayed_dose
    current_image *= SUVbw_scale_factor

    return current_image

def read_dicom_image(dicom_dir=None, series_img_list=None):
    """
    Read a DICOM series and return a SimpleITK image converted to SUV for PET.

    Parameters
    ----------
    dicom_dir : str, optional
        Path to directory containing the DICOM series. Used to obtain file list when
        series_img_list is not provided.
    series_img_list : list of str, optional
        Explicit ordered list of file paths (DICOM instances) that belong to the series.
        If provided, this list is used directly (it will be natsorted).

    Returns
    -------
    image : SimpleITK.Image
        The loaded image (reoriented to 'LPS'). If modality is PET ('PT'), the image
        is converted to SUV via bqml_to_SUV before being returned.
    missing_instances : list[int]
        List of instance numbers detected as missing inside the contiguous range.
    """
    if series_img_list is None:
        if dicom_dir is None:
            raise ValueError("Either dicom_dir or series_img_list must be provided")
        series_reader = sitk.ImageSeriesReader()
        series_img_list = series_reader.GetGDCMSeriesFileNames(dicom_dir)

    missing_instances = check_missing_instances(series_img_list)

    if missing_instances:
        image = manually_construct_image(series_img_list)
    else:
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_img_list)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        image = series_reader.Execute()

    image = sitk.DICOMOrient(image, 'LPS')  # Ensure orientation is in DICOM standard

    # Read first dataset to inspect modality and other tags
    ds = dicom.read_file(series_img_list[0])
    modality = getattr(ds, "Modality", None)
    if modality == 'PT':
        image = bqml_to_SUV(image, ds)
    else:
        raise ValueError("Modality=%s; expected 'PT'" % modality)

    return image, missing_instances


def load_rois(roi_dir):
    """
    Load ROI (region-of-interest) images from a directory into a dictionary of SimpleITK images.

    This function searches the given directory for files ending with '.nii', reads each file
    with SimpleITK, ensures the image is cast to unsigned 8-bit integer, enforces a binary
    mask where appropriate, and orients the image to DICOM LPS orientation. The returned
    dictionary keys are the file basenames without the '.nii' extension.

    Parameters
    ----------
    roi_dir: str
        Path to the directory containing ROI NIfTI files (.nii).

    Returns
    -------
    ROI_dic: dict
        A dictionary mapping ROI base filenames (str, filename without '.nii')
        to SimpleITK Image objects (sitk.Image) that are uint8 and oriented to LPS.
    """
    ROI_dic = {}
    label_shape = sitk.LabelShapeStatisticsImageFilter()
    for roi_file in os.listdir(roi_dir):
        if '.nii' in roi_file:
            roi_name = roi_file.split('.nii')[0]
            ROI_dic[roi_name] = sitk.Cast(sitk.ReadImage(os.path.join(roi_dir, roi_file)), sitk.sitkUInt8)
            label_shape.Execute(ROI_dic[roi_name])
            if len(label_shape.GetLabels()) == 1:
                if 1 not in label_shape.GetLabels():
                    ROI_dic[roi_name] = ROI_dic[roi_name] > 0
                    label_shape.Execute(ROI_dic[roi_name])
            else:
                ValueError('Several labels were detected in the mask. Each label must be in its own image.')
            ROI_dic[roi_name] = sitk.DICOMOrient(ROI_dic[roi_name], 'LPS') # Ensure orientation is in DICOM standard
    return ROI_dic


def correct_origin(image1, image2, tol=1e-3):
    """
    Ensure image2 has the same spatial metadata as image1 when their origins are sufficiently close.

    This function compares the origin coordinates of two image-like objects component-wise and,
    if any component difference is within the given tolerance, copies spatial metadata from image1
    to image2 by calling image2.CopyInformation(image1). The function always returns image2
    (which may have been modified in-place).

    Parameters
    ----------
    image1 : SimpleITK.Image
        Source image from which spatial information will be copied. Must implement
        GetOrigin() -> sequence, GetDimension() -> int, and provide metadata consumable by
        image2.CopyInformation(image1) (for example, a SimpleITK Image).
    image2 : SimpleITK.Image
        Target image that may be updated. Must implement GetOrigin(), GetDimension(), and
        CopyInformation(source_image).
    tol : float, optional
        Tolerance for comparing origin coordinates (default: 1e-3). For each axis i, the
        absolute difference |image1.GetOrigin()[i] - image2.GetOrigin()[i]| is compared
        to this tolerance. If any axis difference is <= tol, metadata is copied.

    Returns
    -------
    image2: SimpleITK.Image
        The (possibly modified) image2. The operation is performed in-place on image2
        via its CopyInformation method.
    """
    if np.any(np.abs([image1.GetOrigin()[_] - image2.GetOrigin()[_] for _ in range(image1.GetDimension())]) <= tol):
        image2.CopyInformation(image1)
    return image2


def interpolate_roi_missing_slices(ref_image, associated_ROI, missing_instances):
    """
    Interpolate missing axial slices in an ROI volume to match a reference image volume.
    This function constructs a new ROI image with the same geometry as ref_image by mapping
    slices from an associated_ROI that has one or more axial slices missing. Missing slices
    are identified by the provided missing_instances list (1-based indices) and are filled
    by simple interpolation along the axial direction using a temporary volume sampled at
    twice the axial spacing and resampled back with label-aware linear interpolation.

    Parameters
    ----------
    ref_image : SimpleITK.Image
            Reference image whose geometry (size, spacing, origin, direction) defines the
            target volume for the returned ROI. The function iterates over axial slices using
            ref_image.GetDepth() and expects axial indexing compatible with slicing
            new_roi[:, :, i].
    associated_ROI : SimpleITK.Image
            ROI image that corresponds to the same volume as ref_image but with some axial
            slices missing (shifted indices). Pixel type/label coding is preserved in the
            returned image. The function assumes associated_ROI has at least as many slices as
            ref_image minus the number of missing instances, and that axial ordering matches.
    missing_instances : iterable of int
            Sequence of 1-based slice indices (relative to ref_image) that are missing from
            associated_ROI. For example, if slice number 100 in ref_image is missing in the
            associated_ROI, include 100 in this list. The implementation converts these to
            zero-based indices internally.
    Returns
    -------
    new_roi: SimpleITK.Image
            A new ROI image with the same size and spatial metadata as ref_image, where
            originally-present slices are copied from associated_ROI and missing slices are
            filled by interpolation. The output pixel type matches associated_ROI.GetPixelID().
    """
    new_roi = sitk.Image(ref_image.GetSize(), associated_ROI.GetPixelID())
    new_roi.CopyInformation(ref_image)
    missing_slices = [n_instance - 1 for n_instance in missing_instances]

    # Since slices are missing, for each missing slice the slices index are shifted by one
    # For example, if slice 100 is missing then true slice 101 is at index 100 in original volume, 101 at 102 etc.
    # It is incremented for each missing slice thus we compute a correspondence dic with form {true_index: index_with_missing_slices}
    slice_correspondence_dic = {}
    increment = 0
    for i in range(ref_image.GetDepth()):
        if i not in missing_slices:
            slice_correspondence_dic[i] = i - increment
        else:
            slice_correspondence_dic[i] = None
            increment += 1

    # First fill with slices that are not missing
    for i in range(ref_image.GetDepth()):
        if i not in missing_slices:
            new_roi[:, :, i] = associated_ROI[:, :, slice_correspondence_dic[i]]
    # Then take care of missing slices
    for i in range(ref_image.GetDepth()):
        if i in missing_slices:
            # We extract a volume with 2 x z_spacing then interpolate back to true spacing to get the interpolated values at missing slices
            if i % 2 == 1:
                # Previous and next slice are even numbers and will be kept when taking only one slice out of two
                new_roi_twice_spacing = new_roi[:, :, ::2]
            else:
                # Previous and next slice are odd numbers and will not be kept when taking only one slice out of two. So we take one slice out of 2 starting from the second slice (index 1)
                new_roi_twice_spacing = new_roi[:, :, 1::2]
            new_roi_res = sitk.Resample(new_roi_twice_spacing, new_roi, sitk.Transform(), sitk.sitkLabelLinear)
            new_roi[:, :, i] = new_roi_res[:, :, i]
    
    return new_roi


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate nn-Unet database')
    parser.add_argument("--input_dir", help='Root directory for input image and ROI dataset', type=str, required=True)
    parser.add_argument("--output_dir", help='Root directory for output image and ROI dataset', type=str, required=True)
    parser.add_argument("--data_info_file", help='file with information on population', type=str, default=None)
    parser.add_argument('--postprocessROI', help='Apply postprocessing on ROI generated semi-automatically', 
                        action='store_true', default=False)
    parser.add_argument('--skullstripping', help='Perform skullstripping', 
                        action='store_true', default=False)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    data_info_file_path = os.path.abspath(args.data_info_file)
    
    patients_list = [patient for patient in natsorted(os.listdir(input_dir)) 
                     if os.path.exists(os.path.join(input_dir, patient, 'Static_PET')) and 
                     os.path.exists(os.path.join(input_dir, patient, 'ROI'))]

    if '.xlsx' in args.data_info_file:
        data_info_file = pd.read_excel(data_info_file_path, index_col=0, dtype={'ID': str})
    elif '.csv' in args.data_info_file:
        data_info_file = pd.read_csv(data_info_file_path, index_col=0, dtype={'ID': str})
    else:
        data_info_file = pd.DataFrame({'ID': patients_list}).set_index('ID')
    
    if args.skullstripping:
        # fix threading correctly
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"          # avoid conflict with libgomp
        env.pop("MKL_SERVICE_FORCE_INTEL", None)    # just in case this is set somewhere
        # find models path
        mri_synthstrip_candidate_dirs = list(set([folder for folder in os.environ["PATH"].split(os.pathsep) if 'mri_synthstrip' in folder]))
        if len(mri_synthstrip_candidate_dirs) == 1:
            # model with csf in brain boundary, otherwise model is synthstrip.nocsf.1.pt
            model_path = os.path.join(mri_synthstrip_candidate_dirs[0], 'synthstrip.1.pt')
            if not os.path.isfile(model_path):
                raise FileNotFoundError("model not found in its supposed place %s" % model_path)
            # check than we can execute the script
            script_path = os.path.join(mri_synthstrip_candidate_dirs[0], 'mri_synthstrip')
            if not os.path.isfile(script_path):
                raise FileNotFoundError("mri_synthstrip script not found in its supposed place %s" % script_path)
            else:
                if not os.access(script_path, os.X_OK):
                    raise RuntimeError('mri_synthstrip script can not be executed (permission problem).')
        else:
            if not mri_synthstrip_candidate_dirs:
                raise FileNotFoundError("mri_synthstrip dir not found")
            else:
                raise RuntimeError("several entries found in PATH for mri_synthstrip. Ensure there is only one.")
    
    lesion_info = {'Seg_detection': {}, 'Seg_volume_mL': {}, 'Threshold_seg': {}}
    for patient in patients_list:
        print(patient)
        static_image, missing_static_instances = read_dicom_image(dicom_dir=os.path.join(input_dir, patient, 'Static_PET'))
        roi_dic = load_rois(os.path.join(input_dir, patient, 'ROI'))
        # If missing slices in image also apply in ROI
        if missing_static_instances:
            for roi in roi_dic:
                if roi_dic[roi].GetSize()[-1] != static_image.GetSize()[-1]:
                    roi_dic[roi] = interpolate_roi_missing_slices(static_image, roi_dic[roi], missing_static_instances)
        # Correct the case where loading saved image results in different origin (approx 1e-5 mm differences)
        for roi in roi_dic:
            roi_dic[roi] = correct_origin(static_image, roi_dic[roi], tol=1e-2)
        if 'TumorBox' in roi_dic:
            label_shape = sitk.LabelShapeStatisticsImageFilter()
            threshold_factor = 1.6
            roi, threshold = tumor_TBR_segmentation(static_image, roi_dic['TumorBox'], roi_dic['Brain'], threshold_factor=threshold_factor)
            if args.postprocessROI:
                # Resample to 1mm isotropic
                _, roi_resampled = IBSI_resampling(static_image, roi, resampledPixelSpacing=(1 ,1, 1))
                # Opening by Reconstruction with 1mm isotropic kernel
                processed_roi_resampled = segmentation_post_treatment(roi_resampled, [1, 1, 1], method='OpeningReconstruction', use_image_spacing=False, select_biggest=False)
                # Resample back to original spacing
                processed_roi = sitk.Resample(processed_roi_resampled, roi, sitk.Transform(), sitk.sitkLabelLinear)
                label_shape.Execute(processed_roi)
                if 1 in label_shape.GetLabels():
                    # Accept processed roi only if we still have a volume after
                    roi = processed_roi
            roi_dic['Tumor_TBR%s_seg' % str(threshold_factor).replace('.', ',')] = roi
            label_shape.Execute(roi_dic['TumorBox'])
            if 1 not in label_shape.GetLabels():
                lesion_type = "no_measurable"
                label_shape.Execute(roi_dic['Tumor_TBR%s_seg' % str(threshold_factor).replace('.', ',')])
                if 1 in label_shape.GetLabels():
                    lesion_vol = label_shape.GetPhysicalSize(1) / 1000 # cm3/mL
                    raise ValueError("Lesion vol should be 0 when TumorBox vol is 0. Got %.3f" % lesion_vol)
                else:
                    lesion_vol = 0
            else:
                label_shape.Execute(roi_dic['Tumor_TBR%s_seg' % str(threshold_factor).replace('.', ',')])
                if 1 not in label_shape.GetLabels():
                    lesion_vol = 0
                else:
                    lesion_vol = label_shape.GetPhysicalSize(1) / 1000 # cm3/mL
                if lesion_vol <= 0.5:
                    lesion_type = "non_measurable"
                else:
                    lesion_type = "measurable"
            lesion_info['Seg_detection'][patient] = lesion_type
            lesion_info['Seg_volume_mL'][patient] = lesion_vol
            lesion_info['Threshold_seg'][patient] = threshold
        # Save all
        os.makedirs(os.path.join(output_dir, patient, 'ROI'), exist_ok=True)
        sitk.WriteImage(static_image, os.path.join(output_dir, patient, 'Static_PET.nii.gz'))
        roi_path = {}
        for roi in roi_dic:
            sitk.WriteImage(sitk.Cast(roi_dic[roi], sitk.sitkUInt8), os.path.join(output_dir, patient, 'ROI', '%s.nii.gz' % roi))
            roi_path[roi] = os.path.join(output_dir, patient, 'ROI', '%s.nii.gz' % roi)

        # Run skull stripping using SynthStrip https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
        if args.skullstripping:
            input_image_path = os.path.join(output_dir, patient, 'Static_PET.nii.gz')
            output_image_path = os.path.join(output_dir, patient, 'Static_PET_skullstripped.nii.gz')
            output_mask_path = os.path.join(output_dir, patient, 'Static_PET_skullstripped_mask.nii.gz')
            subprocess.run(['mri_synthstrip', '-i', input_image_path, '-o', output_image_path, '-m', output_mask_path,
                            '--model', model_path], check=True, env=env)
            # Ensure that values are equal to 0 outside of skullstrip mask. Sometimes we add small values
            skullstrip_mask = sitk.ReadImage(os.path.join(output_dir, patient, 'Static_PET_skullstripped_mask.nii.gz'))
            skullstrip_image = sitk.ReadImage(os.path.join(output_dir, patient, 'Static_PET_skullstripped.nii.gz'))
            sitk.WriteImage(sitk.Mask(skullstrip_image, skullstrip_mask), os.path.join(output_dir, patient, 'Static_PET_skullstripped.nii.gz'))

    # add info about no-measurable/non-measurable/measurable lesion + Seg volume in data_info file
    for col in lesion_info:
        data_info_file[col] = pd.Series(lesion_info[col])
    if '.xlsx' in args.data_info_file:
        data_info_file.to_excel(os.path.join(output_dir, os.path.basename(data_info_file_path)))
    elif '.csv' in args.data_info_file:
        data_info_file.to_csv(os.path.join(output_dir, os.path.basename(data_info_file_path)))
    else:
        raise ValueError