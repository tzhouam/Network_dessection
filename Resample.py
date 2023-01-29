import os
# import yaml
import re
from typing import List
import pathlib
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import time
import pandas as pd
# sitk.SetArrayLayout(sitk.DEPTH_FIRST)
class Resample:
    def __init__(self,base):
        self.base=base
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(base)
        reader.SetFileNames(dicom_names)

        self.img=reader.Execute()
        self.original_size=self.img.GetSize()
        self.original_direc=get_volume_orientation(self.img)
        self.reorient=reorient(self.img,True)
        self.result=resample_volume(self.reorient,[0.8036,0.8036,1.0])
        self.result_size=self.result.GetSize()

def get_volume_orientation(volume: sitk.Image) -> str:
    orient_filter = sitk.DICOMOrientImageFilter()
    cosines = volume.GetDirection()
    return orient_filter.GetOrientationFromDirectionCosines(cosines)

def resample_volume(volume: sitk.Image,
                    new_spacing: List[float],
                    interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Change spacing
    :param volume: Input image to be resampled
    :param new_spacing: For our volumes this should be a 3-item list
                        i.e. (1.0, 1.0, 1.0)
    :param interpolator: Which interpolator to use. Preferred Linear,
                        for segmentation nearest neighbor
    """
    new_size = [int(round(osz*ospc/nspc))
                for osz, ospc, nspc
                in zip(volume.GetSize(), volume.GetSpacing(), new_spacing)]
    minimum_value = sitk.GetArrayFromImage(volume).min()
    return sitk.Resample(volume,
                         new_size,
                         sitk.Transform(),
                         interpolator,
                         volume.GetOrigin(),
                         new_spacing,
                         volume.GetDirection(),
                         int(minimum_value),
                         volume.GetPixelID())

def reorient(volume: sitk.Image,
                    verbose: bool = False) -> sitk.Image:
    """
    If volume is not in the LPS orientation, reorient to LPS
    """
    current_orientation = get_volume_orientation(volume)
    if current_orientation != "RPI":
        if verbose:
            # print(f"Reorienting {current_orientation} to LPS")
            reorient_time = time.time()
        orientation_filter = sitk.DICOMOrientImageFilter()
        orientation_filter.SetDesiredCoordinateOrientation("RPI")
        volume = orientation_filter.Execute(volume)
        if verbose:
            reorient_time_end = time.time()
            # print("Reotienting time:", reorient_time_end - reorient_time)
    return volume