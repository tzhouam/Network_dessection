import dicom2nifti
base='/jhcnas1/maxiaoqi/TEST/MRI_images/Duke-Breast-Cancer-MRI'
sb='/jhcnas1/zhoutaichang/data/'
from Patient import Patient
from joblib import delayed,Parallel
import os
import sys
def k(i):
    pname='Breast_MRI_'+str(i).rjust(3,'0')
    p=Patient(base+'/'+pname,pname,i)
    l=[p.dyn_pre,p.dyn_1,p.dyn_2]
    if  not os.path.isdir(sb+pname):
        os.mkdir(sb+pname)
    dicom2nifti.dicom_series_to_nifti(l[0][0],sb+pname+'/pre')
    dicom2nifti.dicom_series_to_nifti(l[1][0],sb+pname+'/1st')
    dicom2nifti.dicom_series_to_nifti(l[2][0],sb+pname+'/2nd')
Parallel(n_jobs=-1,backend='threading')(delayed(k)(i) for i in range(1,923))


