import os
from Dataloader import Dataloader

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from joblib import Parallel,delayed
dataloader = Dataloader("/jhcnas1/zhoutaichang/Duke")
import numpy as np
import pandas as pd
import torch
# from Dataloader import Dataloader
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
# from Datasaver import save
import matplotlib.pyplot as plt
import pydicom as dicom
# import cv2
data_len=len(dataloader.patient_id)
patients= np.arange(data_len)

def r(a:np.array):
    a=(255*(a-np.min(a))/(np.max(a)-np.min(a))).astype(np.uint8)
    return a
def min_max(a:np.array):
    if np.min(a) == np.max(a):
        # print("Has 0")
        return np.zeros(a.shape).astype(np.float32)
    scaled_image = (np.maximum(a, 0) / a.max()) * 255.0

    return scaled_image

def load_scan_pixel(path):
    try:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        ss=[]
        for s in slices:
            ss.append(s.pixel_array)

        return np.array(ss)
    except:
        print('\n\n\n\nError:\t'+path+'\n\n\n\n')


# save(x-1,'D:/Duke_resized/resized_order/')
# t=load_scan_pixel(''
#                   'C:/Users/zhout/Desktop/Duke/Breast_MRI_400/502.000000-Ph2ax 3d dyn MP-04034'
#                   '')
# d=t[60,271:344,164:239]
#
#
# # 251:294,108:136
# d=zoom(d,(100/d.shape[0],100/d.shape[1]))
# # [59,114:187,303:379]
# # d=cv2.resize(d,(100,100))
# scaled_image = (np.maximum(d, 0) / d.max()) * 255.0
# plt.imshow(scaled_image,cmap='gray', vmin=0, vmax=255)
# plt.title('Original')
# # plt.show()
# plt.savefig('/jhcnas1/zhoutaichang/'+str(x)+'_orignial.png')
# exit()
root='/jhcnas1/zhoutaichang/'
if not os.path.isdir(root+'visual/'):
    os.mkdir(root+'visual/')
def show(x):
    # print(x)
    try:
        patient=root+'visual/Breast_MRI_'+str(x).rjust(3,'0')+'/'
        if not os.path.isdir(patient):
            os.mkdir(patient)
        d1,di,ck,ty=dataloader.load_patient(x-1)
        d=d1[2][ck,...]
        scaled_image = (np.maximum(d, 0) / d.max()) * 255.0

        plt.imshow(scaled_image,cmap='gray', vmin=0, vmax=255)
        plt.title('Original')
        plt.savefig(patient+'Original.png')

        im=zoom(d,(100/d.shape[0],100/d.shape[1]))

        # im=cv2.resize(im,(100,100))
        scaled_image = (np.maximum(im, 0) / im.max()) * 255.0

        plt.imshow(scaled_image,cmap='gray', vmin=0, vmax=255)
        plt.title('Dataloader')
        plt.savefig(patient+'dataloader.png')

        # plt.show()
        #
        a=torch.load(root+'/original/Breast_MRI_'+str(x).rjust(3,'0')+'.pt').numpy()[2][50]
        scaled_image2 = (np.maximum(a, 0) / a.max()) * 255.0

        plt.imshow(scaled_image2,cmap='gray', vmin=0, vmax=255)
        # print((scaled_image==scaled_image2).all())
        plt.title("saver")
        plt.savefig(patient+'datasaver.png')

        # plt.show()
        a=torch.load(root+'/original/Breast_MRI_'+str(x).rjust(3,'0')+'_i.pt').numpy()[2][50]
        scaled_image2 = (np.maximum(a, 0) / a.max()) * 255.0

        plt.imshow(scaled_image2,cmap='gray', vmin=0, vmax=255)
        plt.title("saver_i")
        plt.savefig(patient+'inverse.png')
    except:
        print(x)
        return


Parallel(n_jobs=40)(delayed(show)(x) for x in range(1,923))
# show(400)