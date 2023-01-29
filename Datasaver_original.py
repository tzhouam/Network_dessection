import sys

import torch
from Dataloader import Dataloader
import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from joblib import Parallel,delayed
dataloader = Dataloader()
p = np.random.permutation(dataloader.length)
train=p[:int(0.8*dataloader.length)]
test=p[int(0.8*dataloader.length):]
from Rotate import rotate
def rotation(data,dir,an):
    rm=[
        torch.FloatTensor([[1,0,0],[0,0,1],[0,1,0]]),
        torch.FloatTensor([ [0, 1, 0],[1, 0, 0], [0, 0, 1]]),
        torch.FloatTensor([[0, 1, 0], [0, 0, 1],[1, 0, 0]]),

    ]
    m=rm[dir]
    for _ in range(an):
        data=rotate(data,m)
    return data
import nibabel
# # 数据集分割
data_len = len(dataloader.patient_id)
train_len=int(0.8*data_len)
patients= np.arange(data_len)
# from Resample import resample_image
# annotation=pd.read_excel(io='Annotation_Boxes.xlsx').iloc[:,1:].to_numpy()
# print(annotation.shape)
# print(dataloader.patient_id[0])
# patients, annotation=unison_shuffled_copies(patients, annotation)
# for j in range(0,len(dataloader.patient_id)):
annotation=pd.read_excel(io='Annotation_Boxes.xlsx')

unsave=[]
def min_max(a:np.array):
    if np.min(a) == np.max(a):
        # print("Has 0")
        return np.zeros(a.shape).astype(np.float32)
    scaled_image = (np.maximum(a, 0) / a.max()) * 255.0

    return scaled_image
def patch(data,inverse,x_s,x_e,y_s,y_e,z_s,z_e):
    # return data
    x_s, x_e, y_s, y_e, z_s, z_e=x_s-1,x_e-1,y_s-1,y_e-1,z_s-1,z_e-1
    if inverse:
        temp=x_e
        x_e=data.shape[2]-x_s
        x_s=data.shape[2]-temp
        return data[z_s:z_e, y_s:y_e, x_s:x_e][:,:,::-1]

    return data[z_s:z_e,y_s:y_e,x_s:x_e]
def save(j,root='/jhcnas1/zhoutaichang/prototree_full_image/original/'):
    # if not os.path.isdir(root):
    #     os.mkdir(root)
    print(str(j)+' start')
    data_x,data_x_i,zm,ty=dataloader.load_patient(patients[j])
    d1,d2,d3=dataloader.load_direct(patients[j])
    if ty==-1:
        print("\n\n\n\n\nValue Error: " + str(j) + "\n\n\n\n\n")
        unsave.append(j)

        return
    type=d1

    # for i in range(len(annotation)):
    #     if int(annotation.iloc[i, 0][-3:]) == j:
    #         y_s = annotation.iloc[i, 1]
    #         y_e = annotation.iloc[i, 2]
    #         x_s = annotation.iloc[i, 3]
    #         x_e = annotation.iloc[i, 4]
    #         z_s = annotation.iloc[i, 5]
    #         z_e = annotation.iloc[i, 6]
    #
    # l=['pre','1st','2nd']
    # d=[]
    # for i in range(3):
    #     nz = nibabel.load('/jhcnas1/zhoutaichang/data/Breast_MRI_' + str(j).rjust(3, '0')+'/'+l[i]+'.nii')
    #
    #     data=resample_image(nz)
    #     data_x=data.get_fdata()
    #     d.append(data_x)
    # data=np.array(d)

    # sick=float(sick)
    # print(p.dyn_pre)
    # print(p.dyn_1)
    # print(p.dyn_2)
    try:

        data_x=np.array(data_x)

        # data_x=min_max(data_x)
        data_x_new=[]
        data_x_i_new=[]


        for channel in range(3):
            data_x_new.append(min_max(zoom(data_x[channel],(100/data_x[channel].shape[0],100/data_x[channel].shape[1],100/data_x[channel].shape[2]))))

        path=root
        if not os.path.isdir(path):
            os.mkdir(path)
    #     path=path+'/'+ dataloader.patient_id[j]
    #     path=root+ dataloader.patient_id[j]

        data_x_new=np.array(data_x_new).astype(np.float32)
        tensor_x=torch.from_numpy(data_x_new)

        path=path + dataloader.patient_id[j]

        torch.save(tensor_x, path +'.pt')
    except:
        print("\n\n\n\n\nError: "+str(j)+"\n\n\n\n\n")
        unsave.append(j)
        return


    # np.savez_compressed('/jhcnas1/zhoutaichang/resized/resized/'+dataloader.patient_id[j]+'.npz', data_x_new)

    print(str(j)+' Finish')
Parallel(n_jobs=-1,backend='threading')(delayed(save)(j) for j in range(0,len(dataloader.patient_id)))
# save(0)
# save(358,'')
unsave=pd.DataFrame(unsave)
unsave.to_csv("unsave.csv",index=False)
# save(130)
