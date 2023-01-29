from pydicom import dcmread
import pydicom as dicom
import os
import numpy as np
import pandas as pd
match=[]
import SimpleITK as sitk

annotation=pd.read_excel(io='Annotation_Boxes.xlsx')
from Resample import Resample
def load_scan(path):
    try:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices
    except:
        print('\n\n\n\nError:\t'+path+'\n\n\n\n')
def add_thick(path):
    return dicom.read_file(path + '/' + os.listdir(path)[0]).SliceThickness
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



type = {0: 'm1', 1: 'm2', 2: 'm3', 3: 'm4'}


class Patient:
    def patch(self,path, inverse, x_s, x_e, y_s, y_e, z_s, z_e):
        # return path

        # x_s, x_e, y_s, y_e, z_s, z_e=x_s-1,x_e-1,y_s-1,y_e-1,z_s-1,z_e-1
        x_m, y_m, z_m = x_e + x_s, y_e + y_s, z_e + z_s
        m = max(x_e - x_s, y_e - y_s, z_e - z_s) // 2
        x_s = x_m // 2 - m
        x_e = x_m // 2 + m
        y_s = y_m // 2 - m
        y_e = y_m // 2 + m
        z_s = z_m // 2 - m
        z_e = z_m // 2 + m
        self.z_mean=m
        if inverse:
            temp = x_e
            x_e = path.shape[2] - x_s
            x_s = path.shape[2] - temp
            return path[z_s:z_e, y_s:y_e, x_s:x_e][:, :, ::-1]
        # return path
        return path[z_s:z_e, y_s:y_e, x_s:x_e]
    def __init__(self,path,name,index,inverse=False):
        self.path=path

        self.t1=[]
        self.dyn=[]
        self.dyn_pre=[]
        self.dyn_1=[]
        self.dyn_2=[]
        self.dyn_3=[]
        self.dyn_4=[]
        self.dyn_5=[]
        self.seg=[]
        self.total=len(os.listdir(path))

        self.x_s=-1
        self.x_e=-1
        self.y_s=-1
        self.y_e=-1
        self.z_s=-1
        self.z_e=-1
        self.type=''
        for i in range(len(annotation)):
            if annotation.iloc[i,0]==name:
                self.y_s=annotation.iloc[i,1]
                self.y_e = annotation.iloc[i, 2]
                self.x_s = annotation.iloc[i, 3]
                self.x_e = annotation.iloc[i, 4]
                self.z_s = annotation.iloc[i, 5]
                self.z_e = annotation.iloc[i, 6]
                # self.x_s = 0
                # self.x_e = -1
                # self.y_s = 0
                # self.y_e = -1
                # self.z_s = 0
                # self.z_e = -1

        # for p in os.listdir(path):
        #     if '.ds_store' not in p.lower():
        #         self.path=path + '/' + p



        for scan in os.listdir(self.path):
            if '1st' in scan.lower() and 'dyn' in scan.lower() and 'ph' not in scan.lower():
                self.type=type[0]
            elif '1st' not in scan.lower() and 'dyn' in scan.lower() and 'ph1' in scan.lower() and 'dynamic' not in scan.lower():
                self.type=type[1]
            elif '1st' not in scan.lower() and 'dyn' not in scan.lower() and 'ph1' in scan.lower() and 'dynamic' in scan.lower():
                self.type=type[2]
            elif '1st' not in scan.lower() and 'dyn' not in scan.lower() and 'ph1' in scan.lower() and 'dynamic' not in scan.lower() and 'vibrant' in scan.lower():
                self.type=type[3]

            if ('t1' in scan.lower() or 'ideal' in scan.lower()) and 'dyn' not in scan.lower():
                self.t1.append(self.path+'/'+scan)
                continue
            if 'segmentation' not in scan.lower() and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn.append(self.path+'/'+scan)

            if 'segmentation' in scan.lower():
                self.seg.append(self.path+'/'+scan)
                continue
            if 'pre' in scan.lower() :
                self.dyn_pre.append(self.path+'/'+scan)
                continue
            if ('1st' in scan.lower() or 'ph1' in scan.lower()) and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_1.append(self.path+'/'+scan)
                continue
            if ('2nd' in scan.lower() or 'ph2' in scan.lower()) and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_2.append(self.path+'/'+scan)
                continue
            if ('3rd' in scan.lower() or 'ph3' in scan.lower()) and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_3.append(self.path+'/'+scan)
                continue
            if ('4th' in scan.lower() or 'ph4' in scan.lower()) and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_4.append(self.path+'/'+scan)
                continue
            if ('5th' in scan.lower() or 'ph5' in scan.lower()) and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_5.append(self.path+'/'+scan)
                continue
            if 'segmentation' not in scan.lower() and ('dyn' in scan.lower() or 'vibrant' in scan.lower()):
                self.dyn_pre.append(self.path + '/' + scan)
        # print(path)
        # print(os.listdir(path))

        try:
            # assert len(self.dyn)>=3 and len(self.dyn)<=5
            # assert len(self.t1)==1
            # assert len(self.dyn)==len(self.dyn_pre)+len(self.dyn_1)+len(self.dyn_2)+len(self.dyn_3)+ len(self.dyn_4)+ len(self.dyn_5)
            match.append([index, name, len(self.t1), len(self.dyn), len(self.dyn_pre), len(self.dyn_1), len(self.dyn_2),
                          len(self.dyn_3), len(self.dyn_4), len(self.dyn_5),len(self.seg),self.total,1])
            self.t1_scan = load_scan_pixel(self.t1[0])
            if self.dyn_pre != []:
                r=Resample(self.dyn_pre[0])
                self.or_pre=r.original_direc
                x_s=int(self.x_s*(r.result_size[0]/r.original_size[0]))
                x_e = int(self.x_e * (r.result_size[0] / r.original_size[0]))
                y_s = int(self.y_s * (r.result_size[1] / r.original_size[1]))
                y_e = int(self.y_e * (r.result_size[1] / r.original_size[1]))
                z_s = int(self.z_s * (r.result_size[2] / r.original_size[2]))
                z_e = int(self.z_e * (r.result_size[2] / r.original_size[2]))
                if r.original_direc == 'LPS':
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                if r.original_direc == 'RAS':
                    temp = x_s
                    x_s = r.result_size[0] - x_e
                    x_e = r.result_size[0] - temp
                    temp = y_s
                    y_s = r.result_size[1] - y_e
                    y_e = r.result_size[1] - temp
                if r.original_direc == 'RPI':
                    temp = x_s
                    x_s = r.result_size[0] - x_e
                    x_e = r.result_size[0] - temp
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                self.dyn_pre_scan = self.patch(sitk.GetArrayFromImage(r.result),inverse,x_s,x_e,y_s,y_e,z_s,z_e)
            else:
                self.dyn_pre_scan = np.array([])
            if self.dyn_1 != []:
                r = Resample(self.dyn_1[0])
                self.or_1 = r.original_direc
                x_s = int(self.x_s * (r.result_size[0] / r.original_size[0]))
                x_e = int(self.x_e * (r.result_size[0] / r.original_size[0]))
                y_s = int(self.y_s * (r.result_size[1] / r.original_size[1]))
                y_e = int(self.y_e * (r.result_size[1] / r.original_size[1]))
                z_s = int(self.z_s * (r.result_size[2] / r.original_size[2]))
                z_e = int(self.z_e * (r.result_size[2] / r.original_size[2]))
                if r.original_direc == 'LPS':
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                if r.original_direc == 'RAS':
                    temp = x_s
                    x_s = r.result_size[0] - x_e
                    x_e = r.result_size[0] - temp
                    temp = y_s
                    y_s = r.result_size[1] - y_e
                    y_e = r.result_size[1] - temp
                if r.original_direc == 'RPI':
                    temp = x_s
                    x_s = r.result_size[0] - x_e
                    x_e = r.result_size[0] - temp
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                self.dyn_1_scan = self.patch(sitk.GetArrayFromImage(r.result), inverse, x_s, x_e, y_s, y_e, z_s, z_e)
            else:
                self.dyn_1_scan = np.array([])
            if self.dyn_2 != []:
                r = Resample(self.dyn_2[0])
                self.or_2 = r.original_direc
                x_s = int(self.x_s * (float(r.result_size[0]) / r.original_size[0]))
                x_e = int(self.x_e * (float(r.result_size[0]) / r.original_size[0]))
                y_s = int(self.y_s * (float(r.result_size[1]) / r.original_size[1]))
                y_e = int(self.y_e * (float(r.result_size[1]) / r.original_size[1]))
                z_s = int(self.z_s * (float(r.result_size[2]) / r.original_size[2]))
                z_e = int(self.z_e * (float(r.result_size[2]) / r.original_size[2]))
                if r.original_direc=='LPS':
                    temp = x_s
                    x_s = r.result_size[0] - x_e
                    x_e = r.result_size[0] - temp
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                if r.original_direc=='RAS':
                    # temp = x_s
                    # x_s = r.result_size[0] - x_e
                    # x_e = r.result_size[0] - temp
                    temp = y_s
                    y_s = r.result_size[1] - y_e
                    y_e = r.result_size[1] - temp
                    # temp = z_s
                    # z_s = r.result_size[2] - z_e
                    # z_e = r.result_size[2] - temp

                if r.original_direc=='RPI':
                    # temp = y_s
                    # y_s = r.result_size[1] - y_e
                    # y_e = r.result_size[1] - temp
                    temp = z_s
                    z_s = r.result_size[2] - z_e
                    z_e = r.result_size[2] - temp
                self.dyn_2_scan = self.patch(sitk.GetArrayFromImage(r.result), inverse, x_s, x_e, y_s, y_e, z_s, z_e)
            else:
                self.dyn_2_scan = np.array([])
            # if self.dyn_3 != []:
            #     self.dyn_3_scan = patch(load_scan_pixel(self.dyn_3[0]),inverse,self.x_s,self.x_e,self.y_s,self.y_e,self.z_s,self.z_e)
            # else:
            #     self.dyn_3_scan = np.array([])
            # if self.dyn_4 != []:
            #     self.dyn_4_scan = patch(load_scan_pixel(self.dyn_4[0]),inverse,self.x_s,self.x_e,self.y_s,self.y_e,self.z_s,self.z_e)
            # else:
            #     self.dyn_4_scan = np.array([])
            # if self.dyn_5 != []:
            #     self.dyn_5_scan = patch(load_scan_pixel(self.dyn_5[0]),inverse,self.x_s,self.x_e,self.y_s,self.y_e,self.z_s,self.z_e)
            # else:
            #     self.dyn_5_scan = np.array([])
            # print(self.t1_scan.shape)
            # print(self.dyn_pre_scan.shape)
            # print(self.dyn_1_scan.shape)
            # print(self.dyn_2_scan.shape)
            # print(self.dyn_3_scan.shape)
            # print(self.dyn_4_scan.shape)
            # print(self.dyn_5_scan.shape)

        except:
            # match.append([index,name,len(self.t1),len(self.dyn),len(self.dyn_pre),len(self.dyn_1),len(self.dyn_2),len(self.dyn_3),len(self.dyn_4),len(self.dyn_5)])
            match.append([index, name, len(self.t1), len(self.dyn), len(self.dyn_pre), len(self.dyn_1), len(self.dyn_2),
                          len(self.dyn_3), len(self.dyn_4), len(self.dyn_5), len(self.seg), self.total,0])
    def bright(self):
        r = Resample(self.dyn_2[0])

        x_s = int(self.x_s * (float(r.result_size[0]) / r.original_size[0]))
        x_e = int(self.x_e * (float(r.result_size[0]) / r.original_size[0]))
        y_s = int(self.y_s * (float(r.result_size[1]) / r.original_size[1]))
        y_e = int(self.y_e * (float(r.result_size[1]) / r.original_size[1]))
        z_s = int(self.z_s * (float(r.result_size[2]) / r.original_size[2]))
        z_e = int(self.z_e * (float(r.result_size[2]) / r.original_size[2]))

        print(x_s,x_e,y_s,y_e,z_s,z_e)
        self.dyn_2_scan = sitk.GetArrayFromImage(r.result)

        self.dyn_2_scan[z_s:z_e,y_s:y_e,x_s:x_e]=np.max(self.dyn_2_scan)
        return self.dyn_2_scan

