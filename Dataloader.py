import pydicom as dicom
import pylab as p
from pydicom import dcmread
from Patient import Patient
import os
import gc
from Patient import match
import pandas as pd
class Dataloader:
    def __init__(self,data_path= '/jhcnas1/zhoutaichang/Duke'):

        self.root=data_path
        self.patient_id=[]
        for p in os.listdir(data_path):
            if 'duke' not in p.lower() and 'li' not in p.lower() and '.ds_store' not in p.lower() and 'mri' in p.lower():
                self.patient_id.append(p)
        self.patient_id=sorted(self.patient_id)
        self.patients=[]
        self.loaded=[]
        self.length=len(self.patient_id)
    def load_direct(self,index):
        # print(self.patient_id[index])
        if index>=len(self.patient_id):
            ValueError("The index id out of range, it should be with in the range of patient list")
        patient=Patient(self.root + '/' + self.patient_id[index],self.patient_id[index],index,inverse=False)
        # print(patient.dyn_pre)
        # patient_i = Patient(self.root + '/' + self.patient_id[index], self.patient_id[index], index, inverse=True)
        # self.patients.append(patient)
        # self.patients.append(patient)
        self.loaded.append(index)
        try:
            return patient.or_pre,patient.or_1,patient.or_2
        except:
            return -1,-1,-1
    def load_patient(self,index):
        # print(self.patient_id[index])
        if index>=len(self.patient_id):
            ValueError("The index id out of range, it should be with in the range of patient list")
        patient=Patient(self.root + '/' + self.patient_id[index],self.patient_id[index],index,inverse=False)
        # print(patient.dyn_pre)
        patient_i = Patient(self.root + '/' + self.patient_id[index], self.patient_id[index], index, inverse=True)
        # self.patients.append(patient)
        # self.patients.append(patient)
        # self.loaded.append(index)
        try:
            return [patient.dyn_pre_scan,patient.dyn_1_scan, patient.dyn_2_scan],[patient_i.dyn_pre_scan,patient_i.dyn_1_scan, patient_i.dyn_2_scan],patient.z_mean,patient.type
        except:
            return [-1,-1,-1],[-1,-1,-1],-1,-1
    def least(self,index):
        patient=Patient(self.root + '/' + self.patient_id[index],self.patient_id[index],index,inverse=False)
        return patient.dyn_3==[]
    def clear(self):
        self.loaded=[]
        for p in self.patients:
            del p
            gc.collect()
        self.patients=[]
    def check(self):
        for i in range(0,len(self.patient_id)-1,50):

            for j in range(i,min([i+50,len(self.patient_id)])):
                print(j)
                try:
                    self.load_patient(j)
                except:
                    print(self.patient_id[j],'\tnot available')
            self.clear()

# d=Dataloader()
# d.check()
# d.load_patient(0)
# df=pd.DataFrame(match)
# df.columns=['P_index','name','t1','dyn','dyn_pre','dyn_1','dyn_2','dyn_3','dyn_4','dyn_5','segmentation','total','normal']
# df.to_csv('check.csv',index=False)
