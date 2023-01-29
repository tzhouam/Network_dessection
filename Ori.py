import os
from Dataloader import Dataloader

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from joblib import Parallel,delayed
dataloader = Dataloader("/jhcnas1/zhoutaichang/Duke/")
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
ori={'LPS':[],'RPI':[],'RAS':[]}
def save(i):
    print(i)
    a,b,c=dataloader.load_direct(i)
    if a==-1:
        return
    ori[a].append(i)
    # ori[b].append(i)
    # ori[c].append(i)
Parallel(n_jobs=10,backend='threading')(delayed(save)(j) for j in range(0,len(dataloader.patient_id)))
for i in ori:
    ori[i]=sorted(ori[i])
l=0
for i in ori:
   if l<len(ori[i]):
        l=len(ori[i])
for i in ori:
    while len(ori[i])<l:
        ori[i].append(np.NAN)
li=pd.DataFrame(ori)
li.to_csv('ori.csv',index=False)