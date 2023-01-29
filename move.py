import shutil
import os
base='/jhcnas1/maxiaoqi/TEST/MRI_images/Duke-Breast-Cancer-MRI/'
dest='/jhcnas1/zhoutaichang/Duke/'
from joblib import Parallel,delayed
def r(i):
    pname='Breast_MRI_'+str(i).rjust(3,'0')
    print(pname)
    root=base+pname
    root=root+'/'+os.listdir(root)[0]
    if not os.path.isdir(root):
        return
    for file in os.listdir(root):
        if not os.path.isdir(root+'/'+file):
            continue
        print(shutil.copytree(root+'/'+file,dest+pname+'/'+file))
# Parallel(n_jobs=-1,backend='threading')(delayed(r)(i) for i in range(1,923))
r(1)
r(921)
