import torch
from torch import nn, Tensor
import sys
CUDA_LAUNCH_BLOCKING=1
import platform
if platform.system()=="Linux":
    sys.path.insert(1, '/home/zhoutaichang/Network_dessection/Network_desection/Model/CNN/')
elif platform.system()=="Windows":
    sys.path.insert(1, 'C:/Users/zhout/Desktop/Network_dessection/Network_desection/Model/CNN/')
import matplotlib.pyplot as plt
from Conv_model import CNN
from hyperparameter import Hyperpara
import gc
Hyper=Hyperpara()
model=CNN()
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def min_max(a:np.array):
    if np.min(a)==np.max(a):
        # print("Has 0")
        return np.zeros(a.shape).astype(np.float32)
    scaled_image = (np.maximum(a, 0) / a.max()) * 255.0

    return scaled_image
data_len = len(os.listdir(Hyper.file_path))
batch_size=Hyper.batch_size
train_len=int(0.8*data_len)
validation_s=int(0.7*data_len)
test_s=train_len
patients= np.arange(data_len)
annotation_old=pd.read_excel(io='Annotation_Boxes.xlsx')
files = sorted(os.listdir(Hyper.file_path))
annotation=[]
# assert (np.array(os.listdir(Hyper.file_path))==pd.read_excel(io='Annotation_Boxes.xlsx').iloc[:,0].to_numpy()).all()

def prepare(j:int):
    # print(j)
    return min_max(torch.load(Hyper.file_path + files[patients[j]]).cpu().detach().numpy())


model.load_state_dict(torch.load('cnn_model.pt',map_location="cuda:"+str(Hyper.cuda)))
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if i==0:
        for child in model_children[i].children():
            if type(child) == nn.Conv3d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)

    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv3d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")
for i in range(0,data_len,batch_size):
    # torch.cuda.empty_cache()
    data_x = Parallel(n_jobs=-1, backend='threading')(
        delayed(prepare)(j) for j in range(i, min(i + batch_size, data_len)))
    print("Batch ",i," loaded")

    data_x=np.array(data_x).astype(float)
    # print(data_x.shape)
    train_x_tensor=torch.from_numpy(data_x)
    train_x_tensor = train_x_tensor.type(torch.FloatTensor)
    # Visualize feature maps
    activation = {}
    # print('hooking')
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for j in range(len(conv_layers)):
        # print('hooking ',j)
        conv_layers[j].register_forward_hook(get_activation('conv'+str(j)))
    # print("Outputing")
    output=model(train_x_tensor)
    print("Batch ",i," Outputed")
    for m in range(len(conv_layers)):
    # def save(m):
        if not os.path.isdir('/jhcnas1/zhoutaichang/match/cnn/'):
            os.mkdir('/jhcnas1/zhoutaichang/match/cnn/')
        if not os.path.isdir('/jhcnas1/zhoutaichang/match/cnn/conv'+str(m).rjust(2,'0')):
            os.mkdir('/jhcnas1/zhoutaichang/match/cnn/conv'+str(m).rjust(2,'0'))
        for j in range(min(batch_size,data_len-i)):
            act = activation['conv'+str(m)][j].cpu()

            # act = act.view(-1)
            if i+j>=data_len:
                break
            # print(i+j)

            torch.save(act,'/jhcnas1/zhoutaichang/match/cnn/conv'+str(m).rjust(2,'0')+'/'+files[i+j])
            del act


    # Parallel(n_jobs=-1, backend='threading')(
    #     delayed(save)(m) for m in range(0,len(conv_layers)))
    del activation
    del output
    gc.collect()



model.eval()