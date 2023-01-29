import os.path
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import sys
sys.path.insert(0, '/home/zhoutaichang/Network_dessection/Network_desection')
import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import csv
from Resnet_model import ResNet18
from hyperparameter import Hyperpara
from ResBlock import ResBlock
from Dataloader import Dataloader
from scipy.ndimage import zoom
import joblib
CUDA_LAUNCH_BLOCKING=1
from joblib import Parallel, delayed
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()







np.random.seed(0)
Hyper=Hyperpara()
# INPUT_FEATURES_NUM = Hyper.INPUT_FEATURES_NUM
OUTPUT_FEATURES_NUM = Hyper.OUTPUT_FEATURES_NUM
batch_size = Hyper.batch_size
prev_loss = Hyper.prev_loss
max_epochs = Hyper.max_epochs
learning_rate = Hyper.learning_rate
l2 = Hyper.l2
sequence_len=Hyper.sequence_len
epoch_change=Hyper.epoch_change
momentum=Hyper.momentum

# export CUDA_VISIBLE_DEVICES=1,3
def test(lstm_model,test_x,test_y,p=True):
    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model
    # print('X:'+str(test_x.shape))
    # print('Y:'+str(test_y.shape))
    # prediction on test dataset
    # test_x_tensor = test_x.reshape(-1, sequence_len,
                                   # INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x)  # 变为tensor

    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor)
    pred_y_for_test = pred_y_for_test.cpu().data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y).type(torch.FloatTensor))
    if p:
        print("test loss：{:.5f}".format(loss.item()))
    return pred_y_for_test, loss.item()

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
def min_max(a:np.array):
    if np.min(a)==np.max(a):
        # print("Has 0")
        return np.zeros(a.shape).astype(np.float32)
    scaled_image = (np.maximum(a, 0) / a.max()) * 255.0

    return scaled_image


def accuracy(model,x,y):
    model.eval()  # switch to testing model
    # print('X:'+str(test_x.shape))
    # print('Y:'+str(test_y.shape))
    # prediction on test dataset
    # test_x_tensor = test_x.reshape(-1, sequence_len,
    # INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(x)  # 变为tensor

    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = model(test_x_tensor)
    pred_y_for_test = pred_y_for_test.cpu().data.numpy()
    # print('y:\t',y.shape)
    # print(pred_y_for_test.shape)
    assert len(pred_y_for_test)==len(y)
    count=0
    for i in range(len(y)):
        # print(y[i])
        # print(pred_y_for_test)
        if pred_y_for_test[i][0]>pred_y_for_test[i][1]:
            m=[1,0]
        else:
            m=[0,1]
        if (m==y[i]).all():
            count+=1
    return count/len(pred_y_for_test)

def prepare(j:int,train=True):
    if train:
        return torch.load(Hyper.train + train_files[train_patients[j]]).cpu().detach().numpy()
    else:
        return torch.load(Hyper.test + test_files[test_patients[j]]).cpu().detach().numpy()
if __name__ == '__main__':

    # ----------------- train -------------------
    # force_cudnn_initialization()


    # checking if GPU is available
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:"+str(Hyper.cuda))
        print('Training on GPU{}.'.format(Hyper.cuda))
    else:
        print('No GPU available, training on CPU.')

    # 数据读取&类型转换



    print('Preparing')
    # # 数据集分割

    train_patients= np.arange(len(os.listdir(Hyper.train)))
    test_patients= np.arange(len(os.listdir(Hyper.test)))
    train_len=len(train_patients)
    validation_s=int(0.83*train_len)
    validation_e=train_len
    test_len=len(test_patients)
    annotation_old=pd.read_excel(io='Annotation_Boxes.xlsx')
    train_files = sorted(os.listdir(Hyper.train))
    test_files = sorted(os.listdir(Hyper.test))

    train_annotation=[]
    test_annotation=[]

    # assert (np.array(os.listdir(Hyper.file_path))==pd.read_excel(io='Annotation_Boxes.xlsx').iloc[:,0].to_numpy()).all()
    i = 0
    j = 0
    count=0
    for i in range(len(train_patients)):
        if '_i' in train_files[i].lower():
            train_annotation.append([0, 1])
        else:
            train_annotation.append([1, 0])
    train_annotation=np.array(train_annotation)
    for i in range(len(test_patients)):
        if '_i' in test_files[i].lower():
            test_annotation.append([0, 1])
        else:
            test_annotation.append([1, 0])
    test_annotation=np.array(test_annotation)

    train_patients, train_annotation=unison_shuffled_copies(train_patients, train_annotation)
    test_patients, test_annotation=unison_shuffled_copies(test_patients, test_annotation)
    print("Finished preparing")


    model = ResNet18(3,ResBlock,2,Hyper.width).to(device)

    print('Resnet model:', model)
    # print('model.parameters:', model.parameters)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)


    train_losses=[]
    test_losses=[]
    o=0
    best_epoch=0
    # print(optimizer.param_groups[0])
    for epoch in range(max_epochs):
        # print(epoch)
        for i in range(0,train_len,batch_size):

            x=Parallel(n_jobs=Hyper.worker,backend='threading')(delayed(prepare)(j,True) for j in range(i,min(i+batch_size,train_len)))


            y=train_annotation[range(i,min(i+batch_size,train_len))]

            x=np.array(x)
            y=np.array(y)

            # print(y.shape)

            train_x_tensor=torch.from_numpy(x)
            train_x_tensor=train_x_tensor.type(torch.FloatTensor)
            train_x_tensor=train_x_tensor.to(device)
            train_y_tensor=torch.from_numpy(y)
            train_y_tensor = train_y_tensor.type(torch.FloatTensor)
            train_y_tensor=train_y_tensor.to(device)

            for threshold in epoch_change:
                if epoch>threshold:
                    optimizer=torch.optim.SGD(model.parameters(), lr=epoch_change[threshold][0], weight_decay=epoch_change[threshold][1],momentum=momentum)
            output = model(train_x_tensor).to(device)
            # print(output.shape)
            loss = criterion(output, train_y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        # print('Test')
        train_loss=[]
        weight=[]
        for i in range(validation_s, validation_e, batch_size):
            x = Parallel(n_jobs=Hyper.worker, backend='threading')(delayed(prepare)(j, True) for j in range(i, min(i + batch_size, validation_e)))

            # x=[]
            # for j in range(i, min(i + batch_size, validation_e)):
            #     x.append(prepare(j))
            x=np.array(x)
            y = train_annotation[range(i, min(i + batch_size, validation_e))]
            # print(x.shape)
            # _, test_loss_batch = test(lstm_model=model, test_x=x, test_y=y, p=False)
            test_loss_batch=accuracy(model,x,y)
            train_loss.append(test_loss_batch)
            weight.append(min(i + batch_size, train_len) - i)

        loss=np.average(train_loss,weights=weight)


        if loss > prev_loss:
            torch.save(model.state_dict(), '/jhcnas1/zhoutaichang/resnet_model'+str(Hyper.learning_rate)+'.pt')  # save model parameters to files
            prev_loss = loss
            best_epoch=epoch

        if loss < 1e-4:
            print('Epoch [{}/{}], P: {:.5f}'.format(epoch + 1, max_epochs, loss))
            print("The loss value is reached")
            break
        if (epoch+1)%1==0:
            print("Learning rate:\t"+str(optimizer.param_groups[-1]['lr']),end='\t\t')
            print("L2:\t"+str(optimizer.param_groups[-1]['weight_decay']),end='\t\t')
            print('Epoch: [{}/{}], P:{:.5f}'.format(epoch + 1, max_epochs, loss),end='\t\t')
            train_losses.append(loss.item())
            loss_list=[]
            weight=[]
            for i in range(0,test_len,batch_size):
                x = Parallel(n_jobs=Hyper.worker, backend='threading')(delayed(prepare)(j, False) for j in range(i, min(i + batch_size, test_len)))
                x = np.array(x)

                y=test_annotation[range(i,min(i+batch_size,test_len))]
                # _,test_loss_batch=test(lstm_model=model, test_x=x, test_y=y,p=False)
                test_loss_batch=accuracy(model,x,y)
                loss_list.append(test_loss_batch)
                weight.append(min(i+batch_size,test_len)-i)

            # _,test_loss_epoch=test(lstm_model=model, test_x=test_x, test_y=test_y)
            test_loss_epoch=np.average(loss_list,weights=weight)
            print('Test P:\t{:.5f}'.format(test_loss_epoch))
            test_losses.append(test_loss_epoch)
            model.train()
        plt.figure()
        plt.plot(range(0, epoch+1), train_losses, label='Train')
        plt.plot(range(0, epoch+1), test_losses, label='Test')
        plt.legend()
        plt.title('lr: ' + str(Hyper.learning_rate) + ' l2: ' + str(Hyper.l2))
        plt.xlabel('epoch')
        plt.ylabel('ACC')
        plt.savefig('/jhcnas1/zhoutaichang/res_fig' + str(Hyper.learning_rate) + '.png', format='png')


    # prediction on training dataset
    # pred_y_for_train = model(train_x_tensor).to(device)
    # pred_y_for_train = pred_y_for_train.cpu().view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    print('Best epoch:\t' + str(best_epoch))
    # pred_y_for_test,_=test(lstm_model=model, test_x=test_x, test_y=test_y)
    # ----------------- plot -------------------

