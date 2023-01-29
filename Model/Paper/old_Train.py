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

from hyperparameter import Hyperpara
from monai.networks.nets import DenseNet121, DenseNet,DenseNet169,DenseNet201,DenseNet264

from Dataloader import Dataloader
from scipy.ndimage import zoom
import joblib
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

def prepare(j:int):
    return min_max(torch.load(Hyper.file_path + files[patients[j]]).cpu().detach().numpy())
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
    data_len = len(os.listdir(Hyper.file_path))
    train_len=int(0.8*data_len)
    validation_s=int(0.7*data_len)
    validation_e=validation_s+((train_len-validation_s)//batch_size)*batch_size
    test_s=train_len
    test_e=test_s+((data_len-test_s)//batch_size)*batch_size
    patients= np.arange(data_len)
    annotation_old=pd.read_excel(io='Annotation_Boxes.xlsx')
    files = sorted(os.listdir(Hyper.file_path))
    annotation=[]
    # assert (np.array(os.listdir(Hyper.file_path))==pd.read_excel(io='Annotation_Boxes.xlsx').iloc[:,0].to_numpy()).all()
    i = 0
    j = 0
    while i < len(files):
        while j < len(annotation_old):

            if files[i][:14] == annotation_old.iloc[j, 0]:
                if '_i' in files[i].lower():
                    j += 1
                    annotation.append([0,1])                      # output is ill, normal
                else:
                    annotation.append([1,0])

                break
            j += 1
        i += 1
    annotation=np.array(annotation)

    # print(annotation.shape)
    # print(dataloader.patient_id[0])
    patients, annotation=unison_shuffled_copies(patients, annotation)


    print("Finished preparing")
    # data_x=data_x[:len(data_x)-len(data_x)%sequence_len].reshape(-1,sequence_len,INPUT_FEATURES_NUM)
    # data_y=data_y[:len(data_y)-len(data_y)%sequence_len].reshape(-1,sequence_len,OUTPUT_FEATURES_NUM)
    # data_x,data_y=unison_shuffled_copies(data_x,data_y)
    # train_data_ratio = 0.8  # Choose 80% of the data for training
    # train_data_len = int(len(data_x) * train_data_ratio)
    # # t = np.linspace(0, data_len, data_len)
    # #
    # # train_x = data_x[:train_data_len-train_data_len%sequence_len]
    # # train_y = data_y[:train_data_len-train_data_len%sequence_len]
    # # t_for_training = t[:train_data_len-train_data_len%sequence_len]
    # train_x = data_x[:train_data_len,:,:]
    # train_y = data_y[:train_data_len,:,:]
    # # t_for_training = t[:train_data_len,:,:]
    # test_len=len(data_x)-train_data_len
    # test_x = data_x[train_data_len:len(data_x)-test_len%sequence_len].reshape(-1,sequence_len,INPUT_FEATURES_NUM)
    # test_y = data_y[train_data_len:len(data_x)-test_len%sequence_len]
    # test_y=test_y.reshape(-1,OUTPUT_FEATURES_NUM)
    # # t_for_testing = t[train_data_len:len(data_x)-test_len%sequence_len]
    #
    #
    #
    #
    # train_x_tensor = train_x.reshape(-1, sequence_len,INPUT_FEATURES_NUM)  # set batch size to 1
    # train_y_tensor = train_y.reshape(-1,OUTPUT_FEATURES_NUM)  # set batch size to 1
    #
    # # transfer data to pytorch tensor
    # train_x_tensor = torch.from_numpy(train_x_tensor)
    # train_y_tensor = torch.from_numpy(train_y_tensor)
    #
    # train_x_tensor = train_x_tensor.to(device)
    # train_y_tensor = train_y_tensor.to(device)
    # model = DenseNet169(spatial_dims=3,in_channels=3,out_channels=2).to(device)
    model = DenseNet(spatial_dims=3,in_channels=3,out_channels=2,block_config=(3,6,12,8),growth_rate=16).to(device)

    # model = Conv_model.Conv(3)
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl')
    # my_model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
    # model = torch.nn.parallel.DistributedDataParallel(my_model, find_unused_parameters=True)

    # if torch.cuda.is_available():
    #
    #     model=model.to(device)
    print('Resnet model:', model)
    print('model.parameters:', model.parameters)
    # print('train x tensor dimension:', Variable(train_x_tensor).size())
    # print('train y tensor dimension:', Variable(train_y_tensor).size())

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2,momentum=momentum)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    o = 0
    best_epoch = 0
    # print(optimizer.param_groups[0])
    for epoch in range(max_epochs):
        # print(epoch)
        for i in range(0, train_len, batch_size):
            # print(i)
            # print(i)
            # x = []
            # y = []
            # for j in range(i,min(i+batch_size,train_len)):
            #     file=os.listdir(Hyper.file_path)[patients[j]]
            #     print(file)
            #     data_x=np.load(Hyper.file_path+file)['arr_0']
            #     y.append(annotation[j])
            #
            #     x.append(data_x)
            # Feature_map.py:
            # print()
            x = Parallel(n_jobs=batch_size, backend='threading')(
                delayed(prepare)(j) for j in range(i, min(i + batch_size, train_len)))

            # x=[]
            # for j in range(i,min(i+batch_size,train_len)):
            #     x.append(prepare(j))
            # print(Hyper.file_path + files[patients[j]])
            # except:
            #     print('Error'+str(i))
            #     continue
            y = annotation[range(i, min(i + batch_size, train_len))]
            # # print(y)
            # indexs=[index[1] for index in x]
            # print(np.array(files)[patients[indexs]])
            x = np.array(x)
            y = np.array(y)

            # print(y)
            train_x_tensor = torch.from_numpy(x)
            train_x_tensor = train_x_tensor.type(torch.FloatTensor)
            train_x_tensor = train_x_tensor.to(device)
            train_y_tensor = torch.from_numpy(y)
            train_y_tensor = train_y_tensor.type(torch.FloatTensor)
            train_y_tensor = train_y_tensor.to(device)

            for threshold in epoch_change:
                if epoch > threshold:  # update para in optimizer
                    # optimizer.param_groups[0]['lr']=epoch_change[threshold][0]
                    # optimizer.param_groups[0]['weight_decay']=epoch_change[threshold][1]
                    optimizer = torch.optim.SGD(model.parameters(), lr=epoch_change[threshold][0],
                                                weight_decay=epoch_change[threshold][1], momentum=momentum)

            output = model(train_x_tensor).to(device)
            # print(train_y_tensor)
            # print('Shape\t' + str(output.size()))
            # print('TShape\t' + str(train_y_tensor.size()))
            # train_y_tensor=train_y_tensor.view(output.size())

            loss = criterion(output, train_y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        # print('Test')
        train_loss = []
        train_acc = []
        for i in range(validation_s, validation_e, batch_size):
            x = np.array(Parallel(n_jobs=batch_size, backend='threading')(delayed(lambda j: min_max(torch.load(
                Hyper.file_path + files[patients[j]]).cpu().detach().numpy()))(j) for j in
                                                                          range(i, min(i + batch_size, validation_e))))
            # x=[]
            # for j in range(i, min(i + batch_size, validation_e)):
            #     x.append(prepare(j))
            x = np.array(x)
            y = annotation[range(i, min(i + batch_size, validation_e))]
            # print(x.shape)
            _, test_loss_batch = test(lstm_model=model, test_x=x, test_y=y, p=False)
            test_acc_batch = accuracy(model, x, y)
            train_loss.append(test_loss_batch)
            train_acc.append(test_acc_batch)
        loss = np.average(train_loss)
        acc = np.average(train_acc)

        if loss < prev_loss:
            torch.save(model.state_dict(), '/jhcnas1/zhoutaichang/dense_model.pt')  # save model parameters to files
            prev_loss = loss
            best_epoch = epoch

        if loss < 1e-4:
            print('Epoch [{}/{}], P: {:.5f}'.format(epoch + 1, max_epochs, loss))
            print("The loss value is reached")
            break
        if (epoch + 1) % 1 == 0:
            print("Learning rate:\t" + str(optimizer.param_groups[-1]['lr']), end='\t')
            print('Epoch: [{}/{}], Loss: {:.5f}, Prob: {:.5f}'.format(epoch + 1, max_epochs, loss, acc), end='\t')
            train_losses.append(loss)
            train_accs.append(acc)

            loss_list = []
            acc_list = []
            for i in range(test_s, test_e, batch_size):
                x = np.array(Parallel(n_jobs=batch_size, backend='threading')(delayed(lambda j: min_max(torch.load(
                    Hyper.file_path + files[patients[j]]).cpu().detach().numpy()))(j) for j in
                                                                              range(i, min(i + batch_size, test_e))))
                # x = []
                # for j in range(i, min(i + batch_size, test_e)):
                #     x.append(prepare(j))
                x = np.array(x)
                y = annotation[range(i, min(i + batch_size, test_e))]
                _, test_loss_batch = test(lstm_model=model, test_x=x, test_y=y, p=False)
                test_acc_batch = accuracy(model, x, y)
                loss_list.append(test_loss_batch)
                acc_list.append(test_acc_batch)
            # _,test_loss_epoch=test(lstm_model=model, test_x=test_x, test_y=test_y)
            test_loss_epoch = np.average(loss_list)
            test_acc_epoch = np.average(acc_list)

            print('Test Loss: {:.5f}, Test Prob: {:.5f}'.format(test_loss_epoch, test_acc_epoch))
            test_losses.append(test_loss_epoch)
            model.train()

    # prediction on training dataset
    # pred_y_for_train = model(train_x_tensor).to(device)
    # pred_y_for_train = pred_y_for_train.cpu().view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    print('Best epoch:\t' + str(best_epoch))
    # pred_y_for_test,_=test(lstm_model=model, test_x=test_x, test_y=test_y)
    # ----------------- plot -------------------
    plt.figure()
    plt.plot(range(0, max_epochs), train_losses, label='Train_loss')
    plt.plot(range(0, max_epochs), test_losses, label='Test_loss')
    plt.plot(range(0, max_epochs), train_accs, label='Train_acc')
    plt.plot(range(0, max_epochs), test_accs, label='Test_acc')
    plt.legend()

    plt.xlabel('epoch')
    plt.ylabel('Probability, Loss')
    plt.savefig('/jhcnas1/zhoutaichang/dense_fig.png', format='png')
