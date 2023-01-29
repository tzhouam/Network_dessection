import torch
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import threading

def show_pt(num):
    img=torch.load("D:/Duke_resized/resized_order/Breast_MRI_"+ num +".pt")[:,:,:,0].numpy()    #[3:100:100:100] ->[3:100:100]
    img_i=torch.load("D:/Duke_resized/resized_order/Breast_MRI_"+ num +"_i.pt")[:,:,:,0].numpy()
    img=np.einsum('kij->ijk',img);img_i=np.einsum('kij->ijk',img_i)
    img=(img+2000)/(4000/255);img_i=(img_i+2000)/(4000/255)
    img=img.astype(int);img_i=img_i.astype(int)

    fig=plt.figure()
    plt.subplot(321);plt.imshow(img[:,:,0])
    plt.xlabel("Breast_MRI_"+ num +".pt")
    plt.subplot(323);plt.imshow(img[:,:,1])
    plt.xlabel("Breast_MRI_"+ num +".pt")
    plt.subplot(325);plt.imshow(img[:,:,2])
    plt.xlabel("Breast_MRI_"+ num +".pt")
    plt.subplot(322);plt.imshow(img_i[:,:,0])
    plt.xlabel("Breast_MRI_"+ num +"_i.pt")
    plt.subplot(324);plt.imshow(img_i[:,:,1])
    plt.xlabel("Breast_MRI_"+ num +"_i.pt")
    plt.subplot(326);plt.imshow(img_i[:,:,2]) 
    plt.xlabel("Breast_MRI_"+ num +"_i.pt")
    plt.subplots_adjust(wspace=0.5, hspace=0.5, left=0.1, right=0.9, bottom=0.1, top=0.95)
    fig.canvas.set_window_title("Patient_"+num)
    plt.show()

def show_pt2(num):
    img=torch.load("D:/Duke_resized/resized_order/Breast_MRI_"+ num +".pt")[:,:,:,:].numpy()    #[3:100:100:100] ->[3:100:100]
#    img_i=torch.load("./Breast_MRI_"+ num +"_i.pt")[:,:,:,:].numpy()
    img=np.einsum('lijk->ijkl',img);#img_i=np.einsum('lijk->ijkl',img_i)
    img=(img+2000)/(4000/255);#img_i=(img_i+2000)/(4000/255)
    img=img.astype(int);#img_i=img_i.astype(int)

    mlab.contour3d(img[:,:,:,0]) # img.shape = C*H*W
    mlab.show()
#    mlab.contour3d(img_i[:,:,:,0]) # img.shape = C*H*W
#    mlab.show()



if __name__ == '__main__':
    show_pt("001")