#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import *
import numpy as np 
import pandas as pd 
import os
import sys
import time
import datetime as dtym
from datetime import datetime
import glob
from pathlib import Path
import gc
from sklearn.metrics import roc_auc_score
from functools import reduce
import operator
from sklearn.cluster import KMeans
from data_processing import *
from kmeans_pytorch import kmeans
from sklearn import preprocessing
from utils import *
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# In[2]:


torch.set_num_threads(2)
print('threads:',torch.get_num_threads())


# In[3]:


def calOutShape(input_shape, ksize1 = 3,ksize2 = 3, ksize3 = 3,stride = 1,
                maxpool1 = False, maxpool2 = False, maxpool3 = False, mpl_ksize = 2):
#     print('mpl_ksize in calOutShape:',mpl_ksize)
    mpl_stride = 2
    pad = ksize1//2
    dim1 = int((input_shape[2] - ksize1 + 2 * pad)/stride) + 1
    if maxpool1 == True:
        dim1 = (dim1 - mpl_ksize)//mpl_stride + 1
    pad = ksize2//2
    dim1 = int((dim1 - ksize2 + 2 * pad)/stride) + 1
    if maxpool2 == True:
        dim1 = (dim1 - mpl_ksize)//mpl_stride + 1
    pad = ksize3//2
    dim1 = int((dim1 - ksize3 + 2 * pad)/stride) + 1
    if maxpool3 == True:
        dim1 = (dim1 - mpl_ksize)//mpl_stride + 1
#     print('dim1 :{}'.format(dim1))
    return dim1


class Transform(nn.Module):
    total_num_atoms = 0
    def __init__(self,input_shape, out_planes1 = 8, out_planes2 = 16, out_planes3 = 16, 
                 ksize1 = 3,ksize2 = 3, ksize3 = 3,
                 maxpool1 = False, maxpool2 = False, maxpool3 = False,
                 mpl_ksize = 2, num_channels = 2, activFunc = 'selu', atom_ratio = 0.5):
        
        super(Transform, self).__init__()
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.ksize3 = ksize3
        self.mpl_ksize = mpl_ksize
        self.out_planes1 = out_planes1
        self.out_planes2 = out_planes2
        self.out_planes3 = out_planes3
        self.init_T(input_shape)
        self.maxpool1 = maxpool1
        self.maxpool2 = maxpool2
        self.maxpool3 = maxpool3
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.activFunc = activFunc
        self.activation = activation_funcs[self.activFunc]
        self.i = 1
        self.atom_ratio = atom_ratio #0.5
        self.init_X()
        
    
    def init_T(self,input_shape):
        
        conv = nn.Conv1d(input_shape[1], out_channels = self.out_planes1, kernel_size = self.ksize1, 
                         stride = 1, bias = True)
        self.T1 = conv._parameters['weight']
        conv = nn.Conv1d(in_channels = self.out_planes1, out_channels = self.out_planes2, kernel_size = self.ksize2, 
                         stride = 1, bias = True)
        self.T2 = conv._parameters['weight']
        conv = nn.Conv1d(in_channels = self.out_planes2, out_channels = self.out_planes3, kernel_size = self.ksize3, 
                         stride = 1, bias = True)
        self.T3 = conv._parameters['weight']
        
        
    def init_X(self):
        
        dim1 = calOutShape(self.input_shape,self.ksize1,self.ksize2,self.ksize3, stride = 1,
                           maxpool1 = self.maxpool1, 
                           maxpool2 = self.maxpool2, maxpool3 = self.maxpool3,
                           mpl_ksize = self.mpl_ksize)
        
        X_shape = [self.input_shape[0],self.out_planes3,dim1]
#         print('X shape : ', X_shape)
        self.X  = nn.Parameter(torch.randn(X_shape), requires_grad=True)
        
        self.num_features = self.out_planes3 * dim1 
        self.num_atoms = int(self.num_features*self.atom_ratio)#* self.num_channels) #dim2//2
#         print('self.num_features : ',self.num_features)
#         print('self.num_atoms : ',self.num_atoms)
        Transform.total_num_atoms += self.num_atoms
#         print('Transform.total_num_atoms : ',Transform.total_num_atoms)
        
    def init_T_tilde(self):
        T_shape = [Transform.total_num_atoms,self.num_features]
#         print('T_shape : ', T_shape)

        self.T = nn.Parameter(torch.randn(T_shape), requires_grad=True)
        
        
        
    def forward(self, x):
#         print(self.T1)
        x = F.conv1d(x, weight = self.T1, stride = 1,padding = self.ksize1//2)
#         print('x after conv1 : ', x)
        if self.maxpool1:
            x = F.max_pool1d(x, self.mpl_ksize)
#         print('x after mxpl1 : ', x)
        x = self.activation(x)
#         print('x after activation : ', x)
        x = F.conv1d(x, weight = self.T2, stride = 1,padding = self.ksize2//2)
#         print('x after conv2 : ', x)
        if self.maxpool2:
            x = F.max_pool1d(x, self.mpl_ksize)
#         print('x after mxpl2 : ', x)    
        x = self.activation(x)
#         #added check where should we apply activations
#         x = self.activation(x)
        x = F.conv1d(x, weight = self.T3, stride = 1,padding = self.ksize3//2)
#         print('x after conv3 : ', x)
        
        if self.maxpool3:
            x = F.max_pool1d(x, self.mpl_ksize)
        y = torch.mm(self.T,x.view(x.shape[0],-1).t())
#         print('x after multply : ', x)
#         print('*'*100)
        return x, y
        
          
    def get_params(self):
        return self.T1, self.T2, self.T3, self.X, self.T
    
    
    def X_step(self):
        self.X.data = torch.clamp(self.X.data, min=0)


    
        
        
    def get_TZ_Dims(self):
        return self.num_features, Transform.total_num_atoms, self.input_shape[0]




def prepareChannels(data_source,X, out_planes1 = 8, out_planes2 = 16, out_planes3 = 16,
                 ksize1 = 3, ksize2 = 3, ksize3 = 3, maxpool1 = True, maxpool2 = True, 
                 maxpool3 = True, mpl_ksize = 2, num_classes = 6, 
                 num_channels = 3, activFunc = 'selu',atom_ratio = 0.5):
    channels_trnsfrm = nn.ModuleList()
    if data_source == "3sources":
        print('3sources')
    elif data_source == "BBC":
        print('BBC') 
    elif data_source == "Mfeat":
        print('Mfeat') 
    for nc in range(num_channels): 
#         print('nc : ',nc)
        #input_shape 
        t = Transform(input_shape = (X[nc].shape[0],1,X[nc].shape[1]), out_planes1 = out_planes1, 
                      out_planes2 = out_planes2, out_planes3 = out_planes3, ksize1 = ksize1,
                      ksize2 = ksize2, ksize3 = ksize3, maxpool1 = maxpool1, 
                      maxpool2 = maxpool2, maxpool3 = maxpool3, 
                      mpl_ksize = mpl_ksize, num_channels = num_channels,
                      activFunc = activFunc, atom_ratio = atom_ratio)

        channels_trnsfrm.append(t)
    
    #initializing the T tilde
    for nc in range(num_channels): 
        channels_trnsfrm[nc].init_T_tilde()
        
    return channels_trnsfrm


# In[4]:


class Network(nn.Module): 
    def __init__(self, source, channels_trnsfrm,out_planes1 = 8, out_planes2 = 16, out_planes3 = 16, 
                 ksize1 = 3, ksize2 = 3, ksize3 = 3, maxpool1 = False, maxpool2 = False,
                 maxpool3 = False, mpl_ksize = 2, num_classes = 6, nclusters = 6, 
                 num_channels = 2, activFunc = 'selu',atom_ratio = 0.5, loss_func = 'kmeans'):
        
        super(Network, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.channels_trnsfrm = channels_trnsfrm   
        self.num_classes = num_classes
        self.nclusters = nclusters
        self.num_features, self.num_atoms, self.input_shape = self.channels_trnsfrm[0].get_TZ_Dims()
        Z_shape = [self.num_atoms, self.input_shape]
#         print('Z_shape : ', Z_shape)
        self.Z = nn.Parameter(torch.randn(Z_shape), requires_grad = True)
        self.pred_list = []
        self.tl_features = []
        self.init_TX()
        
        
    def init_TX(self):
        self.T1  = []
        self.T2 = []
        self.T3 = []
        self.T = []
        self.X_list = []
        for nc in range(self.num_channels): 
            T1, T2, T3, X, T = self.channels_trnsfrm[nc].get_params()
#             print('X shape : init_TX', X.shape)
            self.T1.append(T1)
            self.T2.append(T2)
            self.T3.append(T3)
            self.X_list.append(X)
            self.T.append(T)
        self.T1 = torch.stack(self.T1,1)
        self.T2 = torch.stack(self.T2,1)
        self.T3 = torch.stack(self.T3,1)
        #self.X_list = torch.stack(self.X_list,1) 
        self.T = torch.cat(self.T,1) 
#         print('self.T1.shape : ', self.T1.shape)
#         print('self.T2.shape : ', self.T2.shape)
#         print('self.T3.shape : ', self.T3.shape)
#         print('self.T.shape : ', self.T.shape)
        
        
    def forward(self,input_x,clabels):
        
        self.pred_list = []
        self.outp = []
#         print('input_x.shape : ', input_x.shape)
        #to be changed 
        for nc in range(self.num_channels):
            x = input_x[nc]
            x = x.reshape(x.shape[0],1,x.shape[1])
#             print('x.shape : ', x.shape)
            temp_out, temp_outp = self.channels_trnsfrm[nc](x)
#             print('temp_out.shape : {}, temp_outp.shape : {}', temp_out.shape, temp_outp.shape)
            temp_out = temp_out.view(temp_out.size(0),-1)
            self.pred_list.append(temp_out)
            self.outp.append(temp_outp)
            
#         print('self.pred_list length : ', len(self.pred_list))
#         print('self.outp length : ', len(self.outp))
        
        i = 0
        for nc in range(self.num_channels):
            if i == 0:
                self.tl_features = self.outp[nc] 
            i += 1
            self.tl_features += self.outp[nc]
            
        #print('features after FC - Linear TL:',self.tl_features)
        #print('*'*50)
        print('features shape after FC - Linear TL:',self.tl_features.shape)
        clabels, centers = kmeans(X = self.tl_features, num_clusters = self.nclusters)
        onehotencoder = preprocessing.OneHotEncoder()
        h = onehotencoder.fit_transform(np.transpose(clabels.detach().cpu().numpy()).reshape(-1,1)).toarray()
        h = np.transpose(h)
        hinv = np.linalg.pinv(np.dot(h,np.transpose(h)))
        hterm = np.dot(np.transpose(h),np.dot(hinv,h))
        hterm_ = torch.from_numpy(hterm.astype(np.float32))
#         if torch.cuda.is_available:
#             hterm_ = hterm_.to('cuda:0')
        zhterm = torch.mm(hterm_,self.tl_features)

        return self.tl_features, zhterm

    
    
    def X_step(self):
        
        for nc in range(self.num_channels): 
            self.channels_trnsfrm[nc].X_step()
        
        
    def Z_step(self):
    
        self.Z.data = torch.clamp(self.Z.data, min = 0)

    
    def conv_loss_distance(self, batch_idx, batch_size):
        
        self.init_TX()
        
        loss = 0.0
#         print('self.X_list len : ', len(self.X_list), self.X_list[0].shape)
#         print('self.pred_list : ', self.pred_list)
        for i in range(len(self.pred_list)): 
            predictions = self.pred_list[i].view(self.pred_list[i].size(0), -1)
            
            X = self.X_list[i].view(self.X_list[i].size(0),-1)
            #X = X[batch_idx * batch_size : batch_idx * batch_size + batch_size]
            #print('X shape : ', X.shape)
            
            Y = predictions - X[0:predictions.shape[0]]
            loss += Y.pow(2).mean()
            
        return loss
    
        
    def conv_loss_logdet(self):

        loss = 0.0
        
        for T in self.T1:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
            
        for T in self.T2:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
            
        for T in self.T3:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
            
        return loss
        
        
    def conv_loss_frobenius(self):
        
        loss = 0.0
        
        for T in self.T1:
            loss += T.pow(2).sum()
        
        for T in self.T2:
            loss += T.pow(2).sum()
        
        for T in self.T3:
            loss += T.pow(2).sum()
            
        return loss
    

    def loss_distance(self, batch_idx, batch_size):

        loss = 0.0
        
        predictions = self.tl_features
#         print('predictions.shape : ', predictions.shape)
#         print('self.Z.shape : ', self.Z.shape)
#        Z = self.Z[:,batch_idx * batch_size : batch_idx * batch_size + batch_size]
#         print('Z.shape : ', self.Z.shape)
        Y = predictions - self.Z[:,0:predictions.shape[1]] #Z
        loss += Y.pow(2).mean()    
        
        return loss
        
        
    def loss_logdet(self):
        
        loss = 0.0
        
#         T = torch.stack(self.T,1)
#         T = self.T
#         print('T shape in loss_logdet func : ', T.shape)
        
        T = self.T.view(self.T.shape[0],-1)
        U, s, V = torch.svd(T)
        loss = -s.log().sum()
        
        return loss
        
        
    def loss_frobenius(self):
        
        loss = 0.0       
        loss = self.T.pow(2).sum()
        
        return loss


    def loss_kmeans(self, zhterm):
        loss = 0.0
        predictions = self.tl_features
        Y = self.Z[:,0:predictions.shape[1]].float() -  zhterm
        loss = Y.pow(2).mean()
        return loss
    
    
    def computeLoss(self, lam, mu, km_reg, batch_idx, batch_size, zhterm):
        t0 = time.time()
        loss1 = self.conv_loss_distance(batch_idx, batch_size)
        loss2 = self.conv_loss_frobenius() * mu
        loss3 = self.conv_loss_logdet() * lam
        loss4 = self.loss_distance(batch_idx, batch_size)
        loss5 = self.loss_frobenius() * mu
        loss6 = self.loss_logdet() * lam
        loss_ctl = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 
        t1 = time.time()
#         print('Time taken for ctl loss : ', str(dtym.timedelta(seconds = t1 - t0)))
#         print("computed ctl")
        t0 = time.time()
        loss_kmeans = self.loss_kmeans(zhterm) * km_reg
        t1 = time.time()
#         print('Time taken for kmeans loss : ', str(dtym.timedelta(seconds = t1 - t0)))
#         print("computed kmeans")
        loss = loss_ctl + loss_kmeans
        
        return loss, loss_ctl, loss_kmeans

    
    def getTZ(self):
        
        return self.T.view(self.T.shape[0],-1), self.Z
    
    
    def getX(self):
        
        return self.X_list


# In[5]:


def train_on_batch(X, Y, source, lamda, mean, km_reg, out_pl1, out_pl2, out_pl3, 
           ks1, ks2, ks3, maxpl1, maxpl2, maxpl3,
           mpl_ks, batch_size, xstep_flg, zstep_flg, epochs = 50, lr = 0.001, gamma = 0.1, wd = 1e-4, 
           amsFlg = False, activFunc = 'selu', loss_func = 'kmeans', 
           atom_ratio = 0.5):
#     print('epochs : ', epochs)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_loader = DataLoader(MVCData3Sources(X, Y, source), batch_size = batch_size, num_workers = 0, shuffle = True)
#     print('len(train_loader) : ',len(train_loader))
    if source == "Mfeat":
        num_channels = X.shape[0]
        num_classes = np.unique(Y).shape[0]
        n_clusters = num_classes
        print('num_channels : {}, num_classes : {}'.format(num_channels, num_classes))
    # creating channels networks 
    channels_trnsfrm = prepareChannels(data_source = source, X = X,
                                      out_planes1 = out_pl1, out_planes2 = out_pl2, out_planes3 = out_pl3, 
                                      ksize1 = ks1, ksize2 = ks2, ksize3 = ks3, 
                                      maxpool1 = maxpl1, maxpool2 = maxpl2, maxpool3 = maxpl3,
                                      mpl_ksize = mpl_ks, num_classes = num_classes, 
                                      num_channels = num_channels, activFunc = activFunc, 
                                      atom_ratio = atom_ratio)
#     print('channels_trnsfrm :')
#     print(channels_trnsfrm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Network(source, channels_trnsfrm,out_planes1 = out_pl1, out_planes2 = out_pl2, 
                 out_planes3 = out_pl3, ksize1 = ks1, ksize2 = ks2, ksize3 = ks3,
                 maxpool1 = maxpl1, maxpool2 = maxpl2, maxpool3 = maxpl3,
                 mpl_ksize = mpl_ks, num_classes = num_classes, nclusters = n_clusters, num_channels = num_channels, 
                 activFunc = activFunc,atom_ratio = atom_ratio, loss_func = loss_func)
    model.to(device)
#     for name, param in model.named_parameters():
#         print(name, param)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr = lr, 
#                                  betas = (0.9, 0.999), eps = 1e-08, weight_decay = wd, amsgrad = amsFlg)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = wd)#,momentum = 0.9)
    print_every = 10
    steps = 0
    
    train_losses = []
    ctl_losses = []
    kmeans_losses = []
    scores = []
    train_accs = []
    
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
#         print('***EPOCH*** : ',epoch)
        model.train()
        t00 = time.time()
        running_loss = 0.0
        ctl_run_loss = 0.0
        kmeans_run_loss = 0.0
        correct = 0
        total = 0
        ypred = []
        ytrue = []
        data = []
#         temp = []
        for batch_idx, (vw1,vw2,vw3,vw4,vw5,vw6, labels) in enumerate(train_loader):
#             print('batch_idx : ', batch_idx)
#             for vw in views: 
            vw1, vw2, vw3, vw4, vw5, vw6, labels = map(lambda x: Variable(x), [vw1, vw2, vw3, vw4, vw5, vw6, labels])  
#                 temp.append(vw)
#             data.append(temp)
#             temp = []
            data = [vw1.to(device),vw2.to(device),vw3.to(device),vw4.to(device),vw5.to(device),vw6.to(device)]
            labels = labels.to(device)
#             print('data : ',data)
#             print('labels : ',labels)
#             print('='*20)
            labels = labels.float()
    
            # zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            _, zhterm = model(data,labels)
#             print("computing loss!!")
            loss, loss_ctl, loss_kmeans = model.computeLoss(lamda, mean, km_reg, batch_idx, batch_size, zhterm)
#             print("loss computed!!")
            # backward propagation
            loss.backward()

            # gradient step
            optimizer.step()

            # proximal step
            if xstep_flg:
                model.X_step()
            if zstep_flg:
                model.Z_step()
            
            running_loss += loss.item()
            ctl_run_loss += loss_ctl.item()
            kmeans_run_loss += loss_kmeans.item()
#             if batch_idx == 0:
#                 break
        train_losses.append(running_loss/(batch_idx+1))
        ctl_losses.append(ctl_run_loss/(batch_idx+1))
        kmeans_losses.append(kmeans_run_loss/(batch_idx+1))
        t01 = time.time()
        print("Time taken for one epoch : ", str(dtym.timedelta(seconds = t01 - t00)))
        print('*'*100)
        print('Train Epoch: {} \tLoss: {:.4f}'.format(
                    epoch, running_loss/(batch_idx + 1)))
                
    model.eval()
    torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'model': model
            },model_path)             
#     print("reached here !!")
    S_train = [Variable(torch.from_numpy(X[0]).float(), requires_grad=False), 
               Variable(torch.from_numpy(X[1]).float(), requires_grad=False), 
               Variable(torch.from_numpy(X[2]).float(), requires_grad=False),
               Variable(torch.from_numpy(X[3]).float(), requires_grad=False),
               Variable(torch.from_numpy(X[4]).float(), requires_grad=False),
               Variable(torch.from_numpy(X[5]).float(), requires_grad=False)]

    Z_train,hhh_ = model(S_train, Y)
    Z_train = Z_train.cpu().data.numpy()
    Z_train = Z_train.reshape(Z_train.shape[0], -1).T
    print("Shape of Z_train: " + str(Z_train.shape))
    feat_path = base_path + 'features/' + source + '_' + param_path + '.npy'
#     print('feat_path : ', feat_path)
    np.save(feat_path,Z_train)
    
    np.random.seed(seed)
    ypred = KMeans(n_clusters = n_clusters, init="k-means++").fit_predict(Z_train)
    pred_df = pd.DataFrame(ypred, columns = ['Predicted'])
    pred_df['True'] = Y
#     print(ypred)
#     print(pred_df)
    pred_file_name = res_base_path + 'pred_dcfl2_' + loss_func + param_path + '.csv' 
    pred_df.to_csv(pred_file_name, index = None)
    acc = cluster_acc(Y, ypred)
    nmi_val, ari_val = compute_nmi_ari(Y,ypred)
    print('k means acc = %.4f, nmi = %.4f, ari = %.4f' % (acc,nmi_val,ari_val))
    
    scores_dict = {}
    scores_dict['source'] = source  
    scores_dict['seed'] = seed
    scores_dict['wd'] = wd
    scores_dict['lr'] = lr
    scores_dict['epochs'] = epochs
    scores_dict['lamda'] = lamda
    scores_dict['mean'] = mean
    scores_dict['km_reg'] = km_reg
    scores_dict['atm_ratio'] = atom_ratio
    scores_dict['ks1'] = ks1
    scores_dict['ks2'] = ks2
    scores_dict['ks3'] = ks3
    scores_dict['out_pl1'] = out_pl1
    scores_dict['out_pl2'] = out_pl2
    scores_dict['out_pl3'] = out_pl3
    scores_dict['maxpl1'] = maxpl1
    scores_dict['maxpl2'] = maxpl2
    scores_dict['maxpl3'] = maxpl3
    scores_dict['amsFlg'] = amsFlg
    scores_dict['zstep_flg'] = zstep_flg
    scores_dict['xstep_flg'] = xstep_flg
    scores_dict['activFunc'] = activFunc
    scores_dict['loss_func'] = loss_func
    scores_dict['batch_size'] = batch_size
    scores_dict['train_loss'] = train_losses
    scores_dict['ctl_loss'] = ctl_losses
    scores_dict['kmeans_loss'] = kmeans_losses
    scores_dict['ari'] = ari_val
    scores_dict['nmi'] = nmi_val
    scores_dict['cluster_accuracy'] = acc
    scores_dict['datetime'] = datetime.now()
    t1 = time.time()
    print('Time taken for entire training : ', str(dtym.timedelta(seconds = t1 - t0)))
    print('*'*50)
    return scores_dict
    


# In[6]:


def model_configs():
    # define scope of configs
    n_epochs = [40]
    n_batch = [128]
    lam_mu_km_list = [(0.01,0.0001,0.8)]
    learning_rate_list = [1e-4]
    gamma = [1]
    ks_filters = [(2,5,4,3,8,3)]
    wd_list = [0.001]
    mpl_ks_list = [2]
    maxpool_list = [(False, False, False)]
    activations = ['selu']
    atom_ratio = [0.5]
    amsFlg = ['SGD']
    xstep_flg = [True]
    zstep_flg = [True]
    # create configs
    configs = list()
    for i in n_epochs:
        for j in n_batch:
            for m in lam_mu_km_list:
                for l in learning_rate_list:
                    for g in gamma:
                        for ks_f in ks_filters:
                            for wd in wd_list:
                                for mpl_ks in mpl_ks_list:
                                    for maxpools in maxpool_list: 
                                        for activ_func in activations:
                                            for atm in atom_ratio:
                                                for ams in amsFlg:
                                                    for x in xstep_flg:
                                                        for z in zstep_flg:
                                                            cfg = [i, j, m, l, g, ks_f, wd, mpl_ks, maxpools, 
                                                                   activ_func, atm,ams,x,z]
                                                            configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def createParamPath(epochs, batch_size, lamda, mean, km_reg, lr, gamma, ks_f, wd, mpl_ks, maxpools, amsFlg, activFunc, 
                    loss_func, atom_ratio, xstep_flg, zstep_flg):
    out_pl1, ks1, out_pl2, ks2, out_pl3, ks3 = ks_f
    maxpl1, maxpl2, maxpl3 = maxpools
    if custom_batch_size_flag == True:
        param_path = 'final_ep_' + str(epochs) + '_bs' + str(batch_size) + '_lam' + str(lamda) + '_mu' + str(mean) + '_kmreg' + str(km_reg) +                '_lr' + str(lr) + '_gm' + str(gamma) + '_wd' + str(wd) +  '_ks1_' + str(ks1) + '_ks2' + str(ks2) + '_ks3' + str(ks3) +'_opl1' +                 str(out_pl1) + '_opl2' + str(out_pl2) + '_opl3' + str(out_pl3) +'_mpl1' + str(maxpl1)[0] + '_mpl2' + str(maxpl2)[0] + '_mpl3' + str(maxpl3)[0] +             '_ams' + str(amsFlg) + '_atmrat' + str(atom_ratio) + '_af' + activFunc + '_loss' + loss_func + '_x' + str(xstep_flg)[0] + '_z' + str(zstep_flg)[0] 
    else:
        param_path = 'final_ep_' + str(epochs) + '_lam' + str(lamda) + '_mu' + str(mean) + '_kmreg' + str(km_reg) +                '_lr' + str(lr) + '_gm' + str(gamma) + '_wd' + str(wd) +  '_ks1_' + str(ks1) + '_ks2' + str(ks2) + '_ks3' + str(ks3) +'_opl1' +                 str(out_pl1) + '_opl2' + str(out_pl2) + '_opl3' + str(out_pl3) +'_mpl1' + str(maxpl1)[0] + '_mpl2' + str(maxpl2)[0] + '_mpl3' + str(maxpl3)[0] +             '_ams' + str(amsFlg) + '_atmrat' + str(atom_ratio) + '_af' + activFunc + '_loss' + loss_func + '_x' + str(xstep_flg)[0] + '_z' + str(zstep_flg)[0] 
    return param_path


# In[7]:


activation_funcs = {
    'relu' : nn.ReLU(inplace = True),
    'selu' : nn.SELU(inplace = True),
    'leaky_relu' : nn.LeakyReLU(inplace = True),
    'tanh' : nn.Tanh(),
    'softsign' : nn.Softsign(),
    'softmax' : nn.Softmax(dim = 1),
    'sigmoid' : nn.Sigmoid()
}

loss_funcs = {
    'cross_entropy' : nn.CrossEntropyLoss(),
    'hinge' : nn.HingeEmbeddingLoss()
}


# ## read data

# In[8]:


data_path = "../data/"
base_path = '../' 
loss_func = 'kmeans'
source = 'Mfeat'
model_base_path = base_path + 'models/Dcf_l3' #+ str_fl_nm + '/' 
res_base_path = base_path + 'Results/'  + source + '/' + source + '_'

# base_string = 'dcf_l2_smi_sign' + str_fl_nm + '_' + loss_func + '_' 


# In[12]:


configs_list = model_configs()
custom_batch_size_flag = True
config_result_dict = {}
save_flag = 0
seed = 42
# g_cpu = torch.Generator()
# g_cpu.manual_seed(seed)
mode = 'train'
for idx, config in enumerate(configs_list):
    t0 = time.time()
    log_interval = 1 
    cnt = 0 
    epochs, batch_size, lamda_mean_kmreg, lr, gamma, ks_f, wd, mpl_ks, maxpools, activFunc, atom_ratio, amsFlg, xstep_flg, zstep_flg = config
    out_pl1, ks1, out_pl2, ks2, out_pl3, ks3 = ks_f
    maxpl1, maxpl2, maxpl3 = maxpools
    lamda, mean, km_reg = lamda_mean_kmreg
    if custom_batch_size_flag == False:
        batch_size = X[0].shape[0]
    param_path = createParamPath(epochs, batch_size, lamda, mean, km_reg, lr, gamma, ks_f, wd, mpl_ks, maxpools, amsFlg, 
                                 activFunc, loss_func, atom_ratio, xstep_flg, zstep_flg)
    model_path = model_base_path + source + '_model' +  param_path + '_reg' + str(idx) + '.pth'
    print(param_path)
    file_path = data_path + source + '.mat'
    X, Y = getData(file_path)
#     print(X)
    for vw_idx in range(X.shape[0]):
        X[vw_idx] = normalizeData(X[vw_idx].T)
        print(X[vw_idx].shape)
    print('data loaded !!')
    print('X.shape : ', X.shape)
    print('Y.shape : ', Y.shape)
    scores_dict = train_on_batch(X, Y, source, lamda, mean, km_reg, out_pl1, out_pl2, out_pl3, 
           ks1, ks2, ks3, maxpl1, maxpl2, maxpl3, 
           mpl_ks, batch_size, xstep_flg, zstep_flg, epochs = epochs, lr = lr, 
           gamma = gamma, wd = wd, 
           amsFlg = amsFlg, activFunc = activFunc, loss_func = loss_func, 
           atom_ratio = atom_ratio)
    scores_df = pd.DataFrame.from_dict(scores_dict, orient = 'index').T.reset_index().drop(columns = ['index'])
#     print('scores_df : ')
#     print(scores_df.head())
    os.remove(model_path)
    results_file_name = res_base_path + 'res_dcfl3_' + loss_func + param_path + '.csv'
    results_file_name2 = res_base_path + 'res_dcfl3_' + loss_func + '.csv'
    
    if not os.path.exists(results_file_name):
        scores_df.to_csv(results_file_name, sep = ',', index = None)
    else:
        scores_df.to_csv(results_file_name, mode = 'a', sep = ',', index = None,header = None)
        
    if not os.path.exists(results_file_name2):
        scores_df.to_csv(results_file_name2, sep = ',', index = None)
    else:
        scores_df.to_csv(results_file_name2, mode = 'a', sep = ',', index = None,header = None)


# X[0]

# In[10]:


data = sio.loadmat(file_path)


# In[18]:


len(data['data'][0][0].T)


# In[ ]:




