# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 11:15:25 2022

@author: patzo
"""

import scipy.io as spio
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
import os


path=os.getcwd()

#import true labels
labels=spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW3\label.mat'),squeeze_me=True)['trueLabel']
#labels=spio.loadmat(r'M:\OMSA\ISYE6740\HW3\data\label.mat',squeeze_me=True)['trueLabel']
print("labels shape:", labels.shape)

#import data
data_raw=spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW3\data.mat'),squeeze_me=True)['data']
print("data_raw shape:", data_raw.shape)

data=data_raw.T
print("data shape:",data.shape)


#Scale data
ndata=preprocessing.scale(data)
m,n=ndata.shape
print("m=",m,"n=",n)

### PCA on Data
#Build Covariance Matrix
C=np.matmul(ndata.T,ndata)/m
print("C shape:",C.shape)


#Find eigVec and eigVal
d=4
eigVec,_l,_=np.linalg.svd(C)
eigVec_d=eigVec[:,:d]
print("eigen vector d shape:",eigVec_d.shape)

#project data in top 4 principal directions 
pca_data=np.dot(ndata,eigVec_d)
plt.scatter(pca_data[np.where(labels == 2),0],pca_data[np.where(labels == 2),1])
plt.scatter(pca_data[np.where(labels == 6),0],pca_data[np.where(labels == 6),1])
plt.show()


###Appliying EM
k=3


#random seed
seed=4

#initialize first pi value
pi=np.random.random(k)
pi=pi/np.sum(pi)


#initialize mean and covariance 
mu=np.random.randn(k,d)
mu_old=mu.copy()

sigma=[]
for i in range(k):
    dummy=np.random.randn(4,4)
    sigma.append(dummy@dummy.T)

#initialize the posterior 
tau=np.full((m,k),fill_value=0.)

log_likelihoods=[]
total_steps=[]
for x in range(100):
  
    
    #E-step
    for kk in range(k):
        likelihood=mvn.pdf(pca_data,mu[kk],sigma[kk])
        tau[:, kk]=pi[kk]*mvn.pdf(pca_data, mu[kk], sigma[kk])
    #normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, k)))
    

    #M-step
    for kk in range(k):
        
        #update prior
        pi[kk]=np.sum(tau[:, kk])/m
        
        #update mean
        mu[kk]=pca_data.T @ tau[:,kk]/np.sum(tau[:,kk],axis=0)
        
        #update Covariance
        dummy=pca_data-np.tile(mu[kk], (m,1))
        sigma[kk]=dummy.T@np.diag(tau[:,kk]) @ dummy/np.sum(tau[:,kk],axis=0)
     
    tol=0.005
    
    print('-----iteration---',x)
    total_steps.append(x)
    plt.scatter(pca_data[:,0],pca_data[:,1],c=tau)
    plt.axis('scaled')
    plt.draw()
    plt.pause(0.1)
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if x==99:
        print('max iteration reached')
        break
    
    log_likelihoods.append(np.log(np.sum(likelihood)))
    
    
fig=plt.figure()
total_steps=total_steps[:len(total_steps)-1]
ax=fig.add_subplot(111)
ax.set_title('Log-Likehood')
ax.plot(total_steps,log_likelihoods)
plt.show()



#fig, (axs1, axs2) = plt.subplots(1, 2)

#Randomly generat two indies. One for 2 and one for 6
#img1 = mu[0].reshape((28,28))
#img2 = mu[1].reshape((28,28))


#fig.suptitle('Plotting the mean image')
#axs1.imshow(np.rot90(np.fliplr(img1)), cmap="Greys")
#axs2.imshow(np.rot90(np.fliplr(img2)), cmap="Greys")




##K means classification
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
fig, (axs1, axs2) = plt.subplots(1, 2)

#Randomly generat two indies. One for 2 and one for 6
img1 = kmeans.cluster_centers_[0].reshape((28,28))
img2 = kmeans.cluster_centers_[1].reshape((28,28))


fig.suptitle('Plotting the mean image')
axs1.imshow(np.rot90(np.fliplr(img1)), cmap="Greys")
axs2.imshow(np.rot90(np.fliplr(img2)), cmap="Greys")



