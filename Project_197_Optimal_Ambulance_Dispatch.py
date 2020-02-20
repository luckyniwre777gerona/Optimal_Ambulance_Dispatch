import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import qr
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.spatial import distance
from tensorflow import keras
from sklearn.cluster import KMeans
import pandas as pd
import os
import cv2

#%% Load Yale Database
Ni = []                     # Index of people in Yale Database                  
for i in range(0,99):
    Ni+=[i]
        
m = 480
n = 480

dirn = 'Train/'
cnt_tr = 0
cnt_te = 0
count = 0

avgFace = np.zeros((m*n,1))

Xtrain = []
Xtest = []

for i in range(0,99):
    file = dirn + 'Image' + str(i) + '.png'
    image = cv2.imread(file,0)
    image_bin = cv2.imread(file,0)
    for p in range(0,m):
        for q in range(0,n):
            if image_bin[int(p),int(q)] < 80:
                image_bin[int(p),int(q)] = 0
            else:
                image_bin[int(p),int(q)] = 1

    R = np.reshape(image,((m*n),1))        # flatten
    if cnt_tr==0:
        Xtrain = R
        Xtrain_image = image
        Xtrain_bin = image_bin
    else:
        Xtrain = np.hstack((Xtrain,R))
        Xtrain_image += image
        Xtrain_bin += image_bin
    cnt_tr += 1

pCk_temp = np.zeros((m*n,2))
y = 0
for p in range(0,m):
    for q in range(0,n):
        if Xtrain_bin[int(p),int(q)] != 0:
            Xtrain_bin[int(p),int(q)] = 1
            pCk_temp[y] = [p,q]
            y = y + 1
        else:
            Xtrain_bin[int(p),int(q)] = 0

#%% Get Average Face
avgFace = np.sum(Xtrain,axis=1)/(cnt_tr+1)
avgFaceIm = np.reshape(avgFace,(m,n))

#%% Subtract the mean
Xtrain_s = np.zeros(Xtrain.shape)
for j in range(0,cnt_tr):
    Xtrain_s[:,j] = Xtrain[:,j] - avgFace
    if j%10==0:
        print(j)
#%% Compute the SVD
U,S,Vh = np.linalg.svd(Xtrain_s,full_matrices=False)

#%% Item 1 Spare Sensor Placement
#Item 1 a) Reconstucting in sample
#Code 2.1 
#--------------------------------------------------------------------------------------------------------    
#%% Code 2.2
rp = [99]     # input values for no. of modes and sensors

fig,ax = plt.subplots(len(rp),4,figsize=(19,9.2),sharex='all',sharey='all')
ax = np.atleast_2d(ax)

print('P = '+str(rp[0]))
r = rp[0]     # No. of modes
p = rp[0]     # No. of sensors
Psi = U[:,:r]


km = KMeans(n_clusters=99)
km.fit(pCk_temp)

Q,R,P = qr(Psi.T,pivoting=True)
Pr = np.random.choice(np.arange(1,m*n,1),p-1)

pC = np.zeros((p,2))    # qr sensor coordinates
pCr = np.zeros((p,2))    # random sensor coordinates
pCc = np.zeros((p,2))    # current ambulance placement
pCk = np.zeros((p,2))

# Construct measurement matrix
C = np.zeros((p,m*n))
Cr = np.zeros((p,m*n))
    
for j in range(p-1):
    C[j,P[j]] = 1
    Cr[j,Pr[j]] = 1
        
    xp = P[j]%n
    yp = np.ceil(P[j]/n)
        
    xpr = Pr[j]%n
    ypr = np.ceil(Pr[j]/n)
        
    pC[j,:] = np.array([yp,xp])
    pCr[j,:] = np.array([xpr,ypr])

print(len(pC))
print(len(pCr))
print(len(pCc))
print(len(pCk))

pCc[0] = [2*5, 8*5]
pCc[1] = [95*4, 2*5]
pCc[2] = [80*5, 15*5]
pCc[3] = [79*5, 22*5]
pCc[4] = [75*5, 25*5]
pCc[5] = [99*4, 40*5]
pCc[6] = [44*5, 50*5]
pCc[7] = [50*5, 65*5]
pCc[8] = [60*5, 66*5]
pCc[9] = [50*5, 70*5]
pCc[10] = [60*5, 80*5]
pCc[11] = [70*5, 90*4]
pCc[12] = [85*5, 18*5]
pCc[13] = [99*4, 53*5]
for i in range(14,99):
    pCc[i] = [0,0]

temp = km.cluster_centers_

for q in range(0,99):
    g = int(temp[q][0])
    h = int(temp[q][1])
    pCk[q] = [g, h]
#%% Results / Data Gathering Testing
x_coor = []
y_coor = []
# get smallest distance between test points and sensor placements
dist = []
distr = []
distc = []
distk = []
image1 = np.zeros((n,m))

for i in range(0,99):

    P = np.random.choice(100,1, p=[0.015, 0.015,    0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.015,  0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.01,   0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005])
    x_coor.append(P[0])
    if 0 < i <= 33*4.8:
        P = np.random.choice(100,1, p=[ 0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0046,         0.0084,     0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0084,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,     0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,     0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134,         0.0134])
    elif 33*4.8 < i < 81*4.8:
        P = np.random.choice(100,1, p=[ 0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.01,           0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.015,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005])
    else:
        P = np.random.choice(100,1, p=[ 0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.005,          0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125,            0.01125])
    y_coor.append(P[0])
    
    point = [479 - (int(x_coor[int(i)])*4.8), 479 - (int(y_coor[int(i)])*4.8)]
    image1[int(point[0]),int(point[1])] = 1
    dist.append((distance.cdist([point], pC).min()))
    distr.append((distance.cdist([point], pCr).min()))
    distc.append((distance.cdist([point], pCc).min()))
    distk.append((distance.cdist([point], pCk).min()))

total = np.sum(dist)
totalr = np.sum(distr)
totalc = np.sum(distc)
totalk = np.sum(distk)

image_QR = np.zeros((n,m))
image_R = np.zeros((n,m))
image_C = np.zeros((n,m))
image_K = np.zeros((n,m))
for i in range(0,99):
    d = pC[int(i)]
    e = pCr[int(i)]
    f = pCc[int(i)]
    g = pCk[int(i)]
    image_QR[int(d[0]),int(d[1])] = 1
    image_R[int(e[0]),int(e[1])] = 1
    image_C[int(f[0]),int(f[1])] = 1
    image_K[int(g[0]),int(g[1])] = 1

print('')
print('')
print(pC)
print(len(pC))
print('---------------')
print(pCr)
print(len(pCr))
print('---------------')
print(pCc)
print(len(pCc))
print('---------------')
print(pCk)
print(len(pCk))
print('---------------')
print('---------------')
print('---------------')
print('Distance from SVD placed Ambulance:   ' + str(total))
print('---------------')
print('Distance from Randomly placed Ambulance:   ' + str(totalr))
print('---------------')
print('Distance from Ambulance placed near Hospitals:   ' + str(totalc))
print('---------------')
print('Distance from K-means clustering placed Ambulance:   ' + str(totalk))
print('---------------')

plt.figure()
plt.imshow(image_QR)
plt.figure()
plt.imshow(image_R)
plt.figure()
plt.imshow(image_C)
plt.figure()
plt.imshow(image_K)
plt.figure()
plt.imshow(image1)
plt.show()
