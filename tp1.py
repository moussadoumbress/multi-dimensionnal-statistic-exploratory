#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:02:23 2022

@author: gabinfay
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

####

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# ## 
#y_names = [target_names[i] for i in y]

print('moyenne de chaque variable')
print(X.mean(0))
print('ecart type de chaque variable')
print(X.std(0))
print('min de chaque variable')
print(np.min(X, axis=0))
print('max de chaque variable')
print(np.max(X,axis=0))

m = X.shape[0]
k = X.shape[1]
print(f'm = {m}, k = {k}')
print(f'nombre de données : m*k = {m*k}')

###### MNIST ###

mnist = datasets.fetch_openml('mnist_784')
x = mnist.data
y = mnist.target

imgs = x.reshape((70000,28,28))
plt.imshow(imgs[0,:,:])
plt.show()
print(f"c'est un chiffre {y[0]}")
plt.imshow(imgs[1,:,:])
plt.show()
print(f"c'est un chiffre {y[1]}")

print(f'nombre de classes : {len(np.unique(y))}')

print("l'image moyenne :")
plt.imshow(x.mean(0).reshape((28,28)))
plt.show()

### BLOBS ###

## 1 to 3

# plt.close('all')
plt.figure()
plt.title('Génération de 4 clusters')

x,y = datasets.make_blobs(n_samples=1000, n_features=2, centers=4)
cols = ['C0','C1', 'C2', 'C3']
labels = ['0','1','2','3']

for i in np.arange(4):
  plt.scatter(x[:,0][y==i], x[:,1][y==i],color=cols[i],
              s=47,label=labels[i])
plt.legend()
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim((-15,15))
plt.ylim((-15,15))

## 4

x,y = datasets.make_blobs(n_samples=100, n_features=2, centers=2)
xx, yy = datasets.make_blobs(n_samples=500, n_features=2, centers=3)

x = np.concatenate((x,xx), axis=0)
y = np.concatenate((y,yy),axis=0)

# plt.close('all')
plt.figure()
plt.title('Génération de 4 clusters')

cols = ['C0','C1', 'C2', 'C3']
labels = ['0','1','2','3']

for i in np.arange(3):
  plt.scatter(x[:,0][y==i], x[:,1][y==i],color=cols[i],
              s=47,label=labels[i])
plt.legend()
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim((-15,15))
plt.ylim((-15,15))














