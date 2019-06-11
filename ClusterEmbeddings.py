

import os
import sys
import time
import copy
import cPickle

import numpy as np

import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import matplotlib.cm as cm

import plotly.plotly as py

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.art3d import Line3DCollection



def plot2dVectors(x1,x2):
    u = np.sin(np.pi * x1) * np.cos(np.pi * x2)
    v = -np.cos(np.pi * x1) * np.sin(np.pi * x2)


    plt.figure()
    Q = plt.quiver(x1,x2,u,v, units='width',)
    plt.quiverkey(Q,-0.3,0.3,1,'test')
    # ax.set_xlim(-0.6,0.6)
    # ax.set_ylim(-0.6, 0.6)
    # ax.set_zlim(-0.6, 0.6)
    # ax.set_xlim(-2, 1)
    # ax.set_ylim(-2, 1)
    # ax.set_zlim(-2, 2)

    plt.draw()
    plt.show()

def plot3dVectors(x1,x2,x3):



    u = np.sin(np.pi * x1) * np.cos(np.pi * x2) * np.cos(np.pi * x3)
    v = -np.cos(np.pi * x1) * np.sin(np.pi * x2) * np.cos(np.pi * x3)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x1) * np.cos(np.pi * x2) * np.sin(np.pi * x3))




    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Q = ax.quiver(x1, x2, x3, u, v, w, normalize=True, length=0.5)

    # ax.set_xlim(-0.6,0.6)
    # ax.set_ylim(-0.6, 0.6)
    # ax.set_zlim(-0.6, 0.6)
    # ax.set_xlim(-2, 1)
    # ax.set_ylim(-2, 1)
    # ax.set_zlim(-2, 2)

    plt.draw()
    plt.show()

def plotClusterData(x1,x2,x3,c1,c2,c3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1,x2,x3)

    # ax.scatter(c1,c2,c3,c='red',marker='+')

    fig.show()


if __name__ == '__main__':

    np.random.seed(5)
    showNames=True
    dataType = 'nr'
    savePCA = True

    percentile = '80'
    if dataType=='gpcr':
        leftEntities = 223
        rightEntities = 95
        dataTypeDir = "gpcr90-10"

    elif dataType=='nr':
        leftEntities = 54
        rightEntities = 26
        dataTypeDir = "nr90-10"

    elif dataType=='ic':
        leftEntities = 210
        rightEntities = 204
        dataTypeDir = "ic90-10"
    elif dataType=='en':
        leftEntities = 445
        rightEntities = 664
        dataTypeDir = "en90-10"

    # for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for fold in [4]:

        fold = str(fold)
        savepath = 'data/' + dataTypeDir + '/pProcessed/'
        foldPath = 'data/' + dataTypeDir + '/folds/fold' + fold + '/processed/p' + percentile + '/'

        dataset = 'drugs_targets_'

        # f = open(foldPath + 'best_valid_model.pkl')
        f = open(foldPath + 'current_model.pkl')

        embeddings = cPickle.load(f)
        leftop = cPickle.load(f)
        rightop = cPickle.load(f)
        simfn = cPickle.load(f)
        f.close()

        f = open(savepath + dataset + 'idx2entity.pkl')
        idx2entity = cPickle.load(f)
        f.close()

        X = embeddings[0].E.get_value().T

        pca = PCA(n_components=2).fit(X)
        X = pca.transform(X)
        x1 = X[:, 0]
        x2 = X[:, 1]
        # x3 = X[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        target1 = x1[0:rightEntities]
        target2 = x2[0:rightEntities]

        drug1 = x1[rightEntities:rightEntities + leftEntities]
        drug2 = x2[rightEntities:rightEntities + leftEntities]

        relation1 = x1[rightEntities + leftEntities:]
        relation2 = x2[rightEntities + leftEntities:]
        ax.scatter(target1, target2, color='r')
        ax.scatter(drug1, drug2, color='g')
        ax.scatter(relation1, relation2, color='b')

        if showNames:
            count = 0
            for xy in zip(x1, x2):
                ax.annotate(idx2entity[count], xy=xy, textcoords='data')  # <--
                count += 1

        fig.show()

        plot2dVectors(x1, x2)

        if (savePCA):
            X = embeddings[0].E.get_value().T
            pca = PCA(n_components=3).fit(X)
            X = pca.transform(X)
            x1 = X[:, 0]
            x2 = X[:, 1]
            x3 = X[:, 2]
            plot3dVectors(x1, x2, x3)

            f = open(savepath + 'PCA3D', 'w')
            fMeta = open(savepath + 'PCA3DMeta', 'w')
            count = 0
            for x, y, z in zip(x1, x2, x3):
                f.write(str(x) + '\t' + str(y) + '\t' + str(z) + '\n')
                fMeta.write(idx2entity[count] + '\n')
                count += 1

            f.close()
            fMeta.close()









    # estimator = KMeans(n_clusters=3)
    # estimator.fit(X)
    # labels = estimator.labels_
    # c1 = estimator.cluster_centers_[:, 0]
    # c2 = estimator.cluster_centers_[:, 1]
    # c3 = estimator.cluster_centers_[:, 2]
   # plot3dVectors(c1, c2, c3)

    # plotClusterData(x1,x2,x3,c1,c2,c3)

    sys.exit(0)

