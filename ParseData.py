import os, sys
import cPickle

import numpy as np
import scipy.sparse as sp
import train as train

datatype='nonranden'

if  datatype == 'nr':
    datasetDir = 'nr90-10'
elif datatype == 'gpcr':
    datasetDir = 'gpcr90-10'
elif datatype == 'ic':
    datasetDir = 'ic90-10'
elif datatype == 'en':
    datasetDir = 'en90-10'
elif datatype == 'randnr':
    datasetDir = 'semep/random/nr90-10'
    datatype = 'nr'
elif datatype == 'randgpcr':
    datasetDir = 'semep/random/gpcr90-10'
    datatype = 'gpcr'
elif datatype == 'randic':
    datasetDir = 'semep/random/ic90-10'
    datatype = 'ic'
elif datatype == 'nonrandnr':
    datasetDir = 'semep/nonrandom/nr90-10'
    datatype = 'nr'
elif datatype == 'nonrandgpcr':
    datasetDir = 'semep/nonrandom/gpcr90-10'
    datatype = 'gpcr'
elif datatype == 'nonrandic':
    datasetDir = 'semep/nonrandom/ic90-10'
    datatype = 'ic'
elif datatype == 'nonranden':
    datasetDir = 'semep/nonrandom/en90-10'
    datatype = 'en'

# Put the freebase15k data absolute path here
datapath = 'data/'+datasetDir+'/'
dataPrefix = 'drugs_targets_'
processedDataDir = 'pProcessed/'
assert datapath is not None
print(datapath)



if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs


#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []
#read list of all drugs
f = open(datapath+datatype+'_drugs.txt', 'r')
dat = f.readlines()
f.close()
for i in dat:
    entleftlist += [i[:-1]]
#read list of all targets
f = open(datapath+datatype + '_targets.txt', 'r')
dat = f.readlines()
f.close()
for i in dat:
    entrightlist += [i[:-1]]
#supported relations
rellist += ['interactive','nonInteractive']

entleftset = entleftlist
entsharedset = list(set(entleftlist) & set(entrightlist))
entrightset = entrightlist
relset = rellist

entity2idx = {}
idx2entity = {}

# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in entrightset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbright = idx
for i in entsharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbright
for i in entleftset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbleft = idx - (nbshared + nbright)

print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary
for i in relset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - (nbright + nbshared + nbleft)
print "Number of relations: ", nbrel

f = open(datapath+processedDataDir+dataPrefix+'entity2idx.pkl', 'w')
g = open(datapath+processedDataDir+dataPrefix+'idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()



#read similarity matrices
f = open(datapath+datatype+'_matrix_drugs.txt')
dat = f.readlines()
f.close()
drugSimilarity = sp.lil_matrix( ( len(entleftset), len(entleftset) ),dtype='float32')
for i in range(len(entleftset)):
    line = dat[i+1].split(' ')
    for j in range(len(entleftset)):
        simi = float(line[j])
        #drugSimlarity[i,j] = -1+1/simi  #storing distance instead of similarity
        drugSimilarity[i,j] = simi#1-simi

f = open(datapath+datatype+'_matrix_targets.txt')
dat = f.readlines()
f.close()
targetSimilarity = sp.lil_matrix( ( len(entrightset), len(entrightset) ),dtype='float32')
for i in range(len(entrightset)):
    line = dat[i+1].split(' ')
    for j in range(len(entrightset)):
        simi = float(line[j])
        #targetSimilarity[i,j] =-1+1/simi    #storing distance instead of similarity
        targetSimilarity[i,j] = simi#1-simi

f = open(datapath+processedDataDir+dataPrefix+'drugs_simi.pkl', 'w')
g = open(datapath+processedDataDir+dataPrefix+'targets_simi.pkl', 'w')
cPickle.dump(drugSimilarity.tocsr(), f, -1)
cPickle.dump(targetSimilarity.tocsr(), g, -1)
f.close()
g.close()



#################################################
### Creation of the dataset files from 10 folds

# for fold in range(10):
for fold in range(10):
    print("\n*****FOLD: "+str(fold+1)+" DATA PARSING********\n")

    currentFoldPath = datapath + 'folds/fold' + str(fold + 1)+'/'
    print(currentFoldPath)

    unseen_ents=[]
    remove_tst_ex=[]

    for datatyp in ['train', 'test']:
        print datatyp
        f = open(currentFoldPath + 'drugs_targets_%s.txt' % datatyp, 'r')
        dat = f.readlines()
        f.close()

        # Declare the dataset variables
        inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                dtype='float32')
        inpr = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                dtype='float32')
        inpo = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                dtype='float32')
        # Fill the sparse matrices
        ct = 0
        for i in dat:
            lhs, rhs, rel = parseline(i[:-1])
            if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx:
                inpl[entity2idx[lhs[0]], ct] = 1
                inpr[entity2idx[rhs[0]], ct] = 1
                inpo[entity2idx[rel[0]], ct] = 1
                ct += 1
            else:
                if lhs[0] in entity2idx:
                    unseen_ents+=[lhs[0]]
                if rel[0] in entity2idx:
                    unseen_ents+=[rel[0]]
                if rhs[0] in entity2idx:
                    unseen_ents+=[rhs[0]]
                remove_tst_ex+=[i[:-1]]

        if not os.path.isdir(currentFoldPath+'processed/'):
            os.mkdir(currentFoldPath+'processed/')

        f = open(currentFoldPath+'processed/drugs_targets_%s-lhs.pkl' % datatyp, 'w')
        g = open(currentFoldPath+'processed/drugs_targets_%s-rhs.pkl' % datatyp, 'w')
        h = open(currentFoldPath+'processed/drugs_targets_%s-rel.pkl' % datatyp, 'w')
        cPickle.dump(inpl.tocsr(), f, -1)
        cPickle.dump(inpr.tocsr(), g, -1)
        cPickle.dump(inpo.tocsr(), h, -1)
        f.close()
        g.close()
        h.close()

    unseen_ents=list(set(unseen_ents))
    print len(unseen_ents)
    remove_tst_ex=list(set(remove_tst_ex))
    print len(remove_tst_ex)

    for i in remove_tst_ex:
        print i


    print("\n*****FOLD: "+str(fold+1)+" DATA PARSED********\n")





