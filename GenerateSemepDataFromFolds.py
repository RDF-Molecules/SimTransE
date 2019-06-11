import os,sys
import cPickle

import numpy as np
import scipy.sparse as sp

dataset='en'
if dataset == 'nr':
    datasetDir = 'semep/nonrandom/nr90-10'
elif dataset == 'gpcr':
    datasetDir = 'semep/nonrandom/gpcr90-10'
elif dataset == 'ic':
    datasetDir = 'semep/nonrandom/ic90-10'
elif dataset == 'en':
    datasetDir = 'semep/nonrandom/en90-10'
if dataset == 'randnr':
    datasetDir = 'semep/random/nr90-10'
elif dataset == 'randgpcr':
    datasetDir = 'semep/random/gpcr90-10'
    dataset = 'gpcr'
elif dataset == 'randic':
    datasetDir = 'semep/random/ic90-10'
    dataset = 'ic'
elif dataset == 'randen':
    datasetDir = 'semep/random/en90-10'
    dataset = 'en'

drugsTargetsPath = 'data/'+datasetDir+'/folds/'+dataset+'_drugs_targets.txt'
foldsDir = 'data/'+datasetDir+'/folds/'
foldsPath = foldsDir+dataset+'_folds.txt'
realInteractionsPath = 'data/'+datasetDir+'/folds/'+dataset+'_admat_dgc.txt'

print(drugsTargetsPath)
print(foldsPath)
print(realInteractionsPath)



#read drugs and targets entities from drugsTargetsPath
drugsTargetFile = open(drugsTargetsPath, 'r')
fData = drugsTargetFile.readlines()
drugsTargetFile.close()

#read drug entities
drugs = fData[0].split('\t')
drugs[-1] = drugs[-1].split('\n')[0]

#read target entities
targets = fData[1].split('\t')
targets[-1] = targets[-1].split('\n')[0]

#read the set of real interactions
realInteractions = {}
realInteractionsFile = open(realInteractionsPath,'r')
rData = realInteractionsFile.readlines()
i = 0
for line in rData:
    line2 = line.rstrip("\n")
    tok = line2.split("\t")
    j = 0
    #print(len(tok))
    for x in tok:
        interactionValue = int(x)
        if interactionValue == 1:
            realInteractions[drugs[j] + '-' + targets[i]] = 1
        j += 1
    i += 1
realInteractionsFile.close()

#create drug target pairs and assign id to each-that is used to extract all possible interactions from fold file
drugsTargets = {}
count=1
for d in drugs:
    for t in targets:
        drugsTargets[count]=[d,t]
        count = count+1




foldsFile = open(foldsPath,'r')
fData = foldsFile.readlines()
foldsFile.close()


# ct=1
# testFoldIndices = [0,2,4,6,8,]
# #for 10 folds
# for testIndex in testFoldIndices:
#
#     currentFoldPath = foldsDir + 'fold' + str(ct)+'/'
#     ct+=1
#     if not os.path.exists(currentFoldPath):
#         os.mkdir(currentFoldPath)
#     print(currentFoldPath)
#     # read interactionID
#     interactionIDs = fData[0].split(', ')
#
#     totalTestInteractions = 0
#     totalTrainInteractions = 0
#
#     trainDataFile = open(currentFoldPath + '/drugs_targets_train.txt', 'w')
#     testDataFile = open(currentFoldPath + '/drugs_targets_test.txt', 'w')
#     testItem = ''
#     foldFData = fData[1:]
#     for itemIndex in xrange(len(foldFData)):
#         if( testIndex==itemIndex ): #ignore one fold for test data
#             #put data in test file
#             testItem = foldFData[testIndex]
#             startIdx, endIdx = testItem.split('\n')[0].split(' ')
#             startIdx = int(startIdx) - 1
#             endIdx = int(endIdx) - 1
#             testIndex+=1
#             for x in range(startIdx, endIdx+1):
#                 totalTestInteractions += 1
#                 drugTarget = drugsTargets[int(interactionIDs[x])]
#                 if(drugTarget[0]+'-'+drugTarget[1] in realInteractions):
#                     testDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tinteractive\n')
#                 else:
#                     testDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tnonInteractive\n')
#
#         else:
#             trainItem = foldFData[itemIndex]
#             startIdx, endIdx = trainItem.split('\n')[0].split(' ')
#             startIdx = int(startIdx) - 1
#             endIdx = int(endIdx) - 1
#             for x in range(startIdx, endIdx+1):
#                 totalTrainInteractions += 1
#                 drugTarget = drugsTargets[int(interactionIDs[x])]
#                 if(drugTarget[0]+'-'+drugTarget[1] in realInteractions):
#                     trainDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tinteractive\n')
#                 else:
#                     trainDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tnonInteractive\n')
#
#     testDataFile.close()
#     trainDataFile.close()
#
#     print("total test interactions: " + str(totalTestInteractions))
#     print("total train interactions: " + str(totalTrainInteractions))
#






testFoldIndices = [0,1,2,3,4,5,6,7,8,9]
#for 10 folds
for testIndex in testFoldIndices:

    currentFoldPath = foldsDir + 'fold' + str(testIndex+1)+'/'
    if not os.path.exists(currentFoldPath):
        os.mkdir(currentFoldPath)
    print(currentFoldPath)
    # read interactionID
    interactionIDs = fData[0].split(', ')

    totalTestInteractions = 0
    totalTrainInteractions = 0

    trainDataFile = open(currentFoldPath + '/drugs_targets_train.txt', 'w')
    testDataFile = open(currentFoldPath + '/drugs_targets_test.txt', 'w')
    testItem = ''
    foldFData = fData[1:]
    for itemIndex in xrange(len(foldFData)):
        if( testIndex==itemIndex ): #ignore one fold for test data
            #put data in test file
            testItem = foldFData[testIndex]
            startIdx, endIdx = testItem.split('\n')[0].split(' ')
            startIdx = int(startIdx) - 1
            endIdx = int(endIdx) - 1
            for x in range(startIdx, endIdx+1):
                totalTestInteractions += 1
                drugTarget = drugsTargets[int(interactionIDs[x])]
                if(drugTarget[0]+'-'+drugTarget[1] in realInteractions):
                    testDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tinteractive\n')
                else:
                    testDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tnonInteractive\n')

        else:
            trainItem = foldFData[itemIndex]
            startIdx, endIdx = trainItem.split('\n')[0].split(' ')
            startIdx = int(startIdx) - 1
            endIdx = int(endIdx) - 1
            for x in range(startIdx, endIdx+1):
                totalTrainInteractions += 1
                drugTarget = drugsTargets[int(interactionIDs[x])]
                if(drugTarget[0]+'-'+drugTarget[1] in realInteractions):
                    trainDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tinteractive\n')
                else:
                    trainDataFile.write(drugTarget[0] + '\t' + drugTarget[1] + '\tnonInteractive\n')

    testDataFile.close()
    trainDataFile.close()

    print("total test interactions: " + str(totalTestInteractions))
    print("total train interactions: " + str(totalTrainInteractions))





