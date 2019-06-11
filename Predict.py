
import sys
import cPickle
from numpy import interp
from evaluate import *
from sklearn.preprocessing import MinMaxScaler
import collections
from sortedcontainers import SortedDict

def getNovelTriples(validationFilePath,entity2Idx):

    f = open(validationFilePath)
    dat = f.readlines()[1:]
    f.close()

    stitchValidPredictions = {}
    keggValidPredictions = {}
    for i in dat:
        drug, target, stitch, kegg, value = i[:-1].split('\t')
        if (stitch == 'Yes'):
            stitchValidPredictions[drug + '-' + target] = 1
        if (kegg == 'Yes'):
            keggValidPredictions[drug + '-' + target] = 1

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(stitchValidPredictions.values()) + len(keggValidPredictions.values())),
        dtype='float32')
    inpr = sp.lil_matrix(
        (np.max(entity2idx.values()) + 1, len(stitchValidPredictions.values()) + len(keggValidPredictions.values())),
        dtype='float32')
    inpo = sp.lil_matrix(
        (np.max(entity2idx.values()) + 1, len(stitchValidPredictions.values()) + len(keggValidPredictions.values())),
        dtype='float32')

    ct = 0
    for key in stitchValidPredictions:
        leftEnt, rightEnt = key.split('-')
        inpl[entity2idx[leftEnt], ct] = 1
        inpr[entity2idx[rightEnt], ct] = 1
        inpo[0, ct] = 1
        ct += 1

    for key in keggValidPredictions:
        leftEnt, rightEnt = key.split('-')
        inpl[entity2idx[leftEnt], ct] = 1
        inpr[entity2idx[rightEnt], ct] = 1
        inpo[0, ct] = 1
        ct += 1

    idxl = convert2idx(inpl)
    idxr = convert2idx(inpr)
    idxo = convert2idx(inpo)

    return np.concatenate([idxl, idxo, idxr]).reshape(3, idxl.shape[0]).T



def drugTargetInteracts(truePositivePredictions, falsePositivePredictions, predictedIndexesInTestTriples, auc_prediction_lines,
                        leftEntity, rightEntity, targetSize, leftResult,rightResult, test_triples=None,  predictedIndexesInNovelTriples=None, novel_triples=None):
    # here drug-target have interaction
    auc_prediction_lines[rightEntity][leftEntity - targetSize] = 1

    # check if interaction is also interaction in test data
    if (relation == 0):
        truePositivePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]
        if(test_triples is not None):
            tt = np.where((test_triples[:, 0] == leftEntity) & (test_triples[:, 1] == 0) & (test_triples[:, 2] == rightEntity))[0]
            if (len(tt) > 0):
                predictedIndexesInTestTriples += list(tt)
    else:
        if (novel_triples is not None):
            tt = np.where((novel_triples[:, 0] == leftEntity) & (novel_triples[:, 1] == 0) & (novel_triples[:, 2] == rightEntity))[0]
            if (len(tt) > 0):
                truePositivePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]
                predictedIndexesInNovelTriples += list(tt)
            else:
                falsePositivePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]
        else:
            # auc_prediction_lines[rightEntity][leftEntity - targetSize] = resultLInter
            falsePositivePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]


def drugTargetNoInteracts(trueNegativePredictions, falseNegativePredictions, auc_prediction_lines,
                          leftEntity, rightEntity, targetSize,leftResult,rightResult):
    # here drug-target have no interaction
    auc_prediction_lines[rightEntity][leftEntity - targetSize] = 0

    if (relation == 1):
        trueNegativePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]
    else:
        # auc_prediction_lines[rightEntity][leftEntity - targetSize] = leftResult
        falseNegativePredictions += [[idx2entity[leftEntity], idx2entity[rightEntity]]]


if __name__ == '__main__':

    dataType = 'nr'

    if dataType=='gpcr':
        leftThreshold = 0.5  # for gpcr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.339, leftThreshold + 0.3341, leftThreshold + 0.270, leftThreshold + 0.36,
                          leftThreshold + 0.2, leftThreshold + 0.34, leftThreshold + 0.34, leftThreshold + 0.298,
                          leftThreshold + 0.314, leftThreshold + 0.327]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.05, rightThreshold + 0.015, rightThreshold + 0.0,
                           rightThreshold + 0.032, rightThreshold + 0.1, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.03, rightThreshold + 0.05]
        dataTypeDir = 'gpcr90-10'

    elif dataType == 'ic':
        leftThreshold = 0.8  # for ic
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.064, leftThreshold + 0.1, leftThreshold + 0.01, leftThreshold + 0.09,
                          leftThreshold + 0.02, leftThreshold + 0.03, leftThreshold + 0.0, leftThreshold + 0.021,
                          leftThreshold + 0.10,leftThreshold + 0.0]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0,rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0,rightThreshold + 0.0]
        dataTypeDir = 'ic90-10'



    if dataType=='randnr':
        leftThreshold = 0.5  # for nr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.36, leftThreshold + 0.312, leftThreshold + 0.2, leftThreshold + 0.35,
                          leftThreshold + 0.07,leftThreshold +0.1, leftThreshold + 0.1, leftThreshold + 0.35,
                          leftThreshold + 0.27,leftThreshold + 0.38]
        rightThresholds = [rightThreshold + 0.43, rightThreshold + 0.1, rightThreshold + 0.2, rightThreshold + 0.15,
                           rightThreshold + 0.1,rightThreshold + 0.3, rightThreshold + 0.22, rightThreshold + 0.22,
                           rightThreshold + 0.42,rightThreshold + 0.16]
        dataType = 'nr'
        dataTypeDir = 'semep/random/nr90-10'

    if dataType=='nonrandic':
        leftThreshold = 0.5  # for ic
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.34, leftThreshold + 0.39, leftThreshold + 0.37, leftThreshold + 0.34,
                          leftThreshold + 0.357, leftThreshold + 0.33, leftThreshold + 0.364, leftThreshold + 0.35,
                          leftThreshold + 0.35, leftThreshold + 0.36]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.01, rightThreshold + 0.0]
        dataType = 'ic'
        dataTypeDir = 'semep/nonrandom/ic90-10'

    if dataType=='randgpcr':
        leftThreshold = 0.5  # for gpcr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.32, leftThreshold + 0.22, leftThreshold + 0.31, leftThreshold + 0.3,
                          leftThreshold + 0.26, leftThreshold + 0.364, leftThreshold + 0.276, leftThreshold + 0.35,
                          leftThreshold + 0.29, leftThreshold + 0.33]
        rightThresholds = [rightThreshold + 0.04, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.01, rightThreshold + 0.0, rightThreshold + 0.01, rightThreshold + 0.02,
                           rightThreshold + 0.01, rightThreshold + 0.04]
        dataType = 'gpcr'
        dataTypeDir = 'semep/random/gpcr90-10'
    if dataType=='randic':
        leftThreshold = 0.5  # for ic
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.34, leftThreshold + 0.39, leftThreshold + 0.37, leftThreshold + 0.34,
                          leftThreshold + 0.357, leftThreshold + 0.33, leftThreshold + 0.364, leftThreshold + 0.35,
                          leftThreshold + 0.35, leftThreshold + 0.36]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.01, rightThreshold + 0.0]
        dataType = 'ic'
        dataTypeDir = 'semep/random/ic90-10'

    if dataType=='nr':
        leftThreshold = 0.7  # for nr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.11, leftThreshold + 0.0, leftThreshold + 0.08, leftThreshold + 0.05,
                          leftThreshold , leftThreshold , leftThreshold + 0.085, leftThreshold + 0.1415,
                          leftThreshold + 0.04,leftThreshold + 0.358]
        rightThresholds = [rightThreshold + 0.2, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.28,
                           rightThreshold + 0.08,rightThreshold + 0.3, rightThreshold + 0.136, rightThreshold + 0.285,
                           rightThreshold + 0.23,rightThreshold + 0.20]
        dataTypeDir = 'nr90-10'

    if dataType=='nonrandgpcr':
        leftThreshold = 0.5  # for gpcr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.214, leftThreshold + 0.22, leftThreshold + 0.31, leftThreshold + 0.3,
                          leftThreshold + 0.26, leftThreshold + 0.364, leftThreshold + 0.276, leftThreshold + 0.35,
                          leftThreshold + 0.29, leftThreshold + 0.33]
        rightThresholds = [rightThreshold + 0.23, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.01, rightThreshold + 0.0, rightThreshold + 0.01, rightThreshold + 0.02,
                           rightThreshold + 0.01, rightThreshold + 0.04]
        dataType = 'gpcr'
        dataTypeDir = 'semep/nonrandom/gpcr90-10'

    if dataType == 'en':
        leftThreshold = 0.62  # for en
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07]
        rightThresholds = [rightThreshold + 0.07, rightThreshold + 0.01, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0]
        dataTypeDir = 'en90-10'



    if dataType=='randen':
        leftThreshold = 0.6  # for en
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.1, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.01, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0]
        dataType = 'en'
        dataTypeDir = 'semep/random/en90-10'

    if dataType=='transenr':
        leftThreshold = 0.6  # for nr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0, leftThreshold + 0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0.0,leftThreshold +0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0,leftThreshold + 0]
        rightThresholds = [rightThreshold + 0, rightThreshold + 0.0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0]
        dataType = 'nr'
        dataTypeDir = 'transE/nr90-10'

    if dataType=='nonranden':
        leftThreshold = 0.6  # for en
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.1, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07, leftThreshold + 0.07,
                          leftThreshold + 0.07, leftThreshold + 0.07]
        rightThresholds = [rightThreshold + 0.0, rightThreshold + 0.01, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0, rightThreshold + 0.0,
                           rightThreshold + 0.0, rightThreshold + 0.0]
        dataType = 'en'
        dataTypeDir = 'semep/nonrandom/en90-10'
    if dataType=='nonrandnr':
        leftThreshold = 0.5  # for nr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0.36, leftThreshold + 0.312, leftThreshold + 0.4, leftThreshold + 0.35,
                          leftThreshold + 0.07,leftThreshold +0.1, leftThreshold + 0.1, leftThreshold + 0.35,
                          leftThreshold + 0.27,leftThreshold + 0.38]
        rightThresholds = [rightThreshold + 0.4, rightThreshold + 0.05, rightThreshold + 0.2, rightThreshold + 0.15,
                           rightThreshold + 0.1,rightThreshold + 0.3, rightThreshold + 0.22, rightThreshold + 0.22,
                           rightThreshold + 0.42,rightThreshold + 0.16]
        dataType = 'nr'
        dataTypeDir = 'semep/nonrandom/nr90-10'

    if dataType=='transegpcr':
        leftThreshold = 0.6  # for nr
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0, leftThreshold + 0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0.0,leftThreshold +0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0,leftThreshold + 0]
        rightThresholds = [rightThreshold + 0, rightThreshold + 0.0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0]
        dataType = 'gpcr'
        dataTypeDir = 'transE/gpcr90-10'

    if dataType=='transeic':
        leftThreshold = 0.6  # for ic
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0, leftThreshold + 0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0.0,leftThreshold +0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0,leftThreshold + 0]
        rightThresholds = [rightThreshold + 0, rightThreshold + 0.0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0]
        dataType = 'ic'
        dataTypeDir = 'transE/ic90-10'

    if dataType=='transeen':
        leftThreshold = 0.6  # for ic
        rightThreshold = 0.0
        leftThresholds = [leftThreshold + 0, leftThreshold + 0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0.0,leftThreshold +0, leftThreshold + 0, leftThreshold + 0,
                          leftThreshold + 0,leftThreshold + 0]
        rightThresholds = [rightThreshold + 0, rightThreshold + 0.0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0, rightThreshold + 0, rightThreshold + 0,
                           rightThreshold + 0,rightThreshold + 0]
        dataType = 'en'
        dataTypeDir = 'transE/en90-10'

    foldList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # foldList = [1]
    percentileIndices = [0, 1, 2, 3]
    percentileIndices = [0]
    completePrediction = False

    print(dataTypeDir)

    dataset = 'drugs_targets_'
    datapath = 'data/' + dataTypeDir + '/pProcessed/'
    validationFilePath = 'data/' + dataTypeDir + '/' + dataType + '_complement_interactions_validated.txt'


    percentileThresholds = getPercentileThresholds()

    f = open(datapath + dataset + 'entity2idx.pkl')
    entity2idx = cPickle.load(f)
    f.close()

    f = open(datapath + dataset + 'idx2entity.pkl')
    idx2entity = cPickle.load(f)
    f.close()

    novel_triples = getNovelTriples(validationFilePath,entity2idx)

    drugSimilarity = load_file(datapath + dataset + 'drugs_simi.pkl')
    targetSimilarity = load_file(datapath + dataset + 'targets_simi.pkl')

    # create a list of left and right ids
    uniqueLeft = [i + targetSimilarity.shape[0] for i in range(drugSimilarity.shape[0])]
    uniqueRight = [i for i in range(targetSimilarity.shape[0])]

    Nrel = 2
    Nsyn = len(uniqueLeft)+len(uniqueRight)+Nrel

    print("left threshold: " + str(leftThresholds))
    print("right threshold: " + str(rightThresholds))

    percentileNames = ['p80', 'p90', 'p95', 'p98']

    precisions=[]
    recalls = []
    accuracies = []
    sortedPredictions = dict()

    for fd in foldList:
    # for fd in [1,2,3,4]:

        fold = str(fd)

        print("\n")
        print("Prediction for fold: " + fold)

        savepath = 'data/' + dataTypeDir + '/folds/fold' + fold + '/processed/'

        for percentileIndex in percentileIndices:

            percentile = percentileNames[percentileIndex]

            print("\nPercentile: " + percentile)
            percentileLeftThreshold = percentileThresholds[dataType + '-left'][percentileIndex]
            percentileRightThreshold = percentileThresholds[dataType + '-right'][percentileIndex]

            modelPath = savepath + percentile + '/' + 'best_valid_model.pkl'
            modelPath = savepath + percentile + '/' + 'current_model.pkl'
            print(modelPath)
            resultsPath = 'results/' + dataTypeDir + '/' + percentile + '/'
            aucPredictionFileName = 'auc-' + '-fold'+fold+'.fold'


            f = open(modelPath)
            embeddings = cPickle.load(f)
            leftop = cPickle.load(f)
            rightop = cPickle.load(f)
            simfn = cPickle.load(f)
            f.close()


            # Load data
            l = load_file(savepath + dataset + 'test-lhs.pkl')
            r = load_file(savepath + dataset + 'test-rhs.pkl')
            o = load_file(savepath + dataset + 'test-rel.pkl')
            o = o[-Nrel:, :]
            # # from test data
            testIdxl = convert2idx(l)
            testIdxr = convert2idx(r)
            testIdxo = convert2idx(o)
            test_triples = np.concatenate([testIdxl, testIdxo, testIdxr]).reshape(3, testIdxl.shape[0]).T
            # test_triples = novel_triples

            tl = load_file(savepath  + dataset + 'train-lhs.pkl')
            tr = load_file(savepath + dataset + 'train-rhs.pkl')
            to = load_file(savepath + dataset + 'train-rel.pkl')
            to = to[-Nrel:, :]
            # from train data
            idxtl = convert2idx(tl)
            idxtr = convert2idx(tr)
            idxto = convert2idx(to)
            train_triples = np.concatenate([idxtl, idxto, idxtr]).reshape(3, idxtl.shape[0]).T
            if completePrediction:
                prediction_triples = np.concatenate([train_triples,test_triples])
            else:
                prediction_triples = test_triples

            # ranking functions to calculate similarity between entities
            ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=Nsyn)
            rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=Nsyn)


            keggPredictions = []
            stitchPredictions = []

            truePositivePredictions = []
            falsePositivePredictions = []
            falseNegativePredictions = []
            trueNegativePredictions = []

            predictedIndexesInTestTriples = []
            predictedIndexesInNovelTriples = []
            truePositiveInTest = []
            falsePositiveInTest = []
            trueNegativeInTest = []
            falseNegativeInTest = []

            targetSize = len(uniqueRight)
            auc_prediction_lines = np.zeros((targetSize,len(uniqueLeft)),dtype=np.int8)
            auc_actual_lines = np.zeros((targetSize,len(uniqueLeft)),dtype=np.int8)

            scaler = MinMaxScaler()
            resultsArray = np.empty((Nsyn-1,Nsyn-1),dtype=float)
            for rightID in uniqueRight:
                # x=np.argsort(np.argsort((ranklfunc(rightID, 0)[0]).flatten())[::-1]).flatten()
                # x = x.astype(np.float32, copy=False)
                # x = (ranklfunc(rightID, 0)[0]).flatten()  # scores for interaction
                # scoresL = (x-min(x))/(max(x)-min(x))
                scores_lInter = (ranklfunc(rightID, 0)[0]).flatten()  # scores for interaction
                scoresL = interp(scores_lInter, [scores_lInter.min(), scores_lInter.max()], [0, 1])
                resultsArray[rightID] = scoresL

            for leftID in uniqueLeft:
                # lCount += 1;
                # x = np.argsort(np.argsort((rankrfunc(leftID, 0)[0]).flatten())[::-1]).flatten()
                # x = x.astype(np.float32, copy=False)
                # x = (rankrfunc(leftID, 0)[0]).flatten()  # scores for interaction
                # scoresR = (x-min(x))/(max(x)-min(x))
                scores_rInter = (rankrfunc(leftID, 0)[0]).flatten()  # scores for interaction
                scoresR = interp(scores_rInter, [scores_rInter.min(), scores_rInter.max()], [0, 1])


                resultsArray[leftID] =  scoresR


            #predictions
            for triple in prediction_triples:
                leftEntity = triple[0]
                rightEntity = triple[2]
                relation = triple[1]

                # auc_actual_lines[rightEntity][leftEntity - targetSize] = 1-relation

                thresholdLeft = leftThresholds[int(fold)-1]
                thresholdRight = rightThresholds[int(fold)-1]
                resultLInter = resultsArray[leftEntity][rightEntity]
                resultRInter = resultsArray[rightEntity][leftEntity]
                resultInteraction = np.mean([resultLInter, resultRInter])
                # resultInteraction = resultLInter

                # if (resultInteraction >= leftThreshold):
                if (resultLInter >= thresholdLeft and resultRInter >= thresholdRight):
                    sortedPredictions[idx2entity[leftEntity]+'\t'+idx2entity[rightEntity]+'\tinteractive'+'\t'] = resultLInter*100

                    drugTargetInteracts(truePositivePredictions,falsePositivePredictions,predictedIndexesInTestTriples,
                                        auc_prediction_lines,leftEntity,rightEntity,targetSize,resultLInter,resultRInter,
                                        test_triples,predictedIndexesInNovelTriples=predictedIndexesInNovelTriples,novel_triples=None)
                else:
                    drugTargetNoInteracts(trueNegativePredictions, falseNegativePredictions, auc_prediction_lines,leftEntity,
                                          rightEntity, targetSize,resultLInter,resultRInter)

            print("\n")
            # print("STITCH Predictions: "+str(len(stitchPredictions))+"/"+str(len(stitchValidPredictions)))
            # print("KEGG Predictions: "+str(len(keggPredictions))+"/"+str(len(keggValidPredictions)))

            print("\n")
            print("Total TEST positives: " + str(len(np.where((test_triples[:, 1] == 0))[0])))
            print("Total TEST negatives: " + str(len(np.where((test_triples[:, 1] == 1))[0])))
            print("Total TRAIN positives: " + str(len(np.where((train_triples[:, 1] == 0))[0])))
            print("Total TRAIN negatives: " + str(len(np.where((train_triples[:, 1] == 1))[0])))

            print("True positves: "+str(len(truePositivePredictions)))
            print("False positives: "+str(len(falsePositivePredictions)))
            print("True negatives: "+str(len(trueNegativePredictions)))
            print("False negatives: " + str(len(falseNegativePredictions)))

            tpr = len(truePositivePredictions)/float(len(truePositivePredictions)+len(falseNegativePredictions))
            fpr = len(falsePositivePredictions)/float(len(truePositivePredictions)+len(falseNegativePredictions))
            # print("TPR: " + str(tpr))
            # print("FPR: " + str(fpr))

            print("Test Positives: "+str(len(predictedIndexesInTestTriples)))
            print("Novel Positives: "+str(len(predictedIndexesInNovelTriples)))
            precision = float(len(truePositivePredictions) + len(falsePositivePredictions))
            if(precision==0):
                precision=1
            precision = len(truePositivePredictions) / precision
            recall = float(len(truePositivePredictions)+len(falseNegativePredictions))
            if(recall==0):
                recall=1
            recall = len(truePositivePredictions) / recall
            accuracy = (len(truePositivePredictions)+len(trueNegativePredictions))/float( len(truePositivePredictions)+len(trueNegativePredictions)+len(falsePositivePredictions)+len(falseNegativePredictions) )
            print("Precision tp/(tp+fp): " + str(precision))
            print("Recall tp/(tp+fn):" + str(recall))
            print("Accuracy (tp+tn)/(tp+tn+fp+fn): "+ str( accuracy ))
            print('\n\n')
            precisions+=[precision]
            recalls+=[recall]
            accuracies+=[accuracy]
            f = open(resultsPath + aucPredictionFileName, 'w')
            np.savetxt(f,auc_prediction_lines,delimiter='\t',fmt='%1i')
            f.close()

    sortedPredictions = sorted(sortedPredictions.iteritems(), key=lambda (k, v): (v, k), reverse=True)
    print("Top Sorted predictions: ")
    for k, v in sortedPredictions:
        leftEntity, rightEntity,_,_ = k.split('\t')
        leftEntity = entity2idx[leftEntity]
        rightEntity = entity2idx[rightEntity]
        tt = np.where((novel_triples[:, 0] == leftEntity) & (novel_triples[:, 1] == 0) & (novel_triples[:, 2] == rightEntity))[0]
        if (len(tt) > 0):
            print(k + "\t" + str(v)+"\tnovel")
        else:
            # print(k + "\t" + str(v))+"\tnon novel"
            continue
    print(precisions)
    print(recalls)
    print(accuracies)
    print("AVG Precision: " + str(np.mean(precisions)))
    print("AVG Recall: " + str(np.mean(recalls)))
    print("AVG Accuracy: " + str(np.mean(accuracies)))

# get drugs similar to this drug
# similarDrugs = drugSimilarity[leftEntity-targetSize].toarray()[0]
# similarDrugs[leftEntity-targetSize] = -1 #to ignore the similarity value of itself
# similarDrugsIndices = np.where(similarDrugs>percentileLeftThreshold)[0]
#
# #get targets similar to this target
# similarTargets = targetSimilarity[rightEntity].toarray()[0]
# similarTargets[rightEntity] = -1  # to ignore the similarity value of itself
# similarTargetsIndices = np.where(similarTargets>percentileRightThreshold)[0]
#
# # get drugs which interacts with particular target
# resultRInter = resultsArray[rightEntity]
# # get targets which interacts with particular drug
# resultLInter = resultsArray[leftEntity]
#
# # rightScore = np.mean(resultLInter[(similarTargetsIndices)])
# # leftScore = np.mean(resultRInter[similarDrugsIndices+targetSize])
# rightScore = np.sum(resultLInter[similarTargetsIndices] * similarTargets[similarTargetsIndices]) / np.sum(
#     similarTargets[similarTargetsIndices])
# leftScore = np.sum( resultRInter[similarDrugsIndices+targetSize] * similarDrugs[similarDrugsIndices] ) / np.sum(similarDrugs[similarDrugsIndices])
#
# score = np.mean([leftScore,rightScore])
#
# # np.mean([percentileLeftThreshold, percentileRightThreshold])
# # print(str(score))
# if(score>0.55):
#     print("Similarity based prediction: "+str([[idx2entity[leftEntity], idx2entity[rightEntity]]]))
#     drugTargetInteracts(truePositivePredictions, falsePositivePredictions,predictedIndexesInTestTriples,
#                         predictedIndexesInNovelTriples, auc_prediction_lines, leftEntity, rightEntity, targetSize,
#                         test_triples)
#     # drugTargetNoInteracts(trueNegativePredictions, falseNegativePredictions, auc_prediction_lines,
#     #                       leftEntity, rightEntity, targetSize)
# else: