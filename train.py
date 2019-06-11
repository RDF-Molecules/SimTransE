#! /usr/bin/python

from model import *
import evaluate as Evaluate

import os
import sys
import time
import copy
import cPickle
import scipy.sparse as sp


def saveModel(model,path):
    f = open(path, 'w')
    cPickle.dump(model[0], f, -1) #embeddings
    cPickle.dump(model[1], f, -1) #leftop
    cPickle.dump(model[2], f, -1) #rightop
    cPickle.dump(model[3], f, -1) #simfn
    f.close()

def getPercentileThresholds():
    percentileFile = open('data/percentiles.txt', 'r')
    percentileThresholds = {}
    dat = percentileFile.readlines()
    for item in dat:
        pData, values = item[:-1].split('*')
        percentileThresholds[pData] = map(float, values.split(','))
    percentileFile.close()
    return percentileThresholds
def printTriples(triples,idx2entity):
    for interaction in triples:
        print(idx2entity[interaction[0]]+'      interactive     '+idx2entity[interaction[2]])


def normalizeEmbeddings(embeddings,rhoE,rhoL):
    # normalize embeddings
    auxE = embeddings[0].E.get_value()
    idx = np.where(np.sqrt(np.sum(auxE ** 2, axis=0)) > rhoE)
    auxE[:, idx] = (rhoE * auxE[:, idx]) / np.sqrt(np.sum(auxE[:, idx] ** 2, axis=0))
    embeddings[0].E.set_value(auxE)
    auxR = embeddings[1].E.get_value()
    idx = np.where(np.sqrt(np.sum(auxR ** 2, axis=0)) > rhoL)
    auxR[:, idx] = (rhoL * auxR[:, idx]) / np.sqrt(np.sum(auxR[:, idx] ** 2, axis=0))
    embeddings[1].E.set_value(auxR)

def trainingIterationSoft(batchsize,sBatchsize,posl,posr,poso,softl,softr,trainfunc,embeddings,out,outb,nbatches,lremb,lrparam,rhoE,rhoL):
    for i in range(nbatches):
        tmpl = posl[:, i * batchsize:(i + 1) * batchsize]
        tmpr = posr[:, i * batchsize:(i + 1) * batchsize]
        tmpo = poso[:, i * batchsize:(i + 1) * batchsize]

        softnl = softl[:, i * sBatchsize:(i + 1) * sBatchsize]
        softnr = softr[:, i * sBatchsize:(i + 1) * sBatchsize]

        if(softnl.shape[1]>tmpl.shape[1]):
            localBatchsize = softnl.shape[1] / tmpl.shape[1]
            outtmp = []
            for k in range(localBatchsize):
                localsoftnl = softnl[:, k * tmpl.shape[1]:(k + 1) * tmpl.shape[1]]
                localsoftrl = softnr[:, k * tmpl.shape[1]:(k + 1) * tmpl.shape[1]]

                # training iteration
                # learn embeddings according to interactions
                localout = trainfunc(lremb, lrparam, tmpl, tmpr, tmpo, localsoftnl, localsoftrl)
                outtmp += [localout]
        else:
            localBatchsize = tmpl.shape[1] / softnl.shape[1]
            outtmp = []
            for k in range(localBatchsize):
                localtmpl = tmpl[:, k * softnl.shape[1]:(k + 1) * softnl.shape[1]]
                localtmpr = tmpr[:, k * softnl.shape[1]:(k + 1) * softnl.shape[1]]
                localtmpo = tmpo[:, k * softnl.shape[1]:(k + 1) * softnl.shape[1]]
                # training iteration
                # learn embeddings according to interactions
                localout = trainfunc(lremb, lrparam, localtmpl, localtmpr, localtmpo, softnl, softnr)
                outtmp += [localout]

        outtmp = np.mean(outtmp, axis=0)
        out += [outtmp[0] / float(batchsize)]
        outb += [outtmp[1]]

        normalizeEmbeddings(embeddings,rhoE,rhoL)



def trainingIteration(batchsize,nBatchsize,posl,posr,poso,negl,negr,trainfunc,embeddings,out,outb,nbatches,lremb,lrparam,rhoE,rhoL):
    for i in range(nbatches):
        tmpl = posl[:, i * batchsize:(i + 1) * batchsize]
        tmpr = posr[:, i * batchsize:(i + 1) * batchsize]
        tmpo = poso[:, i * batchsize:(i + 1) * batchsize]

        tmpnl = negl[:, i * nBatchsize:(i + 1) * nBatchsize]
        tmpnr = negr[:, i * nBatchsize:(i + 1) * nBatchsize]

        localBatchsize = tmpnl.shape[1] / tmpl.shape[1]
        outtmp = []
        for k in range(localBatchsize):
            localtmpnl = tmpnl[:, k * tmpl.shape[1]:(k + 1) * tmpl.shape[1]]
            localtmprl = tmpnr[:, k * tmpl.shape[1]:(k + 1) * tmpl.shape[1]]

            # training iteration
            # learn embeddings according to interactions
            localout = trainfunc(lremb, lrparam, tmpl, tmpr, tmpo, localtmpnl, localtmprl)
            outtmp += [localout]
        outtmp = np.mean(outtmp, axis=0)
        out += [outtmp[0] / float(batchsize)]
        outb += [outtmp[1]]

        normalizeEmbeddings(embeddings,rhoE,rhoL)


# Experiment function --------------------------------------------------------
def FB15kexp(state, channel,percentileThresholds):

    includeSimilarities = True#percentile!='ws'  #whether to include similarity matrices in learning embeddings

    bestModelName = 'best_valid_model.pkl'
    currentModelName = 'current_model.pkl'

    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)


    # similarity matrices
    drugSimilarity = load_file(state.datapath + state.dataset + 'drugs_simi.pkl')
    targetSimilarity = load_file(state.datapath + state.dataset + 'targets_simi.pkl')

    numberOfLeftEntities = drugSimilarity.shape[0]
    numberOfRightEntities = targetSimilarity.shape[0]

    percentileLeftThreshold  = percentileThresholds[0]
    percentileRightThreshold = percentileThresholds[1]

    f = open(datapath + dataset + 'entity2idx.pkl')
    entity2idx = cPickle.load(f)
    f.close()

    f = open(datapath + dataset + 'idx2entity.pkl')
    idx2entity = cPickle.load(f)
    f.close()

    # Positives
    trainl = load_file(state.foldPath +'/processed/'+ state.dataset + 'train-lhs.pkl')
    trainr = load_file(state.foldPath +'/processed/'+ state.dataset + 'train-rhs.pkl')
    traino = load_file(state.foldPath +'/processed/'+ state.dataset + 'train-rel.pkl')

    trainl = trainl[:state.Nsyn, :]
    trainr = trainr[:state.Nsyn, :]
    traino = traino[-2:, :]

    # Test set
    testl = load_file(state.foldPath +'/processed/'+ state.dataset + 'test-lhs.pkl')
    testr = load_file(state.foldPath +'/processed/'+ state.dataset + 'test-rhs.pkl')
    testo = load_file(state.foldPath +'/processed/'+ state.dataset + 'test-rel.pkl')

    testl = testl[:state.Nsyn, :]
    testr = testr[:state.Nsyn, :]
    testo = testo[-2:, :]

    # Index conversion
    trainlidx = convert2idx(trainl)[:state.neval]
    trainridx = convert2idx(trainr)[:state.neval]
    trainoidx = convert2idx(traino)[:state.neval]
    testlidx = convert2idx(testl)[:state.neval]
    testridx = convert2idx(testr)[:state.neval]
    testoidx = convert2idx(testo)[:state.neval]

    idxl = convert2idx(trainl)
    idxr = convert2idx(trainr)
    idxo = convert2idx(traino)
    idxtl = convert2idx(testl)
    idxtr = convert2idx(testr)
    idxto = convert2idx(testo)

    true_triples=np.concatenate([idxl,idxo,idxr]).reshape(3,idxl.shape[0]).T
    # initialize the soft positive dataset variables
    softl = sp.lil_matrix((state.Nsyn, true_triples.shape[0]), dtype='float64')
    softr = sp.lil_matrix((state.Nsyn, true_triples.shape[0]), dtype='float64')
    softo = sp.lil_matrix((state.Nsyn, true_triples.shape[0]), dtype='float64')
    softTriplesDict = {}

    negative_triples = true_triples[np.where(true_triples[:, 1] == 1)[0]].astype('float64')
    true_triples = true_triples[np.where(true_triples[:, 1] == 0)[0]].astype('float64')


    #initialize the positive dataset variables
    posl = sp.lil_matrix((state.Nsyn,true_triples.shape[0]),dtype='float64')
    posr = sp.lil_matrix((state.Nsyn,true_triples.shape[0]),dtype='float64')
    poso = sp.lil_matrix((state.Nsyn,true_triples.shape[0]),dtype='float64')
    ct = 0
    softLimit = 225
    for triple in true_triples:
        leftEntity = int(triple[0])
        rightEntity = int(triple[2])
        relation = int(triple[1])
        rowValue = 1
        # get drugs similar to this drug
        similarDrugs = drugSimilarity[leftEntity-targetSimilarity.shape[0]].toarray()[0]
        similarDrugs[leftEntity-targetSimilarity.shape[0]] = -1 #to ignore the similarity value of itself
        similarDrugsIndices = np.where(similarDrugs>percentileLeftThreshold)[0]

        # #get targets similar to this target
        similarTargets = targetSimilarity[rightEntity].toarray()[0]
        similarTargets[rightEntity] = -1  # to ignore the similarity value of itself
        similarTargetsIndices = np.where(similarTargets>percentileRightThreshold)[0]

        iCount = 0

        for drug in similarDrugsIndices:
            for target in similarTargetsIndices:
                key = str(drug) + '-' + str(target)
                # inverseKey = str(target)+'-'+str(drug)
                if(key not in softTriplesDict):
                    if iCount <= softLimit:
                        # similarity = np.mean([similarDrugs[drug], similarTargets[target]])
                        softTriplesDict[key] = 1
                        # softTriplesDict[inverseKey] = similarity
                        iCount+=1
                    else:
                        # print("breaking due to soft limit reached")
                        break

        posl[leftEntity,ct] = rowValue
        posr[rightEntity,ct] = rowValue
        poso[relation,ct] = rowValue
        ct+=1
    poso = poso[-1:, :]
    softo = softo[-1:, :]

    lct = 0
    for key in softTriplesDict:
        leftEnt, rightEnt = key.split('-')
        leftEnt = int(leftEnt)
        rightEnt = int(rightEnt)
        # similarity = softTriplesDict[key]
        similarity=1#temporary
        softl[leftEnt + targetSimilarity.shape[0], lct] = similarity
        softr[rightEnt, lct] = similarity
        softo[0, lct] = similarity
        lct+=1

    softl = softl[:,0:lct]
    softr = softr[:,0:softl.shape[1]]
    softo = softo[:,0:softl.shape[1]]
    softlidx = convert2idx(softl)
    softridx = convert2idx(softr)
    softoidx = convert2idx(softo)
    soft_triples = np.concatenate([softlidx,softoidx,softridx]).reshape(3,softlidx.shape[0]).T
    print("positive triples created with shape: ",soft_triples.shape)


    # Initialize the negative dataset variables
    negl = sp.lil_matrix((state.Nsyn, negative_triples.shape[0]), dtype='float64')
    negr = sp.lil_matrix((state.Nsyn, negative_triples.shape[0]), dtype='float64')
    nego = sp.lil_matrix((state.Nsyn, negative_triples.shape[0]), dtype='float64')
    ct = 0
    for triple in negative_triples:
        leftEntity = triple[0]
        rightEntity = triple[2]
        relation = triple[1]
        rowValue = 1

        key = str(int(leftEntity)) + '-' + str(int(rightEntity))
        if(key in softTriplesDict):
            rowValue = softTriplesDict[key]
            rowValue=1#temporary
        else:
            negl[leftEntity, ct] = rowValue
            negr[rightEntity, ct] = rowValue
            # nego[0, ct] = rowValue
            ct += 1
    negl = negl[:, 0:ct]
    negr = negr[:, 0:ct]
    nego = nego[:, 0:ct]

    # Model declaration
    leftop = LayerMat('lin', state.ndim, 1)
    rightop = LayerdMat()

    # embeddings
    if not state.loademb:
        embeddings = Embeddings(np.random, state.Nsyn, state.ndim, 'Emat')
        W = Embeddings(np.random, 1, state.ndim, 'Wmat')
        rel_matricesl = Embeddings(np.random, 1, state.ndim, 'relmatL')
        rel_matricesr = Embeddings(np.random, 1, state.ndim, 'relmatR')
        embeddings = [embeddings, W, rel_matricesl, rel_matricesr]
    else:
        f = open(state.loademb+currentModelName)
        embeddings = cPickle.load(f)
        f.close()

    simfn = eval(state.simfn + 'sim')

    # Evaluate.evaluateSimilarities(embeddings,idxl,idxr,drugSimilarity.get_value(),targetSimilarity.get_value())
    # if(includeSimilarities):
    # setDistancesFromSimilarities(embeddings, targetSimilarity, drugSimilarity, percentileLeftThreshold,
    #                              percentileRightThreshold)

    trainfunc = TrainFn1MemberBi(embeddings, leftop, rightop, simfn, marge=state.marge)
    trainfuncSoft = TrainFn1MemberBiSoft(embeddings, leftop, rightop, simfn, marge=state.marge)

    ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=state.Nsyn)
    rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=state.Nsyn)

    #saveInitial model
    saveModel([embeddings,leftop,rightop,simfn],state.savepath+currentModelName)
    interactionRelationID = 0#entity2idx['interactive']

    # create a list of left and right ids
    uniqueLeft = [i + targetSimilarity.shape[0] for i in range(drugSimilarity.shape[0])]
    uniqueRight = [i for i in range(targetSimilarity.shape[0])]

    out = []
    outb = []
    state.bestvalid = -1
    batchsize = posl.shape[1] / state.nbatches
    nBatchsize = negl.shape[1]/state.nbatches
    sBatchsize = softl.shape[1]/state.nbatches
    print >> sys.stderr, "positives: " + str(posl.shape[1])
    print >> sys.stderr, "unknown: " + str(negl.shape[1])
    print >> sys.stderr, "reinforced: " + str(softl.shape[1])
    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(posl.shape[1])
        posl = posl[:, order]
        posr = posr[:, order]
        poso = poso[:, order]

        order = np.random.permutation(negl.shape[1])
        negl = negl[:, order]
        negr = negr[:, order]

        order = np.random.permutation(softl.shape[1])
        softl = softl[:, order]
        softr = softr[:, order]
        softo = softo[:, order]

        trainingIteration(batchsize,nBatchsize,posl,posr,poso,negl,negr,trainfunc,embeddings,out,outb,state.nbatches,
                          state.lremb,state.lrparam,state.rhoE,state.rhoL)
        # trainingIteration(sBatchsize,nBatchsize,softl,softr,softo,negl,negr,trainfunc,embeddings,out,outb,state.nbatches,
        #                   state.lremb,state.lrparam,state.rhoE,state.rhoL)
        trainingIterationSoft(batchsize, sBatchsize, posl, posr, poso, softl, softr, trainfuncSoft, embeddings, out, outb,
                          state.nbatches,state.lremb, state.lrparam, state.rhoE, state.rhoL)

        print >> sys.stderr, "EPOCH COUNT >> %s Iteration COST >> %s +/- %s, %% updates: %s%%" % (
            str(epoch_count),
            round(np.mean(out), 4),
            round(np.std(out), 4),
            round(np.mean(outb) * 100, 3))

        # Save current model
        saveModel([embeddings, leftop, rightop, simfn], state.savepath + currentModelName)

        if ((epoch_count % state.test_all) == 0 or epoch_count==state.totepochs):
            # model evaluation
            print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                epoch_count,
                round(time.time() - timeref, 3) / float(state.test_all))
            timeref = time.time()
            out = []
            outb = []

            restest = Evaluate.RankingScoreIdx(ranklfunc, rankrfunc, testlidx[np.where(testoidx[:] == 0)],
                                               testridx[np.where(testoidx[:] == 0)],
                                               testoidx[np.where(testoidx[:] == 0)])
            state.test = np.mean(restest[0] + restest[1])
            print >> sys.stderr, "\t\t##### NEW >> TEST: %s" % (state.test)
            # resttrain = Evaluate.RankingScoreIdx(ranklfunc, rankrfunc, trainlidx[np.where(trainoidx[:] == 0)],
            #                                      trainridx[np.where(trainoidx[:] == 0)],
            #                                      trainoidx[np.where(trainoidx[:] == 0)])

            # state.train = np.mean(resttrain[0] + resttrain[1])
            #
            # print >> sys.stderr, "\t\t##### NEW >> TRAIN: %s" % (state.train)
            if state.besttest == -1 or state.test <= state.besttest:
            # if state.besttrain == -1 or state.train <= state.besttrain:

                state.besttest = state.test
                # state.besttrain = state.train

                state.bestepoch = epoch_count

                # Save model best valid model
                saveModel([embeddings, leftop, rightop, simfn], state.savepath + bestModelName)
                # restest = Evaluate.RankingScoreIdx(ranklfunc, rankrfunc, testlidx[np.where(testoidx[:] == 0)],
                #                                    testridx[np.where(testoidx[:] == 0)],
                #                                    testoidx[np.where(testoidx[:] == 0)])
                # state.test = np.mean(restest[0] + restest[1])
                # print >> sys.stderr, "\t\t##### NEW >> TEST: %s" % (state.test)

                print >> sys.stderr, "\t\t##### NEW BEST >> TEST: %s" % (state.test)
                resttrain = Evaluate.RankingScoreIdx(ranklfunc, rankrfunc, trainlidx[np.where(trainoidx[:] == 0)],
                                                     trainridx[np.where(trainoidx[:] == 0)],
                                                     trainoidx[np.where(trainoidx[:] == 0)])

                state.train = np.mean(resttrain[0] + resttrain[1])

                print >> sys.stderr, "\t\t##### NEW  >> TRAIN: %s" % (state.train)

                # restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc, testlidx, testridx, testoidx, true_triples)
                # resvalid = FilteredRankingScoreIdx(ranklfunc, rankrfunc,testlidx, testridx, testoidx, true_triples)


            else:
                # Save current model
                saveModel([embeddings, leftop, rightop, simfn], state.savepath + currentModelName)

            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    return channel.COMPLETE





# Utils ----------------------------------------------------------------------
def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = scipy.sparse.lil_matrix((shape[0], shape[1]),
                                        dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
                                   dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

def convert2idxT(spmat):
    rows,cols = T.nonzero(spmat)
    return rows[T.argsort(cols)]




class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z


# ----------------------------------------------------------------------------



def launch(foldPath='folds/fold1', datapath='data/', dataset='semep', Nent=80, rhoE=1, rhoL=5,
           Nsyn=81, Nrel=1, loadmodel=False, loademb=False,percentileThresholds=[],
           simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.1, lrparam=1.,
           nbatches=100, totepochs=2000, test_all=1, neval=50, seed=123,
           savepath='.', loadmodelBi=False, loadmodelTri=False):
    # Argument of the experiment script
    state = DD()
    state.datapath = datapath
    state.dataset = dataset
    state.Nent = Nent  # number of entities
    state.Nsyn = Nsyn
    state.Nrel = Nrel  # number of relations
    state.loadmodel = loadmodel  # load model from file
    state.loadmodelBi = loadmodelBi
    state.loadmodelTri = loadmodelTri
    state.loademb = loademb  # load embeddings from file
    state.op = 'Tri'  # operator
    state.simfn = simfn  # similarity function
    state.ndim = ndim  # number of dimensions
    state.nhid = nhid
    state.marge = marge  # margin argument
    state.rhoE = rhoE
    state.rhoL = rhoL
    state.lremb = lremb  # learning rate for embeddings
    state.lrparam = lrparam  # learning rate parameter
    state.nbatches = nbatches  # number of batches
    state.totepochs = totepochs  # total epochs
    state.test_all = test_all  #
    state.neval = neval  # number of evaluations
    state.seed = seed  # random seed
    state.savepath = savepath  # file path
    state.foldPath = foldPath
    state.besttrain = -1
    state.besttest= -1
    #doing this to ignore validation in training
    state.valid = 0
    state.bestvalid = 1


    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    # Jobman channel remplacement
    class Channel(object):
        def __init__(self, state):
            self.state = state
            f = open(self.state.savepath + '/orig_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()
            self.COMPLETE = 1

        def save(self):
            f = open(self.state.savepath + '/current_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()

    channel = Channel(state)

    FB15kexp(state, channel, percentileThresholds)





if __name__ == '__main__':


    #theano.config.mode = 'DebugMode'
    datatype = 'randen'
    # datatype = 'nr'
    if datatype == 'nr':
        dataDir = 'nr90-10'
    elif datatype == 'gpcr':
        dataDir = 'gpcr90-10'
    elif datatype == 'ic':
        dataDir = 'ic90-10'
    elif datatype == 'en':
        dataDir = 'en90-10'
    elif datatype == 'randnr':
        dataDir = 'semep/random/nr90-10'
        datatype = 'nr'
    elif datatype == 'nonrandnr':
        dataDir = 'semep/nonrandom/nr90-10'
        datatype = 'nr'
    elif datatype == 'nonrandgpcr':
        dataDir = 'semep/nonrandom/gpcr90-10'
        datatype = 'gpcr'
    elif datatype == 'nonrandic':
        dataDir = 'semep/nonrandom/ic90-10'
        datatype = 'ic'
    elif datatype == 'randgpcr':
        dataDir = 'semep/random/gpcr90-10'
        datatype = 'gpcr'
    elif datatype == 'nonranden':
        dataDir = 'semep/nonrandom/en90-10'
        datatype = 'en'
    elif datatype == 'randen':
        dataDir = 'semep/random/en90-10'
        datatype = 'en'

    datapath = 'data/'+dataDir+'/pProcessed/'
    dataset = 'drugs_targets_'
    percentileNames = ['p80', 'p90', 'p95', 'p98']

    print("Dataset: " + datatype)

    leftEntities =  -1
    rightEntities = -1
    relation = 1
    seed = 234
    # seed = 123

    totalEpochs = 150
    testAll = 150

    nbatches = 150

    percentileThresholds = getPercentileThresholds()

    percentileIndices = [0, 1, 2, 3]
    percentileIndices = [1]
    for percentileIndex in percentileIndices:
        print("Percentile: " + percentileNames[percentileIndex])

        for fold in [1,2,3,4,5,6,7,8,9,10]:
        # for fold in [2]:
            print("Training Iteration for fold: " + str(fold))

            foldPath = 'data/' + dataDir +'/folds/fold'+str(fold)
            savepath = foldPath + '/processed/' + percentileNames[percentileIndex] + '/'
            previousSavePath= ""
            if percentileNames[percentileIndex] not in os.listdir(foldPath+'/processed/'):
                os.mkdir(savepath)

            if (datatype == 'nr'):
                # nr data
                leftEntities = 54
                rightEntities = 26
                totalEpochs = 100
                testAll = 10
                nbatches = 10
                loadEmb = False
                # loadEmb = foldPath + '/processed/' + percentileNames[percentileIndex] + '/'
                # if percentileIndex==0:
                #     loadEmb=False #first percentile (train single model for all percentiles)
                #     totalEpochs = 100
                # else:
                #     loadEmb = foldPath + '/processed/' + percentileNames[percentileIndex - 1] + '/'

                percentileLeftThreshold = percentileThresholds[datatype + '-left'][percentileIndex]
                percentileRightThreshold = percentileThresholds[datatype + '-right'][percentileIndex]
                launch(datapath=datapath, dataset=dataset, foldPath=foldPath, Nent=leftEntities+rightEntities, rhoE=1, rhoL=5,
                       Nsyn=leftEntities+rightEntities+relation, Nrel=relation, loadmodel=False, nbatches=nbatches,
                       loademb=loadEmb, ndim=50, nhid=50, marge=1, lremb=0.001,percentileThresholds=[percentileLeftThreshold,percentileRightThreshold],
                       totepochs=totalEpochs, test_all=testAll, neval=-1, seed=seed, savepath=savepath)

            elif (datatype == 'gpcr'):
                # gpcr data
                leftEntities = 223
                rightEntities = 95
                nbatches = 100
                totalEpochs = 100
                testAll = 10
                percentileLeftThreshold = percentileThresholds[datatype + '-left'][percentileIndex]
                percentileRightThreshold = percentileThresholds[datatype + '-right'][percentileIndex]
                launch(datapath=datapath, dataset=dataset, foldPath=foldPath, Nent=leftEntities + rightEntities, rhoE=1,
                       rhoL=5,Nsyn=leftEntities + rightEntities + relation, Nrel=relation, loadmodel=False,
                       nbatches=nbatches,loademb=False, simfn='Cos', ndim=100, nhid=50, marge=0.25, lremb=0.001,
                       percentileThresholds=[percentileLeftThreshold, percentileRightThreshold],
                       totepochs=totalEpochs, test_all=testAll, neval=-1, seed=seed, savepath=savepath)
            elif (datatype == 'ic'):
                leftEntities = 210
                rightEntities = 204
                nbatches = 100
                totalEpochs = 100
                testAll = 10
                # ic data
                percentileLeftThreshold = percentileThresholds[datatype + '-left'][percentileIndex]
                percentileRightThreshold = percentileThresholds[datatype + '-right'][percentileIndex]
                launch(datapath=datapath, dataset=dataset, foldPath=foldPath, Nent=leftEntities + rightEntities, rhoE=1,
                       rhoL=5, Nsyn=leftEntities + rightEntities + relation, Nrel=relation, loadmodel=False,
                       nbatches=nbatches, loademb=False, simfn='Cos', ndim=100, nhid=50, marge=0.25, lremb=0.001,
                       percentileThresholds=[percentileLeftThreshold, percentileRightThreshold],
                       totepochs=totalEpochs, test_all=testAll, neval=-1, seed=seed, savepath=savepath)
            elif (datatype == 'en'):
                leftEntities = 445
                rightEntities = 664
                nbatches = 100
                totalEpochs = 100
                testAll = 10
                # en data
                percentileLeftThreshold = percentileThresholds[datatype + '-left'][percentileIndex]
                percentileRightThreshold = percentileThresholds[datatype + '-right'][percentileIndex]
                launch(datapath=datapath, dataset=dataset, foldPath=foldPath, Nent=leftEntities + rightEntities, rhoE=1,
                       rhoL=5, Nsyn=leftEntities + rightEntities + relation, Nrel=relation, loadmodel=False,
                       nbatches=nbatches, loademb=False, simfn='Cos', ndim=100, nhid=50, marge=0.25, lremb=0.001,
                       percentileThresholds=[percentileLeftThreshold, percentileRightThreshold],
                       totepochs=totalEpochs, test_all=testAll, neval=-1, seed=seed, savepath=savepath)

            elif (dataType == 'custom'):
                # custom data
                launch(fold,datapath=datapath, dataset=dataset, Nent=24, rhoE=1, rhoL=5, Nsyn=25, Nrel=1, loadmodel=False,
                       loademb=False, op='Bi', simfn='Dot', ndim=20, nhid=50, marge=0.25, lremb=0.001, lrparam=0.001,
                       nbatches=6, totepochs=1000, test_all=10, neval=12, seed=123, savepath=savepath,
                       loadmodelBi=False,
                       loadmodelTri=False, percentile=percentile)

            modelPath = savepath + 'best_valid_model.pkl'
            MR, T10 = Evaluate.RankingEval(datapath=datapath, dataset=dataset, Nsyn=leftEntities+rightEntities+relation,
                                loadmodel=modelPath,foldPath=foldPath )

            print "\n##### MEAN RANK: %s #####\n" % (MR)
            print "\n##### HITS@10: %s #####\n" % (T10)