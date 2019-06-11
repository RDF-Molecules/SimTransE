#! /usr/bin/python
import sys
import cPickle

from train import *

def RankingScoreIdx(sl, sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
    return errl, errr


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
                                   dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data/', dataset='drugs_targets', foldPath='folds/fold1/',
                loadmodel='best_valid_model.pkl', neval='all', Nsyn=14951, n=10):
    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    # Load data
    l = load_file(foldPath +'/processed/'+ dataset + 'test-lhs.pkl')
    r = load_file(foldPath +'/processed/'+ dataset + 'test-rhs.pkl')
    o = load_file(foldPath +'/processed/'+ dataset + 'test-rel.pkl')
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]

    # Convert sparse matrix to indexes
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]

    ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=Nsyn)
    rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=Nsyn)

    restest = RankingScoreIdx(ranklfunc, rankrfunc, idxl, idxr, idxo)
    printResult(restest,n,idxo)
    T10 = np.mean(np.asarray(restest[0] + restest[1]) <= n) * 100
    MR = np.mean(np.asarray(restest[0] + restest[1]))
    return MR,T10

def printResult(res,n,idxo):
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    print "### MICRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
        n, round(dres['microlhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
        n, round(dres['microrhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
        n, round(dres['microghits@n'], 3))

    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellrn = {}
    dictrelrrn = {}
    dictrelgrn = {}

    for i in listrel:
        dictrelres.update({i: [[], []]})

    for i, j in enumerate(res[0]):
        dictrelres[idxo[i]][0] += [j]

    for i, j in enumerate(res[1]):
        dictrelres[idxo[i]][1] += [j]

    for i in listrel:
        dictrellmean[i] = np.mean(dictrelres[i][0])
        dictrelrmean[i] = np.mean(dictrelres[i][1])
        dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        dictrellmedian[i] = np.median(dictrelres[i][0])
        dictrelrmedian[i] = np.median(dictrelres[i][1])
        dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
                                           dictrelres[i][1]) <= n) * 100

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrellmean': dictrellmean})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelgmean': dictrelgmean})
    dres.update({'dictrellmedian': dictrellmedian})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelgmedian': dictrelgmedian})
    dres.update({'dictrellrn': dictrellrn})
    dres.update({'dictrelrrn': dictrelrrn})
    dres.update({'dictrelgrn': dictrelgrn})

    dres.update({'macrolmean': np.mean(dictrellmean.values())})
    dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
    dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
    dres.update({'macrogmean': np.mean(dictrelgmean.values())})
    dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
    dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

    print "### MACRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
        n, round(dres['macrolhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
        n, round(dres['macrorhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
        n, round(dres['macroghits@n'], 3))



def evaluateSimilarities(embeddings,idxtl,idxtr,drugSimilarity,targetSimilarity):
    embedding = embeddings[0].E.get_value()
    lefterrors = []
    idxtl = set(idxtl)
    idxtr = set(idxtr)
    for i in range(len(idxtl)):
        currLeftID = idxtl.pop() - targetSimilarity.shape[0]
        localError = 0
        for k in range(drugSimilarity.shape[0]):
            vec1 = embedding[:, currLeftID]
            vec2 = embedding[:, k + targetSimilarity.shape[0]]
            similarityValue = drugSimilarity[currLeftID, k]
            numerator = np.dot(vec1, vec2)
            denominator = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
            currentSimilarity = numerator / denominator  # cosine similarity
            localError += np.abs(similarityValue - currentSimilarity)
        lefterrors += [localError]

    righterrors = []
    for i in range(len(idxtr)):
        currRightID = idxtr.pop()
        for k in range(targetSimilarity.shape[0]):
            vec1 = embedding[:, currRightID]
            vec2 = embedding[:, k]
            similarityValue = targetSimilarity[currRightID, k]
            numerator = np.dot(vec1, vec2)
            denominator = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
            currentSimilarity = numerator / denominator  # cosine similarity
            localError += np.abs(similarityValue - currentSimilarity)
        righterrors += [localError]

    print(lefterrors)
    print(righterrors)


#filtered evaluation
def RankingEvalFil(datapath='../data/', dataset='umls-test', op='TransE',loadmodel='best_valid_model.pkl', fold=0,
                   Nrel=26, Nsyn=135,percentile='80'):

    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    f.close()

    # Load data
    # l = load_file(datapath + dataset + '-test-lhs.pkl')
    # r = load_file(datapath + dataset + '-test-rhs.pkl')
    # o = load_file(datapath + dataset + '-test-rel.pkl')
    # o = o[-Nrel:, :]

    tl = load_file(datapath + dataset + '-true-lhs.pkl')
    tr = load_file(datapath + dataset + '-true-rhs.pkl')
    to = load_file(datapath + dataset + '-true-rel.pkl')
    to = to[-Nrel:, :]

    # vl = load_file(datapath + dataset + '-valid-lhs.pkl')
    # vr = load_file(datapath + dataset + '-valid-rhs.pkl')
    # vo = load_file(datapath + dataset + '-valid-rel.pkl')
    # vo = vo[-Nrel:, :]


    # #from test data
    # idxl = convert2idx(l)
    # idxr = convert2idx(r)
    # idxo = convert2idx(o)
    #from train data
    idxtl = convert2idx(tl)
    idxtr = convert2idx(tr)
    idxto = convert2idx(to)
    # #from valid data
    # idxvl = convert2idx(vl)
    # idxvr = convert2idx(vr)
    # idxvo = convert2idx(vo)

    ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop,subtensorspec=Nsyn)
    rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop,subtensorspec=Nsyn)

    # true_triples = np.concatenate([idxtl, idxl, idxto, idxo, idxtr, idxr]).\
    #     reshape(3,idxtl.shape[0]  + idxl.shape[0]).T
    true_triples = np.concatenate([idxtl, idxto, idxtr]).\
        reshape(3,idxtl.shape[0]).T

    restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc, idxtl, idxtr, idxto, true_triples)
    T10 = np.mean(np.asarray(restest[0] + restest[1]) <= 10) * 100
    MR = np.mean(np.asarray(restest[0] + restest[1]))
    printResult(restest,10,idxto)


    # similarity matrices
    drugSimilarity = load_file(datapath + dataset + 'drugs_simi.pkl')
    targetSimilarity = load_file(datapath + dataset + 'targets_simi.pkl')
    evaluateSimilarities(embeddings,idxtl,idxtr,drugSimilarity,targetSimilarity)
    return MR, T10


if __name__ == '__main__':
    dataType = "nr"
    sys.argv.insert(1,'data/'+dataType+'/processed/')
    savepath = 'data/'+dataType+'/processed/'
    dataset = 'drugs_targets_'
    percentile = 'p80'
    fold="0"

    print "\n##### EVALUATION #####\n"
    modelPath = savepath + 'best_valid_model(' + percentile + ')(fold' + str(fold) + ').pkl'

    filtered = False
    if(filtered):
        MR, T10 = Evaluate.RankingEvalFil(datapath=sys.argv[1], dataset=dataset, op='Bi',
                                          loadmodel=modelPath, Nrel=1, Nsyn=319,percentile=percentile)
    else:
        MR, T10 = RankingEval(datapath=sys.argv[1], dataset='drugs_targets_',Nsyn=81,fold=fold,
                loadmodel=modelPath,percentile=percentile)

    print "\n##### MEAN RANK: %s #####\n" % (MR)
    print "\n##### HITS@10: %s #####\n" % (T10)




    #
    # modelName='best_valid_model(p98).pkl'
    # print "\n##### EVALUATION #####\n"
    # # MR, T10 = Evaluate.RankingEvalFil(datapath=datapath, dataset=dataset, op='Bi',
    # #                                   loadmodel=savepath + '/best_valid_model(p90).pkl', Nrel=1, Nsyn=81)
    # MR, T10 = Evaluate.RankingEvalFil(datapath=datapath, dataset=dataset, op='Bi',
    #                                   loadmodel=savepath+'/'+modelName, Nrel=1, Nsyn=81)
    #
    # # MR, T10 = Evaluate.RankingEvalFil(datapath=datapath, dataset=dataset, op='Bi',
    # #                                   loadmodel=savepath + '/best_valid_modelws.pkl', Nrel=1, Nsyn=81)
    #
    # print "\n##### MEAN RANK: %s #####\n" % (MR)
    # print "\n##### HITS@10: %s #####\n" % (T10)
