import theano
import theano.sparse as S
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import numpy as np

import scipy
import scipy.sparse

from collections import OrderedDict


# currentSimilarity = 1/(1+scipy.spatial.distance.euclidean( vec1,vec2 ))

def calculateLeftDistance(i, drugSimilarity,embedding, currentLeftID, targetSize, percentileLeftThreshold):
    similarityValue = drugSimilarity[currentLeftID, i]

    if(T.ge(similarityValue,percentileLeftThreshold)):
        vec1 = embedding[:, currentLeftID]
        vec2 = embedding[:, i + targetSize]

        numerator = T.dot(vec1, vec2)
        denominator = T.sqrt(theano.tensor.sum(vec1 ** 2) * theano.tensor.sum(vec2 ** 2))
        currentSimilarity = numerator / denominator  # cosine similarity
        return T.sqr(similarityValue - currentSimilarity)
    else:
        return 0

    # distanceValue = drugSimilarity[p_i,i]
    # eDistance = T.sum(T.sqr(vec1 - vec2))
    #distanceValue = drugSimilarity[p_i,i]
    #eDistance = T.sqrt(T.maximum(T.sum(T.sqr(vec1 - vec2)), 0))
    #eDistance = np.sqrt(np.sum(np.square(vec1-vec2)))
    #eDistance = T.sqrt( T.maximum( T.sum( T.sqr(vec1 - vec2) ),0) )
    #eDistance = T.maximum(T.sum(T.sqr(vec1 - vec2)), 0)
    # currentSimilarity = 1 / (1 + eDistance )
    #return T.sqr(distanceValue-eDistance)
    ##return np.abs(distanceValue-eDistance)

def leftDistanceStep(i, inplidx, drugSimilarity, targetSimilarity, embedding, percentileLeftThreshold):
    targetSize = targetSimilarity.shape[0]
    currentLeftID = inplidx[i]-targetSize
    seq = T.arange(drugSimilarity.shape[0])
    fout,fupdates = theano.scan(fn=calculateLeftDistance,sequences=seq,non_sequences=[drugSimilarity,embedding,currentLeftID,targetSize,percentileLeftThreshold])
    return fout.sum()



def calculateRightDistance(i, targetSimilarity, embedding, currentRightID,percentileRightThreshold):
    similarityValue = targetSimilarity[currentRightID, i]
    if(T.ge(similarityValue,percentileRightThreshold)):
        vec1 = embedding[:, currentRightID]
        vec2 = embedding[:, i]

        numerator = T.dot(vec1, vec2)
        denominator = T.sqrt(theano.tensor.sum(vec1 ** 2) * theano.tensor.sum(vec2 ** 2))
        currentSimilarity = numerator / denominator  # cosine similarity
        return T.sqr(similarityValue - currentSimilarity)
        # return T.abs_(similarityValue - currentSimilarity)
        # return similarityValue - currentSimilarity
        # return currentSimilarity - similarityValue

    else:
        return 0
    # distanceValue = targetSimilarity[p_i,i]
    # eDistance = T.sqrt(T.maximum(T.sum(T.sqr(vec1 - vec2)), 0))
    # return T.abs_(distanceValue-eDistance)

def rightDistanceStep(i, inpridx, targetSimilarity, embedding, percentileRightThreshold):
    currentRightID = inpridx[i]
    seq = T.arange(targetSimilarity.shape[0])
    fout,fupdates = theano.scan(fn=calculateRightDistance,sequences=seq,non_sequences=[targetSimilarity, embedding,currentRightID,percentileRightThreshold])

    return fout.sum()


def TrainFn1MemberBiSoft(embeddings, leftop, rightop, fnsim, marge=1.0):
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[1]
    rel_matricesr = embeddings[2]

    # Inputs
    inpr = S.csr_matrix(name="inpr")
    inpl = S.csr_matrix(name="inpl")
    inpo = S.csr_matrix(name="inpo")
    inpln = S.csr_matrix(name="inpln")
    inprn = S.csr_matrix(name="inprn")
    # inplidx = T.vector(name='left IDs',dtype='int64')
    # inpridx = T.vector(name='right IDs', dtype='int64')
    # drugSimilarity = S.csr_matrix(name="drugSimilarity")
    # targetSimilarity = S.csr_matrix(name="targetSimilarity")
    lr = T.scalar(name="learningRate")
    lrparam = T.scalar(name='lrparam')

    ## Positive triplet
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    relmatricesl = S.dot(rel_matricesl.E, inpo).T
    relmatricesr = S.dot(rel_matricesr.E, inpo).T
    ## Negative triplet
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    # relmatricesln = S.dot(rel_matricesl.E, inpo).T
    # relmatricesrn = S.dot(rel_matricesr.E, inpo).T

    W_m = W.E.reshape((1, W.D))

    # Positive triplet score
    expP = leftop(lhs, relmatricesl) + leftop(rhs, relmatricesr) + leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    #
    # # Negative triple score
    expNl = leftop(lhsn, relmatricesl) + leftop(rhsn, relmatricesr) + leftop(lhsn, rightop(rhsn, W_m))
    # expNr = leftop(lhs, relmatricesl) + leftop(rhsn, relmatricesr) + leftop(lhs, rightop(rhsn, W_m))
    similn = -T.sum(expNl, axis=1)
    # simirn = -T.sum(expNr, axis=1)


    # move the resultant matrices close according to similarity matrices
    # when they are close cost should decrease
    costl, outl = margincost(simi, similn,marge=0)
    # costr, outr = margincost(simi, simirn, marge=0)

    # cost = costl + costr
    cost = costl
    # out = T.concatenate([outl, outr])
    out=outl


    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    d = dict((i, i - lr * j) for i, j in zip(leftop.params + rightop.params, gradientsparams))
    updates = OrderedDict(sorted(d.items()))

    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lr * gradients_embedding
    updates.update({embedding.E: newE})

    gradients_embeddingrelMatl = T.grad(cost, rel_matricesl.E)
    newrelMatl = rel_matricesl.E - lr * gradients_embeddingrelMatl
    updates.update({rel_matricesl.E: newrelMatl})

    gradients_embeddingrelMatr = T.grad(cost, rel_matricesr.E)
    newrelMatr = rel_matricesr.E - lr * gradients_embeddingrelMatr
    updates.update({rel_matricesr.E: newrelMatr})

    gradients_embeddingW = T.grad(cost, W.E)
    newW = W.E - lr * gradients_embeddingW
    updates.update({W.E: newW})

    list_in = [lr, lrparam, inpl, inpr, inpo, inpln, inprn]
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
                            updates=updates, on_unused_input='ignore')
    # return theano.function(list_in, [T.mean(cost), T.mean(out),leftop(lhs, relmatricesl),expP,lhs,simi,similn,cost,
    #                                  cost.shape,out,gradients_embedding],updates=updates, on_unused_input='ignore',
    #                        mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=True))



def TrainFn1MemberBi(embeddings, leftop, rightop, fnsim, marge=1.0):
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[1]
    rel_matricesr = embeddings[2]

    # Inputs
    inpr = S.csr_matrix(name="inpr")
    inpl = S.csr_matrix(name="inpl")
    inpo = S.csr_matrix(name="inpo")
    inpln = S.csr_matrix(name="inpln")
    inprn = S.csr_matrix(name="inprn")
    # inplidx = T.vector(name='left IDs',dtype='int64')
    # inpridx = T.vector(name='right IDs', dtype='int64')
    # drugSimilarity = S.csr_matrix(name="drugSimilarity")
    # targetSimilarity = S.csr_matrix(name="targetSimilarity")
    lr = T.scalar(name="learningRate")
    lrparam = T.scalar(name='lrparam')

    ## Positive triplet
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    relmatricesl = S.dot(rel_matricesl.E, inpo).T
    relmatricesr = S.dot(rel_matricesr.E, inpo).T
    ## Negative triplet
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    # relmatricesln = S.dot(rel_matricesl.E, inpo).T
    # relmatricesrn = S.dot(rel_matricesr.E, inpo).T

    W_m = W.E.reshape((1, W.D))

    # Positive triplet score
    expP = leftop(lhs, relmatricesl) + leftop(rhs, relmatricesr) + leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    #
    # # Negative triple score
    expNl = leftop(lhsn, relmatricesl) + leftop(rhsn, relmatricesr) + leftop(lhsn, rightop(rhsn, W_m))
    # expNr = leftop(lhs, relmatricesl) + leftop(rhsn, relmatricesr) + leftop(lhs, rightop(rhsn, W_m))
    similn = -T.sum(expNl, axis=1)
    # simirn = -T.sum(expNr, axis=1)


    # move the resultant matrices close according to similarity matrices
    # when they are close cost should decrease
    costl, outl = margincost(simi, similn, marge)
    # costr, outr = margincost(simi, simirn, marge)

    # cost = costl + costr
    cost = costl
    # out = T.concatenate([outl, outr])
    out=outl


    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    d = dict((i, i - lr * j) for i, j in zip(leftop.params + rightop.params, gradientsparams))
    updates = OrderedDict(sorted(d.items()))

    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lr * gradients_embedding
    updates.update({embedding.E: newE})

    gradients_embeddingrelMatl = T.grad(cost, rel_matricesl.E)
    newrelMatl = rel_matricesl.E - lr * gradients_embeddingrelMatl
    updates.update({rel_matricesl.E: newrelMatl})

    gradients_embeddingrelMatr = T.grad(cost, rel_matricesr.E)
    newrelMatr = rel_matricesr.E - lr * gradients_embeddingrelMatr
    updates.update({rel_matricesr.E: newrelMatr})

    gradients_embeddingW = T.grad(cost, W.E)
    newW = W.E - lr * gradients_embeddingW
    updates.update({W.E: newW})

    list_in = [lr, lrparam, inpl, inpr, inpo, inpln, inprn]
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
                            updates=updates, on_unused_input='ignore')
    # return theano.function(list_in, [T.mean(cost), T.mean(out),leftop(lhs, relmatricesl),expP,lhs,simi,similn,cost,
    #                                  cost.shape,out,gradients_embedding],updates=updates, on_unused_input='ignore',
    #                        mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=True))



def RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[1]
    rel_matricesr = embeddings[2]

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))#get embeddings for right side
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))

    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    W_m = W.E.reshape((1, W.D))

    expP = leftop(lhs, relmatricesl) + leftop(rhs, relmatricesr) + leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    # simi = -T.sum(Cossim(leftop(lhs, relmatricesl)+leftop(rhs, relmatricesr), leftop(lhs, rightop(rhs, W_m)).T ),axis=1)

    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
                           on_unused_input='ignore')

def RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))

    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T

    W_m = W.E.reshape((1, W.D))

    expP = leftop(lhs, relmatricesl) + leftop(rhs, relmatricesr) + leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    # simi = -T.sum(Cossim(leftop(lhs, relmatricesl)+leftop(rhs, relmatricesr), leftop(lhs, rightop(rhs, W_m)).T ), axis=1)

    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')


def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
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
        il = np.argwhere(true_triples[:, 0] == l).reshape(-1, ) #triples with this leftID
        io = np.argwhere(true_triples[:, 1] == o).reshape(-1, ) #triples with this relation (should return all since there is only one relation in BI)
        ir = np.argwhere(true_triples[:, 2] == r).reshape(-1, ) #triple with this rightID

        inter_l = [i for i in ir if i in io] #return the list of IDs which are present in ir and also in io
        rmv_idx_l = [true_triples[i, 0] for i in inter_l if true_triples[i, 0] != l] #get the leftIDs from true_triples which are  part of inter_l but not equal to l
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i, 2] for i in inter_r if true_triples[i, 2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr





# Similarity functions -------------------------------------------------------
def L1sim(left, right):
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def Dotsim(left, right):
    return T.sum(left * right, axis=1)

def Cossim(left, right):
    numerator = T.dot(left, right)
    denominator = T.sqrt(theano.tensor.sum(left ** 2) * theano.tensor.sum(right ** 2))
    currentSimilarity = numerator / denominator  # cosine similarity
    return currentSimilarity

# -----------------------------------------------------------------------------


# Cost ------------------------------------------------------------------------
def margincost(pos, neg, marge=1.0):
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0
# -----------------------------------------------------------------------------


# Activation functions --------------------------------------------------------
def rect(x):
    return x * (x > 0)


def sigm(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def lin(x):
    return x
# -----------------------------------------------------------------------------




# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({
            self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))
        })
        self.normalize = theano.function([], [], updates=self.updates)


# ----------------------------------------------------------------------------


class LayerdMat(object):
    """

    """

    def __init__(self):
        """
        Constructor.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.

        return x * y


class LayerMat(object):
    """
    Class for a layer with two input vectors, the 'right' member being a flat
    representation of a matrix on which to perform the dot product with the
    'left' vector [Structured Embeddings model, Bordes et al. AAAI 2011].
    """

    def __init__(self, act, n_inp, n_out):
        """
        Constructor.

        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inp: input dimension.
        :param n_out: output dimension.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.
        ry = y.reshape((y.shape[0], self.n_inp, self.n_out))
        rx = x.reshape((x.shape[0], x.shape[1], 1))
        return self.act((rx * ry).sum(1)) #returns  #examplesx#dimsx1


class LayerTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of
    of the 'left member' and 'right member'i.e. translating x by y.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x+y

class Unstructured(object):
    """
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x