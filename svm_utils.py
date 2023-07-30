import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm



# createKmerSet creats a k-mer sets given the list of datastrings X and k(size of string)
def createKmerSet(X, k):
    kmer_sets={}
    len_seq = len(X[0])
    idx = len(kmer_sets)
    for i in range(len(X)):
        x = X[i]
        kmer_x = [x[i:i + k] for i in range(len_seq - k + 1)]
        for kmer in kmer_x:
            if kmer not in kmer_sets:
                kmer_sets[kmer] = idx
                idx += 1
    return kmer_sets


def m_neighbours(kmer, mismatchLength, recurs=0):
    if mismatchLength == 0:
        return [kmer]
    k = len(kmer)
    neighbours = m_neighbours(kmer, mismatchLength - 1, recurs + 1)
    for j in range(len(neighbours)):
        neighbour = neighbours[j]
        for i in range(recurs, k - mismatchLength + 1):
            for l in ['G', 'T', 'A', 'C']:
                neighbours.append(neighbour[:i] + l + neighbour[i + 1:])
    return list(set(neighbours))


def retrieveNeighbours(kmer_set, mismatchLen):
    kmers_list = list(kmer_set.keys())
    kmers = np.array(list(map(list, kmers_list)))
    numberOfKMers = kmers.shape[0]
    neighbours = {}
    for i in range(numberOfKMers):
        neighbours[kmers_list[i]] = []
    for i in tqdm(range(numberOfKMers)):
        kmer = kmers_list[i]
        kmer_neighbours = m_neighbours(kmer, mismatchLen)
        for neighbour in kmer_neighbours:
            if neighbour in kmer_set:
                neighbours[kmer].append(neighbour)
    return neighbours

def getNeighbours(datasets,kmer_size,mismatchLen, datasetIndex=0):
    file_name = str(datasetIndex)+'_'+str(kmer_size)+'_'+str(mismatchLen)+'.p'
    try:
        return pickle.load(open(file_name, 'rb'))
    except:
        kmer_set = createKmerSet(np.concatenate((datasets[datasetIndex][0], datasets[datasetIndex][1]), axis=0), kmer_size)
        neighbours = retrieveNeighbours(kmer_set, mismatchLen)
        pickle.dump([neighbours, kmer_set], open(file_name, 'wb'))
    return neighbours, kmer_set