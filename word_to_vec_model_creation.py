import numpy as np
from gensim.models import Word2Vec
import pandas as pd

def generate_4mers(sequence):
    stride = 1  # Length of k-mer
    k = 4  # Length of k-mer
    n = len(sequence)
    kmers = []

    final_start = n - k + 1
    for i in range(final_start):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    return kmers

def createWord2VecModel():
    texts =np.array([])
    X_test0=pd.read_csv("data/Xte0.csv")
    X_test1=pd.read_csv("data/Xte1.csv")
    X_test2=pd.read_csv("data/Xte2.csv")
    X_train_0=pd.read_csv("data/Xtr0.csv")
    X_train=pd.read_csv("data/Xtr1.csv")
    X_valid=pd.read_csv("data/Xtr2.csv")

    X_train["k_mers"]= X_train["seq"].apply(generate_4mers)
    X_test0["k_mers"]= X_test0["seq"].apply(generate_4mers)
    X_test1["k_mers"]= X_test1["seq"].apply(generate_4mers)
    X_valid["k_mers"]= X_valid["seq"].apply(generate_4mers)

    X_test0["k_mers"]= X_test0["seq"].apply(generate_4mers)
    X_test1["k_mers"]= X_test1["seq"].apply(generate_4mers)
    X_test2["k_mers"]= X_test2["seq"].apply(generate_4mers)
    X_train_0["k_mers"]= X_train_0["seq"].apply(generate_4mers)
    X_train["k_mers"]= X_train["seq"].apply(generate_4mers)
    X_valid["k_mers"]= X_valid["seq"].apply(generate_4mers)

    texts = np.append(texts, X_test0["k_mers"])
    texts = np.append(texts, X_test1["k_mers"])
    texts = np.append(texts, X_test2["k_mers"])
    texts = np.append(texts, X_train_0["k_mers"])
    texts = np.append(texts, X_train["k_mers"])
    texts = np.append(texts, X_valid["k_mers"])

    # Creating a word to vec embedding
    return Word2Vec(sentences=texts, vector_size=1, min_count=1)