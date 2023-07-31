import pandas as pd
import numpy as np
from tqdm import tqdm
from word_to_vec_model_creation import createWord2VecModel, generate_4mers

def kernel(self, X1, X2=None):
    if X2 is None: 
        X2=X1
    n_samples_1 = X1.shape[0]
    n_samples_2 = X2.shape[0]
    K = np.zeros((n_samples_1, n_samples_2))
    for ii in tqdm(range(n_samples_1)):
        for jj in range(n_samples_2):
            K[ii,jj] = rbfSimilarity(X1[ii], X2[jj])
    return K

def rbfSimilarity(x, y):
    norm_fact = (np.sqrt(2 * np.pi) * 0.22360679775) ** len(x)
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * 0.22360679775**2)) / norm_fact
    

class RidgeRegression():
    def __init__(self):
        self.alpha = None

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        if not self.alpha:
            self.alpha = np.ones((n_samples))
        K = kernel(self.X_train, self.X_train)
        self.alphas = np.linalg.solve(K+n_samples*self.alpha*np.eye(n_samples), y)
        return self.alphas

    def predict(self, X):
        K = kernel(X, self.X_train)
        y = np.dot(K, self.alphas)
        return y

    def predict_classes(self, X, threshold=0.5):
        return np.where(np.dot(self.kernel.gram(X, self.X_train), self.alphas)>threshold, 1, 0)
    

def rbfKernel():
    Y0_train = pd.read_csv("data/Ytr0.csv", sep=",", index_col=0).values
    Y1_train = pd.read_csv("data/Ytr1.csv", sep=",", index_col=0).values
    Y2_train = pd.read_csv("data/Ytr2.csv", sep=",", index_col=0).values

    Y0_train = np.where(Y0_train == 0, -1, 1)
    Y1_train = np.where(Y1_train == 0, -1, 1)
    Y2_train = np.where(Y2_train == 0, -1, 1)


    model = createWord2VecModel()
    def flattenEmbeddings(sliceOfString):
        return model.wv[sliceOfString].flatten().tolist()

    def transformToEmbeddings(sequence):
        kmers = generate_4mers(sequence)
        return flattenEmbeddings(kmers)

    X0_train_emb = np.array(pd.read_csv("data/Xtr0.csv", sep=",", index_col=0)["seq"].apply(transformToEmbeddings).tolist()) #chaos_game_representation).tolist())
    X1_train_emb = np.array(pd.read_csv("data/Xtr1.csv", sep=",", index_col=0)["seq"].apply(transformToEmbeddings).tolist()) #chaos_game_representation).tolist())
    X2_train_emb = np.array(pd.read_csv("data/Xtr2.csv", sep=",", index_col=0)["seq"].apply(transformToEmbeddings).tolist()) #chaos_game_representation).tolist())

    split = int(len(X0_train_emb) * .5)
    X_sample,x_validation = X0_train_emb[:split],X0_train_emb[:split]
    y_sample,y_validation = Y0_train[split:],Y0_train[split:]
    rbfModel = RidgeRegression()
    rbfModel.fit(X_sample,y_sample)
    prediction = rbfModel.predict(x_validation)
    val_acc = np.sum(np.squeeze(prediction)==np.squeeze(y_validation)) / len(y_validation)
    print("Dataset 0 Validation accuracy:", val_acc)
    
    
    split = int(len(X1_train_emb) * .5)
    X_sample,x_validation = X1_train_emb[:split],X1_train_emb[:split]
    y_sample,y_validation = Y1_train[split:],Y1_train[split:]
    rbfModel = RidgeRegression()
    rbfModel.fit(X_sample,y_sample)
    prediction = rbfModel.predict(x_validation)
    val_acc = np.sum(np.squeeze(prediction)==np.squeeze(y_validation)) / len(y_validation)
    print("Dataset 1 Validation accuracy:", val_acc)
    
    split = int(len(X2_train_emb) * .5)
    X_sample,x_validation = X2_train_emb[:split],X2_train_emb[:split]
    y_sample,y_validation = Y2_train[split:],Y2_train[split:]
    rbfModel = RidgeRegression()
    rbfModel.fit(X_sample,y_sample)
    prediction = rbfModel.predict(x_validation)
    val_acc = np.sum(np.squeeze(prediction)==np.squeeze(y_validation)) / len(y_validation)
    print("Dataset 2 Validation accuracy:", val_acc)