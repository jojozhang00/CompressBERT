import pathlib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.layers import Input
from keras.models import Sequential, load_model
import torch
from sklearn import random_projection
import csv
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA, KernelPCA
import pickle as pk
from datasets import load_dataset


# set random seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = load_dataset("trec")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device for training: ", device)

# apply sentence encoder
path = 'sentence-transformers/all-mpnet-base-v2'
# path = 'sentence-transformers/msmarco-roberta-base-v2'
# path = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
model = SentenceTransformer(path)
name=path.split('/')[1]

# concat train and test sets (transductive setting)
embeddings_all = model.encode(dataset['train']['text']+dataset['test']['text'])  

dim_origin = embeddings_all.shape[1] 
dim_ls = np.arange(40, dim_origin, 20).tolist() 

# # PCA
# result_PCA = []
# print("PCA:")
# for index in range(len(dim_ls)):

#     new_dim = dim_ls[index]

#     pca = PCA(n_components = new_dim)
#     pca.fit(embeddings_all)

#     # save model
#     pk.dump(pca, open(name+"PCA{}.pkl".format(dim_ls[index]),"wb"))

#KPCA
result_KPCA = []
print("KPCA:")
for index in range(len(dim_ls)):

    new_dim = dim_ls[index]

    kernel_pca = KernelPCA(n_components=new_dim, 
                           kernel="rbf",
                           eigen_solver='randomized')
    kernel_pca.fit(embeddings_all)

    pk.dump(kernel_pca, open('random'+name+"KPCA{}.pkl".format(dim_ls[index]),"wb"))


# # SVD
# result_SVD = []
# print("SVD:")
# for index in range(len(dim_ls)):

#     new_dim = dim_ls[index]

#     svd = TruncatedSVD(n_components=new_dim)
#     svd.fit(embeddings_all)

#     # save model
#     pk.dump(svd, open(name+"SVD{}.pkl".format(dim_ls[index]),"wb"))

# # AE1
# result_AE = []
# print("AE1:")
# for latent_dim in dim_ls:

#     input = Input(shape=(dim_origin,))
#     encoded = layers.Dense(latent_dim, activation='relu')(input)
#     decoded = layers.Dense(dim_origin, activation='sigmoid')(encoded)

#     autoencoder = keras.Model(inputs = input, outputs = decoded)
#     encoder = keras.Model(inputs = input, outputs = encoded)

#     autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
#     autoencoder.fit(embeddings_all, embeddings_all,
#                     epochs = 100,
#                     shuffle = False)
    
#     encoder.save(name+'encoder{}.h5'.format(latent_dim))

# # Guassian random projection
# result_rp = []
# print("Guassian random projection:")
# for index in range(len(dim_ls)):
#     Grp = random_projection.GaussianRandomProjection(n_components = dim_ls[index], random_state = 0)
#     Grp.fit(embeddings_all)

#     # save model
#     pk.dump(Grp, open(name+"Grp{}.pkl".format(dim_ls[index]),"wb"))
