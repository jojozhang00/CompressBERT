#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import torch
import csv
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# import STS-B dataset 
sts_dataset = tf.keras.utils.get_file(
    fname="Stsbenchmark.tar.gz",
    origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
    extract=True)
sts_train = pd.read_table(
    os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-train.csv"),
    error_bad_lines=False,
    skip_blank_lines=True,
    usecols=[4, 5, 6],
    names=["sim", "sent_1", "sent_2"])
sts_test = pd.read_table(
    os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"),
    error_bad_lines=False,
    quoting=csv.QUOTE_NONE,
    skip_blank_lines=True,
    usecols=[4, 5, 6],
    names=["sim", "sent_1", "sent_2"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device for training: ", device)

text1 = sts_train['sent_1'].tolist()
text2 = sts_train['sent_2'].tolist()

# apply sentence embeddings
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer('sentence-transformers/stsb-bert-base')
# stsb-bert-base work better

# get the embeddings
def get_embeddings(text):
    embeddings = model.encode(text)
    # convert numpy array to tensor
    embeddings = torch.tensor(embeddings)
    return embeddings

embeddings1 = get_embeddings(text1) 
embeddings2 = get_embeddings(text2) 

similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_train['sim'])
print("For train set:")
print("Spearman r for sentence-transformers is:", result.correlation)


# define a method for SVD
# select top biggest (count = eigenvalues * percentage) eigenvalues
def k_svd(embeddings, percent):
    U, Sigma, VT = np.linalg.svd(embeddings)   
    # Sigma only contains the values on the diagonal
    count = int(Sigma.shape[0]*percent)
    U = U[:, :count]
    Sigma = Sigma[:count]
    Sigma = np.diag(Sigma)
    VT = VT[:count,:]  
    re_embeddings = np.dot(np.dot(U, Sigma), VT)
    return re_embeddings

percent = np.arange(0.1, 1.1, 0.1).tolist()

# for train set
result_SVD = []

print('Apply SVD:')

for index in range(len(percent)):
    embeddings1_SVD = k_svd(embeddings1, percent[index])
    embeddings2_SVD = k_svd(embeddings2, percent[index])

    embeddings1_SVD = torch.tensor(embeddings1_SVD)
    embeddings2_SVD = torch.tensor(embeddings2_SVD)

    similarity_SVD = torch.cosine_similarity(embeddings1_SVD, embeddings2_SVD, dim=1) 
    result = spearmanr(similarity_SVD.detach().numpy(), sts_train['sim'])
    result_SVD.append(result.correlation)

    print ('Taking {:.2%}'.format(percent[index]), 'of the eigenvalues, spearman r is:', result.correlation)


# concat train and test sets (transductive learning)
sts_all = pd.concat([sts_train, sts_test], axis = 0, ignore_index = True)

text1_all = sts_all['sent_1'].tolist()
text2_all = sts_all['sent_2'].tolist()

embeddings1_all = get_embeddings(text1_all) 
embeddings2_all = get_embeddings(text2_all) 

similarity = torch.cosine_similarity(embeddings1_all, embeddings2_all, dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_all['sim'])
print("For train and test set:")
print("Spearman r for sentence-transformers is:", result.correlation)


# for train and test set
result_SVD_all = []

for index in range(len(percent)):
    embeddings1_SVD = k_svd(embeddings1_all, percent[index]) # take the top biggest eigenvalues
    embeddings2_SVD = k_svd(embeddings2_all, percent[index])

    embeddings1_SVD = torch.tensor(embeddings1_SVD)
    embeddings2_SVD = torch.tensor(embeddings2_SVD)

    similarity_SVD = torch.cosine_similarity(embeddings1_SVD, embeddings2_SVD, dim=1) 
    result = spearmanr(similarity_SVD.detach().numpy(), sts_all['sim'])
    result_SVD_all.append(result.correlation)

    print ('Taking {:.2%}'.format(percent[index]), 'of the eigenvalues, spearman r is:', result.correlation)

# plot line chart
plt.scatter(percent, result_SVD)
plt.plot(percent, result_SVD, label='train set')
plt.scatter(percent, result_SVD_all)
plt.plot(percent, result_SVD_all, label = 'train and test set')
plt.title('spearman r for SVD')
plt.xlabel('percentage')
plt.ylabel('spearman r')
plt.legend()
plt.savefig('./SVD.jpg')