#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn import random_projection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device for training: ", device)

text1 = sts_train['sent_1'].tolist()
text2 = sts_train['sent_2'].tolist()

# apply sentence embeddings
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer('sentence-transformers/stsb-bert-base')
# stsb-bert-base work better
embeddings1 = model.encode(text1)
embeddings1 = torch.tensor(embeddings1)  # convert numpy array to tensor
embeddings2 = model.encode(text2)
embeddings2 = torch.tensor(embeddings2)

similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_train['sim'])
print("The spearmanr result for sentence-transformers is:", result.correlation)


print('Then, apply SVD.')

# apply SVD to the procedure above
# define a method for SVD
# select count = n eigenvalues
def k_svd(embeddings, count):
    U, Sigma, VT = np.linalg.svd(embeddings)   
    # Sigma only contains the values on the diagonal
    U = U[:, :count]
    Sigma = Sigma[:count]
    Sigma = np.diag(Sigma)
    VT = VT[:count,:]  
    re_embeddings = np.dot(np.dot(U, Sigma), VT)
    return re_embeddings

# SVD gives 100 diagonals for matrix Sigma
percent = np.arange(0.1, 1.1, 0.1).tolist()
result_SVD_ls = []

for index in range(len(percent)):
    embeddings1_SVD = k_svd(embeddings1, int(percent[index]*100)) # take the top biggest eigenvalues
    embeddings2_SVD = k_svd(embeddings2, int(percent[index]*100))

    embeddings1_SVD = torch.tensor(embeddings1_SVD)
    embeddings2_SVD = torch.tensor(embeddings2_SVD)

    similarity_SVD = torch.cosine_similarity(embeddings1_SVD, embeddings2_SVD, dim=1) 
    result_SVD = spearmanr(similarity_SVD.detach().numpy(), sts_train['sim'])
    result_SVD_ls.append(result_SVD.correlation)

    print ('Taking {:.2%}'.format(percent[index]), 'of the eigenvalues, the spearmanr result is:', result_SVD.correlation)

# plot line chart
plt.figure(1)
plt.scatter(percent, result_SVD_ls)
plt.plot(percent, result_SVD_ls)
plt.title('spearmanr results for SVD')
plt.xlabel('percentile')
plt.ylabel('spearmanr')
plt.savefig('./SVD.jpg')



print('Then, apply random projection.')

# apply Guassian random projection
# percent = np.arange(0.1, 1.1, 0.1).tolist()
result_rp_ls = []
size = embeddings1.shape[1]

for index in range(len(percent)):
    transformer = random_projection.GaussianRandomProjection(n_components = int(size*percent[index]), random_state = 1)
    #set n-components otherwise error occurs

    embeddings1_rp = transformer.fit_transform(embeddings1)
    embeddings2_rp = transformer.fit_transform(embeddings2)

    embeddings1_rp = torch.tensor(embeddings1_rp)
    embeddings2_rp = torch.tensor(embeddings2_rp)

    similarity_rp = torch.cosine_similarity(embeddings1_rp, embeddings2_rp, dim=1) 
    result_rp = spearmanr(similarity_rp.detach().numpy(), sts_train['sim'])
    result_rp_ls.append(result_rp.correlation)

    print ('Taking {:.2%}'.format(percent[index]), 'of the original dimension as target, the spearmanr result is:', result_rp.correlation)

# plot line chart
plt.figure(2)
plt.scatter(percent, result_rp_ls)
plt.plot(percent, result_rp_ls)
plt.title('spearmanr results for Gaussian random projection')
plt.xlabel('percentile')
plt.ylabel('spearmanr')
plt.savefig('./random_projection.jpg')

