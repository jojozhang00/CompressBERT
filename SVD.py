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

# apply sentence embeddings
model = SentenceTransformer('sentence-transformers/stsb-bert-base')

text1 = sts_train['sent_1'].tolist()
text2 = sts_train['sent_2'].tolist()
embeddings1 = model.encode(text1)
embeddings2 = model.encode(text2)

similarity = torch.cosine_similarity(torch.tensor(embeddings1), torch.tensor(embeddings2), dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_train['sim'])
print("The spearmanr result for sentence-transformers is:", result.correlation)

# for test data
t1 = sts_test['sent_1'].tolist()
t2 = sts_test['sent_2'].tolist()
e1 = model.encode(t1) 
e2 = model.encode(t2) 

# concat train and test sets (transductive learning)
sts_all = pd.concat([sts_train, sts_test], axis = 0, ignore_index = True)
text1_all = sts_all['sent_1'].tolist()
embeddings1_all = model.encode(text1_all) 

# define a method for SVD
def k_svd(data, dim, test):
    U, Sigma, VT = np.linalg.svd(data)   
    # Sigma only contains the values on the diagonal
    # U = U[:, :count]
    # Sigma = Sigma[:count]
    # Sigma = np.diag(Sigma)
    VT = VT[:dim,:]  
    redata = np.dot(test, VT.T)
    return redata

dim_ls = np.arange(10, 100, 10).tolist()
result_ls_aa = []
result_ls_ae = []
result_ls_aee = []

# M_train(train)
print('M_train(train):')
for index in range(len(dim_ls)):
    reduced_1 = k_svd(embeddings1, dim_ls[index], embeddings1)
    reduced_2 = k_svd(embeddings1, dim_ls[index], embeddings2)
    print(reduced_2.shape)

    similarity_SVD = torch.cosine_similarity(torch.tensor(reduced_1), torch.tensor(reduced_2), dim=1) 
    result = spearmanr(similarity_SVD.detach().numpy(), sts_train['sim'])
    result_ls_aa.append(result.correlation)

    print('Taking {}'.format(dim_ls[index]), 'as the projection dim, spearman r is:', result.correlation)

# M_train(test)
print('M_train(test):')
for index in range(len(dim_ls)):
    reduced_1 = k_svd(embeddings1, dim_ls[index], e1)
    reduced_2 = k_svd(embeddings1, dim_ls[index], e2)
    print(reduced_2.shape)

    similarity_SVD = torch.cosine_similarity(torch.tensor(reduced_1), torch.tensor(reduced_2), dim=1) 
    result = spearmanr(similarity_SVD.detach().numpy(), sts_test['sim'])
    result_ls_ae.append(result.correlation)

    print('Taking {}'.format(dim_ls[index]), 'as the projection dim, spearman r is:', result.correlation)    

# M_{train+test}(test)
print('M_train(test):')
for index in range(len(dim_ls)):
    reduced_1 = k_svd(embeddings1_all, dim_ls[index], e1)
    reduced_2 = k_svd(embeddings1_all, dim_ls[index], e2)
    print(reduced_2.shape)

    similarity_SVD = torch.cosine_similarity(torch.tensor(reduced_1), torch.tensor(reduced_2), dim=1) 
    result = spearmanr(similarity_SVD.detach().numpy(), sts_test['sim'])
    result_ls_aee.append(result.correlation)

    print('Taking {}'.format(dim_ls[index]), 'as the projection dim, spearman r is:', result.correlation)    


# plot line chart
plt.scatter(dim_ls, result_ls_aa)
plt.plot(dim_ls, result_ls_aa, label='M_train(train)')
plt.scatter(dim_ls, result_ls_ae)
plt.plot(dim_ls, result_ls_ae, label = 'M_train(test)')
plt.scatter(dim_ls, result_ls_aee)
plt.plot(dim_ls, result_ls_aee, label = 'M_{train+test}(test)')
plt.title('spearman r for SVD')
plt.xlabel('dimension')
plt.ylabel('spearman r')
plt.legend()
plt.savefig('./SVD.jpg')

