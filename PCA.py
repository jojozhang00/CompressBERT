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
from sklearn.decomposition import PCA

# set random seed
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

result_PCA = []
# new_dim_ls = []
new_dim_ls = np.arange(10, 140, 10).tolist()
# percent = np.arange(0.1, 1.1, 0.1).tolist()
# dim = embeddings1.shape[1]

for index in range(len(new_dim_ls)):
    new_dim = int(new_dim_ls[index])
    # new_dim_ls.append(new_dim)
    pca = PCA(n_components = new_dim)
    pca.fit(embeddings1)
    reduced_1 = pca.fit_transform(embeddings1)
    reduced_1 = torch.Tensor(reduced_1)

    reduced_2 = pca.transform(embeddings2)
    reduced_2 = torch.Tensor(reduced_2)

    similarity_PCA = torch.cosine_similarity(reduced_1, reduced_2, dim=1) 
    result = spearmanr(similarity_PCA.detach().numpy(), sts_train['sim'])
    result_PCA.append(result.correlation)

    print ('Taking {}'.format(new_dim_ls[index]), 'of the origin dim, spearman r is:', result.correlation)

# plot line chart
plt.scatter(new_dim_ls, result_PCA)
plt.plot(new_dim_ls, result_PCA)
plt.title('spearman r for PCA')
plt.xlabel('dimension')
plt.ylabel('spearman r')
plt.legend()
plt.savefig('./PCA.jpg')