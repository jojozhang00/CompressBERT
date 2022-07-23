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
import umap

# set random seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# # import STS-B dataset 
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

print("Apply UMAP:")

result_ls = []
new_dim_ls = np.arange(2, 100, 10).tolist()

for index in range(len(new_dim_ls)):

    new_dim = int(new_dim_ls[index])
    print(">> new dim: {}".format(new_dim))

    reducer = umap.UMAP(n_components=new_dim).fit(embeddings1)
    # reducer = umap.UMAP(n_components=new_dim).fit(embeddings1_all)

    reduced_1 = reducer.transform(e1)
    reduced_2 = reducer.transform(e2)

    similarity = torch.cosine_similarity(torch.Tensor(reduced_1), torch.Tensor(reduced_2), dim=1) 
    result = spearmanr(similarity.detach().numpy(), sts_test['sim'])
    result_ls.append(result.correlation)

    print ('Taking {}'.format(new_dim_ls[index]), 'as the new dimension, spearman r is:', result.correlation)


# plot line chart
plt.scatter(new_dim_ls, result_ls)
plt.plot(new_dim_ls, result_ls)
plt.title('spearman r for UMAP')
plt.xlabel('dimension')
plt.ylabel('spearman r')
plt.legend()
plt.savefig('./UMAP.jpg')