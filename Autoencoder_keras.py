#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.layers import Input
from keras.models import Sequential, load_model
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# set random seed
tf.random.set_seed(0)
np.random.seed(0)

tf.config.list_physical_devices('GPU')

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

# # concat train and test sets (transductive learning)
# sts_all = pd.concat([sts_train, sts_test], axis = 0, ignore_index = True)
# text1_all = sts_all['sent_1'].tolist()
# text2_all = sts_all['sent_2'].tolist()
# embeddings1_all = model.encode(text1_all) 
# embeddings2_all = model.encode(text2_all) 

print('Apply autoencoder')

# apply autoencoder based method
dim_ls = np.arange(50, 370, 20).tolist()
new_ls = [int(x/2) for x in dim_ls]
result_AE_ls = []

# stack
for latent_dim in dim_ls:

    input = Input(shape=(768,))
    encoded = layers.Dense(latent_dim*2, activation='relu')(input)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)
    decoded = layers.Dense(latent_dim*2, activation='relu')(encoded)
    decoded = layers.Dense(768, activation='sigmoid')(decoded)

    # model initialization
    # construct autoencoder
    autoencoder = keras.Model(inputs = input, outputs = decoded)
    
    # construct encoder
    encoder = keras.Model(inputs = input, outputs = encoded)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(embeddings1, embeddings1,
                    epochs = 70,
                    shuffle = False,
                    validation_data = (embeddings2, embeddings2))

    embeddings1_encoded = encoder.predict(e1)
    embeddings2_encoded = encoder.predict(e2)

    similarity = torch.cosine_similarity(torch.tensor(embeddings1_encoded), torch.tensor(embeddings2_encoded), dim=1) 
    result_AE = spearmanr(similarity.detach().numpy(), sts_test['sim'])
    result_AE_ls.append(result_AE.correlation)

    print (f'Taking {latent_dim} as the hidden layer dimension for AE, the spearmanr result is:', result_AE.correlation)

print(result_AE_ls)

# plot line chart
plt.figure(1)
plt.scatter(dim_ls, result_AE_ls)
plt.plot(dim_ls, result_AE_ls)
plt.title('spearmanr results for auto-encoder')
plt.xlabel('hidden layer dimension')
plt.ylabel('spearman r')
plt.savefig('./AE_tf.jpg')