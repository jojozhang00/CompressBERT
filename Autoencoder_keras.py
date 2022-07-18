#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device for training: ", device)

text1 = sts_train['sent_1'].tolist()
text2 = sts_train['sent_2'].tolist()

# apply sentence embeddings
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer('sentence-transformers/stsb-bert-base')
# stsb-bert-base work better
embeddings1 = model.encode(text1)
embeddings2 = model.encode(text2)

similarity = torch.cosine_similarity(torch.tensor(embeddings1), torch.tensor(embeddings2), dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_train['sim'])
print("The spearmanr result for sentence-transformers is:", result.correlation)


print('Apply autoencoder')
# apply autoencoder based method
# dataLoader is used to load the dataset 
# for training
percent = np.arange(0.1, 1.1, 0.1).tolist()
dim_ls = []
for index in range(len(percent)):
    dim_ls.append(int(768*percent[index]))
# dim_ls = np.arange(300, 800, 50).tolist()
# dim_ls.append(768)
result_AE_ls = []

# stack
for latent_dim in dim_ls:

    input = Input(shape=(768,))
    encoded = layers.Dense(latent_dim, activation='relu')(input)
    decoded = layers.Dense(768, activation='sigmoid')(encoded)

    # model initialization
    # construct autoencoder
    autoencoder = keras.Model(inputs = input, outputs = decoded)
    
    # construct encoder
    encoder = keras.Model(inputs = input, outputs = encoded)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(embeddings1, embeddings1,
                    epochs = 100,
                    shuffle = False,
                    validation_data = (embeddings2, embeddings2))

    embeddings1_encoded = encoder.predict(embeddings1)
    embeddings2_encoded = encoder.predict(embeddings2)

    similarity = torch.cosine_similarity(torch.tensor(embeddings1_encoded), torch.tensor(embeddings2_encoded), dim=1) 
    result_AE = spearmanr(similarity.detach().numpy(), sts_train['sim'])
    result_AE_ls.append(result_AE.correlation)

    print (f'Taking {latent_dim} as the hidden layer dimension for AE, the spearmanr result is:', result_AE.correlation)

# plot line chart
plt.figure(1)
plt.scatter(dim_ls, result_AE_ls)
plt.plot(dim_ls, result_AE_ls)
plt.title('spearmanr results for auto-encoder')
plt.xlabel('hidden layer dimension')
plt.ylabel('spearman r')
plt.savefig('./AE_tf.jpg')










'''

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   

    self.encoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation='relu'),
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(768, activation='sigmoid'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded

# model initialization
autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(embeddings1, embeddings1,
                epochs=50,
                shuffle=False,
                validation_data=(embeddings2, embeddings2))

# save model
model.save("./model_tfAE.h5")
# load model
model = load_model('./model_tfAE.h5')
# summarize model
model.summary()

'''