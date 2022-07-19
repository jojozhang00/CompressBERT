#!/usr/bin/env python3

# unsupervised greedy layer-wise pretraining for STS benchmark
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras import layers, losses

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


print('Apply greedy layer-wise autoencoder based method')
# print('Train the network on train set and test it on test set')
print('Use train set')
print('Train the network on the first column and test it on the second column')

latent_dim1 = 128
dense1_output_1 = []
dense1_output_2 = []
result_ls = []

# apply autoencoder based method
# define, fit and evaluate the base autoencoder
def create_base_model():
    """ Create a base model: one Dense layer. """
    # Retrieve relevant model configuration
    # Create model instance and add initial layers
    model = Sequential()
    model.add(Dense(128, activation='relu', name="Dense_1"))
    model.add(Dense(768, activation='sigmoid'))
    # Return model
    return model

def add_extra_layer(model):
    """ Add an extra Dense layer to the model. """
    # Define the layer that must be added
    layer_to_add = Dense(64, activation='relu', name="Dense_2")
    # Temporarily assign the output layer to a variable
    output_layer = model.layers[-1]
    # Set all upstream layers to nontrainable
    for layer in model.layers:
        layer.trainable = False
    # Remove output layer and add new layer
    model.pop()
    model.add(layer_to_add)
    model.add(Dense(128, activation='relu'))
    # Re-add output layer
    model.add(output_layer)
	# Save latent layer as another model
    dense2_layer_model  = Model(inputs=model.input, outputs=model.get_layer('Dense_2').output)
	# Use this model to produce output of latent layer
    dense2_output_1 = dense2_layer_model.predict(embeddings1)
    dense2_output_2 = dense2_layer_model.predict(embeddings2)
    print(dense2_output_1.shape)
	# Return trained model, with extra layer
    return model, dense2_output_1, dense2_output_2


def train_model(model, trainX, testX):
    # Compile  model
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # Train model
    model.fit(trainX, trainX,
                    epochs = 5,
                    shuffle = False,
					validation_data = (testX, testX))
    # Return trained model
    return model


model = create_base_model()
# Apply greedy layer-wise training
num_layers_to_add = 1
for i in range(num_layers_to_add):
	# Train and evaluate current model
	model = train_model(model, embeddings1, embeddings2)
	# Add extra layer
	model, dense2_output_1, dense2_output_2 = add_extra_layer(model)

similarity = torch.cosine_similarity(torch.tensor(dense2_output_1), torch.tensor(dense2_output_2), dim=1) 
result = spearmanr(similarity.detach().numpy(), sts_train['sim'])
result_ls.append(result.correlation)

print (f'Taking {64} as the hidden layer dimension for AE, the spearmanr result is:', result.correlation)