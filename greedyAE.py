#!/usr/bin/env python3

# unsupervised greedy layer-wise pretraining for STS benchmark
import pandas as pd
import numpy as np
import os
import csv
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
text2_all = sts_all['sent_2'].tolist()
embeddings1_all = model.encode(text1_all) 
embeddings2_all = model.encode(text2_all) 

print('Apply greedy layer-wise autoencoder based method')

dense1_output_1 = []
dense1_output_2 = []

# apply autoencoder based method
def create_base_model(dim1):
    """ Create a base model: one Dense layer. """
    # Retrieve relevant model configuration
    # Create model instance and add initial layers
    model = Sequential()
    model.add(Dense(dim1, activation='relu', name="Dense_1"))
    model.add(Dense(768, activation='sigmoid'))
    # Return model
    return model

def add_extra_layer(model, dim2):
    """ Add an extra Dense layer to the model. """
    # Define the layer that must be added
    layer_to_add = Dense(dim2, activation='relu', name="Dense_2")
    # Temporarily assign the output layer to a variable
    output_layer = model.layers[-1]
    # Set all upstream layers to nontrainable
    for layer in model.layers:
        layer.trainable = False
    # Remove output layer and add new layer
    model.pop()
    model.add(layer_to_add)
    model.add(Dense(2*dim2, activation='relu'))
    # Re-add output layer
    model.add(output_layer)
	# Save latent layer as another model
    dense2_layer_model  = Model(inputs=model.input, outputs=model.get_layer('Dense_2').output)
	# Use this model to produce output of latent layer
    dense2_output_1 = dense2_layer_model.predict(e1)
    dense2_output_2 = dense2_layer_model.predict(e2)
	# Return trained model, with extra layer
    return model, dense2_output_1, dense2_output_2

def train_model(model, trainX, testX):
    # Compile  model
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # Train model
    model.fit(trainX, trainX,
                    epochs = 70,
                    shuffle = False,
					validation_data = (testX, testX))
    # Return trained model
    return model

dim_ls = np.arange(100, 800, 50).tolist()
new_ls = [int(x/2) for x in dim_ls]
result_ls = []

# Apply greedy layer-wise training
for index in range(len(dim_ls)):
	model = create_base_model(dim_ls[index])
	# Train and evaluate current model
	model = train_model(model, embeddings1_all, embeddings2_all)
	# Add extra layer
	model, dense2_output_1, dense2_output_2 = add_extra_layer(model, new_ls[index])

	similarity = torch.cosine_similarity(torch.tensor(dense2_output_1), torch.tensor(dense2_output_2), dim=1) 
	result = spearmanr(similarity.detach().numpy(), sts_test['sim'])
	result_ls.append(result.correlation)

	print (f'Taking {new_ls[index]} as the hidden layer dimension for AE, the spearmanr result is:', result.correlation)

print(result_ls)

# plot line chart
plt.figure()
plt.scatter(new_ls, result_ls)
plt.plot(new_ls, result_ls)
plt.title('spearman r for greedy auto-encoder')
plt.xlabel('hidden layer dimension')
plt.ylabel('spearman r')
plt.savefig('./greedy_AE.jpg')