#!/usr/bin/env python3

# use pytorch to construct the network

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
from torch.utils.data import DataLoader,TensorDataset

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



print('Then, apply autoencoder based method')

# apply autoencoder based method
# dataLoader is used to load the dataset 
# for training
 
dataset = TensorDataset(embeddings1)
loader = DataLoader(dataset, batch_size = 32, shuffle = False)
num_batch = len(loader)
print(f"Data loaded: {num_batch} batches")
percent = np.arange(0.1, 1.1, 0.1).tolist()


# creating a PyTorch class
# 768 ==> dim/2 ==> 768
class AE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
          
        # building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(768, dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(dim, int(dim/2)),
            # torch.nn.ReLU()
        )
          
        # building an linear decoder with Linear
        # layer followed by Relu activation function
        # the Sigmoid activation function
        # outputs the value between 0 and 1
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(int(dim/2), dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(dim, 768),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

dim_ls = np.arange(300, 800, 50).tolist()
dim_ls.append(768)
new_ls = [int(x/2) for x in dim_ls]
result_AE_ls =[]

for dim in dim_ls:
    # model initialization
    model_AE = AE(dim = dim).to(device)
    
    # validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    
    # using an adam optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model_AE.parameters(),
                                lr = 1e-3,
                                weight_decay = 1e-8)

    epochs = 200 # converge
    outputs = []
    losses = []

    print("Start training...")
    for epoch in range(epochs):
        model_AE.train()
        total_loss = 0
        for _, sentence in enumerate(loader):
            sentence = sentence[0].to(device)
            
            # output of Autoencoder
            _, reconstructed = model_AE(sentence)
            
            # calculating the loss function
            optimizer.zero_grad()
            loss = loss_function(reconstructed, sentence)
            
            # the gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # storing the losses in a list for plotting
        
        losses.append(total_loss/num_batch)

        print("Epoch %d: Training loss %.4f" %(epoch, total_loss/num_batch))
        # outputs.append((epochs, sentence, reconstructed))

    print("Training finished!")

   
    # plot the relationship between losses and epoches
    print("Plot training loss")
    plt.figure(1)
    plt.plot(list(range(epochs)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./AE_loss.jpg')

  

    # save the model
    torch.save(model_AE.state_dict(), './AE.pth')


    # load the model
    model_AE = AE(dim = dim).to(device)
    model_AE.load_state_dict(torch.load('./AE.pth'))
    model_AE.eval()

    # get the results
    dataset1 = TensorDataset(embeddings1)
    loader1 = DataLoader(dataset1, batch_size = 1, shuffle = False)
    dataset2 = TensorDataset(embeddings2)
    loader2 = DataLoader(dataset2, batch_size = 1, shuffle = False)

    embeddings1_AE = []
    embeddings2_AE = []

    for _, sentence in enumerate(loader1):
        sentence = sentence[0].to(device)    
        encoded, _ = model_AE(sentence)   
        embeddings1_AE.append(encoded[0].tolist())
        
    for _, sentence in enumerate(loader2):
        sentence = sentence[0].to(device)    
        encoded, _ = model_AE(sentence)
        embeddings2_AE.append(encoded[0].tolist())

    # print(embeddings1_AE) # list里面有很多tensor

    embeddings1_AE = torch.Tensor(embeddings1_AE)
    embeddings2_AE = torch.Tensor(embeddings2_AE)

    similarity = torch.cosine_similarity(embeddings1_AE, embeddings2_AE, dim=1) 
    result_AE = spearmanr(similarity.detach().numpy(), sts_train['sim'])
    result_AE_ls.append(result_AE.correlation)

    print (f'Taking {dim} as the hidden layer dimension for AE, the spearmanr result is:', result_AE.correlation)

# plot line chart
plt.figure(2)
plt.scatter(dim_ls, result_AE_ls)
plt.plot(dim_ls, result_AE_ls)
plt.title('spearmanr results for auto-encoder')
plt.xlabel('hidden layer dimension')
plt.ylabel('spearmanr')
plt.savefig('./AE.jpg')
