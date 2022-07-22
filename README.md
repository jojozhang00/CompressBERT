# Compress BERT
Currently, the mission is to apply and compare different dimensionality reduction methods. The goal is to find effective algoritms for sentence embeddings. The following shows the results for some methods.

After applying sentence transformers: \
train data size (5711, 768) \
test data size (1379, 768) 

Note: \
transductive learning: train on the 1st coulumn of train and test data (7090, 768), test on test set \
inductive learning: train on the 1st coulumn train data (5711, 768), test on test set 

## Reuslts
The results represent scores that spearman r is stable at and corresponding rough projection dimension (for reference).

### PCA
For transductive learning:\
spearman r: 0.832 ~ 0.850, projection dimension: 50 ~ 710 \
For inductive learning:\
spearman r: 0.833 ~ 0.849, projection dimension: 50 ~ 730 \
Note:\
Good performance and works very fast.

### UMAP
For transductive learning:\
spearman r: 0.540 ~ 0.550, projection dimension: 12 ~ 92 (results have obvious flunctuation)\
For inductive learning:\
spearman r: 0.533 ~ 0.547, projection dimension: 12 ~ 92 (same as above)\
Note: \
Poor performance and works obviously slowlier as the dimension grows. The offical website suggests reasonable projection dimension in range(2, 100), thus larger dimensions are not in the consideration and they need quite much time to compute. 

### Gaussian random projection
For transductive learning:\
spearman r: 0.837 ~ 0.851, projection dimension: 70 ~ 730 \
For inductive learning:\
spearman r: 0.837 ~ 0.851, projection dimension: 70 ~ 730 \
Note:\
Good performance and works fast.

### greedy layer-wise autoencoder (2 layers) (keras)
epoch = 70 \
For transductive learning:\
spearman r: 0.823 ~ 0.837, projection dimension: 150 ~ 375 (results have flunctuation)\
For inductive learning:\
spearman r: 0.825 ~ 0.840, projection dimension: 150 ~ 375 (same as above) \
Note:\
Relatively good performance and works neither slowly nor fast.

### autoencoder (2 layers) (keras)
epoch = 70
