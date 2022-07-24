# Compress BERT
Currently, the mission is to apply and compare different dimensionality reduction methods. The goal is to find effective algoritms for sentence embeddings. 

After applying sentence transformers: \
train data size (5711, 768) \
test data size (1379, 768) 

Note: \
transductive learning: train on the 1st coulumn of train and test data (7090, 768), test on test set \
inductive learning: train on the 1st coulumn train data (5711, 768), test on test set 

## Reuslts
The results represent scores that spearman r is stable at and corresponding rough projection dimension (for reference). Projection dimension larger than 375 is not considered here. 


Training speed:\
Gaussian random projection ~ PCA > SVD >> UMAP > AE(1 layer) > AE(2 layers) ~ greedy AE 

Performance:\
SVD ~ Gaussian random projection ~ PCA > AE(1 layer) > AE(2 layers) > greedy AE >> UMAP 

Problems: \
1.UMAP should perform much better? \
2.Autoencoder with 2 layers (keras) plot: increase first and then decline?


### PCA
For transductive learning:\
spearman r: 0.837 ~ 0.848, projection dimension: 70 ~ 350 \
For inductive learning:\
spearman r: 0.840 ~ 0.848, projection dimension: 70 ~ 350  \
Note:\
Good performance and works very fast.

### SVD
For transductive learning:\
spearman r: 0.837 ~ 0.850, projection dimension: 50 ~ 350 \
For inductive learning:\
spearman r: 0.833 ~ 0.849, projection dimension: 50 ~ 350  \
Note:\
Good performance and works fast. Spearnman r reaches 0.83 with smaller projection dimension than PCA and Gaussian random projection.

### UMAP
For transductive learning:\
spearman r: 0.540 ~ 0.550, projection dimension: 12 ~ 92 (results have obvious flunctuation)\
For inductive learning:\
spearman r: 0.533 ~ 0.547, projection dimension: 12 ~ 92 \
Note: \
Poor performance and works obviously slowlier as the dimension grows. The offical website suggests reasonable projection dimension in range(2, 100), thus larger dimensions are not in the consideration and they need quite much time to compute. 

### Gaussian random projection
For transductive learning:\
spearman r: 0.837 ~ 0.850, projection dimension: 70 ~ 360 \
For inductive learning:\
spearman r: 0.837 ~ 0.850, projection dimension: 70 ~ 360 \
Note:\
Good performance and works fast.

### greedy layer-wise autoencoder (2 layers) (keras)
epoch = 70 (losses haven't converge) \
For transductive learning:\
spearman r: 0.823 ~ 0.837, projection dimension: 150 ~ 375 (results have flunctuation)\
For inductive learning:\
spearman r: 0.825 ~ 0.840, projection dimension: 150 ~ 375 \
Note:\
Relatively good performance.

### autoencoder (1 layer) (keras)
epoch = 70 (losses haven't converge) \
For transductive learning:\
spearman r: 0.839 ~ 0.847, projection dimension: 70 ~ 350 (results have flunctuation)\
For inductive learning:\
spearman r: 0.839 ~ 0.847, projection dimension: 70 ~ 350 \
Note:\
Good performance.

### autoencoder (2 layer) (keras)
epoch = 70 (losses haven't converge) \
spearman r: 0.830 ~ 0.845, projection dimension: 50 ~ 350 (results increase and then decline with flunctuation)\
For inductive learning:\
spearman r: 0.830 ~ 0.845, projection dimension: 50 ~ 350 \
Note:\
Relatively good performance.
