# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging

import csv
import pandas as pd
import torch
import argparse
import numpy as np
from tensorflow import keras
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import pickle as pk
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

# Set PATHs
# PATH_TO_SENTEVAL = './SentEval'
# PATH_TO_SENTEVAL = '/Users/fan1218/Desktop/SentEval'
# PATH_TO_DATA = '/Users/fan1218/Desktop/SentEval/data'
PATH_TO_SENTEVAL = '/LOCAL/gaifan/nlp/SentEval'
PATH_TO_DATA = '/LOCAL/gaifan/nlp/SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# import parameters
import itertools
import yaml

with open('parameters.yaml', 'r') as stream:
    try:
        inputdict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

total_list = [inputdict[key] for key in inputdict]
combinations = list(itertools.product(*total_list))
print(combinations)

dim_ls = []
result_na=0
result_na_ls=[]
result_PCA=[]
result_KPCA=[]
result_Grp=[]
result_AE=[]
result_SVD=[]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--method", type=str,
                        choices=['na','SVD', 'PCA', 'KPCA', 'Grp', 'AE'],
                        default='PCA',
                        help="Which dimenionality reduction method to use")
    parser.add_argument("--dim", type=int,
                        choices=np.arange(40, 768, 20).tolist(),
                        default=40,
                        help='What dim to projedct onto')   
    args = parser.parse_args()

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        embeddings = []

        for sent in batch:
            with torch.no_grad():
                sentvec = model.encode(sent, show_progress_bar=False)
            embeddings.append(sentvec)

        embeddings = np.vstack(embeddings)

        if args.method == 'na': 
            return embeddings
        # Apply different methods
        elif args.method == 'PCA':
            pca = pk.load(open(name+"PCA{}.pkl".format(args.dim),'rb'))
            reduced=pca.transform(embeddings)
            return reduced
        elif args.method == 'KPCA':
            kernel_pca = pk.load(open('random'+name+"KPCA{}.pkl".format(args.dim),'rb'))
            reduced = kernel_pca.transform(embeddings)
            return reduced
        elif args.method == 'SVD':
            svd = pk.load(open(name+"SVD{}.pkl".format(args.dim),'rb'))
            reduced = svd.transform(embeddings)
            return reduced
        elif args.method == 'Grp':
            Grp = pk.load(open(name+"Grp{}.pkl".format(args.dim),'rb'))
            reduced = Grp.transform(embeddings)       
            return reduced
        elif args.method == 'AE':
            AE = keras.models.load_model(name+'encoder{}.h5'.format(args.dim))
            reduced = AE.predict(embeddings)     
            return reduced            

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                    'tenacity': 3, 'epoch_size': 2}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    
    for i in combinations:
        args.model_name_or_path = i[0]
        args.method = i[1]
        args.dim = i[2]

        model = SentenceTransformer(args.model_name_or_path)
        name=args.model_name_or_path.split('/')[1]

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['TREC']
        results = se.eval(transfer_tasks)
        print(args.dim)
        print(args.method)
        print(results)
        acc = results['TREC']['acc']
            
        # save the results
        if args.method == 'na': 
            dim = args.dim
            dim_ls.append(dim)
            result_na = acc
            result_na_ls.append(acc)
        elif args.method == 'PCA':           
            result_PCA.append(acc)
        elif args.method == 'KPCA':
            result_KPCA.append(acc)
        elif args.method == 'SVD':
            result_SVD.append(acc)
        elif args.method == 'Grp':
            result_Grp.append(acc)
        elif args.method == 'AE':
            result_AE.append(acc)

    # save results in csv
    dict = {'KPCA': result_KPCA}
    df = pd.DataFrame(data=dict)
    df.to_csv('TREC_trans_KPCA.csv')

    # # save results in csv
    # dict = {'dim': dim_ls, 
    #         'na': result_na_ls, 
    #         'PCA': result_PCA,
    #         'KPCA': result_KPCA,
    #         'Grp': result_Grp,
    #         'AE': result_AE,
    #         'SVD': result_SVD}
    # df = pd.DataFrame(data=dict)
    # df.to_csv('random_TREC_trans_results_paraphrase-xlm-r-multilingual-v1.csv')

    # # plot 
    # plt.axhline(y=result_na, linewidth=1, linestyle='--', label="without reduction")
    # plt.plot(dim_ls, result_PCA, label = "PCA")
    # plt.plot(dim_ls, result_KPCA, label = "kernel PCA")
    # plt.plot(dim_ls, result_Grp, label = "Gaussian random projection")
    # plt.plot(dim_ls, result_AE, label = "autoencoder with 1 layer")
    # plt.plot(dim_ls, result_SVD, label = "SVD")
    # plt.xlabel('dimensionality')
    # plt.ylabel('acc')
    # plt.legend()
    # plt.savefig('./random_TREC_trans_paraphrase-xlm-r-multilingual-v1.jpg')

if __name__ == "__main__":
    main()
