import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

import seaborn as sns
import numpy as np
import pickle
from rdkit import Chem, DataStructs
from stack_cvae.batch_data import GeneratorData 
from utils import canonical_smiles
import matplotlib.pyplot as plt
import time
import os
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import matplotlib.pyplot as plt
import warnings
from stack_cvae.batch_model import stack_CVAE
warnings.filterwarnings(action='ignore')

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    ## optional
    parser.add_argument('-c', metavar='condition', type=list, default=[464.086, 5.549, 92.35], help="Properties for Generate, [MW,logP,TPSA]")
    parser.add_argument('-d', metavar='Drug', type=str, default='Sorafenib', help="A Drug Name for Generate")
    parser.add_argument('-s', metavar='sample',type=int, default=1000, help="Number of samples want to generate sample.")
    parser.add_argument('-p', metavar='path',type=str, default='model', help="model's path want to generate sample.")
    parser.add_argument('-n', metavar='models name',type=list, default=['model_0','model_100','model_200','model_300','model_400','model_500'], help="model's name want to generate sample.")
    parser.add_argument('-ss', metavar='save sample',type=bool, default=True, help="save samples.")
    parser.add_argument('-sf', metavar='save sample graph figure',type=bool, default=False, help="save samples graph.")
    return parser.parse_args()
def main():
    args = get_args()
    
    gen_data_path = '../data/chembl_smiles_prop.txt'
    batch_size = 1
    latent_size = 200
    unit_size = 512
    n_rnn_layer = 3
    stack_width = 50
    stack_depth = 10
    layer_type = 'GRU'
    lr = 0.0001

    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                             cols_to_read=[0,1,2,3], keep_header=True,dif_len=True)

    my_generator = RNNVAE(input_size=gen_data.n_characters,layer_type = layer_type,hidden_size = unit_size,
                          latent_size = latent_size,output_size=gen_data.n_characters,max_sentence_length=10, prop_size=3,
                          batch_size=batch_size, num_layers=n_rnn_layer, lr=lr, has_stack=True, stack_width=stack_width, stack_depth=stack_depth)
                      
    my_generator = my_generator.cuda()

    path = args.p
    prop = args.c
    names = args.n

    pre_dup = []
    valid = []
    valid_o=[]
    overap = []
    dup = []
    save_sample = args.ss
    save_fig = args.sf
    for i in names:
        my_generator.load_model("./{}/{}.pth".format(path,i))
        smiles=[]
        for _ in range(1000):
            generated, latent= my_generator.evaluate(data=gen_data, prop=prop, prime_str = '<', predict_len=120)
            tmp = generated[1:-1]
            smiles.append(tmp)
        print ('number of trial : ', len(smiles))
        smiles = [s.split('>')[0] for s in smiles] 
        if save_sample:
            with open('./{}/graph/result{}.txt'.format(path,i), 'w') as w:
                w.write('SMILES\n')
                for smile in smiles:
                    try:
                        w.write('%s\n' %(smile))
                    except:
                        continue
        ms = [Chem.MolFromSmiles(s) for s in smiles]
        ms = [m for m in ms if m is not None]
        pre_dup.append(len(ms))
        print('epoch:',i)
        print ('number of valid smiles : ', len(ms))
        dup_num = len(ms)
        ms = list(set(Chem.MolToSmiles(m) for m in ms))
        dup_num -= len(ms)
        valid.append(len(ms))
        print('number of smiles that duplicated ones : ', dup_num)
        train_smiles = gen_data.inputs
        train_smiles = [t[1:].split('E')[0] for t in train_smiles]
        duplication = list(set(train_smiles).intersection(ms))
        dup_num += len(duplication)
        print('number of smiles that Overlapping with training data : ', len(duplication))
        overap.append(len(duplication))
        ms = list(set(ms).difference(set(duplication)))
        print('number of valid smiles (after remove duplicated ones): ', len(ms))
        
        valid_o.append(len(ms))
        dup.append(dup_num)
        
    plt.figure()
    plt.plot(epochs,pre_dup,color='blue',label='validated SMILES')
    plt.plot(epochs,valid,color='orange',label='validated and unique SMILES')
    plt.plot(epochs,valid_o,color='gray',label='validated and unique SMILES not overlapped with training data')
    plt.legend()
    plt.title(args.d)
    if save_fig:
        plt.show()
        plt.savefig("{}/graph/rate.png".format(path))
        plt.clf()
    print('epochs:',)
    print('Not remove duplicated ones: ',pre_dup)
    print('duplicated ones: ', dup)
    print('After remove duplicated ones: ',valid)
    print(valid_o)
    
if __name__=="__main__":
    main()
