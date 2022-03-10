import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from batch_model import RNNVAE
import numpy as np
import pickle
from rdkit import Chem, DataStructs
from batch_data import GeneratorData 
from utils import canonical_smiles
import matplotlib.pyplot as plt
import time
import os


import warnings
warnings.filterwarnings(action='ignore') 

gen_data_path = '/BiO2/DrugDesign/ReLeaSE/data/chembl_smiles_prop.txt'

"""
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
		  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
		  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
"""
# D : <Pad>
#tokens = ['D','<', '>', '#', ')', '(', '+', '-', '/', '.', '1', '3', '2', '5', '4', '7',
#		  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
#		  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']
'''		  
tokens = ['<', '>', '#', ')', '(', '+', '-', '/', '.', '1', '3', '2', '5', '4', '7',
		  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
		  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r','!','^']
'''		  
		  
batch_size = 128  #128
latent_size = 200
unit_size = 512
n_rnn_layer = 3 # 환희님:3 , 수현언니:1
num_eqochs = 500 #500
stack_width = 50
stack_depth = 10
prop=[464.086, 5.549, 92.35]
prop_size = 3
#prop=[5.549]
#prop_size = 1

start = time.time()
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
						 cols_to_read=[0,1,2,3], keep_header=True)
end = time.time()
print("\nbatch_size : ", batch_size)
print("prop_size : ", prop_size)
print("time : ",end - start)
print("all_characters : ",gen_data.all_characters)
print("char2idx : ",gen_data.char2idx)
print("n_characters : ",gen_data.n_characters)

# gen_data.n_characters = 43

layer_type = 'GRU'
lr = 0.001

my_generator = RNNVAE(input_size=gen_data.n_characters,layer_type = layer_type,hidden_size = unit_size,
					  latent_size = latent_size,output_size=gen_data.n_characters,max_sentence_length=10, prop_size=prop_size,
					  batch_size=batch_size, num_layers=n_rnn_layer, lr=lr, has_stack=True, stack_width=stack_width, stack_depth=stack_depth)
					  
my_generator = my_generator.cuda()
my_generator.load_model('/BiO2/DrugDesign/ReLeaSE/stack_cvae/CVAE_20210413_largeData_batch128_prop3/model_179.pth')
print(my_generator)

model_path = '/BiO2/DrugDesign/ReLeaSE/stack_cvae/CVAE_20210413_largeData_batch128_prop3'
if not os.path.exists(model_path):
	os.makedirs(model_path)   


my_generator.fit(gen_data, num_eqochs, prop, start_epoch = 180, save_dir=model_path)


'''
RNNVAE(
  (data_encoder): Embedding(42, 1500)
  (eocoder_rnn): GRU(1500, 1500)
  (decoder_rnn): GRU(1500, 1500)
  (encoder_target_decoder): Linear(in_features=1500, out_features=42, bias=True)
  (decoder_target_decoder): Linear(in_features=1500, out_features=42, bias=True)
  (hidden2mean): Linear(in_features=1500, out_features=1000, bias=True)
  (hidden2log): Linear(in_features=1500, out_features=1000, bias=True)
  (latent2hidden): Linear(in_features=1000, out_features=1500, bias=True)
  (output2vocab): Linear(in_features=1500, out_features=42, bias=True)
  (criterion): MSELoss()
)
'''