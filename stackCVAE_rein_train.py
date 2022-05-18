import sys
import os
from os import path
sys.path.append(path.abspath('./stack_cvae'))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
import pickle
from rdkit import Chem, DataStructs
from stack_cvae.batch_data import GeneratorData 
from utils import canonical_smiles
import matplotlib.pyplot as plt
from stackCVAE_BA_RA_reinforcement import BA_Reinforcement_3 #
from tqdm import tqdm, trange
from BA_reward_CVAE import PIN, get_pSeqDict, get_disease_target_genes, calc_biological_reward_v2
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from DeepPurpose import DTI as models 
from DeepPurpose import utils
from stack_cvae.batch_model import stack_CVAE #
import os
from RAscore import RAscore_XGB #For XGB based models
use_cuda = torch.cuda.is_available()

import warnings
warnings.filterwarnings(action='ignore') 

drug = "sora"

drug_BA_dict = {"sora":6.48, "suni" : 8.93, "dasa" : 7.81}

gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
ppi_path = "./data_ppi/BIOGRID-ORGANISM-Homo_sapiens-3.5.176.tab2.txt"
mt_path = "./data_ppi/HUMAN_9606_idmapping.dat"
pseq_path = "./data_ppi/uniprot_sprot.fasta"
disGene_path = "./data_ppi/cancer_gene_census.csv"


target_plist_path = "./data_ppi/groupA_protein_list.txt"
target_pseq_dict_path = "./data_ppi/groupA.pickle"

groupD_plist_path = "./data_ppi/groupD_protein_list.txt" # 
groupD_pseq_dict_path = "./data_ppi/groupD.pickle" # group D : 붙지 않음

def estimate_and_update(generator, j,n_to_generate):
    generated = []
    canonical = []
    invalid_smiles = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data,prop=[464.825, 3.8, 92.35], predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    
    pbar = tqdm(range(len(unique_smiles)))
    for i in pbar:
        sm = unique_smiles[i]
        try:
            sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=False))
            if len(sm) == 0:
                invalid_smiles.append(sm)
            else:
                canonical.append(sm)
        except:
            invalid_smiles.append(sm)
    return canonical


gen_data_path = './data/chembl_smiles_prop.txt'
batch_size = 1
latent_size = 200
unit_size = 512
n_rnn_layer = 3
stack_width = 50
stack_depth = 10
layer_type = 'GRU'
lr = 0.0001


gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', cols_to_read=[0,1,2,3], keep_header=True, dif_len = True)
my_generator = stack_CVAE(input_size=gen_data.n_characters,layer_type = layer_type,hidden_size = unit_size,
                      latent_size = latent_size,output_size=gen_data.n_characters,max_sentence_length=10, prop_size=3,
                      batch_size=batch_size, num_layers=n_rnn_layer, lr=lr, has_stack=True, stack_width=stack_width, stack_depth=stack_depth)
                      
my_generator = my_generator.cuda()
print(my_generator)

### pretrained generate model path
generator_model = './model/pretrain_model.pth'

my_generator.load_model(generator_model)	

print(my_generator)


print("my generator load success")

# Setting up some parameters for the experiment
n_to_generate = 200
n_policy =10		#10
n_iterations = 501   #1000


def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma


def get_reward_BA_continued(smiles, target_pSeq_dict,groupD_pSeq_dict, ppi_graph, dis_Prot_list,
                  DeepPurpose_model,RAscore_model, minPSeqLen=500, invalid_reward=0.0):
    print("smiles: " + str(smiles))

    target_reward = 0.0
    groupD_reward = 0.0
    RAscore_reward = 0.0
    g_reward = 0.0
    ra_reward = 0.0
    target_reward = calc_biological_reward_v2(smiles, target_pSeq_dict, DeepPurpose_model)
    groupD_reward = calc_biological_reward_v2(smiles, groupD_pSeq_dict, DeepPurpose_model)
    RAscore_reward = RAscore_model.predict(smiles)
    print('target_reward : ', target_reward)
    print('groupD_reward : ', groupD_reward)
    print('RAscore_reward : ', RAscore_reward)
    
    
    if target_reward > 5.0:
        final_biol_reward = (target_reward - 4.0)**2 + 1.0
    else:
        final_biol_reward = 1.0
    print('target_reward : ', final_biol_reward)  
    
    if groupD_reward <= 5.5:
        g_reward = 6.0
    else:
        g_reward = 1.0  
    final_biol_reward += g_reward
    print('groupD_biol_reward : ', g_reward)
    
    ra_reward = 1.0 + RAscore_reward*5
    final_biol_reward += ra_reward
    print('RAscore_biol_reward : ', ra_reward)
    print('final_biol_reward (target) : ', final_biol_reward)

    return final_biol_reward   
        
   
    
pin = PIN(ppi_path, mt_path)
ppi_graph = pin.constructPPI_graph()
GS_Prot_dict = pin.construct_GSprot_dict()

maxProtSeqLength = 500
target_pSeq_dict = get_pSeqDict(target_plist_path, pseq_path, target_pseq_dict_path, maxProtSeqLength)
groupD_pseq_dict = get_pSeqDict(groupD_plist_path, pseq_path, groupD_pseq_dict_path, maxProtSeqLength)
dis_Prot_list, geneSet = get_disease_target_genes(disGene_path, GS_Prot_dict)
DeepPurpose_model = models.model_pretrained(model = 'Daylight_AAC_DAVIS')   
RAscore_model = RAscore_XGB.RAScorerXGB()

print("target_pSeq_dict: " + str(len(target_pSeq_dict.keys())))
print("groupD_pseq_dict: " + str(len(groupD_pseq_dict.keys())))
print("dis_Prot_list: " + str(dis_Prot_list))
print("geneSet: " + str(geneSet))

#reinforcement model

RL_BA = BA_Reinforcement_3(my_generator, get_reward_BA_continued, 
                           target_pSeq_dict,groupD_pseq_dict, ppi_graph, dis_Prot_list,
                           DeepPurpose_model, RAscore_model,minPSeqLen=maxProtSeqLength)
                           
print("my reinforcement load success")   
rewards = []
rl_losses = []

path = 'sorafenib'

paths = {}
paths['Average_reward'] = './{}_result/plot/Average_reward'.format(path)
paths['Loss'] = './{}_result/plot/Loss'.format(path)
paths['smiles'] = './{}_result/smiles'.format(path)
paths['checkpoints'] = './{}_result/checkpoints'.format(path)
paths['after_saveModel'] = './{}_result/after_saveModel'.format(path)
paths['MW'] = './{}_result/graph/MW'.format(path)
paths['TPSA'] = './{}_result/graph/TPSA'.format(path)
paths['logP'] = './{}_result/graph/logP'.format(path)
paths['RAscore'] = './{}_result/graph/RAscore'.format(path)
paths['bindingA'] = './{}_result/graph/bindingA'.format(path)
paths['bindingD'] = './{}_result/graph/bindingD'.format(path)

for p in paths:
    print(p)
    if not os.path.exists(paths[p]):
        print(p)
        os.makedirs(paths[p]) 

for i in range(0,n_iterations):
    for j in trange(n_policy, desc='Policy gradient...'):
        print(j)
        #Call the policy_gradient function to train
        cur_reward, cur_loss = RL_BA.policy_gradient(gen_data,std_smiles=True) # our modification : binding affinity
        rewards.append(simple_moving_average(rewards, cur_reward)) 
        rl_losses.append(simple_moving_average(rl_losses, cur_loss))
    
    plt.plot(rewards)
    plt.xlabel('Training iteration')
    plt.ylabel('Average reward')
    plt.savefig(paths['Average_reward']+'/{}_iterations_AR.png'.format(i))
    plt.clf()
    plt.plot(rl_losses)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.savefig(paths['Loss']+'/{}_iterations_Loss.png'.format(path,i))
    plt.clf()
   
    smiles_cur = estimate_and_update(RL_BA.generator, i, n_to_generate)
    print('{} Sample trajectories:'.format(i))
    for sm in smiles_cur:
        print(sm)
    with open(paths['smiles']+"/{}_iteration_after_reinforce_smiles.pickle".format(i),"wb") as f:
        pickle.dump(smiles_cur,f)
    if i % 20 == 0:
        model_path = paths['checkpoints']+"/checkpoint_lr{}_{}epoch".format(path,lr,i)
        my_generator.save_model(model_path)

my_generator.save_model(model_path)


smiles_cur = estimate_and_update(RL_BA.generator, i, n_to_generate)

with open(paths['after_saveModel']+"/smiles.pickle".format(path),"wb") as f:
    pickle.dump(smiles_cur,f)

