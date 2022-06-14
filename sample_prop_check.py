from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DeepPurpose import DTI as models 
from DeepPurpose import utils
import pickle
from RAscore.RAscore import RAscore_XGB #For XGB based models
import argparse
import warnings
warnings.filterwarnings(action='ignore')

def get_args():
    parser = argparse.ArgumentParser()
    ## optional
    parser.add_argument('-c', metavar='condition', type=list, default=[464.086, 5.549, 92.35], help="Properties for Generate, [MW,logP,TPSA]")
    parser.add_argument('-d', metavar='Drug', type=str, default='Sorafenib', help="A Drug Name for Generate")
    parser.add_argument('-s', metavar='sample',type=int, default=1000, help="Number of samples want to generate sample.")
    parser.add_argument('-p', metavar='path',type=str, default='sorafenib_result', help="model's path want to generate sample.")
    parser.add_argument('-n', metavar='models name',type=list, default=['model_100','model_200','model_300','model_400','model_500'], help="model's name want to generate sample.")
    parser.add_argument('-ss', metavar='save sample',type=bool, default=True, help="save samples.")
    parser.add_argument('-sf', metavar='save sample graph figure',type=bool, default=False, help="save samples graph.")
    return parser.parse_args()
    
def plot_mw(mw_result, path, name, mw):
    MW = mw 
    cnt = 0
    a = 40

    sns.distplot(sorted(mw_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(MW,color='red')
    plt.axvline(MW-a,color='black', linestyle='--')
    plt.axvline(MW+a,color='black', linestyle='--')

    for mw in mw_result:
        if mw>(MW-a) and mw<(MW+a):
            cnt += 1
    s = (cnt/len(mw_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_Mol_WE.png")
    plt.clf()
    
def plot_logP(lo_result, path, name,logp):
    logP = logp
    cnt = 0
    a = 0.5

    sns.distplot(sorted(lo_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(logP,color='red')
    plt.axvline(logP-a,color='black', linestyle='--')
    plt.axvline(logP+a,color='black', linestyle='--')

    for lo in lo_result:
        if lo>(logP-a) and lo<(logP+a):
            cnt += 1
    s = (cnt/len(lo_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_clogP.png")
    plt.clf()
                
def plot_TPSA(tp_result, path, name,tpsa):
    TPSA = tpsa
    cnt = 0
    a = 10

    sns.distplot(sorted(tp_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(TPSA,color='red')
    plt.axvline(TPSA-a,color='black', linestyle='--')
    plt.axvline(TPSA+a,color='black', linestyle='--')

    for tp in tp_result:
        if tp>(TPSA-a) and tp<(TPSA+a):
            cnt += 1
    s = (cnt/len(tp_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_TPSA.png")
    plt.clf()
    
def plot_binding_affinity_A(tp_result, path, name):
    binding = 6.48 #sora
    cnt = 0
    a = 0.2
    sns.distplot(sorted(tp_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(binding,color='red')
    
    for tp in tp_result:
        if tp>(binding-a) and tp<(binding+a):
            cnt += 1
    s = (cnt/len(tp_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_bindingA.png")
    plt.clf()
    
def plot_binding_affinity_D(tp_result, path, name):
    binding = 5.2
    cnt = 0
    
    sns.distplot(sorted(tp_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(binding,color='red')
    
    for tp in tp_result:
        if tp<binding:
            cnt += 1
    s = (cnt/len(tp_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_bindingD.png")
    plt.clf()
    
def plot_RAscore(tp_result, path, name):
    binding = 0.9
    cnt = 0
    
    sns.distplot(sorted(tp_result),hist=False,color='blue')
    plt.legend()
    plt.axvline(binding,color='red')
    
    for tp in tp_result:
        if tp>binding:
            cnt += 1
    s = (cnt/len(tp_result))*100
    s = "\n% in range : {:.2f}%".format(s)
    plt.title(s)
    plt.savefig(path+name+"_RAscore.png")
    plt.clf()
def main():
    args= get_args()  
    names = args.n
    path = args.p
    prop = args.c
    train_data = []
    with open('/BiO2/DrugDesign/ReLeaSE/data/chembl_22_clean_1576904_sorted_std_final.smi', 'r') as f:
        for l in f:
            tmp = l.split('\t')
            train_data.append(tmp[0])
    target_pseq_dict_path = "./data_ppi/groupA.pickle" #sora
    with open(target_pseq_dict_path, 'rb') as f:
        data = pickle.load(f)
        groupA_pseq = list(data.values())
    grouD_pseq_dict_path = "/BiO2/DrugDesign/ReLeaSE/data_ppi/groupD.pickle" 
    with open(grouD_pseq_dict_path, 'rb') as f:
        data = pickle.load(f)
        groupD_pseq = list(data.values())
           
    BA_model = models.model_pretrained(model = 'Daylight_AAC_DAVIS')
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    successes = []
    only_successes = []
    unique_successes = []

    location = "./{}/graph/".format(path)
    for name in names:
        print(name)
        success = []
        invalid_mol = 0
        
        path = location+"result_{}.txt".format(name)
        with open(path, 'r') as f:
            f.readline() # 첫 줄 날리기
            for smiles in f:
                smiles = smiles.strip()
                mol = Chem.MolFromSmiles(smiles)
                try:
                    Chem.GetSSSR(mol)
                    success.append(smiles)
                except:
                    invalid_mol += 1
        
        print(success[:3])
        print('success : ', len(success))
        print('fail : ', invalid_mol)

        only_success = set(success)
        print('중복 제거 : ', len(only_success))

        unique_success = [x for x in only_success if x not in train_data]
        print('중복 제거+훈련데이터 제거 : ', len(unique_success))
        print('\n')
        successes.append(len(success))
        only_successes.append(len(only_success))
        unique_successes.append(len(unique_success))
        
        bindA_result, bindD_result, RAscore_result, SAscore_result, mw_result, lo_result,tp_result = [], [], [], [], [], [], []
        
        
        for SMILES in unique_success:
            drug = []
            for _ in range(len(groupA_pseq)):
                drug.append(SMILES)		
            df_data = utils.data_process_repurpose_virtual_screening(drug, groupA_pseq, BA_model.drug_encoding, BA_model.target_encoding, 'virtual screening')
            
            BAs = BA_model.predict(df_data) 
            bindA = np.mean(BAs)

            drug = []
            for _ in range(len(groupD_pseq)):
                drug.append(SMILES)		
            df_data = utils.data_process_repurpose_virtual_screening(drug, groupD_pseq, BA_model.drug_encoding, BA_model.target_encoding, 'virtual screening')
            
            BAs = BA_model.predict(df_data) 
            bindD = np.mean(BAs)

            RAscore = xgb_scorer.predict(SMILES)
            
            mol = Chem.MolFromSmiles(SMILES)
            mw = MolDescriptors.CalcExactMolWt(mol)
            clogp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            bindA_result.append(bindA)
            bindD_result.append(bindD)
            RAscore_result.append(RAscore)
            mw_result.append(mw)
            lo_result.append(clogp)
            tp_result.append(tpsa)

        
        data_df = pd.DataFrame(unique_success, columns=['smiles'])
        data_df['Weights'] = mw_result
        data_df['LogP'] = lo_result
        data_df['TPSA'] = tp_result
        data_df['bindA'] = bindA_result
        data_df['bindD'] = bindD_result
        data_df['RAscore'] = RAscore_result
        data_df.to_csv(location+name+".csv")
        plot_binding_affinity_A(bindA_result, location+'/bindingA/', name)
        plot_binding_affinity_D(bindD_result, location+'/bindingD/', name)
        plot_mw(mw_result, location+'/MW/', name,prop[0])
        plot_logP(lo_result, location+'/logP/', name,prop[1])
        plot_TPSA(tp_result, location+'/TPSA/', name,prop[2])
        plot_RAscore(RAscore_result, location+'/RAscore/', name)
    print(successes)
    print(only_successes)
    print(unique_successes)