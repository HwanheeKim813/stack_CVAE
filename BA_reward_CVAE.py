import argparse
import pandas as pd
from math import isnan
import numpy as np
import itertools
import requests
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

## from ligand_prediction
import torch
# from BA_prediction import predict_class, predict_affinity
import pickle

## for progress bar
import time
from tqdm import tqdm, trange
import multiprocessing as mp

## for DeepPurpose model
from DeepPurpose import DTI as models 
from DeepPurpose import utils

## for HiddenPrints function
import os, sys

pSeq_dict = {} ## cache to reduce API call

## protein-protein interaction network data
class PIN():
    def __init__(self, PPI_path, ID_mapping_table_path):
        """
        Constructor for the Protein-Protein Interaction object.

        Parameters
        ----------
        PPI_path: path which contains PPI file
            ex) "./data_ppi/BIOGRID-ORGANISM-Homo_sapiens-3.5.176.tab2.txt"
        ID_mapping_table_path: path which contains ID mapping table (Biogrid ID - uniprot ID)
            ex) "./data_ppi/HUMAN_9606_idmapping.dat"

        Returns
        -------
        object of dictionary type PPI (Protein, interacting Proteins)
        """
        self.PPI_path = PPI_path
        self.ID_mapping_table_path = ID_mapping_table_path

    ## load protein id mapping table
    def load_IDmapping_file(self):
        mt_df = pd.read_csv(self.ID_mapping_table_path, delimiter="\t", header=None)
        mt_df.columns = ["uniprotKB", "mtype", "other"]

        ## select biogrid - uniprotKB mapping
        sel_mt_df = mt_df.loc[mt_df['mtype'] == 'BioGrid']
        sel_mt_df['other'] = sel_mt_df['other'].astype(str).astype(int)  ## dtype from obj to int
        mt_dict = dict(zip(sel_mt_df.other, sel_mt_df.uniprotKB))

        return mt_dict

    ## load protein-protein interaction database from BioGrid
    def load_PPI_file(self):
        ppi_df = pd.read_csv(self.PPI_path, delimiter="\t")
        col_list = ["BioGRID ID Interactor A", "BioGRID ID Interactor B", "Official Symbol Interactor A", "Official Symbol Interactor B"]
        ppi_df = ppi_df[col_list]

        return ppi_df


    ## construct gene symbol - proteins dict
    def construct_GSprot_dict(self):
        mt_dict = self.load_IDmapping_file()
        ppi_df = self.load_PPI_file()

        ppi_df['uniprotKB_A'] = ppi_df['BioGRID ID Interactor A'].map(mt_dict)
        ppi_df['uniprotKB_B'] = ppi_df['BioGRID ID Interactor B'].map(mt_dict)

        ## uniprotKB to gene symbol
        GS_Prot_dict = {}
        for index, row in ppi_df.iterrows():
            up_A = row['uniprotKB_A']
            up_B = row['uniprotKB_B']
            gs_A = row['Official Symbol Interactor A']
            gs_B = row['Official Symbol Interactor B']

            gs_A = str(gs_A).upper() # capitalize
            gs_B = str(gs_B).upper() # capitalize

            if gs_A in GS_Prot_dict.items():
                prot_set = GS_Prot_dict[gs_A]
                prot_set.add(up_A)
                GS_Prot_dict[gs_A] = prot_set

            else:
                prot_set = set()
                prot_set.add(up_A)
                GS_Prot_dict[gs_A] = prot_set

            if gs_B in GS_Prot_dict.items():
                prot_set = GS_Prot_dict[gs_B]
                prot_set.add(up_B)
                GS_Prot_dict[gs_B] = prot_set

            else:
                prot_set = set()
                prot_set.add(up_B)
                GS_Prot_dict[gs_B] = prot_set

        print("# unique protein A: " + str(ppi_df['uniprotKB_A'].nunique()))
        print("# unique protein B: " + str(ppi_df['uniprotKB_B'].nunique()))
        print("# unique gene A: " + str(ppi_df['Official Symbol Interactor A'].nunique()))
        print("# unique gene B: " + str(ppi_df['Official Symbol Interactor B'].nunique()))

        return GS_Prot_dict


    def constructPPI_graph(self):
        mt_dict = self.load_IDmapping_file()  ## BioGRID ID Interactor A --> unitprot ID
        ppi_df = self.load_PPI_file()

        ppi_df['uniprotKB_A'] = ppi_df['BioGRID ID Interactor A'].map(mt_dict)
        ppi_df['uniprotKB_B'] = ppi_df['BioGRID ID Interactor B'].map(mt_dict)

        ## PPI graph
        ppi_graph = nx.from_pandas_edgelist(ppi_df, 'uniprotKB_A', 'uniprotKB_B')

        return ppi_graph




    def constructPPI(self):
        mt_dict = self.load_IDmapping_file() ## BioGRID ID Interactor A --> unitprot ID
        ppi_df = self.load_PPI_file()

        ppi_df['uniprotKB_A'] = ppi_df['BioGRID ID Interactor A'].map(mt_dict)
        ppi_df['uniprotKB_B'] = ppi_df['BioGRID ID Interactor B'].map(mt_dict)

        ## uniprotKB to gene symbol
        uniprotKB_GeneSymbol_dict = {}
        for index, row in ppi_df.iterrows():
            up_A = row['uniprotKB_A']
            up_B = row['uniprotKB_B']
            gs_A = row['Official Symbol Interactor A']
            gs_B = row['Official Symbol Interactor B']
            uniprotKB_GeneSymbol_dict[up_A] = gs_A
            uniprotKB_GeneSymbol_dict[up_B] = gs_B

        #print("uniprotKB_GeneSymbol_dict: " + str(len(uniprotKB_GeneSymbol_dict.keys())))

        final_ppi = ppi_df[["uniprotKB_A", "uniprotKB_B"]]

        final_ppi_dict = {}
        for index, row in final_ppi.iterrows():
            p_A = row['uniprotKB_A']
            p_B = row['uniprotKB_B']

            if p_A in final_ppi_dict.keys():
                if final_ppi_dict[p_A]:
                    pSet = final_ppi_dict[p_A]
                    pSet.add(p_B)
                else:
                    pSet = set()
                    pSet.add(p_B)

                # pSet = {x for x in pSet if str(x) != 'nan'}
                final_ppi_dict[p_A] = pSet

            else:
                pSet = set()
                pSet.add(p_B)

                # pSet = {x for x in pSet if str(x) != 'nan'}
                final_ppi_dict[p_A] = pSet

            if p_B in final_ppi_dict.keys():
                if final_ppi_dict[p_B]:
                    pSet = final_ppi_dict[p_B]
                    pSet.add(p_A)
                else:
                    pSet = set()
                    pSet.add(p_A)

                # pSet = {x for x in pSet if str(x) != 'nan'}
                final_ppi_dict[p_B] = pSet

            else:
                pSet = set()
                pSet.add(p_A)

                # pSet = {x for x in pSet if str(x) != 'nan'}
                final_ppi_dict[p_B] = pSet

        for p, pSet in final_ppi_dict.items():
            final_ppi_dict[p] = {x for x in pSet if str(x) != 'nan'}

        print("# interactions in final PPI: " + str(final_ppi.shape))
        print("# proteins in final PPI : " + str(len(final_ppi_dict.keys())))

        # remove 'nan' key
        final_ppi_dict = dict((k, v) for k, v in final_ppi_dict.items() if not (type(k) == float and isnan(k)))

        return final_ppi_dict


## search protein sequence via restful api of www.uniprot.org
def getProteinSeq(uniprotID):
    # uniprotID = "Q6A162"
    url = "https://www.uniprot.org/uniprot/" + str(uniprotID) + ".fasta"
    print("url: " + str(url))
    ret = requests.get(url)

    ##
    if ret.status_code == 200:
        ret_text = ret.text
        header = str(ret_text).split('\n')[0]
        protein_sequence_with_N = ret_text.replace(header, "")
        protein_sequence = protein_sequence_with_N.replace('\n', "")
        protein_sequence_length = len(protein_sequence)

    ## no return value
    else:
        protein_sequence = ''
        protein_sequence_length = 0

    return protein_sequence, protein_sequence_length



## load protein sequence from file
def loadProteinSeq(pseq_path, proteinList, length):
    file = open(pseq_path,"r")
    tmp = file.readlines()
    for i in range(len(tmp)):
        tmp[i] = tmp[i].strip()
        if "|" in tmp[i]:
            protein = tmp[i].split("|")[1]
            tmp_seq = ''
        else:
            if protein in proteinList:
                tmp_seq += tmp[i]
                pSeq_dict[protein] = tmp_seq
    dict_key = list(pSeq_dict.keys())
    for i in range(len(pSeq_dict.keys())):
        if len(pSeq_dict[dict_key[i]]) > length:
            del pSeq_dict[dict_key[i]]
    return pSeq_dict

def calc_biological_reward_directTargets(smiles, proteins, ppi_dict, clf_model, reg_model, minPSeqLen=1000):
    pSeq_SMILES_list = []

    all_PseqDict = {}
    for p in proteins:
        if p in pSeq_dict.keys():
            pSeq = pSeq_dict[p]
        else:
            ret_pSeq = getProteinSeq(p)
            pSeq = ret_pSeq[0]
            pSeq_dict[p] = pSeq

        pSeqLen = len(pSeq)
        print("p: " + p + "\t" + "seq len: " + str(pSeqLen))
        if pSeqLen <= minPSeqLen and pSeqLen > 0:
            all_PseqDict[p] = pSeq

    pCnt = 0
    for p, seq in all_PseqDict.items():
        pSeq_SMILES = []
        pSeq_SMILES.append(seq)
        pSeq_SMILES.append(smiles)
        pSeq_SMILES_list.append(pSeq_SMILES)
        pCnt += 1

    if len(pSeq_SMILES_list) > 0:
        print(pSeq_SMILES_list)

        outputs, clf_predictions = clf_model.predict_(pSeq_SMILES_list)
        reg_predictions = reg_model.predict_(pSeq_SMILES_list)

        clf_predictions_np = clf_predictions.cpu().numpy()
        reg_predictions_np = reg_predictions.cpu().numpy()

        ## Definition of biological reward
        indice_clf_1 = np.where(clf_predictions_np == 1)
        clf_1_cnt = len(indice_clf_1[0])
        print("clf_1_cnt: " + str(indice_clf_1[0]))
        avg_reg_nPs = np.mean(reg_predictions_np[indice_clf_1[0]])

        if np.isnan(avg_reg_nPs):
            b_reward = 0.0
        else:
            b_reward = avg_reg_nPs + clf_1_cnt

    else:
        b_reward = 0.0

    ## add minimum reward value
    if b_reward < 0:
        b_reward_final = 1.0
    else:
        b_reward_final = b_reward + 1.0

    print("Final biological reward: " + str(b_reward) + "\t ==> " + str(b_reward_final))
    return b_reward_final



## calculate binding affinity based reward
def calc_biological_reward(smiles, proteinID, final_ppi_dict, clf_model, reg_model, minPSeqLen=1000):
    pSeq_SMILES_list = []
    alpha = 0.5

    if proteinID in final_ppi_dict.keys():
        nPset = final_ppi_dict[proteinID]
        plist = []
        plist.append(proteinID)
        plist.extend(nPset)

        all_PseqDict = {}
        for p in plist:
            if p in pSeq_dict.keys():
                pSeq = pSeq_dict[p]
            else:
                ret_pSeq = getProteinSeq(p)
                pSeq = ret_pSeq[0]
                pSeq_dict[p] = pSeq

            pSeqLen = len(pSeq)
            print("p: " + p + "\t" + "seq len: " + str(pSeqLen))

            if p == proteinID:
                if pSeqLen > minPSeqLen or pSeqLen == 0:
                    break

            if pSeqLen <= minPSeqLen and pSeqLen > 0:
                all_PseqDict[p] = pSeq

        idx_proteinID = 0
        pCnt = 0
        for p, seq in all_PseqDict.items():
            pSeq_SMILES = []
            pSeq_SMILES.append(seq)
            pSeq_SMILES.append(smiles)
            pSeq_SMILES_list.append(pSeq_SMILES)

            if p == proteinID:
                idx_proteinID = pCnt

            pCnt += 1


    if len(pSeq_SMILES_list) > 0:

        outputs, clf_predictions = clf_model.predict_(pSeq_SMILES_list)
        reg_predictions = reg_model.predict_(pSeq_SMILES_list)

        clf_predictions_np = clf_predictions.cpu().numpy()
        reg_predictions_np = reg_predictions.cpu().numpy()


        indice_clf_1 = np.where(clf_predictions_np == 1)
        avg_reg_nPs = np.mean(reg_predictions_np[indice_clf_1[0]])
        if np.isnan(avg_reg_nPs) or avg_reg_nPs == None:
            avg_reg_nPs = 0

        ## if classification is 1, calculate b_reward
        if clf_predictions_np[idx_proteinID] == 1:
            b_reward = alpha*reg_predictions_np[idx_proteinID] + (1.0-alpha)*avg_reg_nPs
        else:
            b_reward = (1.0-alpha)*avg_reg_nPs

    else:
        b_reward = 0

    ## add minimum reward value
    if b_reward < 0:
        b_reward_final = 1.0
    else:
        b_reward_final = b_reward + 1.0

    print("Final biological reward: " + str(b_reward) + "\t ==> " + str(b_reward_final))
    return b_reward_final



def calc_biological_reward_for_DTG(smiles, pSeq_dict, ppi_graph, drug_Prot_dict_alreadyknown, clf_model, reg_model):
    pSeq_SMILES_list = []
    numSelTopProt = 10

    plist = []
    for p, pSeq in pSeq_dict.items():
        plist.append(p)
        pSeq_SMILES = []
        pSeq_SMILES.append(pSeq)
        pSeq_SMILES.append(smiles)
        pSeq_SMILES_list.append(pSeq_SMILES)


    topProtList = []
    if len(pSeq_SMILES_list) > 0:
        BA_dict = calc_BA(clf_model, reg_model,pSeq_SMILES_list,plist)

        ## sort and select top N
        sorted_BA_dict = Counter(BA_dict)
        for k, v in sorted_BA_dict.most_common(numSelTopProt):
            topProtList.append(k)

    ## calculate distance between top Proteins & disease targets
    shortest_path_dist_dict_DT = {}
    for topP in topProtList:
        for drug, DTPlist in drug_Prot_dict_alreadyknown.items():
            for drugP in DTPlist:
                key = str(topP + " -> " + drugP)
                if topP in ppi_graph.nodes() and drugP in ppi_graph.nodes():
                    shortest_path_dist = nx.shortest_path_length(ppi_graph, source=topP, target=drugP)
                    shortest_path_dist_dict_DT[key] = shortest_path_dist

    mean_SP_DT = np.array([int(shortest_path_dist_dict_DT[k]) for k in shortest_path_dist_dict_DT]).mean()

    b_reward_DT = float(float(100) / mean_SP_DT)
    print("b_reward for drugs (sorafenib & regorafenib): " + str(b_reward_DT))

    return b_reward_DT


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
        


def calc_biological_reward_v2(smiles, pSeq_dict, BA_model):

    pSeq_SMILES_list = []
    plist = []
    
    for p, pSeq in pSeq_dict.items():
        plist.append(p)
        pSeq_SMILES = []
        pSeq_SMILES.append(pSeq)
        pSeq_SMILES.append(smiles)
        pSeq_SMILES_list.append(pSeq_SMILES)

    if len(pSeq_SMILES_list) > 0:
        BA_dict = calc_BA(BA_model,pSeq_SMILES_list,plist)

    mean_SP = np.array([float(BA_dict[k]) for k in BA_dict]).mean()
    b_reward = float(mean_SP)
    
    return b_reward



def calc_BA(BA_model, chunk, plist_chunk):
    with HiddenPrints():
        BA_dict = {}
        for pSeq_SMILES, proteinID in zip(chunk, plist_chunk):
            pSeq = pSeq_SMILES[0]
            SMILES = pSeq_SMILES[1]
            pSeq = [pSeq]
            SMILES = [SMILES]
            df_data = utils.data_process_repurpose_virtual_screening(SMILES, pSeq, BA_model.drug_encoding, BA_model.target_encoding, 'virtual screening')
            ba = BA_model.predict(df_data) 
            BA_dict[proteinID] = ba[0]
    
    return BA_dict



def calc_BA_by_mp(clf_model, reg_model, plist, pSeq_SMILES_list):
    reward_dict = {}
    final_selProt = []
    ncpu = 1
    manager = mp.Manager()
    ret_df_dict = manager.dict()
    mp_list = []
    
    for tid in range(0, len(chunks)):
        chunk = chunks[tid]
        plist_chunk = plist_chunks[tid]

        start_i = tid * chunkSize
        end_i = start_i + chunkSize - 1

        if tid == (len(chunks) - 1):
            end_i = len(pSeq_SMILES_list)

        print("start_i: " + str(start_i) + "\t" + "end_i: " + str(end_i) + "\t" + "tid: " + str(tid))
        mp_task = mp.Process(target=calc_BA, args=(tid, clf_model, reg_model, chunk, plist_chunk, ret_df_dict))
        mp_task.start()
        mp_list.append(mp_task)
        
    for mp_task in mp_list:
        mp_task.join()
    print("end join")
    for tid in range(0, len(mp_list)):
        b_reward_dict = ret_df_dict[tid]
        reward_dict.update(b_reward_dict)

    print("complete to MP: " + str(len(reward_dict.items())))

    return reward_dict




## load pre-trained model
def load_checkpoint(filepath, USE_CUDA):
    if USE_CUDA:
        checkpoint = torch.load(filepath, map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model



def get_pSeqDict(plist_path, pseq_path, pseq_dict_path, maxProtSeqLength):
    ## 1. get all protein sequence (only G protein coupled receptor)
    ## 1,226 proteins, 500 length
    maxProtSeqLength = 5000
    with open(plist_path, 'r') as f:
        proteinList = [line.strip() for line in f]

    print("proteinList: " + str(len(proteinList)))

    if os.path.isfile(pseq_dict_path):
        print("Load from pickle")
        with open(pseq_dict_path, 'rb') as handle:
            pSeq_dict = pickle.load(handle)
    else:
        print("Load from fasta")
        pSeq_dict = loadProteinSeq(pseq_path, proteinList, maxProtSeqLength)

        ## 1. save dict to pickle
        with open(pseq_dict_path, 'wb') as handle:
            pickle.dump(pSeq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pSeq_dict


def get_disease_target_genes(disGene_path, GS_Prot_dict):
    ## 2. get disease specific target genes
    cgc_df = pd.read_csv(disGene_path)
    cgc_df = cgc_df.fillna('')
    cgc_df_sel = cgc_df[
        cgc_df['Tumour Types(Somatic)'].str.contains('hepatocellular') | cgc_df['Tumour Types(Germline)'].str.contains(
            'hepatocellular')]
    geneSet = set(cgc_df_sel['Gene Symbol'].values.tolist())

    dis_Prot_list = []
    for gene in geneSet:
        protein = list(GS_Prot_dict[gene])[0]
        dis_Prot_list.append(protein)

    return dis_Prot_list, geneSet



def main(args):
    ppi_path = args.ppi_path
    mt_path = args.mt_path
    pseq_path = args.pseq_path
    pseq_dict_path = args.pseq_dict_path
    disGene_path = args.disGene_path
    plist_path = args.plist_path

    ## get disease related genes
    ## liver cancer
    ####################################################################################################################
    ## 1. get all protein sequence and load them into memory
    ## 2. get disease specific target genes (TGs)
    ## 3. calcualte BA between smiles & all proteins
    ## 4. function to calculate distance between two genes on the gene network: D(Gi, TGi)
    ## 5. calcualte meanD
    ## 6. get distribution of meanD and transformation meanD into reward
    ####################################################################################################################

    pin = PIN(ppi_path, mt_path)
    ppi_graph = pin.constructPPI_graph()
    GS_Prot_dict = pin.construct_GSprot_dict()

    maxProtSeqLength = 500
    pSeq_dict = get_pSeqDict(plist_path, pseq_path, pseq_dict_path, maxProtSeqLength)
    dis_Prot_list, geneSet = get_disease_target_genes(disGene_path, GS_Prot_dict)

    print("dis_Prot_list: " + str(dis_Prot_list))


    print("dis_Prot_list: " + str(len(dis_Prot_list)))
    print(dis_Prot_list)

    print("geneSet: " + str(len(geneSet)))
    print(geneSet)

    print("\n\n")
    print("pSeq_dict: " + str(len(pSeq_dict.keys())))


    example_smiles = 'CN(C)C1C(=O)SN(NCc2ccccn2)(C(C)=NC(N)=N)N1Cc1ccc(Cl)cc1'

    ## for validation
    ## target of sorafenib
    dis_Prot_list_alreadyKnown_s = ['P15056', 'P04049', 'P35916', 'P35968', 'P36888', 'P09619', 'P10721', 'P11362',
                                    'P07949', 'P17948']

    ## target of regorafenib
    dis_Prot_list_alreadyKnown_r = ['P17948', 'P35968', 'P35916', 'P10721', 'P16234', 'P09619', 'P11362', 'P21802',
                                    'Q02763', 'Q16832', 'P04629', 'P29317', 'P04049', 'P15056', 'Q15759', 'P42685',
                                    'P00519', 'P07949']

    drug_Prot_dict_alreadyknown = {}
    drug_Prot_dict_alreadyknown['sorafenib'] = dis_Prot_list_alreadyKnown_s
    drug_Prot_dict_alreadyknown['regorafenib'] = dis_Prot_list_alreadyKnown_r

    DeepPurpose_model = models.model_pretrained(model = 'Daylight_AAC_DAVIS') 
    main_start = time.time()
    calc_biological_reward_v2(example_smiles, pSeq_dict, DeepPurpose_model)
    main_end1 = time.time()
    main_end2 = time.time()
    print(main_start , main_end1, main_end2)



if __name__ == '__main__':
    help_str = "python BA_reward.py --ppi_path [ppi_path] --mt_path [mapping_table_path]" + "\n"
    ppi_path = "./data_ppi/BIOGRID-ORGANISM-Homo_sapiens-3.5.176.tab2.txt"
    mt_path = "./data_ppi/HUMAN_9606_idmapping.dat"
    pseq_path = "./data_ppi/uniprot_sprot.fasta"
    pseq_dict_path = "./data_ppi/sorafenib_protein_seq.pickle"
    disGene_path = "./data_ppi/cancer_gene_census.csv"
    protein_list = "./data_ppi/sorafenib_protein_list.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi_path", type=str, default=ppi_path, help=help_str)
    parser.add_argument("--mt_path", type=str, default=mt_path, help=help_str)
    parser.add_argument("--pseq_path", type=str, default=pseq_path, help=help_str)
    parser.add_argument("--pseq_dict_path", type=str, default=pseq_dict_path, help=help_str)
    parser.add_argument("--disGene_path", type=str, default=disGene_path, help=help_str)
    parser.add_argument("--plist_path", type=str, default=protein_list, help=help_str)
    args = parser.parse_args()

    main(args)





