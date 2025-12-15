import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict, defaultdict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdmolops
import networkx as nx
from Bio import SeqIO

from utils import *

def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# print(res_weight_table)

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# feature dim
ATOM_DIM = 101
BOND_DIM = 11
FG_DIM = 73
FG_EDGE_DIM = ATOM_DIM

ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}

PATT = {
    'HETEROATOM': '[!#6]',
    'DOUBLE_TRIPLE_BOND': '*=,#*',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]'
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}



# one-hot encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list
   
def get_atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
            'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()] +
        [atom.IsInRing()]
    )


def get_bond_feature(bond):
    return np.array(
        one_of_k_encoding(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
        [bond.GetIsConjugated()] +
        one_of_k_encoding(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
    )


def get_fg_feature(fg_prop):
    return np.array(
        one_of_k_encoding_unk(fg_prop['#C'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#O'], range(6)) +  # 0-5, 5+
        one_of_k_encoding_unk(fg_prop['#N'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#P'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#S'], range(6)) +
        [fg_prop['#X'] > 0] +
        [fg_prop['#UNK'] > 0] +
        one_of_k_encoding_unk(fg_prop['#SINGLE'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#DOUBLE'], range(8)) +  # 0-6, 6+
        one_of_k_encoding_unk(fg_prop['#TRIPLE'], range(8)) +
        one_of_k_encoding_unk(fg_prop['#AROMATIC'], range(8)) +
        [fg_prop['IsRing']]
    )

#generate moelcular graph

def mol_to_graphs(key, dataset):
    
    dataset_path = 'data/' + dataset + '/'
    df = pd.read_csv(dataset_path+ "davis_smiles.csv")
    smile = df.loc[df["Drug_ID"] == key, "SMILES"].values[0]
    mol = Chem.MolFromSmiles(smile)
       
    fgs = []  
  
    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]  
    flag = True  
    while flag:
        flag = False
        for i in range(len(rings)):
            if len(rings[i]) == 0: continue
            for j in range(i+1, len(rings)):
                shared_atoms = rings[i] & rings[j]
                if len(shared_atoms) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if len(r) > 0]
    
    marks = set()
    for patt in PATT.values():  
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  
    for atom in marks: 
        fgs.append({atom})
        atom2fg[atom] = [len(fgs)-1]
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in marks and a2 in marks: 
            assert a1 != a2
            assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
           
            fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
            fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = atom2fg[a1]
        elif a1 in marks: 
            assert len(atom2fg[a1]) == 1
           
            fgs[atom2fg[a1][0]].add(a2)
            atom2fg[a2].extend(atom2fg[a1])
        elif a2 in marks:
            # add a1 to a2's FG
            assert len(atom2fg[a2]) == 1
            fgs[atom2fg[a2][0]].add(a1)
            atom2fg[a1].extend(atom2fg[a2])
        else:  # both atoms are unmarked
            # add single bond to fgs
            fgs.append({a1, a2})
            atom2fg[a1].append(len(fgs)-1)
            atom2fg[a2].append(len(fgs)-1)
    tmp = []
    for fg in fgs:
        if len(fg) == 0: continue
        if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing(): continue  
        tmp.append(fg)
    fgs = tmp
    

    fgs.extend(rings)  # final FGs: rings + FGs (not in rings)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]
    for i in range(len(fgs)): # update atom2fg
        for atom in fgs[i]:
            atom2fg[atom].append(i)

  
    atom_features, bond_list, bond_features = [], [], []
    fg_prop = [defaultdict(int) for _ in range(len(fgs))]  
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature(atom).tolist())
        elem = atom.GetSymbol()
        if elem in ['C', 'O', 'N', 'P', 'S']:
            key = '#'+elem
        elif elem in ['F', 'Cl', 'Br', 'I']:
            key = '#X'
        else:
            key = '#UNK'
        for fg_idx in atom2fg[atom.GetIdx()]:
            fg_prop[fg_idx][key] += 1
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_list.extend([[a1, a2], [a2, a1]])
        bond_features.extend([get_bond_feature(bond).tolist()] * 2)
        key = '#'+str(bond.GetBondType())
        for fg_idx in (set(atom2fg[a1]) & set(atom2fg[a2])):
            fg_prop[fg_idx][key] += 1
            if bond.IsInRing():
                fg_prop[fg_idx]['IsRing'] = 1
  
    fg_features, fg_edge_list, fg_edge_features = [], [], []
    for i in range(len(fgs)):
        fg_features.append(get_fg_feature(fg_prop[i]).tolist())
        for j in range(i+1, len(fgs)):
            shared_atoms = list(fgs[i] & fgs[j])
            if len(shared_atoms) > 0:
                fg_edge_list.extend([[i, j], [j, i]])
                if len(shared_atoms) == 1:
                    fg_edge_features.extend([atom_features[shared_atoms[0]]] * 2)
                else:  # two rings shared 2 atoms
                    assert len(shared_atoms) == 2
                    ef = [(i+j)/2 for i, j in zip(atom_features[shared_atoms[0]], atom_features[shared_atoms[1]])]
                    fg_edge_features.extend([ef] * 2)
    

    atom2fg_list = []
    for fg_idx in range(len(fgs)):
        for atom_idx in fgs[fg_idx]:
            atom2fg_list.append([atom_idx, fg_idx])

    return atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list

def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
   
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    
    return pssm_mat

# target feature for target graph

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    target_feature= np.concatenate((pro_hot, pro_property), axis=1)
    return np.concatenate((np.transpose(pssm, (1, 0)), target_feature), axis=1)
    

def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature


#generate protein graphs

def target_to_graph(target_key, target_sequence, contact_dir, msa_path):
    target_size = len(target_sequence)

    # Load contact map
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)

    # Add self-loops
    contact_map += np.eye(contact_map.shape[0])

    # Get edges (contacts above threshold)
    index_row, index_col = np.where(contact_map >= 0.5)

    # Build edge list
    target_edge_index = np.stack([index_row, index_col], axis=1)

    # Remove edges that point outside the sequence length
    valid_mask = (target_edge_index[:, 0] < target_size) & (target_edge_index[:, 1] < target_size)
  #  if not np.all(valid_mask):
   #     print(f"[WARNING] {target_key}: removed {(~valid_mask).sum()} invalid edges "
    #          f"(target_size={target_size}, max_index={target_edge_index.max()})")
    target_edge_index = target_edge_index[valid_mask]

    #Make graph undirected (if needed)
    target_edge_index = np.unique(
        np.concatenate([target_edge_index, target_edge_index[:, [1, 0]]], axis=0),
        axis=0
    )

    # Features
    target_feature = target_to_feature(target_key, target_sequence, msa_path)

    return target_size, target_feature, target_edge_index



# to judge whether the required files exist
def valid_target(key, dataset):
    
    dataset_path = 'data/' + dataset + '/'
   # aln_dir = 'data/' + dataset + '/aln/'
    contact_dir = dataset_path + '/pconsc4/'
    
    contact_file = os.path.join(contact_dir, key + '.npy')
    #aln_file = os.path.join(aln_dir, key + '.aln')
    # print(contact_file, aln_file)
    if os.path.exists(contact_file) :
        return True
    else:
        return False


#def data_to_csv(csv_file, datalist):
 #   with open(csv_file, 'w') as f:
  #      f.write('compound_iso_smiles,drug_smiles,smile_keys, target_sequence,target_key,affinity\n')
   #     for data in datalist:
    #        f.write(','.join(map(str, data)) + '\n')
            
           
def create_dataset_for_test(dataset, split):
    
    dataset_path = f'data/{dataset}/'
    
    df = pd.read_csv(dataset_path+ f'{dataset}_smiles.csv')
    drugs_dict = dict(zip(df["Drug_ID"], df["SMILES"]))
    
    dfT = pd.read_csv(dataset_path+ f'{dataset}_kinases.csv')
    proteins_dict = dict(zip(dfT["Protein_ID"], dfT["FASTA"]))
    
    msa_path = dataset_path + 'aln/'
    contact_path = dataset_path + 'pconsc4/'
    msa_list = []
    contact_list = []
    for key in proteins_dict:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contact_path, key + '.npy'))

   # drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    smile_keys= []
    # smiles
    for d in drugs_dict.keys():
        #lg = Chem.MolToSmiles(Chem.MolFromSmiles(drugs_dict[d]), isomericSmiles=True)
        #drugs.append(lg)
        drug_smiles.append(drugs_dict[d])
        smile_keys.append(d)
    # seqs
    for t in proteins_dict.keys():
        prots.append(proteins_dict[t])
        prot_keys.append(t)


   # compound_iso_smiles = drugs
    #target_key = prot_keys
    #drug_key= smile_keys

    # create smile graph
    mol_graph = {}
    for key in smile_keys:
        g = mol_to_graphs(key, dataset)
        mol_graph[key] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in prot_keys:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins_dict[key], contact_path, msa_path)
        target_graph[key] = g


        
    # count the number of  proteins with aln and contact files
   # print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    #if len(smile_graph) == 0 or len(target_graph) == 0:
     #   raise Exception('no protein or drug, run the script for datasets preparation.')

    df_test = pd.read_csv(dataset_path + f"split_data/test_{split}.csv")
    df_test.columns = df_test.columns.str.strip()
    print('Test entries:', len(df_test))

    affinity_col = 'Affinity_pKd' if dataset == 'davis' else 'KIBA_affinity'

    test_drug_id, test_prot_id, affinity = list(df_test['Drug_ID']), list(df_test['Protein_ID']), list(df_test[affinity_col])
        
    #test_proteins_vec= [seq_cat(t) for t in test_prot_seq]
    #test_smiles_vec = [smi_cat(t) for t in test_drugs_smiles]
    #test_smiles_vec = test_drugs_smiles
    #test_proteins_vec= [prot_tok_encoding(t) for t in test_prot_seq]
    #test_proteins_vec= test_prot_seq
    
    test_drug_id, test_prot_id, affinity =  np.asarray(test_drug_id), np.asarray(test_prot_id), np.asarray(affinity)

    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', drug_key= test_drug_id, target_key=test_prot_id,
                               y= affinity, mol_graph=mol_graph, target_graph=target_graph)

                               
    return test_dataset

   
                               

def create_dataset(dataset, split):
  
    dataset_path = f'data/{dataset}/'
    
    df = pd.read_csv(dataset_path+ f'{dataset}_smiles.csv')
    drugs_dict = dict(zip(df["Drug_ID"], df["SMILES"]))
    
    dfT = pd.read_csv(dataset_path+ f'{dataset}_kinases.csv')
    proteins_dict = dict(zip(dfT["Protein_ID"], dfT["FASTA"]))
    
    
    msa_path = dataset_path + 'aln/'
    contact_path = dataset_path + 'pconsc4/'
    msa_list = []
    contact_list = []
    for key in proteins_dict:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contact_path, key + '.npy'))

   # drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    smile_keys= []
    
    # smiles
    for d in drugs_dict.keys():
        #lg = Chem.MolToSmiles(Chem.MolFromSmiles(drugs_dict[d]), isomericSmiles=True)
        #drugs.append(lg)
        drug_smiles.append(drugs_dict[d])
        smile_keys.append(d)
    # seqs
    for t in proteins_dict.keys():
        prots.append(proteins_dict[t])
        prot_keys.append(t)


   # compound_iso_smiles = drugs
    #target_key = prot_keys
    #drug_key= smile_keys

    # create smile graph
    mol_graph = {}
    for key in smile_keys:
        g = mol_to_graphs(key, dataset)
        mol_graph[key] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in prot_keys:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins_dict[key], contact_path, msa_path)
        target_graph[key] = g

        
    # count the number of  proteins with aln and contact files
   # print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    #if len(smile_graph) == 0 or len(target_graph) == 0:
     #   raise Exception('no protein or drug, run the script for datasets preparation.')

  
    df_train = pd.read_csv(dataset_path + f"split_data/train_{split}.csv")
    df_train.columns = df_train.columns.str.strip()
    print('Train entries:', len(df_train))

    affinity_col = 'Affinity_pKd' if dataset == 'davis' else 'KIBA_affinity'
    
    train_drug_id, train_prot_id, affinity = list(df_train['Drug_ID']), list(df_train['Protein_ID']), list(df_train[affinity_col])
    train_drug_id, train_prot_id, affinity =  np.asarray(train_drug_id), np.asarray(train_prot_id), np.asarray(affinity)

    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drug_key= train_drug_id, target_key=train_prot_id,
                               y= affinity, mol_graph=mol_graph, target_graph=target_graph)

         
    df_val = pd.read_csv(dataset_path + f"split_data/val_{split}.csv")
    df_val.columns = df_val.columns.str.strip()
    print('Validation entries:', len(df_val))

    val_drug_id, val_prot_id, affinity = list(df_val['Drug_ID']), list(df_val['Protein_ID']), list(df_val[affinity_col])       
    val_drug_id, val_prot_id, affinity =  np.asarray(val_drug_id), np.asarray(val_prot_id), np.asarray(affinity)

    val_dataset = DTADataset(root='data', dataset=dataset + '_' + 'val', drug_key= val_drug_id, target_key=val_prot_id,
                               y= affinity, mol_graph=mol_graph, target_graph=target_graph)

    return train_dataset, val_dataset
     