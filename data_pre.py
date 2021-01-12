# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:31:47 2020

@author: 华阳
"""
import pandas as pd
from keras.preprocessing import sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from gensim.models import Word2Vec
from graph_features import atom_features
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from collections import defaultdict


model = word2vec.Word2Vec.load('model_300dim.pkl')
t = np.mean(sentences2vec("CCC1=CC=C(C=C1)C2=C(C=NN2C)C3=NN(C4=C3C(=NC=N4)N5CC(C5)(F)F)C", model, unseen='UNK'),0)


#MolVS for standardization and normalization of molecules

def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    return npfp

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]
    
seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}

seq_rdic2 = ['A','B','C','D','E','F']
seq_dic2 = {w: i+1 for i,w in enumerate(seq_rdic2)}

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency

def gnn(atom_feature, adj):
    fea = np.matmul(adj, atom_feature)
    output = np.mean(fea,0)
    return output

def buling(seq):
    length = len(seq)
    result = seq
    if length < 2500:
        for i in range(2500-length):
            result += str(0)
    else:
        result = seq[0:2500]
    return result

def split_sequence(seq):
    protein = np.zeros(2500)
    for i in range(len(seq)):
        protein[i] = int(seq[i])
    return protein
    
dicts = {"H":"A","R":"A","K":"A",
         "D":"B","E":"B","N":"B","Q":"B",
         "C":"C","X":"C",
         "S":"D","T":"D","P":"D","A":"D","G":"D","U":"D",
         "M":"E","I":"E","L":"E","V":"E",
         "F":"F","Y":"F","W":"F"}
        
if __name__ == "__main__":

    import os
    """
    atom_feature ,adj = mol_features("C(C(C(=O)[O-])[O-])(C(=O)[O-])[O-].C(C(C(=O)[O-])[O-])(C(=O)[O-])[O-].[K+].[K+].[Sb+3].[Sb+3]")
    print(atom_feature)
    print(adj)
    """
    DATASET = "GPCR"
    with open("GPCR/data/test.txt","r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    Drug_morgan,Drug_mol2vec,Drug_Mpnn,proteins,interactions = [],[],[], [], []
    t= 0
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequences, interaction = data.strip().split(" ")
        Drug_morgan.append(morgan_fp(smiles))
        Drug_mol2vec.append(np.mean(sentences2vec(smiles, model, unseen='UNK'),0))
        #Drug_mol2vec.append(np.mean(sentences2vec("CCC1=CC=C(C=C1)C2=C(C=NN2C)C3=NN(C4=C3C(=NC=N4)N5CC(C5)(F)F)C", model, unseen='UNK'),0))
        #atom_feature, adj = smile_to_graph(smiles)
        #mpnn = gnn(atom_feature, adj)
        #Drug_Mpnn.append(mpnn)
        #Drug_adj.append(adj)
        
        
        #print(morgan_fp(smiles))
        newsequence = ''
    #print(sequence)
        for i in range(len(sequences)):
            newsequence += dicts[sequences[i]]
        newsequence2 = ''
        for i in range(len(newsequence)):
            newsequence2 += str(seq_dic2[newsequence[i]])
            
        protein_fea = buling(newsequence2)
        protein_feature = split_sequence(protein_fea)
        proteins.append(protein_feature)
        if float(interaction)==0:
            t+=1
        interactions.append(np.array(float(interaction)))
    print(t)
    dir_input = ('GPCR/G_test/')
    #os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'morgan', Drug_morgan)
    np.save(dir_input + 'mol2vec', Drug_mol2vec)
    #np.save(dir_input + 'Drug_Mpnn', Drug_Mpnn)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    
    print('The preprocess of ' + DATASET + ' dataset has finished!')
