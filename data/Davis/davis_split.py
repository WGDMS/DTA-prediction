import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import random
import ast
import csv


def split_S1(data):
    print("--------Data split S1 ----------")
    # Split data into 80% training+validation and 20% testing
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Split the training+validation data into 70% training and 10% validation
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42)  
    # Note: 0.125 of 80% = 10% of the full dataset

    train_data.to_csv("train_S1.csv", index=False)
    val_data.to_csv("val_S1.csv", index=False)
    test_data.to_csv("test_S1.csv", index=False)

    print(f"Train split: {len(train_data)} data")
    print(f"Val split: {len(val_data)} data")
    print(f"Test split: {len(test_data)} data")
    print("--------Data split S1 completed----------")
    
def split_S2(data):
    
    print("--------Data split S2 ----------")
    clusters = {}
    with open("drug_clusters_chembert.txt", "r") as f:
        for line in f:
            cluster_name, drugs = line.strip().split(": ")
            clusters[cluster_name] = drugs.split(", ")
    
    all_drugs = [drug for drugs in clusters.values() for drug in drugs]
    total_drugs = len(all_drugs)
    test_target = math.floor(total_drugs * 0.2)
    val_target = math.floor(total_drugs * 0.1)
    print(f"S2 - Total drugs: {total_drugs}, Test target: {test_target}, Val target: {val_target}")

    # Shuffle clusters
    cluster_list = list(clusters.items())
    random.seed(42)
    random.shuffle(cluster_list)
    
    test_clusters, val_clusters, train_clusters = [], [], []
    test_count, val_count = 0, 0

    for cluster_name, drugs in cluster_list:
        cluster_size = len(drugs)
        
        # fill test set
        if test_count + cluster_size <= test_target:
            test_clusters.extend([(drug, cluster_name) for drug in drugs])
            test_count += cluster_size
        # fill validation set
        elif val_count + cluster_size <= val_target:
            val_clusters.extend([(drug, cluster_name) for drug in drugs])
            val_count += cluster_size
        # Remaining go to train
        else:
            train_clusters.extend([(drug, cluster_name) for drug in drugs])
    print(f"S2 - from clusters: Train drugs: {len(train_clusters)}, Test drugs: {len(test_clusters)}, Val drugs: {len(val_clusters)}")
    
    #pd.Series(train_clusters).to_csv('train_drugs_S2.txt', index=False, header=False)    
    #pd.Series(test_clusters).to_csv('test_drugs_S2.txt', index=False, header=False)
    #pd.Series(val_clusters).to_csv('val_drugs_S2.txt', index=False, header=False)
  
    # Create splits with cluster IDs
    def create_split(data, cluster_drugs):
        drug_ids = [drug for drug, _ in cluster_drugs]
        cluster_ids = [cluster_id for _, cluster_id in cluster_drugs]
        split = data[data["Drug_ID"].isin(drug_ids)].copy()
        split["Drug_Cluster_ID"] = split["Drug_ID"].map(dict(zip(drug_ids, cluster_ids)))
        return split

    test_set = create_split(data, test_clusters)
    val_set = create_split(data, val_clusters)
    train_set = create_split(data, train_clusters)

    # check no overlapping
    test_drugs = set(test_set["Drug_ID"])
    val_drugs = set(val_set["Drug_ID"])
    train_drugs = set(train_set["Drug_ID"])
    assert len(test_drugs & val_drugs) == 0
    assert len(test_drugs & train_drugs) == 0
    assert len(val_drugs & train_drugs) == 0

    train_set.to_csv("train_S2.csv", index=False)
    val_set.to_csv("val_S2.csv", index=False)
    test_set.to_csv("test_S2.csv", index=False)

    print(f"Train split: {len(train_set)} data")
    print(f"Val split: {len(val_set)} data")
    print(f"Test split: {len(test_set)} data")
    print("--------Data split S2 completed----------")

   
def split_S3(data):
    print("--------Data split S3 ----------")
    clusters = {}
    with open("cluster_protT5.txt", "r") as f:
        for line in f:
            cluster_name, proteins = line.strip().split(": ")
            clusters[cluster_name] = proteins.split(", ")
    
    all_proteins = [protein for proteins in clusters.values() for protein in proteins]
    total_proteins = len(all_proteins)
    test_target = math.floor(total_proteins * 0.2)
    val_target = math.floor(total_proteins * 0.1)
    print(f"S3 - Total proteins: {total_proteins}, Test target: {test_target}, Val target: {val_target}")

    # Shuffle clusters 
    cluster_list = list(clusters.items())
    random.seed(42)
    random.shuffle(cluster_list)
    
    test_clusters, val_clusters, train_clusters = [], [], []
    test_count, val_count = 0, 0

    for cluster_name, proteins in cluster_list:
        cluster_size = len(proteins)
        
        #  fill test set
        if test_count + cluster_size <= test_target:
            test_clusters.extend([(protein, cluster_name) for protein in proteins])
            test_count += cluster_size
        #  fill validation set
        elif val_count + cluster_size <= val_target:
            val_clusters.extend([(protein, cluster_name) for protein in proteins])
            val_count += cluster_size
        # Remaining go to train
        else:
            train_clusters.extend([(protein, cluster_name) for protein in proteins])
    
    print(f"S3 - From clusters: Train proteins: {len(train_clusters)}, Test proteins: {len(test_clusters)}, Val proteins: {len(val_clusters)}")
    
    #pd.Series([protein for protein, _ in test_clusters]).to_csv('test_proteins_S3.txt', index=False, header=False)
    #pd.Series([protein for protein, _ in val_clusters]).to_csv('val_proteins_S3.txt', index=False, header=False)
  
    # Create splits with cluster IDs
    def create_split(data, cluster_proteins):
        protein_ids = [protein for protein, _ in cluster_proteins]
        cluster_ids = [cluster_id for _, cluster_id in cluster_proteins]
        split = data[data["Protein_ID"].isin(protein_ids)].copy()
        split["Protein_Cluster_ID"] = split["Protein_ID"].map(dict(zip(protein_ids, cluster_ids)))
        return split

    test_set = create_split(data, test_clusters)
    val_set = create_split(data, val_clusters)
    train_set = create_split(data, train_clusters)

    # check no overlapping
    test_proteins = set(test_set["Protein_ID"])
    val_proteins = set(val_set["Protein_ID"])
    train_proteins = set(train_set["Protein_ID"])
    assert len(test_proteins & val_proteins) == 0
    assert len(test_proteins & train_proteins) == 0
    assert len(val_proteins & train_proteins) == 0

    train_set.to_csv("train_S3.csv", index=False)
    val_set.to_csv("val_S3.csv", index=False)
    test_set.to_csv("test_S3.csv", index=False)
    
    print(f"Train split: {len(train_set)} data")
    print(f"Val split: {len(val_set)} data")
    print(f"Test split: {len(test_set)} data")
    print("--------Data split S3 completed----------")
  
def split_S4(data):
    print("--------Data split S4 ----------")
    
    drug_clusters = {}
    with open("drug_clusters_chembert.txt", "r") as f:
        for line in f:
            cluster_name, drugs = line.strip().split(": ")
            drug_clusters[cluster_name] = drugs.split(", ")
    
    protein_clusters = {}
    with open("cluster_protT5.txt", "r") as f:
        for line in f:
            cluster_name, proteins = line.strip().split(": ")
            protein_clusters[cluster_name] = proteins.split(", ")
    
    all_drugs = [drug for drugs in drug_clusters.values() for drug in drugs]
    all_proteins = [protein for proteins in protein_clusters.values() for protein in proteins]
    
    total_drugs = len(all_drugs)
    total_proteins = len(all_proteins)
    
    test_drug_target = math.floor(total_drugs * 0.2)
    val_drug_target = math.floor(total_drugs * 0.1)
    
    test_protein_target = math.floor(total_proteins * 0.2)
    val_protein_target = math.floor(total_proteins * 0.1)
    
    print(f"S4 - Total drugs: {total_drugs}, Test drug : {test_drug_target}, Val drug : {val_drug_target}")
    print(f"S4 - Total proteins: {total_proteins}, Test protein : {test_protein_target}, Val protein : {val_protein_target}")
    
    # Shuffle clusters
    random.seed(42)
    
    drug_cluster_list = list(drug_clusters.items())
    random.shuffle(drug_cluster_list)
    
    protein_cluster_list = list(protein_clusters.items())
    random.shuffle(protein_cluster_list)
    
    test_drugs, val_drugs, train_drugs = [], [], []
    test_proteins, val_proteins, train_proteins = [], [], []
    
    test_drug_count, val_drug_count = 0, 0
    test_protein_count, val_protein_count = 0, 0
    
    for cluster_name, drugs in drug_cluster_list:
        cluster_size = len(drugs)
        if test_drug_count + cluster_size <= test_drug_target:
            test_drugs.extend([(drug, cluster_name) for drug in drugs])
            test_drug_count += cluster_size
        elif val_drug_count + cluster_size <= val_drug_target:
            val_drugs.extend([(drug, cluster_name) for drug in drugs])
            val_drug_count += cluster_size
        else:
            train_drugs.extend([(drug, cluster_name) for drug in drugs])
    
    for cluster_name, proteins in protein_cluster_list:
        cluster_size = len(proteins)
        if test_protein_count + cluster_size <= test_protein_target:
            test_proteins.extend([(protein, cluster_name) for protein in proteins])
            test_protein_count += cluster_size
        elif val_protein_count + cluster_size <= val_protein_target:
            val_proteins.extend([(protein, cluster_name) for protein in proteins])
            val_protein_count += cluster_size
        else:
            train_proteins.extend([(protein, cluster_name) for protein in proteins])
    
    print(f"S4 - From clusters: Train drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}, Val drugs: {len(val_drugs)}")
    print(f"S4 - From clusters: Train proteins: {len(train_proteins)}, Test proteins: {len(test_proteins)}, Val proteins: {len(val_proteins)}")
    
    test_drug_set = {drug for drug, _ in test_drugs}
    val_drug_set = {drug for drug, _ in val_drugs}
    test_protein_set = {protein for protein, _ in test_proteins}
    val_protein_set = {protein for protein, _ in val_proteins}
    
    test_set = data[(data['Drug_ID'].isin(test_drug_set)) & (data['Protein_ID'].isin(test_protein_set))].copy()
    val_set = data[(data['Drug_ID'].isin(val_drug_set)) & (data['Protein_ID'].isin(val_protein_set))].copy()
    train_set = data[(~data['Drug_ID'].isin(test_drug_set | val_drug_set)) & (~data['Protein_ID'].isin(test_protein_set | val_protein_set))].copy()
    
    # Map cluster IDs
    drug_cluster_dict = dict(test_drugs + val_drugs + train_drugs)
    protein_cluster_dict = dict(test_proteins + val_proteins + train_proteins)
    
    test_set["Drug_Cluster_ID"] = test_set["Drug_ID"].map(drug_cluster_dict)
    test_set["Protein_Cluster_ID"] = test_set["Protein_ID"].map(protein_cluster_dict)
    
    val_set["Drug_Cluster_ID"] = val_set["Drug_ID"].map(drug_cluster_dict)
    val_set["Protein_Cluster_ID"] = val_set["Protein_ID"].map(protein_cluster_dict)
    
    train_set["Drug_Cluster_ID"] = train_set["Drug_ID"].map(drug_cluster_dict)
    train_set["Protein_Cluster_ID"] = train_set["Protein_ID"].map(protein_cluster_dict)
    
    train_set.to_csv("train_S4.csv", index=False)
    val_set.to_csv("val_S4.csv", index=False)
    test_set.to_csv("test_S4.csv", index=False)
    
    print(f"Train split: {len(train_set)} data")
    print(f"Val split: {len(val_set)} data")
    print(f"Test split: {len(test_set)} data")
    print("--------Data split S4 completed----------")

# Load the formatted CSV file
input_file = "davis_affinity_data.csv"  
data = pd.read_csv(input_file)

split_S1(data)
split_S2(data)
split_S3(data)
split_S4(data)