import os
import random
import numpy as np
import json
from collections import OrderedDict
from Bio import SeqIO

# Modified from the DGraphDTA repository:
# Repository: https://github.com/595693085/DGraphDTA


def seq_format(proteins_dic, output_dir):
    
    for key, value in proteins_dic.items():
        with open(os.path.join(output_dir, key + '.fasta'), 'w') as f:
            f.writelines('>' + key + '\r\n')
            f.writelines(value + '\r\n')



def HHblitsMSA(bin_path, db_path, input_dir, output_dir):
    print(f"Database path: {db_path}")
    print(f"Files in database directory: {os.listdir(db_path)}")
    #database file path
    #https://wwwuser.gwdguser.de/~compbiol/uniclust/2023_02/
    db_path = '/home/samankweda/Documents/UniRef30_2023_02/UniRef30_2023_02'  # hhblits dataset for msa
    
    for fas_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, fas_file)
        output_file = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.hhr')  # ignore
        output_file_a3m = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.a3m')
        if os.path.exists(output_file) and os.path.exists(output_file_a3m):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        output_file_a3m = output_file_a3m.replace('(', '\(').replace(')', '\)')
        cmd = (
            f"{bin_path} -maxfilt 100000 -realign_max 100000 "
            f"-d {db_path} -all -B 100000 -Z 100000 -n 3 -e 0.001 "
            f"-i {process_file} -o {output_file} -oa3m {output_file_a3m} -cpu 8"
        )
        print(f"Running command: {cmd}")
        os.system(cmd)
        
def HHfilter(bin_path, input_dir, output_dir):
    file_prefix = []
    # print(input_dir)
    for file in os.listdir(input_dir):
        if 'a3m' not in file:
            continue
        temp_prefix = file.split('.a3m')[0]
        if temp_prefix not in file_prefix:
            file_prefix.append(temp_prefix)

    for msa_file_prefix in file_prefix:
        file_name = msa_file_prefix + '.a3m'
        process_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' -id 90 -i ' + process_file + ' -o ' + output_file
        print(cmd)
        os.system(cmd)


def reformat(bin_path, input_dir, output_dir):
    # print('reformat')
    for a3m_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, a3m_file)
        output_file = os.path.join(output_dir, a3m_file.split('.a3m')[0] + '.fas')
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' ' + process_file + ' ' + output_file + ' -r'
        print(cmd)
        os.system(cmd)


def convertAlignment(bin_path, input_dir, output_dir):
    # print('convertAlignment')
    for fas_file in os.listdir(input_dir):
        process_file = input_dir + '/' + fas_file
        output_file = output_dir + '/' + fas_file.split('.fas')[0] + '.aln'
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = 'python ' + bin_path + ' ' + process_file + ' fasta ' + output_file
        print(cmd)
        os.system(cmd)


def alnFilePrepare():
    import json
    from collections import OrderedDict
    print('aln file prepare ...')
    #datasets = ['davis', 'kiba']
    datasets = ['davis']
    for dataset in datasets:
        seq_dir = os.path.join('data/', dataset, 'seq')
        msa_dir = os.path.join('data/', dataset, 'msa')
        filter_dir = os.path.join('data/', dataset, 'hhfilter')
        reformat_dir = os.path.join('data/', dataset, 'reformat')
        aln_dir = os.path.join('data, dataset, 'aln')
        # pconsc4_dir = os.path.join('data', dataset, 'pconsc4')
        protein_path = os.path.join('data/', dataset)
     #   proteins = json.load(open(os.path.join(protein_path, 'proteins.txt')), object_pairs_hook=OrderedDict)

        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        if not os.path.exists(msa_dir):
            os.makedirs(msa_dir)
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)
        if not os.path.exists(reformat_dir):
            os.makedirs(reformat_dir)
        if not os.path.exists(aln_dir):
            os.makedirs(aln_dir)
        
        dataset_path = 'data/' + dataset + '/'
        dfT = pd.read_csv(dataset_path+ "davis_kinases.csv")
        proteins_dict = dict(zip(dfT["Protein_ID"], dfT["FASTA"]))
    
        
        HHblits_bin_path = 'hhsuite/build/src/hhblits'  # HHblits bin path
        HHblits_db_path = '/home/samankweda/Documents/UniRef30_2023_02'  # hhblits dataset for msa
       
        HHfilter_bin_path = 'hhsuite/build/src/hhfilter'  # HHfilter bin path
        reformat_bin_path = 'hhsuite/scripts/reformat.pl'  # reformat bin path
        convertAlignment_bin_path = 'hhsuite/build/ccmpred/scripts/convert_alignment.py'  # ccmpred convertAlignment bin path

        # check the programs used for the script
        if not os.path.exists(HHblits_bin_path):
            raise Exception('Program HHblits was not found. Please specify the run path.')

        if not os.path.exists(HHfilter_bin_path):
            raise Exception('Program HHfilter was not found. Please specify the run path.')

        if not os.path.exists(reformat_bin_path):
            raise Exception('Program reformat was not found. Please specify the run path.')

        if not os.path.exists(convertAlignment_bin_path):
            raise Exception('Program convertAlignment was not found. Please specify the run path.')

        seq_format(proteins_dict, seq_dir)
        HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
        HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
        reformat(reformat_bin_path, filter_dir, reformat_dir)
        convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)

        print('aln file prepare over.')
        print(f"Database path: {HHblits_db_path}")
        print(f"Files in database directory: {os.listdir(HHblits_db_path)}")
        
if __name__ == '__main__':
    alnFilePrepare()
  
  