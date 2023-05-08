import pysam
import pickle
from itertools import chain

def create_map(bam_path, output_path):
    chromosome_to_reads = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        for chromosome in bam_file.references:
            read_ids = []
            for read in bam_file.fetch(chromosome):
                read_id = read.query_name
                read_ids.append(read_id)
            chromosome_to_reads[chromosome] = read_ids
    # for k,v in chromosome_to_reads.items():
        # print(k, len(v))
    with open(output_path+'/chromosome_to_reads.pkl','wb') as pickle_file:
        pickle.dump(chromosome_to_reads, pickle_file)

def create_split(output_path):
    with open(output_path+'/chromosome_to_reads.pkl', 'rb') as pickle_file:
        chromosome_to_reads = pickle.load(pickle_file)
        
    train_chromosomes = [str(ch) for ch in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','MT']]
    test_chromosomes = [str(ch) for ch in [1]]
    
    train_readids = join_lists_in_dict(chromosome_to_reads, train_chromosomes)
    test_readids = join_lists_in_dict(chromosome_to_reads, test_chromosomes)
    
    train_set = set(train_readids)
    test_set = set(test_readids)
    print('training size', len(train_readids))
    print('testing size', len(test_readids))
    print('OVERLAP', len(train_set & test_set))
    
    # Removing reads that mapped to both sets from the training data (supplementary reads)
    train_set = train_set - test_set
    train_readids = list(train_set)
    print('training size', len(train_readids))
    print('testing size', len(test_readids))
    print('OVERLAP AFTER CORRECTION', len(train_set & test_set))
    assert (len(train_set & test_set) == 0)
    
    
    #Pickle files
    with open(output_path+'/train_readids.pkl','wb') as pickle_file:
        pickle.dump(train_readids, pickle_file)
    with open(output_path+'/test_readids.pkl','wb') as pickle_file:
        pickle.dump(test_readids, pickle_file)
        
    #Txt files
    with open(output_path+'/train_readids.txt','w') as txt_file:
        for readid in train_readids:
            txt_file.write(readid + '\n')
    with open(output_path+'/test_readids.txt','w') as txt_file:
         for readid in test_readids:
            txt_file.write(readid + '\n')    
                        
def join_lists_in_dict(input_dict, keys_to_use):
    combined_list = list(chain.from_iterable(input_dict[key] for key in keys_to_use))
    return combined_list

create_map(bam_path = snakemake.input.bam_path, output_path = 'splits/'+snakemake.wildcards.experiment_name)
create_split(output_path = 'splits/'+snakemake.wildcards.experiment_name)

