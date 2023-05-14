import pysam
import numpy as np
import argparse

def main(bam_path, output_path, train_chromosomes, test_chromosomes, validation_chromosomes):
    chromosome_to_reads = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        for chromosome in bam_file.references:
            read_ids = []
            for read in bam_file.fetch(chromosome):
                read_id = read.query_name
                read_ids.append(read_id)
            chromosome_to_reads[chromosome] = read_ids
    
    train_readids = np.concatenate([chromosome_to_reads[ch] for ch in train_chromosomes]) if train_chromosomes else np.array([])
    test_readids = np.concatenate([chromosome_to_reads[ch] for ch in test_chromosomes]) if test_chromosomes else np.array([])
    validation_readids = np.concatenate([chromosome_to_reads[ch] for ch in validation_chromosomes]) if validation_chromosomes else np.array([])
    
    train_set = set(train_readids)
    test_set = set(test_readids)
    validation_set = set(validation_readids)
        
    # Removing reads that mapped to both sets from the training data (supplementary reads)
    train_set = train_set - test_set
    train_set = train_set - validation_set
    test_set = test_set - validation_set
    assert (len(train_set & test_set) == 0)
    assert (len(train_set & validation_set) == 0)
    assert (len(test_set & validation_set) == 0)
    train_readids = list(train_set)
    test_readids = list(test_set)
    validation_readids = list(validation_set)
    
    #Txt files
    with open(output_path+'/train_readids.txt','w') as txt_file:
        for readid in train_readids:
            txt_file.write(readid + '\n')
    with open(output_path+'/test_readids.txt','w') as txt_file:
         for readid in test_readids:
            txt_file.write(readid + '\n')    
    with open(output_path+'/validation_readids.txt','w') as txt_file:
        for readid in validation_readids:
            txt_file.write(readid + '\n')  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a map from a bam file.')
    parser.add_argument('--bam_path', required=True, help='The path to the bam file.')
    parser.add_argument('--output_path', required=True, help='The output path for the final txt files.')
    parser.add_argument('--train_chromosomes', nargs='*', required=True, help='List of chromosomes for the training split. E.g. 2 3 4 5 X MT')
    parser.add_argument('--test_chromosomes', nargs='*', required=True, help='List of chromosomes for the testing split. E.g. 1')
    parser.add_argument('--validation_chromosomes', nargs='*', required=True, help='List of chromosomes for the validation split. E.g. 2 3 4 5 X MT')

    args = parser.parse_args()

    main(
        bam_path=args.bam_path, 
        output_path=args.output_path, 
        train_chromosomes=args.train_chromosomes, 
        test_chromosomes=args.test_chromosomes, 
        validation_chromosomes=args.validation_chromosomes
    )
