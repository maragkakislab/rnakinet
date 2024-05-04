# RNAkinet
RNAkinet is a project dedicated to detecting 5eu-modified reads directly from the raw nanopore sequencing signal. Furthermore, it offers tools to calculate transcript halflives.

# Usage
## Installation
```sh
pip install rnakinet
```
## Predict 5EU in your fast5 files
```sh
rnakinet-inference --path <path_to_folder_containing_fast5s> --output <predictions_name.csv>
```
This creates a csv file with columns `read_id` - the read id, `5eu_mod_score` - the raw prediction score from 0 to 1, `5eu_modified_prediction` - Boolean column, True if the read is predicted to be modified by 5EU, False otherwise

Nvidia GPU is recommended to run this command. If you want to run inference on a CPU-only machine, use the `--use-cpu` option. This will substantially increase runtime.

### Example
```sh
rnakinet-inference --path data/experiment/fast5_folder --output preds.csv
```

## Calculate transcript halflives
```sh
rnakinet-predict-halflives --transcriptome-bam <path_to_transcriptome_alignment.bam> --predictions <predictions_name.csv> --tl <experiment_tl> --output <halflives_name.csv>
```

The `--tl` parameter is the duration for which the cells were exposed to 5EU in hours

The `--predictions` parameter is the output file of the 5EU prediction step described above

This creates a csv file with columns `transcript` - the transcript identifier from your BAM file, `reads` - the amount of reads available for the given transcript, `percentage_modified` - the percentage of reads of the given transcript that were predicted to contain 5EU, `pred_t5` - the predicted halflife of the given transcript

### Example
```sh
rnakinet-predict-halflives --transcriptome-bam alignments/experiment/transcriptome_alignment.bam --predictions preds.csv --tl 2.0 --output halflives.csv
```

Note that the calculated halflives `pred_t5` are the most reliable for transcripts with high read count. 
The following plot shows correlation of halflives computed from RNAkinet predictions with experimentaly measured halflives [1] as we increase read count requirement.
We recommend users to acknowledge this and put more confidence in halflife predictions for transcripts with high read count, and less confidence for transcripts with low read count.

<img src="https://github.com/maragkakislab/rnakinet/assets/30112906/91b9e9e3-4a9b-4c1f-a0e2-9ef5eef045e7)" width="400" height="400">

[1] Eisen,T.J., Eichhorn,S.W., Subtelny,A.O., Lin,K.S., McGeary,S.E., Gupta,S. and Bartel,D.P.
(2020) The Dynamics of Cytoplasmic mRNA Metabolism. Mol. Cell, 77, 786-799.e10.



