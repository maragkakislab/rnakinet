# RNAModif
RNAModif is a project dedicated to detecting 5eu-modified reads directly from the raw nanopore sequencing signal. Furthermore, it offers tools to calculate transcript decay rates.

# Usage
## Clone the repo
```sh
git clone https://github.com/maragkakislab/RNAModif.git -b release_v1
cd RNAModif
```
## Install dependencies
```sh
conda env create -f deploy.yaml
conda activate rnakinet_deploy
```
## Predict 5EU in your fast5 files
```sh
python3 rnamodif/workflow/scripts/inference_complete.py --path <path_to_folder_containing_fast5s> --out-csv <predictions_name.csv>
```
This creates a csv file with columns `read_id` - the read id, `5eu_mod_score` - the raw prediction score from 0 to 1, `5eu_modified_prediction` - Boolean column, True if the read is predicted to be modified by 5EU, False otherwise

### Example
```sh
python3 rnamodif/workflow/scripts/inference_complete.py --path my_data/experiment/fast5 --out-csv preds.csv
```

## Calculate transcript decay rates
```sh
python3 rnamodif/workflow/scripts/decay_rate_complete.py --transcriptome-bam <path_to_transcriptome_alignment.bam> --predictions <predictions_name.csv> --tl <your_tl> --output <halflives_name.csv>
```

The `--tl` parameter is the duration for which the cells were exposed to 5EU in hours

The `--predictions` parameter is the output file of the 5EU prediction step described above

This creates a csv file with columns `transcript` - the transcript identifier from your BAM file, `reads` - the amount of reads available for the given transcript, `percentage_modified` - the percentage of reads of the given transcript that were predicted to contain 5EU, `pred_t5` - the predicted halflife of the given transcript

### Example
```sh
python3 rnamodif/workflow/scripts/decay_rate_complete.py --transcriptome-bam alignments/experiment/transcriptome_alignment.bam --predictions preds.csv --tl 2.0 --output halflives.csv
```

Note that the calculated decay rates `pred_t5` are the most reliable for transcripts with at least 200 reads available

