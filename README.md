# RNAkinet

RNAkinet detects reads labeled with a nucleoside analog (5-ethynyluridine, 5EU, or 4-thiouridine, 4SU; 4SU models also apply to 4-thiouracil, 4TU) directly from the raw nanopore sequencing signal. Furthermore, it offers tools to calculate transcript halflives.

# Usage

## Installation

```sh
pip install rnakinet
```

RNAkinet requires Python 3.10 or newer.

On older NVIDIA GPUs (e.g. V100), install a CUDA 12 build of PyTorch first: the default PyTorch build uses CUDA 13, which no longer supports them.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install rnakinet
```

## Predict labeled reads in POD5 files

```sh
rnakinet-inference --path <pod5_file_or_directory> --model-name <model_name> --output <predictions_name.csv>
```

This creates a csv file with columns `read_id` - the read id, `mod_score` - the raw prediction score from 0 to 1, `mod_prediction` - Boolean column, True if the read is predicted to be labeled by the model's analog (5EU or 4SU), False otherwise. Earlier versions named these columns `5eu_mod_score` and `5eu_modified_prediction`; `rnakinet-predict-halflives` accepts both.

Nvidia GPU is recommended to run this command. If you want to run inference on a CPU-only machine, use the `--use-cpu` option. This will substantially increase runtime.

FAST5 input is not currently supported. Support will be reimplemented in a future release; for now, inference can still be run by converting Fast5 reads to POD5 first.

Pass one or more POD5 files or directories via `--path`. Use `--model-name` to select a packaged pretrained model for your analog and flow-cell chemistry, for example `r9_5EU_v1.0` or `r10_5EU_v2.0`; `rnakinet-inference --help` lists all packaged models. The packaged models currently detect 5EU; other models (e.g. 4SU) can be used with `--model-path`.

Users who have trained their own RNAkinet models can use `--model-path` (must be used in conjunction with `--arch`) to run inference on custom models in place of `--model-name`.


## Calculate transcript halflives

```sh
rnakinet-predict-halflives --transcriptome-bam <path_to_transcriptome_alignment.bam> --predictions <predictions_name.csv> --tl <experiment_tl> --output <halflives_name.csv>
```

The `--tl` parameter is the duration for which the cells were exposed to the analog (5EU or 4SU) in hours

The `--predictions` parameter is the output file of the prediction step described above

This creates a csv file with columns `transcript` - the transcript identifier from your BAM file, `reads` - the amount of reads available for the given transcript, `percentage_modified` - the percentage of reads of the given transcript that were predicted to be labeled, `pred_t5` - the predicted halflife of the given transcript

### Example

```sh
rnakinet-predict-halflives --transcriptome-bam alignments/experiment/transcriptome_alignment.bam --predictions preds.csv --tl 2.0 --output halflives.csv
```

Note that the calculated halflives `pred_t5` are the most reliable for transcripts with high read count. 

The following plots show correlation of halflives computed from RNAkinet predictions with experimentaly measured halflives [1] as we increase read count requirement.
We recommend users to acknowledge this and put more confidence in halflife predictions for transcripts with high read count, and less confidence for transcripts with low read count.

<img src="https://github.com/user-attachments/assets/b01b062a-1b64-4de4-b076-fb0c4ebb84e8" width="400" height="400">
<img src="https://github.com/user-attachments/assets/1c2c0dd7-3ee4-4806-81d0-a757b3ac8dcd" width="400" height="400">

[1] Eisen,T.J., Eichhorn,S.W., Subtelny,A.O., Lin,K.S., McGeary,S.E., Gupta,S. and Bartel,D.P. (2020) The Dynamics of Cytoplasmic mRNA Metabolism. Mol. Cell, 77, 786-799.e10.

# Model training

The models are trained and evaluated with a separate Snakemake workflow,
which is not part of this package.

# Cite

Vlastimil Martinek, Jessica Martin, Cedric Belair, Matthew J Payea, Sulochan Malla, Panagiotis Alexiou, Manolis Maragkakis, Deep learning and direct sequencing of labeled RNA captures transcriptome dynamics, NAR Genomics and Bioinformatics, Volume 6, Issue 3, September 2024, lqae116, https://doi.org/10.1093/nargab/lqae116
