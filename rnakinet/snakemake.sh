#!/bin/bash

source myconda
conda activate rnakinet_snakemake_base



snakemake -pr --keep-going --rerun-incomplete --latency-wait 120 --use-conda --use-envmodules -s Snakefile --profile snakemake_profile --rerun-triggers mtime --configfile config/config.yml --conda-frontend conda