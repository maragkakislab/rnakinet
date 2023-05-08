# RNAModif
Project to detect 5eu-modified reads from raw nanopore sequencing signal

## Repository structure

[data_utils](rnamodif/data_utils/)
- Helper functions for processing and loading data

[inference](rnamodif/inference)
- Snakemake pipeline to predict fast5 reads

[preprocessing](rnamodif/preprocessing)
- Snakemake pipeline to preprocess files for training purposes

[training](rnamodif/training)
- Jupyter notebooks for individual training experiments and logging

[Result visualization](rnamodif/Result_visualization.ipynb)
- Jupyter notebook to visualize predictions

[Model](rnamodif/model.py)
- Architecture and optimization logic for the trained model


