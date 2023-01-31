# RNAModif
## How to run m6a modification detection

### Install the package

Run the following commands to prepare for running the script
```sh
python3 -m venv virtualenv
source virtualenv/bin/activate
git clone https://github.com/MartinekV/RNAModif.git
cd RNAModif/
git clone https://github.com/biodlab/RODAN.git
pip install -e .
pip install -r RODAN/requirements.txt
pip install -r requirements.txt
```

Then run the following command from the RNAModif/ folder
```sh
python3 rnamodif/evaluation/run.py --datadir <path> --outfile <file> --model <model>
```
the --datadir argument is a folder containing your fast5 files (can be nested)

the --outfile is a csv file where you want to output the results

the --model argument is a choice of a trained model. Options are: m6a_v3 (for m6a), 5eu_v1 (for 5eu), s4u_v1 (for s4u)

Example
```sh
python3 rnamodif/evaluation/run.py --datadir ../datastorage/experiment/fast5filesfolder/ --outfile results.csv --model s4u_v1
```

The resulting csv contains row for each READ_ID, and prediction for whether this read is modified or not.

OPTIONAL

You can also tweak the batch size and number of workers processing the data.
```sh
python3 rnamodif/evaluation/run.py --batchsize 128 --workers 10 --model .... --datadir ........ --outfile .....
```

### Limitations
Currently the method only predicts modifications for the whole read, not specific positions.