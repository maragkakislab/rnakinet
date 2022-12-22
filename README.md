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
python3 rnamodif/evaluation/run.py --datadir <path> --outfile <file>
```
the --datadir argument is a folder containing your fast5 files (can be nested)
the --outfile is a csv file where you want to output the results

Example
```sh
python3 rnamodif/evaluation/run.py --datadir ../datastorage/experiment/fast5filesfolder/ --outfile results.csv
```

The resulting csv contains row for each READ_ID, and prediction if this read is m6a modified.

OPTIONAL

You can also tweak the batch size and number of workers processing the data.
There are two models available v1 and v2. You can specifiy them with the model parameter.
v1 is recommended, but feel free to experiment with v2
```sh
python3 rnamodif/evaluation/run.py --batchsize 128 --workers 10 --model v2 --datadir ........ --outfile .....
```