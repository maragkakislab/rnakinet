#TODO preprocess fast5 files into numpy arrays for faster loading

from rnamodif.data_utils.datamap import experiment_files
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from pathlib import Path
import os
import paramiko

def init_client(password):
    ssh = paramiko.SSHClient()
    server = 'skirit.ics.muni.cz'
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username='martinekv', password=password)
    return ssh

    # cmd_to_execute = 'mkdir nanopore/test'
    # ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)


def run_prep(experiment, password):
    ssh_client = init_client(password)
    
    files = experiment_files[experiment]
    ssh_client.exec_command(f'mkdir nanopore_custom/{experiment}')
    
    for fast5 in files:
        print(f'{fast5.stem} file begin')
        with get_fast5_file(fast5, mode='r') as f5:
            filestem = fast5.stem
            ssh_client.exec_command(f'mkdir nanopore_custom/{experiment}/{filestem}')
            for read in f5.get_reads():
                raw = read.get_raw_data(scale=True)
                
                prefix = Path(f'../../meta/martinekv/nanopore_custom/{experiment}/{filestem}')
                
                filename = Path(f'{prefix}/{experiment}:{filestem}:{read.read_id}.npy')
                with open(filename, 'wb') as f:
                    np.save(f, raw)
    print('DONE')

    
    