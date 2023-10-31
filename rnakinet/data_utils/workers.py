import torch
import time
    
def worker_init_fn_inference(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
    
    pos_per_worker = len(dataset.files)//total_workers
    
    if(current_worker == total_workers -1): #Last worker
        dataset.files = dataset.files[pos_per_worker*current_worker:]
    else:
        dataset.files = dataset.files[pos_per_worker*current_worker: pos_per_worker*(current_worker+1)]
       
    assert(len(dataset.files)>0)
