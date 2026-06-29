import torch

def worker_init_fn_inference(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers

    current_worker = worker_id
    pos_per_worker = len(dataset.files) // total_workers

    if current_worker == total_workers - 1:  # Last worker
        dataset.files = dataset.files[pos_per_worker * current_worker :]
    else:
        dataset.files = dataset.files[pos_per_worker * current_worker : pos_per_worker * (current_worker + 1)]
    assert len(dataset.files) > 0

def worker_init_fn_train(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers

    for pod5, readids in dataset.pod5_to_readids.items():
        ids_per_worker = len(readids) // total_workers
        old_ids = dataset.pod5_to_readids[pod5]
        if worker_id == total_workers - 1:  # Last worker
            dataset.pod5_to_readids[pod5] = old_ids[ids_per_worker * worker_id :]
        else:
            dataset.pod5_to_readids[pod5] = old_ids[ids_per_worker * worker_id : ids_per_worker * (worker_id + 1)]
        assert len(dataset.pod5_to_readids[pod5]) > 0
