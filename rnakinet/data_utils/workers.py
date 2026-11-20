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

def worker_init_fn_experiments(worker_id):
    """Give each worker every num_workers-th record batch of every
    experiment, so the workers' shares are disjoint and each spans the whole
    run, and each worker streams from every experiment."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers

    for batches in dataset.experiments:
        mine = sorted(batches)[worker_id::total_workers]
        assert mine, 'fewer record batches in an experiment than workers'
        kept = {b: batches[b] for b in mine}
        # update in place: the dataset's generators use these dicts
        batches.clear()
        batches.update(kept)
