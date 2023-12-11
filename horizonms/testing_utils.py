import torch
from .engine import utils


def create_testing_dataloader(dataset_class, dataset_params,  
                      test_batch_size=None, workers=0, distributed=False):
    # Data loading code
    print("Loading data")  
    dataset_test = dataset_class(**dataset_params)
    print(f"#test = {len(dataset_test)}")    

    print("Creating testing data loaders")
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    if test_batch_size is None:
        test_batch_size = 1

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size,
        sampler=test_sampler, num_workers=2*workers,
        collate_fn=utils.collate_fn, pin_memory=True)
    
    return dataset_test, test_loader