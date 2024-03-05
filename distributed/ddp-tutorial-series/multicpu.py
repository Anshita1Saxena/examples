import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

# Wrapper around Python's native multiprocessing GPU
import torch.multiprocessing as mp
# Modules takes input data and distributes it across all the GPUs
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time


'''
Script for single CPU setup where batchsize=32 remains the same, 
and 64 steps are divided into 8 steps in the distributed setup, 
so training load is divided across 8 cores of single CPU.
Since this machine has single CPU with 8 cores, so we have a 8 logging statement 
for every epoch.
'''


def ddp_setup(rank, world_size):
    """
    Setting group is necessary so that all the process can discover
    and communicate with each other.
    Args:
        rank: Unique identifier of each process (0, world_size-1)
        world_size: Total number of processes 
    """
    # Running rank-0 process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id,
        save_every: int,
        rank
    ) -> None:
        self.gpu_id = gpu_id
        self.rank = rank
        # self.model = model.to(rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # Wrapping it via DDP for CPUs
        self.model = DDP(model)
        # Wrapping it via DDP for GPUs
        # self.model = DDP(model, device_ids=[self.rank])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[CPU: {self.rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            '''
            Adding the condition self.rank==0 helps in removing the unnecessary 
            redundancy because all of the copies are the same. 
            It will only collect the checkpoint from only one process- CPU 0
            '''
            if self.rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    '''
    Sampler ensures that the input batch is chunked across 
    all the CPUs without any overlapping samples.
    '''
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, train_data, optimizer, device, save_every, rank)
    start_time = time.time()
    trainer.train(total_epochs)
    # To cleanly exit from the training group
    end_time = time.time()
    print(f'CPU: {rank} took Total time: {end_time - start_time} seconds')
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    # world_size = torch.cuda.device_count()
    world_size = os.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Takes a function and spawns that across all of processes in the distributed group
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
