import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    print('rank={}, size={}'.format(rank, size))
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)  # blocking send
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)  # non-blocking send
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)  # blocking recv
        print('Rank ', rank, ' recv data ', tensor[0])
        req = dist.irecv(tensor=tensor, src=0)  # non-blocking recv
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
