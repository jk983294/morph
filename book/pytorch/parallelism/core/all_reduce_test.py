import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    print('rank={}, size={}'.format(rank, size))
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # dist.broadcast(tensor, src, group): Copies tensor from src to all other processes.
    # dist.reduce(tensor, dst, op, group): Applies op to all tensor and stores the result in dst.
    # dist.all_reduce(tensor, op, group): Same as reduce, but the result is stored in all processes.
    # dist.scatter(tensor, scatter_list, src, group): Copies the i^th  tensor scatter_list[i] to the i^th process.
    # dist.gather(tensor, gather_list, dst, group): Copies tensor from all processes in dst.
    # dist.all_gather(tensor_list, tensor, group): Copies tensor from all processes to tensor_list, on all processes.
    # dist.barrier(group): block all processes in group until each one has entered this function.
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
