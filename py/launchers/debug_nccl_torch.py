"""
Independent, non-JAX cross-check: does torch.distributed's NCCL all_reduce
give the correct result across the local GPUs? If this ALSO gives a wrong
answer, that's airtight confirmation the problem is NCCL/hardware/driver,
not anything in JAX or our training code.

Run with torchrun (NOT plain python):
    torchrun --standalone --nproc_per_node=4 launchers/debug_nccl_torch.py
"""

import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # rank 0 -> 1.0, rank 1 -> 2.0, rank 2 -> 3.0, rank 3 -> 4.0
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")

    sum_tensor = value.clone()
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)

    mean_tensor = value.clone()
    dist.all_reduce(mean_tensor, op=dist.ReduceOp.AVG)

    expected_sum = sum(range(1, world_size + 1))
    expected_mean = expected_sum / world_size

    print(
        f"[rank {rank}] local_value={value.item()}  "
        f"all_reduce SUM={sum_tensor.item()} (expected {expected_sum})  "
        f"all_reduce AVG={mean_tensor.item()} (expected {expected_mean})  "
        f"SUM correct={sum_tensor.item() == expected_sum}  "
        f"AVG correct={abs(mean_tensor.item() - expected_mean) < 1e-6}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
