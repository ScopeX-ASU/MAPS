import argparse
import os
import subprocess
from multiprocessing import Manager, Pool, Queue

import torch


def metacoupler_launcher(queue):
    # While there are tasks in the queue, each process will fetch and execute one
    while not queue.empty():
        try:
            rand_seed, gpu_id, each_step, include_perturb, script = (
                queue.get_nowait()
            )  # Get task in order from the queue
            print("this is the random seed: ", rand_seed, flush=True)
            print("this is the gpu id: ", gpu_id, flush=True)
            print("this is the each step: ", each_step, flush=True)
            print("this is the include perturb: ", include_perturb, flush=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            pres = ["python3", script]
            exp = [
                f"--random_seed={rand_seed}",
                f"--gpu_id={gpu_id}",
                f"--each_step={each_step}",
                f"--include_perturb={include_perturb}",
            ]
            subprocess.call(pres + exp)
        except Exception as e:
            print(f"Error fetching task from queue: {e}")


# Wrapper function to allow passing `queue` without using a lambda
def worker_process(queue):
    metacoupler_launcher(queue)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--device-name",
        type=str,
        default="bending",
        help="device name, e.g., bending, crossing, mdm, etc. Default is bending.",
    )
    argparse.add_argument(
        "--num-devices",
        type=int,
        default=8,
        help="number of photonic devices to generate. Default is 8.",
    )
    argparse.add_argument(
        "--num-gpus", type=int, default=4, help="number of GPUs to use. Default is 4."
    )
    argparse.add_argument(
        "--each-step",
        action="store_true",
        default=False,
        help="whether to save each step of the device generation. Default is False.",
    )
    argparse.add_argument(
        "--include-perturb",
        action="store_true",
        default=False,
        help="whether to apply permittivity perturbations in the device generation. Default is False.",
    )

    args = argparse.parse_args()

    num_gpus = args.num_gpus  # Number of GPUs
    assert num_gpus <= torch.cuda.device_count(), "num_gpus exceeds available GPUs"
    each_step = int(args.each_step)
    include_perturb = int(args.include_perturb)
    taskid_begin, taskid_end = (0, args.num_devices)
    script = f"data/fdfd/generate_{args.device_name}.py"

    # Manager's queue allows inter-process communication for tasks
    manager = Manager()
    queue = manager.Queue()

    # Populate queue with tasks, ordered by task ID
    for seed in range(taskid_begin, taskid_end):
        queue.put((seed, seed % num_gpus, each_step, include_perturb, script))

    with Pool(4) as p:
        # Each process runs `metacoupler_launcher`, pulling tasks from the queue
        p.map(worker_process, [queue] * 4)
