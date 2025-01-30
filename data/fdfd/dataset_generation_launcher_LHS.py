import os
from multiprocessing import Pool, Queue, Manager
import subprocess
import random
from scipy.stats import qmc
import numpy as np
import h5py

script = 'data/fdfd/generate_bending_LHS.py'

def generate_lhs_samples(n_samples, n_dimensions, lower_bounds=None, upper_bounds=None, seed=None):
    """
    Generate Latin Hypercube Samples (LHS) with specified dimensions and sample size.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - n_dimensions (int): Number of dimensions for each sample.
    - lower_bounds (array-like, optional): Lower bounds for each dimension. Defaults to 0 for all dimensions.
    - upper_bounds (array-like, optional): Upper bounds for each dimension. Defaults to 1 for all dimensions.
    - seed (int or np.random.Generator, optional): Seed for reproducibility. Defaults to None.

    Returns:
    - samples (np.ndarray): Array of shape (n_samples, n_dimensions) containing the LHS samples.
    """
    # Initialize the LatinHypercube sampler
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)

    # Generate samples in the unit hypercube
    unit_samples = sampler.random(n=n_samples)

    # If bounds are not specified, default to [0, 1] for all dimensions
    if lower_bounds is None:
        lower_bounds = np.zeros(n_dimensions)
    if upper_bounds is None:
        upper_bounds = np.ones(n_dimensions)

    # Scale samples to the specified bounds
    samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return samples

def metacoupler_launcher(queue):
    # While there are tasks in the queue, each process will fetch and execute one
    while not queue.empty():
        try:
            rand_seed, gpu_id, port_len, init_weight_idx = queue.get_nowait()  # Get task in order from the queue
            print("this is the random seed: ", rand_seed, flush=True)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            pres = ['python3', script]
            exp = [
                f"--random_seed={rand_seed}", 
                f"--gpu_id={gpu_id}",
                f"--port_len={port_len}",
                f"--init_weight_idx={init_weight_idx}",
            ]
            subprocess.call(pres + exp)
        except Exception as e:
            print(f"Error fetching task from queue: {e}")

# Wrapper function to allow passing `queue` without using a lambda
def worker_process(queue):
    metacoupler_launcher(queue)

if __name__ == "__main__":
    num_gpus = 4  # Number of GPUs
    # taskid_begin, taskid_end = (0, 615)
    # taskid_begin, taskid_end = (615, 1230)
    # taskid_begin, taskid_end = (0, 1230)
    taskid_begin, taskid_end = (0, 1230)
    # taskid_begin, taskid_end = (10025, 10050)
    # int(region_s * res) + 1
    assert taskid_begin == 0, "taskid_begin must be 0"
    port_len_options = list(range(160, 180, 2))
    port_lens = [round(length / 100, 2) for length in port_len_options]
    num_per_port_len = (taskid_end - taskid_begin) // len(port_lens)
    target_size = 256 / 50
    init_weight_dict = {}
    for i in range(len(port_lens)):
        region_s = round(target_size - 2 * port_lens[i], 2)
        dim = (int(region_s * 25) + 1) ** 2
        lower_bounds = [-0.2] * dim
        upper_bounds = [0.2] * dim
        lhs_samples = generate_lhs_samples(num_per_port_len, dim, lower_bounds, upper_bounds, seed=42)
        init_weight_dict[port_lens[i]] = lhs_samples

    # store the dict to a h5 file
    with h5py.File('./data/fdfd/init_weight_dict.h5', 'w') as f:
        for key, value in init_weight_dict.items():
            f.create_dataset(str(key), data=value)
    

    # Manager's queue allows inter-process communication for tasks
    manager = Manager()
    queue = manager.Queue()
    # Populate queue with tasks, ordered by task ID
    for seed in range(taskid_begin, taskid_end):
        # queue.put((seed, seed % num_gpus, port_len))
        port_len_index = (seed - taskid_begin) // num_per_port_len
        port_len = port_lens[port_len_index]
        init_weight_idx = (seed - taskid_begin) % num_per_port_len
        queue.put((seed, seed % num_gpus, port_len, init_weight_idx))  # Convert init_weight to a list for subprocess


    with Pool(20) as p:
        # Each process runs `metacoupler_launcher`, pulling tasks from the queue
        p.map(worker_process, [queue] * 20)
