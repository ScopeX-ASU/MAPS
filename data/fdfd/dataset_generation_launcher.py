import os
from multiprocessing import Pool
import subprocess

script = 'data/fdfd/generate_metacoupler.py'

def metacoupler_launcher(args):
    rand_seed, gpu_id = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pres = [
        'python3',
        script
    ]
    exp = [
        f"--random_seed={rand_seed}",
        f"--gpu_id={gpu_id}"
        ]
    # subprocess.call(pres + exp, stderr=wfid, stdout=wfid)
    subprocess.call(pres + exp)


if __name__ == "__main__":
    num_gpus = 4  # Number of GPUs
    # taskid_begin, taskid_end = (0, 8)
    # taskid_begin, taskid_end = (8, 16)
    # taskid_begin, taskid_end = (16, 32)
    # taskid_begin, taskid_end = (32, 48)
    # taskid_begin, taskid_end = (48, 64)
    # taskid_begin, taskid_end = (64, 80)
    taskid_begin, taskid_end = (80, 90)
    # taskid_begin, taskid_end = (90, 100)

    # Create a list of tasks, each with a random seed and corresponding GPU
    tasks = [
        (seed, seed % num_gpus) for seed in range(taskid_begin, taskid_end)
    ]
    # tasks = [
    #     (0, 0 % num_gpus)
    # ]

    with Pool(24) as p:
        p.map(metacoupler_launcher, tasks)