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
    # taskid_begin, taskid_end = (0, 20)
    taskid_begin, taskid_end = (20, 40)
    # taskid_begin, taskid_end = (40, 60)
    # taskid_begin, taskid_end = (60, 80)
    # taskid_begin, taskid_end = (80, 96)
    # taskid_begin, taskid_end = (96, 100)

    # Create a list of tasks, each with a random seed and corresponding GPU
    tasks = [
        (seed, seed % num_gpus) for seed in range(taskid_begin, taskid_end)
    ]

    with Pool(24) as p:
        p.map(metacoupler_launcher, tasks)