import os
import subprocess
from multiprocessing import Manager, Pool, Queue

# script = "data/fdfd/generate_mdm.py"
# script = "data/fdfd/generate_wdm.py"
# script = "data/fdfd/generate_tdm.py"
# script = "data/fdfd/generate_bending.py"
# script = "data/fdfd/generate_crossing.py"
# script = "data/fdfd/generate_od.py"
script = "data/fdfd/generate_mmi.py"


def metacoupler_launcher(queue):
    # While there are tasks in the queue, each process will fetch and execute one
    while not queue.empty():
        try:
            rand_seed, gpu_id, each_step, include_perturb = (
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
    num_gpus = 4  # Number of GPUs
    # taskid_begin, taskid_end = (0, 615)
    # taskid_begin, taskid_end = (615, 1230)
    # taskid_begin, taskid_end = (25, 50)
    # taskid_begin, taskid_end = (25, 50)
    taskid_begin, taskid_end = (0, 1)

    # task_list = [16, 37]

    # Manager's queue allows inter-process communication for tasks
    manager = Manager()
    queue = manager.Queue()
    each_step = 0
    include_perturb = 0
    # Populate queue with tasks, ordered by task ID
    for seed in range(taskid_begin, taskid_end):
        queue.put((seed, seed % num_gpus, each_step, include_perturb))
    # for idx, seed in enumerate(task_list):
    #     queue.put((seed, idx % num_gpus, each_step, include_perturb))

    with Pool(4) as p:
        # Each process runs `metacoupler_launcher`, pulling tasks from the queue
        p.map(worker_process, [queue] * 4)
