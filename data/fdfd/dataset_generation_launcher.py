import os
from multiprocessing import Pool, Queue, Manager
import subprocess

script = 'data/fdfd/generate_mdm.py'

def metacoupler_launcher(queue):
    # While there are tasks in the queue, each process will fetch and execute one
    while not queue.empty():
        try:
            rand_seed, gpu_id = queue.get_nowait()  # Get task in order from the queue
            print("this is the random seed: ", rand_seed, flush=True)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            pres = ['python3', script]
            exp = [f"--random_seed={rand_seed}", f"--gpu_id={gpu_id}"]
            subprocess.call(pres + exp)
        except Exception as e:
            print(f"Error fetching task from queue: {e}")

# Wrapper function to allow passing `queue` without using a lambda
def worker_process(queue):
    metacoupler_launcher(queue)

if __name__ == "__main__":
    num_gpus = 4  # Number of GPUs
    # taskid_begin, taskid_end = (250, 260)
    # taskid_begin, taskid_end = (8, 16)
    # taskid_begin, taskid_end = (16, 24)
    # taskid_begin, taskid_end = (94, 98)
    task_list = [0]

    # Manager's queue allows inter-process communication for tasks
    manager = Manager()
    queue = manager.Queue()

    # Populate queue with tasks, ordered by task ID
    # for seed in range(taskid_begin, taskid_end):
    #     queue.put((seed, seed % num_gpus))
    for idx, seed in enumerate(task_list):
        queue.put((seed, idx % num_gpus))

    # Use Pool(20) to allow 20 concurrent processes fetching tasks from the queue
    with Pool(8) as p:
        # Each process runs `metacoupler_launcher`, pulling tasks from the queue
        p.map(worker_process, [queue] * 8)
