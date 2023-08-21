import time
from typing import Any, List
import copy
from gradio_client import Client
from tqdm import tqdm


class MultiClient(object):
    def __init__(self, worker_addrs, synced_worker=False) -> None:
        self.clients = [Client(addr) for addr in worker_addrs]
        self.synced_worker = synced_worker

    def predict(self, tasks: List[List], max_retries: int = 3) -> List[Any]:
        assert len(tasks) >= 1, "No predict tasks!"
        num_tasks = len(tasks)
        if self.synced_worker and len(tasks) % len(self.clients) != 0:
            num_dummy_tasks = len(self.clients) - len(tasks) % len(self.clients)            
            tasks.extend([copy.deepcopy(tasks[-1]) for _ in range(num_dummy_tasks)])

        pbar = tqdm(total=len(tasks))
        jobs = {
            client: (i, client.submit(*(tasks[i]), api_name="/predict"))
            for i, client in enumerate(self.clients)
            if i < len(tasks)
        }
        results = {}
        retries = {i: 0 for i in range(len(tasks))}

        while jobs:
            for client, (i, job) in list(jobs.items()):
                if job.done():
                    pbar.update(1)
                    del jobs[client]
                    try:
                        result = job.result()
                        results[i] = result
                    except Exception as e:
                        print("Job failed with error:", e)
                        if retries[i] < max_retries:
                            print("Retrying job...")
                            retries[i] += 1
                            new_job = client.submit(
                                *tasks[i], api_name="/predict")
                            jobs[client] = (i, new_job)
                            continue  # Skip the rest of the loop
                        else:
                            results[i] = None

                    new_i = len(results) + len(jobs)
                    if new_i < len(tasks):
                        new_task = tasks[new_i]
                        new_job = client.submit(
                            *new_task, api_name="/predict")
                        jobs[client] = (new_i, new_job)
            time.sleep(1)
        pbar.close()

        predicts = [results[i] for i in range(num_tasks)]

        return predicts
