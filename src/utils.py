import time
from typing import Any, List
from gradio_client import Client
from tqdm import tqdm


class MultiClient(object):
    def __init__(self, worker_addrs) -> None:
        self.clients = [Client(addr) for addr in worker_addrs]

    def predict(self, tasks: List[List]) -> List[Any]:
        pbar = tqdm(total=len(tasks))
        # Start the first batch of jobs
        jobs = {
            client: (i, client.submit(*(tasks[i]), api_name="/predict"))
            for i, client in enumerate(self.clients)
            if i < len(tasks)
        }

        # Initialize a dict for results
        results = {}

        while jobs:
            # Check each job's status
            for client, (i, job) in list(jobs.items()):
                if job.done():
                    pbar.update(1)
                    # If job is finished, remove it from the dict
                    del jobs[client]
                    # Get the result (or handle the error)
                    try:
                        result = job.result()
                        # Save the result at the right position
                        results[i] = result
                    except Exception as e:
                        print("Job failed with error:", e)
                    # Start a new job if there are still tasks left
                    if tasks:
                        new_i = len(results) + len(jobs)
                        if new_i < len(tasks):
                            new_task = tasks[new_i]
                            new_job = client.submit(
                                *new_task, api_name="/predict")
                            jobs[client] = (new_i, new_job)

            # Sleep for a while to avoid busy waiting
            time.sleep(0.1)
        pbar.close()

        # Sort results by task index
        predicts = [results[i] for i in sorted(results)]

        return predicts