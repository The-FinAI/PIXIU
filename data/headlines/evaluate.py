import numpy as np
from sklearn.metrics import f1_score
import json

from src.utils import MultiClient

GENERATION_CONFIG = [
    0.1,  # int | float (numeric value between 0 and 1) in 'Temperature' Slider component
    0.75,  # int | float (numeric value between 0 and 1) in 'Top p' Slider component
    40,  # int | float (numeric value between 0 and 100) in 'Top k' Slider component
    1,  # int | float (numeric value between 1 and 4) in 'Beams Number' Slider component
    512,  # int | float (numeric value between 1 and 2000) in 'Max New Tokens' Slider component
    1,  # int | float (numeric value between 1 and 300) in 'Min New Tokens' Slider component
    1.2,  # int | float (numeric value between 1.0 and 2.0) in 'Repetition Penalty' Slider component
]

worker_addrs = [
    f"http://127.0.0.1:{8888 + i}" for i in range(4)
]

clients = MultiClient(worker_addrs)

with open("data/headlines/test.jsonl") as f:
    data = f.readlines()
    data = [json.loads(val) for val in data]

results = []
labels = []
texts = []
data = data
results = clients.predict(
    [[datum["conversations"][0]["value"]] + [GENERATION_CONFIG] for datum in data]
)
labels = [
    datum["label"] for datum in data
]
texts = [
    datum["text"]
    for datum in data
]

types = [datum["label_type"] for datum in data]

label_results = {}

for rel, lab, typ in zip(results, labels, types):
    if typ not in label_results:
        label_results[typ] = ([], [])
    label_results[typ][0].append(rel)
    label_results[typ][1].append(lab)

all_f1s = []
for key, val in label_results.items():
    print (key)
    f1 = f1_score(val[0], val[1], average='weighted', labels=['Yes', 'No'])
    all_f1s.append(f1)
    print("F1-Score: ", f1)

print ("Average F1:", np.mean(all_f1s))
