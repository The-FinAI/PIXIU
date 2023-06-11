import json
import re

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
    f"http://127.0.0.1:{8888 + i}" for i in range(8)
]

clients = MultiClient(worker_addrs)

with open("data/FinQA/test.jsonl") as f:
    data = f.readlines()
    data = [json.loads(val) for val in data]

results = []
labels = []
texts = []
data = data
results = clients.predict(
    [[datum["conversations"][0]["value"]] + [GENERATION_CONFIG] for datum in data])
labels = [
    datum["label"] for datum in data
]
texts = [
    datum["text"]
    for datum in data
]
for re, la in zip(results, labels):
    print (re, la)

em = sum([1 if re == la else 0 for re, la in zip(results, labels)])
accuracy = em / len(results)

print("Accuracy: ", accuracy)
