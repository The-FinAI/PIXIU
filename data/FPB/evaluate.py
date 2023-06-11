from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score
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
    f"http://127.0.0.1:{8888 + i}" for i in range(4)
]

clients = MultiClient(worker_addrs)

def process_text(entity_string, text):
    # Initialize
    entity_list = [(", ".join(val.split(", ")[:-1]), val.split(", ")[-1]) for val in entity_string.split("\n")]
    text_words = text.split()
    labels = ['O'] * len(text_words)
    # text_lower = text.lower()
    text_lower = text

    # Create a list to store the start index of each word
    word_indices = [0]
    for word in text_words[:-1]:
        word_indices.append(word_indices[-1] + len(word) + 1)

    # Iterate over the entity list
    print (entity_list)
    for entity, entity_type in entity_list:
        entity_lower = entity

        # Find start and end index of each occurrence of the entity in the text
        start = 0
        while True:
            start = text_lower.find(entity_lower, start)
            if not entity or start == -1: break  # No more occurrence
            end = start + len(entity) - 1

            # Find the words included in this occurrence
            start_word = next(i for i, ind in enumerate(word_indices) if ind >= start)
            end_word = next(i for i, ind in enumerate(word_indices) if ind > end)

            # Label the words
            labels[start_word] = 'B-' + entity_type
            for i in range(start_word+1, end_word):
                labels[i] = 'I-' + entity_type

            # Move to the next character after the occurrence
            start = end + 1

    return labels

def extract_text_from_prompt(prompt):
    match = re.search(r'Text:\s*(.*?\s*)Answer:', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

with open("data/fpb/test.jsonl") as f:
    data = f.readlines()
    data = [json.loads(val) for val in data]

results = []
labels = []
texts = []
data = data[::10]
results = clients.predict(
    [[datum["conversations"][0]["value"]] + [GENERATION_CONFIG] for datum in data])
labels = [
    datum["label"] for datum in data
]
texts = [
    extract_text_from_prompt(datum["conversations"][0]["value"])
    for datum in data
]

accuracy = accuracy_score(results, labels)
print("Accuracy: ", accuracy)

# F1-Score
# Note: labels should be provided for multi-class/multi-label classification
f1 = f1_score(results, labels, average='weighted', labels=['positive', 'neutral', 'negative'])
print("F1-Score: ", f1)
