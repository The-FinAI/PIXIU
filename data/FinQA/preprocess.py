import json
import random
from tqdm import tqdm

from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

def preprocess_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def build_instructions(dataset, templates):
    results = []
    lengths = []

    for datum in tqdm(dataset):
        table = datum['table']
        table = "\n".join(["|"+"|".join(val)+"|" for val in table])
        pre_text = ' '.join(datum['pre_text'])
        post_text = ' '.join(datum['post_text'])
        context = f"Context: {pre_text}\n{table}\n{post_text}"
        question = datum['qa']['question']
        answer = str(datum['qa']['exe_ans'])
        template = random.choice(templates)
        template = "Please answer the given financial question based on the context."
        temp = {"id": f"finqa{len(results)}",
                "conversations": [{
                    "from": "human", "value": f"{template}\n{context}\nQuestion: {question}\nAnswer:",},
                    {"from": "assistant", "value": answer}]}
        tokens = tokenizer.encode(temp["conversations"][0]["value"])
        if len(tokens) > 2000:
            continue
        results.append(temp)
    return results

def dump_json(dataset, name):
    results = []
    for datum in dataset:
        results.append(json.dumps(datum))
    with open(f"data/FinQA/{name}.json", "w") as f:
        f.writelines("\n".join(results))

def dump_evaluate_data(data, dataset_name):
    for datum in data:
        datum["text"] = datum["conversations"][0]["value"].split("\n")[1]
        datum["label"] = datum["conversations"][1]["value"]
    d = [json.dumps(val) for val in data]
    with open(f"data/FinQA/{dataset_name}.jsonl", "w") as f:
        f.writelines("\n".join(d))

train_dataset = preprocess_data("data/FinQA/raw/train.json")
valid_dataset = preprocess_data("data/FinQA/raw/dev.json")
test_dataset = preprocess_data("data/FinQA/raw/test.json")

instructions = [
    "Given the financial data and expert analysis, please answer this question: ",
    "Can you provide the answer to this expert-authored finance question: ",
    "Here's a deep financial question based on a financial report, could you help answer it: ",
    "Considering the analysis of a large corpus of financial documents, please answer the following question: ",
    "A question has been posed by a financial expert. Can you provide an answer: ",
    "Based on the financial report data, please respond to this question: ",
    "Here's a question that requires understanding of complex numerical reasoning. Can you answer it: ",
    "Please provide a response to this finance-focused question based on a financial report: ",
    "Answer this question that requires deep understanding of financial data: ",
    "This is a complex multi-step numerical reasoning question. Can you provide the answer: "
]

train_dataset = build_instructions(train_dataset, instructions)
valid_dataset = build_instructions(valid_dataset, instructions)
test_dataset = build_instructions(test_dataset, instructions)
print (len(test_dataset))

dump_json(train_dataset, "train")
dump_json(valid_dataset, "valid")
dump_evaluate_data(test_dataset, "test")
