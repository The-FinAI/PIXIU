import json

from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

def preprocess_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def build_instructions(dataset, templates, truncate=True, inference=False):
    results = []
    lengths = []
    for datum in dataset:
        context = f"{datum['annotation'].get('amt_pre_text', '')} {datum['annotation'].get('amt_table', '')} {datum['annotation']['amt_post_text']}"
        diags = []
        ans = []
        pros = []
        for index in range(len(datum['annotation']['dialogue_break'])):
            diags.append(f"{datum['annotation']['dialogue_break'][index]}")
            ans.append(f"{datum['annotation']['exe_ans_list'][index]}")
            pros.append(f"{datum['annotation']['exe_ans_list'][index]}")

        pre_diags = ""

        for index, (diag, an) in enumerate(zip(diags, ans)):
            template = templates[0]
            prompt = f"{template}\nContext: {context}\nConversations: {pre_diags}\nQuestion: {diag}\nAnswer:"
            pre_diags = pre_diags+f"\nq{index}: "+diag
            temp = {"id": f"convfinqa{len(results)}",
                    "conversations": [{
                        "from": "human", "value": prompt,},
                        {"from": "assistant", "value": an}]}
            pre_diags = pre_diags + " " + (an if not inference else "{answer"+str(index)+"}")
            tokens = tokenizer.encode(temp["conversations"][0]["value"])
            if inference:
                temp["qid"] = index
                if "qa" in datum:
                    temp["ori_question"] = datum.get("qa")["question"]
                else:
                    temp["ori_question"] = datum["qa_0"]["question"] + datum["qa_1"]["question"]
            if len(tokens) > 2000 and truncate:
                continue
            results.append(temp)
    return results

def dump_json(dataset, name):
    results = []
    for datum in dataset:
        results.append(json.dumps(datum))
    with open(f"data/ConvFinQA/{name}.json", "w") as f:
        f.writelines("\n".join(results))

def dump_evaluate_data(data, dataset_name):
    for datum in data:
        datum["text"] = datum["conversations"][0]["value"].split("\n")[1]
        datum["label"] = datum["conversations"][1]["value"]
    d = [json.dumps(val) for val in data]
    with open(f"data/ConvFinQA/{dataset_name}.jsonl", "w") as f:
        f.writelines("\n".join(d))

train_dataset = preprocess_data("data/ConvFinQA/raw/train.json")
test_dataset = preprocess_data("data/ConvFinQA/raw/dev.json")
train_size = int(len(train_dataset) * 0.8)
train_dataset, valid_dataset = train_dataset[:train_size], train_dataset[train_size:]

instructions = [
    "In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and posttext from a company's financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response:",
    "Considering the series of finance questions and their answers, can you answer the following final question: ",
    "Keeping in mind the context of previous questions and answers, please respond to this final finance question: ",
    "In light of the previous financial inquiries and their responses, please answer this final question: ",
    "Here's a series of finance-based questions. Using the context of previous questions and answers, please answer the final question: ",
    "In context of the given financial questions, please provide an answer to the final query: ",
    "Given the chain of financial questions and answers, please respond to the final question: ",
    "This is the last question in a series of finance-related questions. Use the previous questions as context to answer: ",
    "This is a series of financial queries, where later questions might depend on previous ones. Please answer the final question: ",
    "With the series of finance-related questions and answers as context, can you provide a response to the final question: "
]


train_dataset = build_instructions(train_dataset, instructions)
valid_dataset = build_instructions(valid_dataset, instructions)
test_dataset = build_instructions(test_dataset, instructions, False, True)
print (len(test_dataset))

dump_json(train_dataset, "train")
dump_json(valid_dataset, "valid")
dump_evaluate_data(test_dataset, "test")
