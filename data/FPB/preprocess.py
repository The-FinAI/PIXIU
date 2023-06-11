import json
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name):
    with open(f"data/FPB/raw/{dataset_name}.json") as f:
        data = f.readlines()
    data = [json.loads(val) for val in data]
    data = [{k: v for k, v in val.items() if k != 'is_classification'} for val in data]
    for index, datum in enumerate(data):
        datum["id"] = f"fpb{index}"
        datum["conversations"] = datum["conversations:"]
        datum.pop("conversations:")
    return data

def dump_llama_data(data, dataset_name):
    d = [json.dumps(val) for val in data]
    with open(f"data/FPB/{dataset_name}.json", "w") as f:
        f.writelines("\n".join(d))

def dump_evaluate_data(data, dataset_name):
    for datum in data:
        datum["text"] = datum["conversations"][0]["value"].split("\n")[1]
        datum["label"] = datum["conversations"][1]["value"]
    d = [json.dumps(val) for val in data]
    with open(f"data/FPB/{dataset_name}.jsonl", "w") as f:
        f.writelines("\n".join(d))

prompts = [
    "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. For instance, 'The company's stocks plummeted following the scandal.' would be classified as negative.",
    "Determine the sentiment expressed in the following financial news sentence. Is it positive, negative, or neutral? For example, 'Despite initial concerns, the company reported better than expected quarterly earnings.' would be considered positive.",
    "Identify the sentiment in this sentence taken from a financial news piece. Indicate whether it is positive, negative, or neutral. An example of a neutral sentiment would be 'The company released its financial statements for the second quarter.'.",
    "What sentiment is conveyed in the provided sentence from a financial news report? Your options are positive, negative, or neutral. As an illustration, 'The company's recent investment in renewable energy has yielded impressive returns.' would be labeled as positive.",
    "Based on the sentence from the financial news report, identify whether the sentiment is positive, negative, or neutral. For instance, 'The tech firm's profits have been steadily declining this year.' would be considered negative.",
    "Evaluate the sentiment of the provided sentence from a financial news source. The sentiment could be positive, negative, or neutral. For example, 'The recent merger between the two leading firms had no significant impact on the market.' is a neutral sentiment.",
    "Examine the sentiment of this statement extracted from a financial news piece. Indicate whether it is positive, negative, or neutral. For example, 'The newly introduced government regulations are expected to boost the real estate market.' would be viewed as positive.",
    "Identify the sentiment in this sentence from a financial news report. Classify it as positive, negative, or neutral. For instance, 'The central bank's decision to increase interest rates has sparked concern among investors.' would be considered negative.",
    "Determine the sentiment of the provided sentence from a financial news article. Your answer should be either negative, positive, or neutral. As an example, 'Investors are unsure about the potential effects of the upcoming economic policy.' would be viewed as neutral.",
    "Assess the sentiment in the given sentence from a financial news publication. Is it positive, negative, or neutral? For example, 'The latest jobs report shows the economy is steadily recovering.' would be classified as positive."
]


id_ = 0

def shrink_data(data):
    samples = {}
    for val in data:
        text = val["conversations"][0]["value"].split("\n")[1]
        prompt = val["conversations"][0]["value"].split("\n")[0]
        label = val["conversations"][1]["value"]
        if text not in samples:
            samples[text] = []
        samples[text].append({"text": text, "label": label})

    samples = {key: val[0] for key, val in samples.items()}

    return samples.values()


def build_prompts(samples, id_):
    new_data = []
    for sample in samples:
        for prompt in prompts:
            temp = {"id": f"fpb{id_}", "conversations": [
                {"from": "human", "value": f"{prompt}\n{sample['text']}"},
                {"from": "agent", "value": f"{sample['label']}"},
            ]}
            new_data.append(temp)
            id_ += 1
    return new_data, id_



train_data = load_dataset("train")
train_data = shrink_data(train_data)
valid_data = load_dataset("valid")
valid_data = shrink_data(valid_data)
test_data = load_dataset("test")
test_data = shrink_data(test_data)
print (len(train_data), len(valid_data), len(test_data))
all_data = []
all_data.extend(train_data)
all_data.extend(valid_data)
all_data.extend(test_data)
data = all_data
print (len(all_data), all_data[0])

# Separate data by sentiment
positive_data = [item for item in data if item['label'] == 'positive']
negative_data = [item for item in data if item['label'] == 'negative']
neutral_data = [item for item in data if item['label'] == 'neutral']

# Split data into train and test sets
positive_train, positive_test = train_test_split(positive_data, test_size=277, random_state=42)
negative_train, negative_test = train_test_split(negative_data, test_size=116, random_state=42)
neutral_train, neutral_test = train_test_split(neutral_data, test_size=577, random_state=42)

# Combine splits into final train and test sets
train_data = positive_train + negative_train + neutral_train
test_data = positive_test + negative_test + neutral_test
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

print (len(train_data), len(test_data))

train_data, id_= build_prompts(train_data, id_)
valid_data, id_= build_prompts(valid_data, id_)
test_data, id_= build_prompts(test_data, id_)

dump_llama_data(train_data, "train")
dump_llama_data(valid_data, "valid")
dump_evaluate_data(test_data, "test")
