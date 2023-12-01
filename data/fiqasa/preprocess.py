import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data(filename):
    data = pd.read_csv(f"data/fiqasa/raw/{filename}")
    score = data["score"].tolist()
    label = []
    for s in score:
        if s < -0.1:
            label.append("Negative")
        elif s < 0.1:
            label.append("Neutral")
        else:
            label.append("Positive")
    data["label"] = label
    return data

def dump_llama_data(data, dataset_name):
    d = [json.dumps(val) for val in data]
    with open(f"data/fiqasa/{dataset_name}.json", "w") as f:
        f.writelines("\n".join(d))

def dump_evaluate_data(data, dataset_name):
    for datum in data:
        datum["text"] = datum["conversations"][0]["value"].split("\n")[1]
        datum["label"] = datum["conversations"][1]["value"]
    d = [json.dumps(val) for val in data]
    with open(f"data/fiqasa/{dataset_name}.jsonl", "w") as f:
        f.writelines("\n".join(d))

prompts = [
    "What is the sentiment of the following financial {category}: Positive, Negative, or Neutral?",
    "Given this financial {category}, how would you classify its sentiment: Positive, Negative, or Neutral?",
    "Can you identify the sentiment expressed in this financial {category}: Is it Positive, Negative, or Neutral?",
    "Could you classify the sentiment of the financial {category}: Is it Positive, Negative, or Neutral?",
    "Analyze this financial {category} and categorize its sentiment: Positive, Negative, or Neutral?",
    "What type of sentiment does this financial {category} convey: Positive, Negative, or Neutral?",
    "Based on this financial {category}, what's the sentiment: Positive, Negative, or Neutral?",
    "Assess this financial {category} and determine its sentiment: Positive, Negative, or Neutral?",
    "What sentiment is represented in this financial {category}: Positive, Negative, or Neutral?",
    "Given the sentiment of this financial {category}, is it Positive, Negative, or Neutral?"
]

id_ = 0

def build_instructions(data, id_, with_type=False):
    results = []
    for index, row in data.iterrows():
        text = row["sentence"]
        cate = row["type"]
        ans = row["label"]
        for prompt in prompts:
            ptext = prompt.format(category=cate)
            temp = {"id": f"fiqasa{id_}", "conversations": [
                {"from": "human", "value": ptext+"\nText: "+text+"\nAnswer:"},
                {"from": "agent", "value": ans}]}
            results.append(temp)
            id_ += 1
    return results, id_


train_data = load_data("train.csv")
valid_data = load_data("valid.csv")
test_data = load_data("test.csv")

df = pd.concat([train_data, valid_data, test_data])

negative_df = df[df['label'] == 'Negative']
positive_df = df[df['label'] == 'Positive']
neutral_df = df[df['label'] == 'Neutral']

negative_df = shuffle(negative_df, random_state=42)
positive_df = shuffle(positive_df, random_state=42)
neutral_df = shuffle(neutral_df, random_state=42)

# Now, split each class into train and test
negative_train = negative_df.iloc[76:]  # 76 is the number of desired test instances
negative_test = negative_df.iloc[:76]
positive_train = positive_df.iloc[141:]  # 141 is the number of desired test instances
positive_test = positive_df.iloc[:141]
neutral_train = neutral_df.iloc[18:]  # 18 is the number of desired test instances
neutral_test = neutral_df.iloc[:18]

train_df = pd.concat([negative_train, positive_train, neutral_train])
test_data = pd.concat([negative_test, positive_test, neutral_test])

train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)

train_data, id_ = build_instructions(train_data, id_)
valid_data, id_ = build_instructions(valid_data, id_)
test_data, id_ = build_instructions(test_data, id_)

dump_llama_data(train_data, "train")
dump_llama_data(valid_data, "valid")
dump_evaluate_data(test_data, "test")
