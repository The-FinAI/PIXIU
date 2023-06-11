import json

def preprocess_conll_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    sentences = []
    sentence = []
    for line in lines:
        if line == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            word, pos, chunk, entity = line.strip().split()
            sentence.append((word, entity))
    if sentence:
        sentences.append(sentence)

    return extract_entities(sentences)


def extract_entities(sentences):
    sentence_words = []
    sentence_labels = []
    sentence_triplets = []

    for sentence in sentences:
        words = []
        labels = []
        entities = []
        current_entity_words = []
        current_label = None

        for i, (word, entity) in enumerate(sentence):
            words.append(word)

            if entity in ['I-LOC', 'I-PER', 'I-ORG']:
                if current_label == entity:
                    labels.append('I-' + entity[2:])
                    current_entity_words.append(word)
                else:
                    if current_entity_words:
                        entities.append((' '.join(current_entity_words), current_label[2:]))
                        labels[-len(current_entity_words):] = ['B-' + current_label[2:]] + ['I-' + current_label[2:]] * (len(current_entity_words) - 1)
                    current_entity_words = [word]
                    current_label = entity
                    labels.append('I-' + entity[2:])
            else:
                if current_entity_words:
                    entities.append((' '.join(current_entity_words), current_label[2:]))
                    labels[-len(current_entity_words):] = ['B-' + current_label[2:]] + ['I-' + current_label[2:]] * (len(current_entity_words) - 1)
                current_entity_words = []
                current_label = None
                labels.append('O')

        if current_entity_words:
            entities.append((' '.join(current_entity_words), current_label[2:]))
            labels[-len(current_entity_words):] = ['B-' + current_label[2:]] + ['I-' + current_label[2:]] * (len(current_entity_words) - 1)

        if any(b_label in labels for b_label in ['B-LOC', 'B-PER', 'B-ORG']):
            sentence_words.append(words)
            sentence_labels.append(labels)
            sentence_triplets.append(entities)

    return sentence_words, sentence_labels, sentence_triplets


def build_instructions(dataset, templates):
    results = []
    llama_results = []
    for text, labels, triplets in zip(*dataset):
        answer = "\n".join([", ".join([str(v) for v in val]) for val in triplets])
        for template in templates:
            temp = {"id": f"ner{len(results)}",
                    "conversations": [{
                        "from": "human", "value": f"{template}\nText: {' '.join(text)}\nAnswer:",},
                        {"from": "assistant", "value": answer}]}
            temp_one = dict(temp)
            temp_one["label"] = labels
            temp_one["text"] = ' '.join(text)
            llama_results.append(temp)
            results.append(temp_one)
    return results, llama_results

def dump_json(dataset, name):
    results = []
    for datum in dataset[0]:
        results.append(json.dumps(datum))
    with open(f"data/ner/{name}.jsonl", "w") as f:
        f.writelines("\n".join(results))
    results = []
    for datum in dataset[1]:
        results.append(json.dumps(datum))
    with open(f"data/ner/{name}.json", "w") as f:
        f.writelines("\n".join(results))

train_dataset = preprocess_conll_data("data/ner/FIN5.txt")
test_dataset = preprocess_conll_data("data/ner/FIN3.txt")
train_size = int(len(train_dataset[0]) * 0.8)
valid_dataset = (train_dataset[0][train_size:], train_dataset[1][train_size:], train_dataset[2][train_size:])
train_dataset = (train_dataset[0][:train_size], train_dataset[1][:train_size], train_dataset[2][:train_size])

instructions = [
    "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'. For instance, in \"Elon Musk, CEO of SpaceX, announced the launch from Cape Canaveral.\", the entities would be: 'Elon Musk, PER\nSpaceX, ORG\nCape Canaveral, LOC'",
    "When analyzing sentences from U.S. SEC filings, identify specific named entities indicating a person ('PER'), an organization ('ORG'), or a location ('LOC'). The answer should follow the format: 'entity name, entity type'. For example, in \"Microsoft, founded by Bill Gates and Paul Allen in Albuquerque.\", the entities are: 'Microsoft, ORG\nBill Gates, PER\nPaul Allen, PER\nAlbuquerque, LOC'",
    "In sentences from financial agreements found in U.S. SEC filings, your task is to identify entities signifying a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required format for the answer is: 'entity name, entity type'. For example, in \"Larry Page and Sergey Brin started Google in Menlo Park.\", the entities would be: 'Larry Page, PER\nSergey Brin, PER\nGoogle, ORG\nMenlo Park, LOC'",
    "In the sentences extracted from U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). Your answer should be in this format: 'entity name, entity type'. For instance, in \"Jeff Bezos announced the expansion of Amazon to Seattle.\", the entities would be: 'Jeff Bezos, PER\nAmazon, ORG\nSeattle, LOC'",
    "Analyze the sentences from financial agreements in U.S. SEC filings, and identify the specific named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required format for the answer is: 'entity name, entity type'. For example, in \"Tim Cook, CEO of Apple Inc, revealed a new store opening in San Francisco.\", the entities would be: 'Tim Cook, PER\nApple Inc, ORG\nSan Francisco, LOC'",
    "In sentences from U.S. SEC filings, your task is to identify entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The format for the answer should be: 'entity name, entity type'. For instance, in \"Mary Barra, CEO of General Motors, announced the construction of a new factory in Detroit.\", the entities would be: 'Mary Barra, PER\nGeneral Motors, ORG\nDetroit, LOC'",
    "When reading sentences from financial agreements in U.S. SEC filings, identify specific named entities indicating a person ('PER'), an organization ('ORG'), or a location ('LOC'). The answer should follow the format: 'entity name, entity type'. For example, in \"Mark Zuckerberg, founder of Facebook, announced new privacy measures from their headquarters in Menlo Park.\", the entities are: 'Mark Zuckerberg, PER\nFacebook, ORG\nMenlo Park, LOC'",
    "In sentences from financial agreements present in U.S. SEC filings, your goal is to identify entities representing a person ('PER'), an organization ('ORG'), or a location ('LOC'). The format for the answer is: 'entity name, entity type'. For example, in \"Warren Buffett, CEO of Berkshire Hathaway, announced a new acquisition at a press conference in Omaha, Nebraska.\", the entities would be: 'Warren Buffett, PER\nBerkshire Hathaway, ORG\nOmaha, LOC\nNebraska, LOC'",
    "In the sentences extracted from financial agreements in U.S. SEC filings, identify specific named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required format for the answer is: 'entity name, entity type'. For instance, in \"Jamie Dimon, CEO of JPMorgan Chase, commented on the financial outlook during a meeting in New York City.\", the entities would be: 'Jamie Dimon, PER\nJPMorgan Chase, ORG\nNew York City, LOC'",
    "In the sentences from financial agreements in U.S. SEC filings, find specific named entities that symbolize a person ('PER'), an organization ('ORG'), or a location ('LOC'). Answers should be provided in this format: 'entity name, entity type'. For example, in \"Satya Nadella, CEO of Microsoft, introduced a new initiative at the annual meeting in Redmond, Washington.\", the entities are: 'Satya Nadella, PER\nMicrosoft, ORG\nRedmond, LOC\nWashington, LOC'"
]
print (len(test_dataset[0]))

train_dataset = build_instructions(train_dataset, instructions)
valid_dataset = build_instructions(valid_dataset, instructions)
test_dataset = build_instructions(test_dataset, instructions)

dump_json(train_dataset, "train")
dump_json(valid_dataset, "valid")
dump_json(test_dataset, "test")
