"""
FLARE
"""
from itertools import zip_longest
from lm_eval.base import MultipleChoiceTask, Task, rf
from lm_eval.metrics import mean, matthews_corrcoef
import numpy as np
from .utils import process_text
from seqeval.metrics import f1_score as entity_score
from sklearn.metrics import accuracy_score, f1_score


_CITATION = """
@misc{xie2023pixiu,
      title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance}, 
      author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
      year={2023},
      eprint={2306.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class FPB(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "chancefocus/flare-fpb"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": doc["question"],  # The query prompt.
            "choices": doc["options"],  # The list of choices.
            "gold": doc["gold"],  # The integer used to index into the correct element of `"choices"`.
        }

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "f1": (np.argmax(results), gold),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "f1": True,
        }

    def weighted_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(preds, golds, average='weighted', labels=[0, 1, 2])
        return f1

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "f1": self.weighted_fi,
        }


class FIQASA(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "chancefocus/flare-fiqasa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": doc["question"],  # The query prompt.
            "choices": doc["options"],  # The list of choices.
            "gold": doc["gold"],  # The integer used to index into the correct element of `"choices"`.
        }

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "f1": (np.argmax(results), gold),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "f1": True,
        }

    def weighted_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(preds, golds, average='weighted', labels=[0, 1, 2])
        return f1

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "f1": self.weighted_fi,
        }


class NER(Task):
    VERSION = 1
    DATASET_PATH = "chancefocus/flare-ner"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        cont_request = rf.greedy_until(ctx, {"until": "\n\n"})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        text = doc["text"]
        pred = process_text(results[0], text)

        return {
            "entity_f1": (pred, doc["label"], results[0])
        }

    def higher_is_better(self):
        return {
            "entity_f1": True,
        }

    @classmethod
    def entity_f1(cls, items):
        preds, golds, _ = zip(*items)
        f1 = entity_score(preds, golds)
        return f1

    def aggregation(self):
        return {
            "entity_f1": self.entity_f1,
        }


class FinQA(Task):
    VERSION = 1
    DATASET_PATH = "chancefocus/flare-finqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        cont_request = rf.greedy_until(ctx, {"until": "\n\n"})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        gold = doc["answer"]

        acc = 1.0 if results[0].strip() == gold else 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class StockMovement(MultipleChoiceTask):
    VERSION = 1
    # DATASET_PATH = "chancefocus/flare-fpb"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "mcc": (np.argmax(results), gold),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "mcc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "mcc": matthews_corrcoef,
        }


class StockMovementBigData(StockMovement):
    DATASET_PATH = "chancefocus/flare-sm-bigdata"

class StockMovementACL(StockMovement):
    DATASET_PATH = "chancefocus/flare-sm-acl"

class StockMovementCIKM(StockMovement):
    DATASET_PATH = "chancefocus/flare-sm-cikm"

SM_TASKS = {
    "flare_sm_bigdata": StockMovementBigData,
    "flare_sm_acl": StockMovementACL,
    "flare_sm_cikm": StockMovementCIKM,
}

class Headlines(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "chancefocus/flare-headlines"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": doc["question"],  # The query prompt.
            "choices": doc["options"],  # The list of choices.
            "gold": doc["gold"],  # The integer used to index into the correct element of `"choices"`.
        }

    def should_decontaminate(self):
         return True

    def doc_to_decontamination_query(self, doc):
         return doc["text"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        return {
            "avg_f1": (doc["label_type"], int(results[0] != "Yes"), gold, results),
        }

    def higher_is_better(self):
        return {
            "avg_f1": True,
        }

    @classmethod
    def label_avg(cls, items):
        labels, preds, golds, rels = zip(*items)
        label_set = set(labels)
        labels = np.array(labels)
        preds = np.array(preds)
        golds = np.array(golds)
        all_f1s = []
        for l in label_set:
            pds = preds[labels == l]
            gds = golds[labels == l]
            f1 = f1_score(pds, gds, average='weighted', labels=[0, 1])
            print (l, f1)
            all_f1s.append(f1)
        return np.mean(all_f1s)

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        cont_request = rf.greedy_until(ctx, {"until": "\n\n"})
        return cont_request

    def aggregation(self):
        return {
            "avg_f1": self.label_avg,
        }
