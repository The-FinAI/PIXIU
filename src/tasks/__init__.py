from pprint import pprint
from typing import List, Union

import json
import lm_eval.base

from . import flare

TASK_REGISTRY = {
    "flare_fpb": flare.FPB,
    "flare_fiqasa": flare.FIQASA,
    "flare_ner": flare.NER,
    "flare_finqa": flare.FinQA,
    "flare_convfinqa": flare.ConvFinQA,
    "flare_headlines": flare.Headlines,
    "flare_finer_ord": flare.FinerOrd,
    "flare_fomc": flare.FOMC,
    "flare_german": flare.German,
    "flare_australian": flare.Australian,
    "flare_ectsum": flare.ECTSUM,
    "flare_edtsum": flare.EDTSUM,
    "flare_es_multifin": flare.ESMultiFin,
    "flare_es_efp": flare.ESEFP,
    "flare_es_efpa": flare.ESEFPA,
    "flare_es_fns": flare.ESFNS,
    "flare_es_tsa": flare.ESTSA,
    "flare_es_financees": flare.ESFinancees,
    **flare.SM_TASKS,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

_EXAMPLE_JSON_PATH = "split:key:/absolute/path/to/data.json"


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return

    def create_json_task():
        splits = task_name.split("=", 1)
        if len(splits) != 2 or not splits[1]:
            raise ValueError(
                "json tasks need a path argument pointing to the local "
                "dataset, specified like this: json="
                + _EXAMPLE_JSON_PATH
                + ' (if there are no splits, use "train")'
            )

        json_path = splits[1]
        if json_path == _EXAMPLE_JSON_PATH:
            raise ValueError(
                "please do not copy the example path directly, but substitute "
                "it with a path to your local dataset"
            )
        return lambda: json.JsonPerplexity(json_path)

    TASK_REGISTRY[task_name] = create_json_task()


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
