import numpy as np
import re

invalid_ppl_mentions = [
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question"
]

def remove_citation(text):
    # text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r"\s*\[\d+\]\s*","", text)
    if text.startswith("According to , "):
        text = text.replace("According to , ", "According to the search results, ")
    return text

def is_invalid_ppl(text):
    return np.any([text.lower().startswith(mention.lower()) for mention in invalid_ppl_mentions])

def is_invalid_paragraph_ppl(text):
    return len(text.strip())==0 or np.any([mention.lower() in text.lower() for mention in invalid_ppl_mentions])

def perplexity_ai_abstain_detect(generation):
    output = remove_citation(generation)
    if is_invalid_ppl(output):
        return True
    valid_paras = []
    for para in output.split("\n\n"):
        if is_invalid_paragraph_ppl(para):
            break
        valid_paras.append(para.strip())

    if len(valid_paras) == 0:
        return True
    else:
        return False

def generic_abstain_detect(generation):
    return generation.startswith("I'm sorry") or "provide more" in generation

def is_response_abstained(generation, fn_type):
    if fn_type == "perplexity_ai":
        return perplexity_ai_abstain_detect(generation)

    elif fn_type == "generic":
        return generic_abstain_detect(generation)

    else:
        return False

