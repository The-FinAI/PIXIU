def no_prompt(ctx):
    return ctx

def finma_prompt(ctx):
    return f'Human: \n{ctx}\n\nAssistant: \n'

MODEL_PROMPT_MAP = {
    "no_prompt": no_prompt,
    "finma_prompt": finma_prompt
}