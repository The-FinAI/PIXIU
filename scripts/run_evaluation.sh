export PYTHONPATH='/path/to/PIXIU/src:/path/to/Projects/PIXIU/src/lm-evaluation-harness'

python src/eval.py \
    --model hf-causal-experimental \
    --tasks flare_fpb \
    --model_args use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full \
    --no_cache \
    --batch_size 20 \
    --num_fewshot 0 \
    --limit 100 \
    --write_out
