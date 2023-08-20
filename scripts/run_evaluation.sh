pixiu_path='/data/hanweiguang/Projects/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,7"

# flare_fpb,flare_fiqasa,flare_finqa
# flare_german,flare_australian
python src/eval.py \
    --model hf-causal-experimental \
    --tasks flare_sm_cikm \
    --model_args use_accelerate=True,pretrained=EleutherAI/gpt-neox-20b,tokenizer=EleutherAI/gpt-neox-20b,use_fast=True,max_gen_toks=32,dtype=float16 \
    --batch_size 4 \
    --model_prompt 'no_prompt' \
    --num_fewshot 0 \
    --limit 2 \
    --write_out \
