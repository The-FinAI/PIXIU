pixiu_path='/workspace/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

python src/eval.py \
    --model hf-causal-experimental \
    --tasks flare_edtsum,flare_ectsum \
    --model_args use_accelerate=True,pretrained=EleutherAI/gpt-neox-20b,tokenizer=EleutherAI/gpt-neox-20b,use_fast=True,max_gen_toks=1024,dtype=float16 \
    --no_cache \
    --batch_size 4 \
    --model_prompt 'no_prompt' \
    --num_fewshot 0 \
    --limit 200 \
    --write_out 
