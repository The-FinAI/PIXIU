pixiu_path='/root/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

python src/eval.py \
    --model hf-causal-llama \
    --tasks flare_edtsum,flare_ectsum \
    --model_args use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full,use_fast=False,max_gen_toks=1024,dtype=float16 \
    --no_cache \
    --batch_size 4 \
    --model_prompt 'finma_prompt' \
    --num_fewshot 0 \
    --write_out 
