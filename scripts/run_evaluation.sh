pixiu_path='/root/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

python src/eval.py \
    --model hf-causal-vllm \
    --tasks flare_en_fintern \
    --model_args use_accelerate=True,pretrained=llama-2-7b-chat-hf,tokenizer=llama-2-7b-chat-hf,use_fast=False,max_gen_toks=1024,dtype=float16 \
    --no_cache \
    --batch_size 2 \
    --model_prompt 'finma_prompt' \
    --num_fewshot 0 \
    --write_out 
