pixiu_path='/data/chenzhengyu/projects/PIXIU_fingpt/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,4"

python src/eval.py \
    --model hf-causal-vllm \
    --tasks flare_en_fintern \
    --model_args use_accelerate=True,pretrained=/data/chenzhengyu/my_belle/base_models/llama-2-7b-chat-hf,tokenizer=/data/chenzhengyu/my_belle/base_models/llama-2-7b-chat-hf,use_fast=False,max_gen_toks=1024,dtype=float16 \
    --no_cache \
    --batch_size 2 \
    --model_prompt 'finma_prompt' \
    --num_fewshot 0 \
    --write_out 
