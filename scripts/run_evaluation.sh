pixiu_path='/data/hanweiguang/Projects/PIXIU'
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,7"

# flare_fpb,flare_fiqasa,flare_finqa
# flare_german,flare_australian
# model_args是HuggingFaceAutoLM的参数，用`,`连接
# 下载模型`https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download`到`src/metrics/BARTScore/bart_score.pth`
python src/eval.py \
    --model hf-causal-experimental \
    --tasks flare_german,flare_australian,flare_sm_bigdata,flare_sm_acl,flare_sm_cikm \
    --model_args use_accelerate=True,pretrained=EleutherAI/gpt-neox-20b,tokenizer=EleutherAI/gpt-neox-20b,use_fast=True,max_gen_toks=64,dtype=float16 \
    --batch_size 4 \
    --model_prompt 'no_prompt' \
    --num_fewshot 0 \
    --limit 200 \
    --write_out \
