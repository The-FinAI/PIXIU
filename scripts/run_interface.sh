export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export PYTHONPATH='...'

model_name_or_path='...'
ckpt_path='...'

python src/interface.py \
    --model_name_or_path $model_name_or_path \
    --ckpt_path $ckpt_path \
    --llama \
    --local_rank $1
    # --use_lora \
