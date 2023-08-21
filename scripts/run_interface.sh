export CUDA_VISIBLE_DEVICES='0,1,2,3,4,7'
export PYTHONPATH='.../PIXIU/src'

model_name_or_path='...'

python src/interface.py \
    --model_name_or_path $model_name_or_path \
    --llama
