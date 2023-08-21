export CUDA_VISIBLE_DEVICES='0,1,2,3,4,7'
export PYTHONPATH='/data/hanweiguang/Projects/PIXIU/src'

model_name_or_path='ChanceFocus/finma-7b-trade'

python src/interface.py \
    --model_name_or_path $model_name_or_path \
    --llama
