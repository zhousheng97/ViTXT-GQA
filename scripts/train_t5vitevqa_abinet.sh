CUDA_VISIBLE_DEVICES=$1 python tools/run.py --tasks textvideoqa --datasets vtextgqa --model t5vitevqa --config configs/t5vitevqa_abinet.yml --seed 13 --save_dir save/$2  training_parameters.data_parallel True

# --resume_file $3 --resume True