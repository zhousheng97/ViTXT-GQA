CUDA_VISIBLE_DEVICES=$1 python tools/run.py --tasks textvideoqa --datasets vtextgqa --model t2s --config configs/t2s_abinet.yml --seed 13 --save_dir save/$2  training_parameters.data_parallel True

# --resume_file $3 --resume True