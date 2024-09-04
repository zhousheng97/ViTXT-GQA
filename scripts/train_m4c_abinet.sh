CUDA_VISIBLE_DEVICES=$1 python tools/run.py --tasks textvideoqa --datasets vtextgqa --model m4c --config configs/m4c_abinet.yml --seed 13 --save_dir save/$2  training_parameters.data_parallel True

# --resume_file $3 --resume True