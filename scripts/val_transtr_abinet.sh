CUDA_VISIBLE_DEVICES=$1 python tools/run.py --tasks textvideoqa --datasets vtextgqa --model transtr --config configs/transtr_abinet.yml --save_dir save/$2 --resume_file $3 --run_type $4  training_parameters.data_parallel True 
#  --evalai_inference 1