test_model_prefix=path # Examples: logs/ACON/UCIHAR/22_11_2024_20_12_22

CUDA_VISIBLE_DEVICES=0 python main.py \
 --experiment_description ACON \
 --run_description UCIHAR \
 --da_method ACON \
 --dataset UCIHAR \
 --num_runs 5 \
 --lr 0.01 \
 --cls_trade_off 1 \
 --domain_trade_off 1 \
 --entropy_trade_off 0.01 \
 --align_t_trade_off 1 \
 --align_s_trade_off 1 \
 --phase test \
 --test_model_prefix ${test_model_prefix}