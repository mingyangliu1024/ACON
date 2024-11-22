
CUDA_VISIBLE_DEVICES=0 python main.py \
 --experiment_description ACON \
 --run_description CAP \
 --da_method ACON \
 --dataset CAP \
 --num_runs 5 \
 --lr 0.001 \
 --cls_trade_off 1 \
 --domain_trade_off 1 \
 --entropy_trade_off 0.01 \
 --align_t_trade_off 1 \
 --align_s_trade_off 1