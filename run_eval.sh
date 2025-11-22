#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run_eval.py \
    --gt_audio /path/to/folder/contains/gt_audio \
    --gt_cache /path/to/folder/to/save/gt_cache \
    --pred_audio /path/to/folder/contains/generated_audio \
    --pred_cache /path/to/folder/to/save/pred_cache \
    --clap_model_path /path/to/ckpt/laion_clap/xxx.pt \
    --audio_length=10 --pred_batch_size 32 --gt_batch_size 32 --num_workers 4 \
    --recompute_pred_cache --recompute_gt_cache