#!/bin/bash

set +x
set -eo pipefail

export MODEL='llama3.1-8b'
export bucket_name=llm_pruning_us_central2_b
export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}"

export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B'
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"

export CONVERTED_CHECKPOINT_PATH="gs://$bucket_name/model_ckpts/maxtext/${MODEL}"
export CONVERTED_CHECKPOINT="${CONVERTED_CHECKPOINT_PATH}/0/items"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

export PYTHONPATH='/home/zephyr/maxtext':$PYTHONPATH
python3 -u /home/zephyr/gcs-bucket/lm-evaluation-harness/scripts/test_orbax_eval.py \
    /home/zephyr/gcs-bucket/maxtext/MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test \
    per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=8192 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    attention="dot_product" \
    --hf_model_path=${HF_MODEL_PATH}

# decode example
idx=0
TOKENIZER='/home/zephyr/gcs-bucket/maxtext/assets/tokenizer_llama3.tiktoken'
# python3 -m MaxText.decode \
#     /home/zephyr/gcs-bucket/maxtext/MaxText/configs/base.yml \
#     load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     tokenizer_type=tiktoken \
#     tokenizer_path=$TOKENIZER \
#     per_device_batch_size=1 \
#     run_name=runner_$(date +%Y-%m-%d-%H-%M) \
#     max_prefill_predict_length=4 \
#     max_target_length=16 \
#     model_name=$MODEL \
#     dataset_type=synthetic \
#     async_checkpointing=false \
#     scan_layers=false \
#     attention=dot_product \
#     prompt="I love to" 

# python3 -m MaxText.decode MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken tokenizer_type=tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false scan_layers=false model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"
