#!/usr/bin/env bash
set -o nounset                              # Treat unset variables as an error
set -e

base_path=$(pwd -P)

src_path="${base_path}/src"
output_path="${base_path}/outputs"
data_path="${base_path}/data"


src_mapped_path="/workspace/ds_doc_qa"
data_mapped_path="/data"
output_mapped_path="/results"

bert_dir="${data_mapped_path}/bert_base"
trivia_qa_dir="${data_mapped_path}/sample_data"
bert_base_vocab="${bert_dir}/vocab.txt"
bert_base_config="${bert_dir}/bert_config.json"
bert_base_ckpt="${bert_dir}/bert_model.ckpt"

converted_train_data_path="${output_mapped_path}/topk_8_max-seq_384_max-short-ans_10_lower-case_true"

# Defines the training command.
CMD="/bin/bash ${src_mapped_path}/script/train_bert_doc_qa.sh ${src_mapped_path} ${bert_base_ckpt} ${bert_base_config} ${bert_base_vocab} ${trivia_qa_dir} ${converted_train_data_path} ${output_mapped_path}/ckpt triviaqa_run v2.0 1 3e-4 3 384 adam 64 true 10 8 true h3-pos-mml 1.0 h2-pos-mml"

NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"0"}

docker run --rm \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -v ${src_path}:${src_mapped_path} \
    -v ${data_path}:${data_mapped_path} \
    -v ${output_path}:${output_mapped_path} \
    --name nvidia_tf \
    ds_doc_qa:v1.0 $CMD
