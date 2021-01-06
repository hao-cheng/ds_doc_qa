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
bert_base_ckpt="${bert_dir}/model.ckpt"


# Converts the data for DocQA model.
CMD="/bin/bash ${src_mapped_path}/script/bert_doc_data_conversion.sh ${src_mapped_path} ${trivia_qa_dir} ${bert_base_vocab} true ${output_mapped_path} 384 64 10 true 8 128"

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
