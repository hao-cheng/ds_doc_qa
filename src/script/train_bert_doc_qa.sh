#!/usr/bin/env bash
# This script contains sample run for SQUAD.

set -o nounset                              # Treat unset variables as an error
set -e

src_dir=$1
model_ckpt=$2
model_config=$3
vocab_file=$4
data_dir=$5
converted_train_file_path=$6
output_base_dir=$7
exp_prefix=${8:-"uncased_bert_base"}
squad_version=${9:-"v2.0"}

batch_size=${10:-12}
learning_rate=${11:-3e-5}
num_train_epochs=${12:-2.0}
max_seq_length=${13:-384}

optimizer_type=${14:-"adam"}
max_query_length=${15:-64}
do_lower_case=${16:-"true"}

max_short_answers=${17:-10}
max_num_doc_feature=${18:-8}
filter_null_doc=${19:-"true"}

global_loss=${20:-"h3-pos-mml"}
local_obj_alpha=${21:-0.0}
local_loss=${22:-"h2-pos-mml"}

test_run=${23:-"true"}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

if [ "${test_run}" = "true" ]; then
  output_dir="${output_base_dir}/${exp_prefix}_docqa_test_run"
else
  rand_suffix=`shuf -i1-1000 -n1`
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  output_dir="${output_base_dir}/${exp_prefix}_docqa_test_run"
fi

mkdir -p ${output_dir}

# For document-level QA, there are always negative passages.
v2_w_neg="true"

(
  echo "src_dir=${src_dir}"
  echo "vocab_file=${vocab_file}"
  echo "bert_config_file=${model_config}"
  echo "init_checkpoint=${model_ckpt}"
  echo "batch_size=${batch_size}"
  echo "learning_rate=${learning_rate}"
  echo "num_train_epochs=${num_train_epochs}"
  echo "max_seq_length=${max_seq_length}"
  echo "max_query_length=${max_query_length}"
  echo "global_loss=${global_loss}"
  echo "local_obj_alpha=${local_obj_alpha}"
  echo "local_loss=${local_loss}"
  echo "max_num_doc_feature=${max_num_doc_feature}"
  echo "max_short_answers=${max_short_answers}"

)> ${output_dir}/exp.config

export PYTHONPATH="${src_dir}"

train_file="${data_dir}/train-${squad_version}.json"


python3 ${src_dir}/run_doc_qa.py \
  --vocab_file=${vocab_file} \
  --bert_config_file=${model_config} \
  --init_checkpoint=${model_ckpt} \
  --do_train=True \
  --train_file=${train_file} \
  --train_file_dir=${converted_train_file_path} \
  --do_predict=False \
  --debug=False \
  --train_batch_size=${batch_size} \
  --predict_batch_size=${batch_size} \
  --learning_rate=${learning_rate} \
  --num_train_epochs=${num_train_epochs} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --max_num_doc_feature=${max_num_doc_feature} \
  --max_short_answers=${max_short_answers} \
  --filter_null_doc=${filter_null_doc} \
  --doc_stride=128 \
  --global_loss=${global_loss} \
  --local_obj_alpha=${local_obj_alpha} \
  --local_loss=${local_loss} \
  ${case_flag} \
  --version_2_with_negative=${v2_w_neg} \
  --output_dir=${output_dir} |& tee ${output_dir}/log.log
