#!/usr/bin/env bash
# This script contains sample run for evaluation.

set -o nounset                              # Treat unset variables as an error
set -e

src_dir=$1
model_ckpt=$2
model_config=$3
vocab_file=$4
data_dir=$5
output_base_dir=$6
exp_prefix=${7:-"uncased_bert_base"}
squad_version=${8:-"v2.0"}

batch_size=${9:-12}
max_seq_length=${10:-384}
max_query_length=${11:-64}
do_lower_case=${12:-"true"}

sum=${13:-"true"}
max_para=${14:-8}
use_doc_score=${15:-"true"}

test_run=${16:-"true"}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

if [ "${test_run}" = "true" ]; then
  output_dir="${output_base_dir}/eval_${exp_prefix}_docqa"
else
  rand_suffix=`shuf -i1-1000 -n1`
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  output_dir="${output_base_dir}/eval_${exp_prefix}_docqa_${DATESTAMP}_${rand_suffix}"
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
  echo "max_seq_length=${max_seq_length}"
  echo "max_query_length=${max_query_length}"
  echo "sum=${sum}"
  echo "max_para=${max_para}"
  echo "use_doc_score=${use_doc_score}"

)> ${output_dir}/exp.config

export PYTHONPATH="${src_dir}"
predict_file="${data_dir}/dev-${squad_version}.json"

# Evaluates the trained model.
python3 ${src_dir}/run_doc_qa.py \
  --vocab_file=${vocab_file} \
  --bert_config_file=${model_config} \
  --init_checkpoint=${model_ckpt} \
  --do_train=False \
  --do_predict=True \
  --predict_file=${predict_file} \
  --predict_batch_size=${batch_size} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --doc_stride=128 \
  ${case_flag} \
  --version_2_with_negative=${v2_w_neg} \
  --output_dir=${output_dir} |& tee ${output_dir}/eval_log.log

# Extracts answers from n-best.
python3 ${src_dir}/qa_utils/extract_answers.py \
  --nbest_file=${output_dir}/nbest_predictions.json \
  --predictions_file=${output_dir}/dev_predictions.json \
  --sum=${sum} \
  --max_para=${max_para} \
  --use_doc_score=${use_doc_score}

python3 ${src_dir}/qa_utils/triviaqa_evaluation.py \
  --dataset_file ${data_dir}/sample-wikipedia-dev.json \
  --prediction_file ${output_dir}/dev_predictions.json \
  --out_file ${output_dir}/dev.metrics

cat ${output_dir}/dev.metrics
