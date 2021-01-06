#!/usr/bin/env bash
set -o nounset                              # Treat unset variables as an error
set -e


src_dir=$1
data_dir=$2
vocab_file=$3
do_lower_case=$4
output_base_dir=$5

max_seq_length=${6:-384}
max_query_length=${7:-64}
max_short_answers=${8:-10}
filter_null_doc=${9:-"true"}
keep_topk=${10:-50}
doc_stride=${11:-128}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

split_name="train"
json_file="${data_dir}/${split_name}-v2.0.json"

echo "Convert data into textline format"
output_dir="${output_base_dir}/topk_${keep_topk}_max-seq_${max_seq_length}_max-short-ans_${max_short_answers}_lower-case_${do_lower_case}"
mkdir -p ${output_dir}

python3 ${src_dir}/convert_example_to_textline.py \
  --vocab_file=${vocab_file} \
  --json_file=${json_file} \
  --split_name=${split_name} \
  --keep_topk=${keep_topk} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --max_short_answers=${max_short_answers} \
  --filter_null_doc=${filter_null_doc} \
  --doc_stride=${doc_stride} \
  ${case_flag} \
  --output_dir=${output_dir} |& tee ${output_dir}/log_${split_name}.log
