# Probabilistic Assumptions Matter: Improved Models for Distantly-Supervised Document-Level Question Answering

This repository includes the codes and models for the paper
[Probabilistic Assumptions Matter: Improved Models for Distantly-Supervised Document-Level Question Answering](https://www.aclweb.org/anthology/2020.acl-main.501).
```
@InProceedings{Cheng2020ACL,
  author    = {Hao Cheng and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
  title     = {Probabilistic Assumptions Matter: Improved Models for Distantly-Supervised Document-Level Question Answering},
  booktitle = {Proc. Annual Conference of the Association for Computational Linguistics (ACL)},
  year      = {2020},
  pages     = {5657-–5667},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.501},
}
```

## Requirements
* Python >= 3.6
* Tensorflow 1.14

The code in this repo has been tested with Tensorflow 1.14 on a single V100-16GB.
We highly recommend using the docker file for creating the enviroment.
All the following sample commands are based on using docker.

Build the docker image:
```
cd docker_file
docker build -t ds_doc_qa:v1.0 -f Dockerfile.cuda11 .
```

In addition, please download the uncased bert-base model into `data/bert_base`.
```
cd data
unzip uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 bert_base
```

## Data Processing
We use the [data processing script](https://github.com/allenai/document-qa) for converting the [TriviaQA-Wiki split](http://nlp.cs.washington.edu/triviaqa) into the SQuAD format.
For all experiments in the paper, we use the top-8 paragraphs for each question as the input document.

For NarrativeQA, we use the processed data as in [Min et al. (2019)](https://www.aclweb.org/anthology/D19-1284).

Both datasets have to be converted in the SQuAD-V2 format for training. Please refer to folder `data/sample_data` for example.

## Data conversion for training document-level QA model.
In order to train the document-level QA model, we do an additional data conversion step to faciliate a sampling-based training.
```
bash doc_qa_data_conversion.sh
```
* A sample input based on TriviaQA-wiki is provided in `data/sample_data`.
* The folder `data/bert_base` should contain the BERT vocab file (`vocab.txt`).
* The converted data is located at `outputs/topk_8_max-seq_384_max-short-ans_10_lower-case_true`.

For details, please take a look at the data conversion script located at `src/script/bert_doc_data_conversion.sh`.

## Training
Run the following command to train the model on the converted sample data.
```
bash train_docqa.sh
```
* By default (a learning rate of `2e-4` for `2` epochs), it trains a QA model with the multi-objective formulation, i.e.,
  `H3-D position-based MML` + `H2-P position-based MML`.
* The model checkpoint is saved in `outputs/ckpt/triviaqa_run_docqa_test_run/model_dir`.

If you would like to try different distant supervision hypotheses discussed in our paper, please refer to `src/script/train_bert_doc_qa.sh`.
* `global_loss`: see available document-level losses in the function `document_level_loss_builder` in `src/run_doc_qa.py`
* `local_loss`: see available paragraph-level losses defined in the function `paragraph_level_loss_builder` in `src/run_doc_qa.py`.

## Inference & Evaluation
To evaluate the trained model, run the following eval script to perform the inference and evaluation on the sampled dev data.
```
bash eval_docqa.sh
```

By default, the eval script carries out inference with `SUM` inference with document-level probabilities, i.e. aggreting probabilties of the same string over all paragraphs for ranking.
* If you would like to try the `MAX` inference, you can set the argument `sum=false` in the evaluation script `src/script/eval_bert_doc_qa.sh`.
* If paragraph-level probabilities are preferred, set the argument `use_doc_score=false` in the evaluation script `src/script/eval_bert_doc_qa.sh`.
