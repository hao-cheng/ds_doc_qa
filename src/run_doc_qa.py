# coding=utf-8
#!/usr/bin/env python
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified based on run_squad.py from the original BERT repo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import time
import six
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
import glob
try:
    from scipy.special import logsumexp
except:
    from scipy.misc import logsumexp

from utils.data_utils import (
    FeatureWriter,
    InputFeatureContainer,
    read_squad_examples_from_generator,
    convert_examples_to_features,
)

from utils.loss_helper import (
    doc_span_loss,
    doc_pos_loss,
    par_span_loss,
    par_pos_loss,
)


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("global_loss", None,
                    "The global objective")

flags.DEFINE_integer("anneal_steps", 20000,
                    "The number of annealing steps for switching from MML to Hard EM.")

flags.DEFINE_string("local_loss", None,
                    "The local objective")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string("train_file_dir", None,
                    "converted file path for training.")


flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "rand_seed", 12345, "The rand seed for shuffling during training")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("filter_null_doc", True,
                  "Whether to filter out no-answer document.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("max_num_doc_feature", 12,
                     "Max number of document features allowed.")

flags.DEFINE_integer("predict_batch_size", 12,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_float(
    "local_obj_alpha", 0.0,
    "Trades off between clean global and noisy local objectives.")

flags.DEFINE_bool(
    "label_cleaning", True,
    "Performs global label cleaning.")

flags.DEFINE_bool(
    "debug", False,
    "If true we process a tiny dataset.")

flags.DEFINE_bool(
    "posterior_distillation", False,
    "If true we distill teacher supervision in training.")

flags.DEFINE_string(
    "pd_loss", "sqerr",
    "The distance function between teacher and student predictions.")

flags.DEFINE_bool(
    "doc_normalize", False,
    "If true, performs document-level normalization."
)

flags.DEFINE_bool(
    "add_null_simple", False,
    "If true, we add the null span as possible in all cases.")


flags.DEFINE_integer("max_short_answers", 10,
                     "The maximum number of distinct short answer positions.")

flags.DEFINE_integer("max_num_answer_strings", 80,
                     "The maximum number of distinct short answer strings.")

flags.DEFINE_integer("max_paragraph", 4,
                     "The maximum numbr of paragraph allowed in a document.")

flags.DEFINE_string("no_answer_string", "",
                    "The string is used for as no-answer string.")

## Modified configuration parameters.
flags.DEFINE_string("device", "gpu",
                    "The main device is used for training.")

flags.DEFINE_integer("num_cpus", 4,
                     "The number of cpus is used.")

flags.DEFINE_string("initializer", "Xavier",
                    "The initializer is used for parameters.")

flags.DEFINE_bool(
    "shuffle_data", True,
    "If True, shuffles the training data locally.")



def _build_initializer(initializer):
    """Builds initialization method for the TF model."""
    if initializer == 'Uniform':
        tf.logging.info('Using random_uniform_initializer')
        tf_initializer = tf.random_uniform_initializer(
            -0.1, 0.1, dtype=tf.float32
        )
    elif initializer == 'Gaussian':
        tf.logging.info('Using truncated_normal_initializer')
        tf_initializer = tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32
        )
    elif initializer == 'Xavier':
        tf.logging.info('Using xavier_initializer')
        tf_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32
        )
    else:
        raise ValueError('Unknown initializer {0}!'.format(initializer))

    return tf_initializer


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def document_level_loss_builder(start_logits, start_positions_list,
                                end_logits, end_positions_list,
                                answer_positions_mask, answer_index_list,
                                seq_length, global_loss, null_ans_index=0):
    """Builds the loss function for the document-level QA model."""
    tf.logging.info("Using %s for global objective" % global_loss)

    # Here, we assume the null answer has index 0.
    not_null_ans = tf.cast(answer_index_list > null_ans_index, tf.int64)
    doc_answer_positions_mask = not_null_ans * answer_positions_mask
    positive_par_mask = tf.cast(
        tf.reduce_max(doc_answer_positions_mask, axis=-1), dtype=tf.float32)

    if global_loss == 'h2-pos-mml':
        # H2 document-level position-based MML loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h2_mml'
        )
    elif global_loss == 'h2-pos-hard_em':
        # H2 document-level position-based HardEM loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, seq_length,
            loss_type='h2_hard_em'
        )
    elif global_loss == 'h2-pos-hard_em_anneal':
        # H2 document-level position-based HardEM with annealing.
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h2_hard_em'
        ) + is_warmup * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h2_mml'
        )
    elif global_loss == 'h2-span-mml':
        # H2 document-level span-based MML loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h2-span-hard_em':
        # H2 document-level span-based HardEM loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_hard_em'
        )
    elif global_loss == 'h2-span-hard_em_anneal':
        # H2 document-level span-based HardEM with annealing.
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_hard_em'
        ) + is_warmup * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h3-span-mml':
        # H3 document-level span-based MML loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h3-span-hard_em':
        # H3 document-level span-based HardEM loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        )
    elif global_loss == 'h3-span-hard_em_anneal':
        # H3 docuemnt-level span-based HardEM with annealing:
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        ) + is_warmup * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h3-pos-mml':
        # H3 document-level position-based MML loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h3_mml'
        )
    elif global_loss == 'h3-pos-hard_em':
        # H3 document-level position-based HardEM loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h3_hard_em'
        )
    elif global_loss == 'h3-pos-hard_em_anneal':
        # H3 docuemnt-level position-based hardEM with annealing.
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h3_hard_em'
        ) + is_warmup * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h3_mml'
        )
    elif global_loss == 'h1':
        # H1 document-level loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, seq_length, loss_type='h1'
        )
    else:
        raise ValueError("Unknown global loss %s" % global_loss)

    return total_loss


def paragraph_level_loss_builder(start_logits, start_positions_list,
                                 end_logits, end_positions_list,
                                 answer_positions_mask, seq_length,
                                 local_loss):
    tf.logging.info("Using %s for local objective" % local_loss)

    if local_loss  == 'h2-pos-mml':
        # H2 paragraph-level position-based MML loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, seq_length,
            loss_type='h2_mml',
        )
    elif local_loss == 'h2-span-mml':
        # H2 paragraph-level span-based MML loss.
        total_loss = par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h2-pos-hard_em':
        # H2 paragraph-level pos-based HardEM loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, seq_length,
            loss_type='h2_hard_em',
        )
    elif local_loss == 'h2-pos-hard_em_anneal':
        # H2 paragraph-level pos-based HardEM loss with annealing.
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            seq_length, loss_type='h2_hard_em',
        ) + is_warmup * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            seq_length, loss_type='h2_mml',
        )
    elif local_loss == 'h2-span-hard_em':
        # Paragraph-level span-based marginalization loss.
        total_loss = par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            loss_type='h2_hard_em',
        )
    elif local_loss == 'h2-span-hard_em_anneal':
        # H2 paragraph-level span-based HardEM loss with annealing.
        global_step = tf.train.get_or_create_global_step()
        global_steps_int = tf.cast(global_step, tf.int32)
        is_warmup = tf.cast(global_steps_int < FLAGS.anneal_steps, tf.float32)
        total_loss = (1.0 - is_warmup) * par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            loss_type='h2_hard_em',
        ) + is_warmup * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h1':
        # Paragraph-level H1 loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, seq_length,
            loss_type='h1'
        )
    else:
        raise ValueError("Unknown local_loss %s" % local_loss)

    return total_loss


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            start_positions_list = features["start_position_list"]
            end_positions_list = features["end_position_list"]
            answer_positions_mask = features["position_mask"]

            total_loss = paragraph_level_loss_builder(
                start_logits,
                start_positions_list,
                end_logits,
                end_positions_list,
                answer_positions_mask,
                seq_length,
                FLAGS.local_loss,
            )

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["start_position_list"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )
        name_to_features["end_position_list"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )
        name_to_features["position_mask"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def doc_normalization(results, unique_id_to_qid):
    """Normalizes each RawResult using the qid."""

    qid_to_start_logits = collections.defaultdict(list)
    qid_to_end_logits = collections.defaultdict(list)

    for result in results:
        qid = unique_id_to_qid.get(result.unique_id, None)
        if not qid:
            raise ValueError("Unknown qid for unique_id {0}".format(
                result.unique_id))
        qid_to_start_logits[qid].extend(result.start_logits)
        qid_to_end_logits[qid].extend(result.end_logits)

    # Normalizes the scores.
    qid_to_doc_start_score = collections.defaultdict(float)
    qid_to_doc_end_score = collections.defaultdict(float)

    for qid in qid_to_start_logits:
        start_logits = qid_to_start_logits[qid]
        end_logits = qid_to_end_logits[qid]

        qid_to_doc_start_score[qid] = logsumexp(start_logits)
        qid_to_doc_end_score[qid] = logsumexp(end_logits)

    def normalize(scores, normalization_score):
        return [score - normalization_score for score in scores]

    def normalized_result_generator(result_list):
        """The generator function for normalizing the results."""
        for result in result_list:
            qid = unique_id_to_qid[result.unique_id]
            doc_start_score = qid_to_doc_start_score[qid]
            doc_end_score = qid_to_doc_end_score[qid]
            yield RawResult(
                unique_id=result.unique_id,
                start_logits=normalize(result.start_logits, doc_start_score),
                end_logits=normalize(result.end_logits, doc_end_score)
            )

    # Iterates for the second time to apply the document level normalization.
    new_results = [norm_result
                   for norm_result in normalized_result_generator(results)]

    return new_results


def compute_doc_norm_score(results, unique_id_to_qid):
    """Computes the document-level normalization score."""

    qid_to_start_logits = collections.defaultdict(list)
    qid_to_end_logits = collections.defaultdict(list)

    for result in results:
        qid = unique_id_to_qid.get(result.unique_id, None)
        if not qid:
            raise ValueError("Unknown qid for unique_id {0}".format(
                result.unique_id))
        qid_to_start_logits[qid].extend(result.start_logits)
        qid_to_end_logits[qid].extend(result.end_logits)

    # Normalizes the scores.
    qid_to_doc_score = collections.defaultdict(float)

    for qid in qid_to_start_logits:
        start_logits = qid_to_start_logits[qid]
        end_logits = qid_to_end_logits[qid]

        qid_to_doc_score[qid] = logsumexp(start_logits) + logsumexp(end_logits)

    unique_id_to_doc_score = dict([
        (result.unique_id, qid_to_doc_score[unique_id_to_qid[result.unique_id]])
        for result in results
    ])
    return unique_id_to_doc_score


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      prob_transform_func, unique_id_to_doc_score):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "doc_score"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        null_doc_score = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if FLAGS.version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
                    null_doc_score = unique_id_to_doc_score[result.unique_id]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            doc_score=unique_id_to_doc_score[result.unique_id]))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    doc_score=null_doc_score))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "doc_score"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    doc_score=pred.doc_score))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        doc_score=null_doc_score))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_score=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = prob_transform_func(total_scores)

        if not best_non_null_entry:
            tf.logging.info("No non-null guess")
            best_non_null_entry = _NbestPrediction(
                text="empty", start_logit=0.0, end_logit=0.0, doc_score=0.0
                )

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["doc_score"] = entry.doc_score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    tf.logging.info("Dumps predictions")
    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _compute_exp(scores):
    """Computes expoent score over normalized logits."""
    if not scores:
        return []

    return [math.exp(x) for x in scores]


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train and not FLAGS.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")

    if FLAGS.do_predict and not FLAGS.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

    if FLAGS.max_num_answer_strings < FLAGS.max_short_answers:
        raise ValueError(
            "The max_num_answer_strings (%d) must be bigger than "
            "max_short_answers (%d)" % (FLAGS.max_num_answer_strings,
                                        FLAGS.max_short_answers)
        )

    if FLAGS.local_obj_alpha > 0.0:
        tf.logging.info("Using local_obj_alpha=%f" % FLAGS.local_obj_alpha)


class DocQAModel(object):
    """Document QA model."""

    def __init__(self, bert_config, mode):
        # Makes those variables as local variables.
        self.max_seq_length = FLAGS.max_seq_length
        self.max_num_answers = FLAGS.max_short_answers
        self.max_num_answer_strings = FLAGS.max_num_answer_strings

        self._inputs, self._outputs = self.build_model(mode, bert_config)

        self._fetch_var_names = []
        self._fetch_var_names.append('loss_to_opt')
        if mode != 'TRAIN':
            self._fetch_var_names += ['start_logits', 'end_logits']

    def check_fetch_var(self):
        """Checks whether all variables to fetch are in the output dict."""
        # Checks whether required variables are in the outputs_.
        for var_name in self._fetch_var_names:
            if var_name not in self._outputs:
                raise ValueError(
                    '{0} is not in the output list'.format(var_name))

    def build_model(self, mode, bert_config, use_one_hot_embeddings=False):
        """Builds the model based on BERT."""
        input_ids = tf.placeholder(
            tf.int64, name='input_ids', shape=[None, self.max_seq_length]
        )
        input_mask = tf.placeholder(
            tf.int64, name='input_mask', shape=[None, self.max_seq_length]
        )
        segment_ids = tf.placeholder(
            tf.int64, name="segment_ids", shape=[None, self.max_seq_length]
        )

        is_training = (mode == 'TRAIN')

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        inputs = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

        return inputs, outputs

    def build_loss(self):
        """Builds loss variables."""

        start_logits = self._outputs["start_logits"]
        end_logits = self._outputs["end_logits"]

        input_shape = modeling.get_shape_list(self._inputs['input_ids'],
                                              expected_rank=2)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Builds input placeholder variables.
        start_positions_list = tf.placeholder(
            tf.int32, name="start_position_list",
            shape=[None, self.max_num_answers]
        )
        end_positions_list = tf.placeholder(
            tf.int32, name="end_position_list",
            shape=[None, self.max_num_answers]
        )
        answer_positions_mask = tf.placeholder(
            tf.int64, name="answer_positions_mask",
            shape=[None, self.max_num_answers]
        )

        answer_index_list = tf.placeholder(
            tf.int64, name="answer_index_list",
            shape=[None, self.max_num_answers]
        )

        # Builds the input-variable map.
        self._inputs['start_positions_list'] = start_positions_list
        self._inputs['end_positions_list'] = end_positions_list
        self._inputs['answer_positions_mask'] = answer_positions_mask
        self._inputs['answer_index_list'] = answer_index_list

        # ===============================
        # Document-level losses.
        # ===============================
        total_loss = document_level_loss_builder(
            start_logits,
            start_positions_list,
            end_logits,
            end_positions_list,
            answer_positions_mask,
            answer_index_list,
            seq_length,
            FLAGS.global_loss,
            null_ans_index=0,
        )

        # ===============================
        # Paragraph-level losses.
        # ===============================
        if FLAGS.local_obj_alpha > 0.0:
            total_loss += FLAGS.local_obj_alpha * paragraph_level_loss_builder(
                start_logits,
                start_positions_list,
                end_logits,
                end_positions_list,
                answer_positions_mask,
                seq_length,
                FLAGS.local_loss,
            )

        self._outputs['loss_to_opt'] = total_loss

    def build_opt_op(self, learning_rate, num_train_steps, num_warmup_steps,
                     use_tpu=False):
        """Builds optimization operator for the model."""
        loss_to_opt = self._outputs['loss_to_opt']

        return optimization.create_optimizer(
            loss_to_opt, learning_rate, num_train_steps, num_warmup_steps,
            use_tpu
        )

    def _run_model(self, session, feed_dict, opt_op):
        """Performans a forward and backward pass of the model."""

        fetches = [self._outputs[var_name]
                   for var_name in self._fetch_var_names]
        fetches.append(opt_op)

        all_outputs = session.run(fetches, feed_dict)

        fetched_var_dict = dict([
            (var_name, all_outputs[idx])
            for idx, var_name in enumerate(self._fetch_var_names)
        ])

        return fetched_var_dict

    def _build_feed_dict(self, inputs_dict):
        """Builds feed dict for inputs."""
        feed_dict_list = []
        for input_name, input_var in self._inputs.items():
            if input_name not in inputs_dict:
                raise ValueError('Missing input_name: {0}'.format(input_name))
            feed_dict_list.append((input_var, inputs_dict[input_name]))
        return dict(feed_dict_list)

    def one_step(self, session, inputs_dict, opt_op):
        """Trains, evaluates, or infers the model with one batch of data."""
        feed_dict = self._build_feed_dict(inputs_dict)
        fetched_dict = self._run_model(session, feed_dict, opt_op)

        return fetched_dict

    def initialize_from_checkpoint(self, init_checkpoint):
        """Initializes model variables from init_checkpoint."""
        variables_to_restore = tf.trainable_variables()

        initialized_variable_names = []
        if init_checkpoint:
            tf.logging.info(
                "Initializing the model from {0}".format(init_checkpoint))
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 variables_to_restore, init_checkpoint
             )

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in variables_to_restore:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


def run_epoch(model, session, data_container, learning_rate, opt_op, mode,
              model_saver=None, model_dir=None, verbose=True, eval_func=None,
              num_train_steps=None):
    """Runs one epoch over the data."""
    start_time = time.time()

    process_sample_cnt = 0

    # Starts a new epoch.
    data_container.start_epoch()

    total_loss = 0.0

    for iter_count, feature_dict in enumerate(data_container):

        fetched_dict = model.one_step(session, feature_dict, opt_op)
        total_loss += fetched_dict['loss_to_opt']

        process_sample_cnt += feature_dict['num_sample']

        if (iter_count) % 1000 == 0:
            if verbose:
                tf.logging.info(
                    'iter {:d}:, {:.3f} examples per second'.format(
                        iter_count,
                        process_sample_cnt / (time.time() - start_time)
                    )
                )
                tf.logging.info(
                    'loss {:.3f}'.format(fetched_dict['loss_to_opt'])
                )

            global_step = session.run(tf.train.get_global_step())
            if model_saver:
                model_saver.save(
                    session,
                    os.path.join(model_dir, 'model.ckpt'),
                    global_step=global_step
                )

            if num_train_steps is not None and global_step > num_train_steps:
                tf.logging.info(
                    'Reaches the num_train_steps {0}'.format(num_train_steps))
                break


    tf.logging.info(
        'time for one epoch: {:.3f} secs'.format(time.time() - start_time)
    )
    tf.logging.info('iters over {0} num of samples'.format(process_sample_cnt))

    eval_metric = {'total_loss': total_loss}
    output_dict = {}

    return total_loss, eval_metric, output_dict


def train_model(train_data_container, bert_config, learning_rate,
                num_train_steps, num_warmup_steps,
                init_checkpoint, rand_seed=12345):
    """ Training wrapper function."""
    if FLAGS.device == 'cpu':
        session_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=FLAGS.num_cpus,
            inter_op_parallelism_threads=FLAGS.num_cpus,
            allow_soft_placement=True
        )
    else:
        session_config = tf.ConfigProto(
            intra_op_parallelism_threads=FLAGS.num_cpus,
            inter_op_parallelism_threads=FLAGS.num_cpus,
            allow_soft_placement=True
        )
        session_config.gpu_options.allow_growth = True

    if not os.path.exists(FLAGS.output_dir):
        raise ValueError(
            'output_dir ({0}) does not exist!'.format(FLAGS.output_dir))

    model_dir = os.path.join(FLAGS.output_dir, "model_dir")
    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        tf.set_random_seed(rand_seed)
        np.random.seed(rand_seed)

        model = DocQAModel(bert_config, 'TRAIN')
        model_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)

        # This is needed for both TRAIN and EVAL.
        model.build_loss()
        model.check_fetch_var()

        # This operation is only needed for TRAIN phase.
        opt_op = model.build_opt_op(learning_rate, num_train_steps,
                                    num_warmup_steps)

        # Loads pretrain model parameters if specified.
        if init_checkpoint:
            model.initialize_from_checkpoint(init_checkpoint)

        session.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        for it in range(math.ceil(FLAGS.num_train_epochs)):
            tf.logging.info('Train Iter {0}'.format(it))

            _, train_metric, _ = run_epoch(
                model, session, train_data_container, None,
                opt_op, 'TRAIN', eval_func=None, model_saver=model_saver,
                model_dir=model_dir, num_train_steps=num_train_steps,
            )

            tf.logging.info('\n'.join([
                'train {}: {:.3f}'.format(metric_name, metric_val)
                for metric_name, metric_val in train_metric.items()
            ]))

            tf.logging.info('Saves the current model.')
            global_step = session.run(tf.train.get_global_step())
            model_saver.save(
                session,
                os.path.join(model_dir, 'model.ckpt'),
                global_step=global_step
            )

            if global_step > num_train_steps:
                tf.logging.info(
                    'Reaches the num_train_steps {0}'.format(num_train_steps))
                break

        tf.logging.info('Saves the final model.')
        global_step = session.run(tf.train.get_global_step())
        model_saver.save(
            session,
            os.path.join(model_dir, 'model.ckpt'),
            global_step=global_step
        )
        tf.logging.info('Training model done!')

    return True


def prediction_generator(estimator, predict_input_fn):
    """Given the input fn and estimator, yields one result."""
    for cnt, result in enumerate(estimator.predict(
            predict_input_fn, yield_single_examples=True)):
        if cnt % 1000 == 0:
            tf.logging.info("Processing example: %d" % cnt)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        yield RawResult(unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=10,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train and FLAGS.global_loss:

        # Prepare Training Data
        num_train_files = 0
        train_files = []
        for input_dir in FLAGS.train_file_dir.split(","):
            train_files.extend(glob.glob(os.path.join(input_dir, "*.tf_record_*")))

        tf.logging.info("Reading tfrecords from %s" % "\n".join(train_files))

        num_train_files = len(train_files)

        if num_train_files < 1:
            raise ValueError(
                "Can not find train files from %s" % FLAGS.train_file_dir)

        train_data_container = InputFeatureContainer(
            train_files, FLAGS.max_num_doc_feature, 1, True,
            FLAGS.rand_seed, single_pos_per_dupe=False,
            allow_null_doc=(not FLAGS.filter_null_doc),
        )

        num_train_features = train_data_container.num_sample
        num_train_steps = int(num_train_features * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num doc examples = %d", num_train_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_model(train_data_container=train_data_container,
                    bert_config=bert_config,
                    learning_rate=FLAGS.learning_rate,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                    init_checkpoint=FLAGS.init_checkpoint,
                    rand_seed=FLAGS.rand_seed)

    elif FLAGS.do_train and FLAGS.local_loss:
        # This is the training for paragraph-level models.
        train_examples = read_squad_examples_from_generator(
            input_file=FLAGS.train_file, is_training=True)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
        )

        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        # If TPU is not available, this will fall back to normal Estimator on
        # CPU or GPU.
        predict_batch_size = FLAGS.predict_batch_size

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=predict_batch_size
        )

        eval_examples = read_squad_examples_from_generator(
            input_file=FLAGS.predict_file, is_training=False)
        eval_record_filename = os.path.join(FLAGS.output_dir, "eval.tf_record")
        eval_feature_filename = os.path.join(FLAGS.output_dir, "eval.features")

        unique_id_to_qid = collections.defaultdict()
        tf.logging.info("Converting examples into records and features.")

        eval_writer = FeatureWriter(
            filename=eval_record_filename,
            is_training=False)

        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature,
            unique_id_to_qid=unique_id_to_qid
        )
        eval_writer.close()
        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_record_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = [
            raw_result
            for raw_result in prediction_generator(estimator, predict_input_fn)
        ]

        tf.logging.info("Done prediction!")
        del estimator
        del predict_input_fn

        if FLAGS.doc_normalize:
            tf.logging.info("Performs document level normalization.")
            normalized_results = doc_normalization(
                all_results, unique_id_to_qid
            )

            # For document-level normalization, each score is a log-prob.
            prob_trans_func = _compute_exp
        else:
            tf.logging.info("Performs paragraph level normalization.")
            normalized_results = all_results

            # For paragraph-level normalization, each score is a logit.
            prob_trans_func = _compute_softmax

        unique_id_to_doc_score = compute_doc_norm_score(all_results, unique_id_to_qid)
        output_prediction_file = os.path.join(
            FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            FLAGS.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            FLAGS.output_dir, "null_odds.json")

        write_predictions(eval_examples, eval_features, normalized_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file,
                          prob_trans_func, unique_id_to_doc_score)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
