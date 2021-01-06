#!/usr/bin/env python
"""This file contains data help for Document-level QA."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import itertools
import json
import random

import tensorflow as tf
import numpy as np
try:
    from scipy.special import logsumexp
except:
    from scipy.misc import logsumexp

import tokenization


FLAGS = tf.flags.FLAGS


class SquadExample(object):
    """A single training/test example for simple sequence classification.
    Each example has one question, the corresponding answer(s), and a single
     paragraph as the evidence.

     For examples without an answer, the start and end position are -1.
    """

    __slots__ = ('qas_id', 'question_text', 'doc_tokens',
                 'orig_answer_text', 'start_position', 'end_position',
                 'orig_answer_text_list', 'start_position_list',
                 'end_position_list', 'is_impossible', 'qid', 'title',)

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 qid=None,
                 title=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False,
                 orig_answer_text_list=None,
                 start_position_list=None,
                 end_position_list=None):
        self.qas_id = qas_id

        if qid:
            # For TriviaQA, there is question id which can be used for document
            # level normalization.
            self.qid = qid
        else:
            self.qid = qas_id

        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

        self.is_impossible = is_impossible

        self.title = title

        # This is an extension to support weak supervisions.
        self.orig_answer_text_list = orig_answer_text_list
        self.start_position_list = start_position_list
        self.end_position_list = end_position_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        if self.start_position and self.start_position_list:
            s += ", start_positions: [{0}]".format(
                ",".join(self.start_position_list))
        else:
            s += ", start_positions: []"
        return s


def convert_string_key_to_int_key(orig_dict):
    """Converts the key from string to integer."""
    return dict([(int(key), val) for key, val in orig_dict.items()])


class InputFeatures(object):
    """A single set of features of data."""

    __slots__ = ('unique_id', 'example_index', 'doc_span_index', 'tokens',
                 'token_to_orig_map', 'token_is_max_context', 'input_ids',
                 'input_mask', 'segment_ids', 'qid', 'start_position',
                 'end_position', 'is_impossible', 'start_position_list',
                 'end_position_list', 'position_mask', 'answer_index_list',
                 'num_answer')

    __dict_attr__ = ('token_to_orig_map', 'token_is_max_context',)

    def __init__(self,
                 unique_id=None,
                 example_index=None,
                 doc_span_index=None,
                 tokens=None,
                 token_to_orig_map=None,
                 token_is_max_context=None,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,
                 qid=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 start_position_list=None,
                 end_position_list=None,
                 answer_index_list=None,
                 position_mask=None):

        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        # self.is_impossible = is_impossible

        # Here, we still assume answer index 0 stands for null.
        if answer_index_list:
            self.is_impossible = all_null_answer(answer_index_list, 0)
        else:
            self.is_impossible = True

        # Adds the qid for normalization.
        if not qid:
            raise ValueError("qid can not be None!")

        self.qid = qid

        if position_mask:
            self.start_position_list = start_position_list
            self.end_position_list = end_position_list
            self.position_mask = position_mask
            self.answer_index_list = answer_index_list
        else:
            # Pads the raw start and end positions.
            (self.start_position_list, self.end_position_list,
             self.position_mask, self.answer_index_list
            ) = InputFeatures.make_fixed_length(
                start_position_list, end_position_list, answer_index_list
            )

        # Truncates possible answers if there are too many possibles.
        self.num_answer = 0
        if self.start_position_list:
            self.num_answer = sum(self.position_mask)

    def to_json(self):
        """Serializes the object into the json form."""
        return json.dumps(
            dict([(attr, getattr(self, attr)) for attr in self.__slots__]))

    @classmethod
    def load_from_json(cls, json_string):
        """Loads the object from json string."""
        attr_dict = json.loads(json_string)
        return cls(
            unique_id=attr_dict["unique_id"],
            example_index=attr_dict["example_index"],
            doc_span_index=attr_dict["doc_span_index"],
            tokens=attr_dict["tokens"],
            token_to_orig_map=convert_string_key_to_int_key(
                attr_dict["token_to_orig_map"]),
            token_is_max_context=convert_string_key_to_int_key(
                attr_dict["token_is_max_context"]),
            input_ids=attr_dict["input_ids"],
            input_mask=attr_dict["input_mask"],
            segment_ids=attr_dict["segment_ids"],
            qid=attr_dict["qid"],
            start_position=attr_dict["start_position"],
            end_position=attr_dict["end_position"],
            is_impossible=attr_dict["is_impossible"],
            start_position_list=attr_dict["start_position_list"],
            end_position_list=attr_dict["end_position_list"],
            answer_index_list=attr_dict["answer_index_list"],
            position_mask=attr_dict["position_mask"]
        )


    @staticmethod
    def make_fixed_length(start_position_list, end_position_list,
                          answer_index_list):
        """Returns three fixed length lists: start, end, and mask."""
        if start_position_list is None:
            return None, None, None, None

        # Truncates possible answers if there are too many possibles.
        len_ans = min(len(start_position_list), FLAGS.max_short_answers)

        # Initializes all lists.
        position_mask = [1 if kk < len_ans else 0
                         for kk in range(FLAGS.max_short_answers)]
        start_positions = list(position_mask)
        end_positions = list(position_mask)
        answer_indices = list(position_mask)

        for ii, (start, end, ans_ind) in enumerate(itertools.islice(
                zip(start_position_list, end_position_list,
                               answer_index_list), len_ans)):
            start_positions[ii] = start
            end_positions[ii] = end
            answer_indices[ii] = ans_ind

        if not start_position_list:
            raise ValueError("No answer positions!")

        return start_positions, end_positions, position_mask, answer_indices


class DocInputFeatures(object):
    """A single set of features of data."""

    __slots__ = ('unique_id', 'feature_list', 'num_feature', 'qid',
                 'num_unique_answer_str', 'answer_string_to_id')

    def __init__(self, unique_id=None, feature_list=None, num_feature=None,
                 qid=None, answer_string_to_id=None):
        """Initializer."""
        assert feature_list is not None

        self.unique_id = unique_id
        self.feature_list = list(feature_list)
        self.num_feature = num_feature
        self.num_unique_answer_str = len(answer_string_to_id) if answer_string_to_id is not None else 0
        self.answer_string_to_id = answer_string_to_id

        if self.num_unique_answer_str < 2:
            tf.logging.info("unique_id={0} has {1} unique answer string".format(
                unique_id, self.num_unique_answer_str
            ))

            for feature in feature_list:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (feature.unique_id))
                tf.logging.info("example_index: %s" % (feature.example_index))
                tf.logging.info("doc_span_index: %s" % (feature.doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [x for x in feature.tokens]))
                if feature.is_impossible:
                    tf.logging.info("impossible example")
                tf.logging.info("start_position_list: %s" % " ".join(
                    [str(x) for x in feature.start_position_list]))
                tf.logging.info("end_position_list: %s" % " ".join(
                    [str(x) for x in feature.end_position_list]))
                for start, end, ans_ind in zip(feature.start_position_list,
                                               feature.end_position_list,
                                               feature.answer_index_list):
                    answer_text = " ".join(
                        feature.tokens[start:(end + 1)])
                    tf.logging.info("start_position: %d" % (start))
                    tf.logging.info("end_position: %d" % (end))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))
                    tf.logging.info("answer index: %d" % ans_ind)
            tf.logging.info("answer_string_to_id: %s" %
                            json.dumps(answer_string_to_id))

            raise ValueError("There should at least two unique answer strings!")

        # Adds the qid for normalization.
        if not qid:
            raise ValueError("qid can not be None!")

        self.qid = qid

    def to_json(self):
        """Serializes the object into the json form."""
        attr_val_list = []
        for attr in self.__slots__:
            if attr == 'feature_list':
                continue
            attr_val_list.append((attr, getattr(self, attr)))

        attr_val_list.append((
            'feature_list', [feature.to_json() for feature in self.feature_list]
        ))
        return json.dumps(dict(attr_val_list))

    @classmethod
    def load_from_json(cls, json_string, keep_topk=None):
        """Loads the object from json string."""
        attr_dict = json.loads(json_string)
        return cls(
            unique_id=attr_dict["unique_id"],
            num_feature=attr_dict["num_feature"],
            qid=attr_dict["qid"],
            answer_string_to_id=attr_dict["answer_string_to_id"],
            feature_list=[
                    InputFeatures.load_from_json(feature_json)
                    for feature_json in attr_dict["feature_list"]
            ][:keep_topk]
        )


PseudoDocFeature = collections.namedtuple(
    "PseudoDocFeature",
    [
        "unique_id",
        "input_ids",
        "input_mask",
        "segment_ids",
        "start_positions",
        "end_positions",
        "answer_positions_mask",
        "answer_ids",
        "notnull_answer_mask",
    ]
)

def all_null_answer(answer_indices, null_answer_index):
    """Checks whether all answer indices equal to null."""
    return all([ans_ind == null_answer_index for ans_ind in answer_indices])


def read_feature_from_file(fn_list, keep_topk=None):
    feature_list = []
    for feature_filename in fn_list:
        tf.logging.info("Reading features from %s" % feature_filename)
        if not tf.gfile.Exists(feature_filename):
            raise ValueError(
                'Feature file {0} doesn not exist'.format(feature_filename))

        with open(feature_filename, mode='r', encoding="utf8") as fin:
            feature_list.append([
                DocInputFeatures.load_from_json(line.strip(),
                                                keep_topk=keep_topk)
                for line in fin
            ])

    return sum(feature_list, [])


def uniform_sample(pos_feature_ids, neg_feature_ids, max_num_doc_feature,
                   shuffler, single_pos_per_dupe):
    """Performs uniform sampling from positive and negative samples."""
    if single_pos_per_dupe:
        shuffler(pos_feature_ids)
        shuffler(neg_feature_ids)
        required_num_neg = max_num_doc_feature - 1
        if len(neg_feature_ids) < required_num_neg:
            selected_feature_ids = pos_feature_ids + neg_feature_ids
        else:
            selected_pos_ids = [pos_feature_ids[0]]
            selected_neg_ids = neg_feature_ids[:required_num_neg]
            selected_feature_ids = selected_pos_ids + selected_neg_ids

        selected_feature_ids = selected_feature_ids[:max_num_doc_feature]
        shuffler(selected_feature_ids)

        if np.intersect1d(pos_feature_ids, selected_feature_ids).size < 1:
            raise ValueError("Not positive feature is selected")
    else:
        num_feature = len(pos_feature_ids) + len(neg_feature_ids)
        nfeat_sample = min(max_num_doc_feature, num_feature)
        selected_feature_ids = np.random.choice(
            num_feature, nfeat_sample, replace=False)
        while np.intersect1d(pos_feature_ids, selected_feature_ids).size < 1:
            selected_feature_ids = np.random.choice(
                num_feature, nfeat_sample, replace=False)
        selected_feature_ids = list(selected_feature_ids)

    return selected_feature_ids


class InputFeatureContainer(object):
    """Data Container for InputFeature."""

    def __init__(self, feature_filename, max_num_doc_feature, batch_size, shuffle_data,
                 rand_seed, is_training=True, null_answer_index=0,
                 topk_for_train=None, rank_sample=False,
                 single_pos_per_dupe=True, allow_null_doc=False):

        np.random.seed(rand_seed)
        self.shuffle_data = shuffle_data
        self.input_feature_list = []
        self.batch_size = batch_size
        self.feature_filename = feature_filename
        self.max_num_doc_feature = max_num_doc_feature
        self.processed_sample_cnt = 0
        self.is_training = is_training
        self.rng = random.Random(rand_seed)
        self.null_answer_index = null_answer_index
        self.single_pos_per_dupe = single_pos_per_dupe
        self.rank_sample = rank_sample
        self.allow_null_doc = allow_null_doc

        if topk_for_train:
            tf.logging.info("Keeps topk-%d for training" % topk_for_train)

        if type(feature_filename) == str:
            self.feature_list = read_feature_from_file([feature_filename],
                                                       keep_topk=topk_for_train)
        elif type(feature_filename) == list:
            self.feature_list = read_feature_from_file(feature_filename,
                                                       keep_topk=topk_for_train)

        else:
            raise ValueError("Unknown feature_filename type %s" % feature_filename)

        self.num_sample = len(self.feature_list)

        tf.logging.info("Num of examples %d" % self.num_sample)

    def start_epoch(self, batch_size=None):
        """Prepares for a new epoch."""
        tf.logging.info("The data reader has processed {0} features".format(
            self.processed_sample_cnt))

        tf.logging.info("Starts a new epoch!")

        self.processed_sample_cnt = 0
        if self.shuffle_data:
            self.rng.shuffle(self.feature_list)

        tf.logging.info(
            'The new epoch will have batch_size: {0}'.format(self.batch_size))


    def _process_feature(self, document):
        """Converts a document into multiple passage features."""
        is_negative = lambda x: all_null_answer(x, self.null_answer_index)

        pos_feature_ids, neg_feature_ids = [], []
        for ii, feature in enumerate(document.feature_list):
            if feature.is_impossible:
                neg_feature_ids.append(ii)
            else:
                pos_feature_ids.append(ii)

        selected_feature_ids = uniform_sample(
            pos_feature_ids, neg_feature_ids, self.max_num_doc_feature,
            self.rng.shuffle, self.single_pos_per_dupe)

        input_ids = []
        input_mask = []
        segment_ids = []

        start_positions_list = []
        end_positions_list = []
        answer_positions_mask = []
        answer_index_list = []
        notnull_answer_mask = []

        has_answer = False

        for feat_id in selected_feature_ids:
            feature = document.feature_list[feat_id]
            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)

            if self.is_training:
                start_positions_list.append(
                    feature.start_position_list)
                end_positions_list.append(
                    feature.end_position_list)
                answer_positions_mask.append(
                    feature.position_mask)
                answer_index_list.append(
                    feature.answer_index_list)
                notnull_answer_mask.append(
                    [0 if ans_id == self.null_answer_index else 1
                     for ans_id in feature.answer_index_list])

            # Answer index 0 is reserved for null.
            if not is_negative(feature.answer_index_list):
                has_answer = True

        if (not self.allow_null_doc) and (not has_answer):
            raise ValueError("No answer document!")

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'start_positions_list': start_positions_list,
            'end_positions_list': end_positions_list,
            'answer_positions_mask': answer_positions_mask,
            'answer_index_list': answer_index_list,
            'notnull_answer_mask': notnull_answer_mask,
            'num_sample': len(input_ids),
        }

    def __iter__(self):
        """Iterates over the dataset."""

        for feature in self.feature_list:
            self.processed_sample_cnt += 1

            yield self._process_feature(feature)


class DocFeatureJsonWriter(object):
    """Writes DocFeature into json line."""

    def __init__(self, filename):
        self.filename = filename
        self.num_features = 0
        self._writer = open(filename, mode="wt", encoding="utf8")

    def process_doc_feature(self, feature):
        self._writer.write(feature.to_json())
        self._writer.write("\n")
        self.num_features += 1

    def close(self):
        self._writer.close()


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature(
                [feature.start_position])
            features["end_positions"] = create_int_feature(
                [feature.end_position])
            features["start_position_list"] = create_int_feature(
                feature.start_position_list)
            features["end_position_list"] = create_int_feature(
                feature.end_position_list)
            features["position_mask"] = create_int_feature(
                feature.position_mask)

            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def get_position_lists(answers_list, char_to_word_offset, doc_tokens):
    """Reads a list of answers and creates lists for the unique ones."""
    start_position_list = []
    end_position_list = []
    answer_text_list = []

    for answer in answers_list:
        orig_answer_text = answer["text"]
        answer_offset = int(answer["answer_start"])
        answer_length = len(orig_answer_text)
        start_position = char_to_word_offset[answer_offset]
        end_position = char_to_word_offset[answer_offset + answer_length - 1]

        # Skips duplicate answers.
        if (start_position in start_position_list or
                end_position in end_position_list):
            continue

        # Checks whether the answer text can be recovered.
        # If not, skips the current answer.
        actual_text = " ".join(
            doc_tokens[start_position:(end_position + 1)])
        cleaned_answer_text = " ".join(
            tokenization.whitespace_tokenize(orig_answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            tf.logging.warning(
                "Could not find answer: '%s' vs. '%s'",
                actual_text, cleaned_answer_text)
            continue

        start_position_list.append(start_position)
        end_position_list.append(end_position)
        answer_text_list.append(orig_answer_text)

    return start_position_list, end_position_list, answer_text_list


def squad_example_generator(input_data, is_training, keep_topk=None,
                            use_answer_list_for_impossible=True):
    """A generator for squad example."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    counter = 0
    stop_processing = False
    for entry in input_data:

        if stop_processing:
            break

        for paragraph in entry["paragraphs"]:
            if FLAGS.debug and counter >= 1000:
                tf.logging.info("[Debugging]: only keeps 1000 examples.")
                stop_processing = True
                break
            paragraph_text = paragraph["context"]

            doc_title = paragraph.get("doc_title", None)

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]

                # id contains the rank information.
                _, rank = qa["id"].split("-")
                if keep_topk and int(rank) > keep_topk:
                    break

                qid = None
                if "qid" in qa:
                    qid = qa["qid"]
                else:
                    # TODO(chenghao): This is debug report for TriviaQA.
                    raise ValueError("No qid in qa")

                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                start_position_list = None
                end_position_list = None
                orig_answer_text_list = None
                if is_training:
                    if FLAGS.version_2_with_negative:
                        if not (type(qa["is_impossible"]) is bool):
                            raise ValueError("is_impossible is not bool")

                        is_impossible = qa["is_impossible"]

                    # Over-writes is_impossible, if answer list is not empty.
                    if use_answer_list_for_impossible:
                        if qa["answers"]:
                            is_impossible = False
                        else:
                            is_impossible = True

                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = int(answer["answer_start"])
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[
                            answer_offset + answer_length - 1]

                        # For weak-supervision datasets, there might be multiple
                        # answers.
                        (start_position_list, end_position_list,
                         orig_answer_text_list) = get_position_lists(
                             qa["answers"], char_to_word_offset, doc_tokens
                         )

                        if not start_position_list:
                            continue

                        # Only add answers where the text can be exactly
                        # recovered from the document. If this CAN'T happen it's
                        # likely due to weird Unicode stuff so we will just skip
                        # the example.
                        #
                        # Note that this means for training mode, every example
                        # is NOT guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning(
                                "Could not find answer: '%s' vs. '%s'",
                                actual_text, cleaned_answer_text)
                            continue

                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = FLAGS.no_answer_string

                        start_position_list = [-1]
                        end_position_list = [-1]
                        orig_answer_text_list = [FLAGS.no_answer_string]

                # TODO(chenghao): Fix this.
                counter += 1
                yield SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    qid=qid,
                    title=doc_title,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    orig_answer_text_list=orig_answer_text_list,
                    start_position_list=start_position_list,
                    end_position_list=end_position_list,
                )


def read_squad_examples_from_generator(input_file, is_training, keep_topk=None,
                                       use_answer_list_for_impossible=True):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding="utf8") as reader:
        input_data = json.load(reader)["data"]

    tf.logging.info("Reading examples in using generator!")

    examples = [
        example for example in squad_example_generator(
            input_data, is_training, keep_topk=keep_topk,
            use_answer_list_for_impossible=use_answer_list_for_impossible)
    ]
    tf.logging.info("Done Reading.")

    return examples


_DocSquadExample = collections.namedtuple(
    "DocSquadExample", ["qid", "example_list"]
)


def _is_old_example(proposed_example, target_example_list, token_window=10):
    """Checks whether the proposed example is old."""
    if proposed_example.qid == target_example_list[0].qid:
        return True

    def _map_tokens_to_string(tokens):
        return " ".join(tokens[:token_window] + tokens[-token_window:])

    token_string = _map_tokens_to_string(proposed_example.doc_tokens)
    return any([
        token_string == _map_tokens_to_string(example.doc_tokens)
        for example in target_example_list
    ])


def _convert_example_with_target_example(example, target_example_list):
    """Converts the example as a negative case for the target example."""
    if _is_old_example(example, target_example_list):
        return None

    return SquadExample(
        qas_id=example.qas_id,
        question_text=target_example_list[0].question_text,
        doc_tokens=example.doc_tokens,
        qid=target_example_list[0].qid,
        title=example.title,
        orig_answer_text="",
        start_position=0,
        end_position=0,
        is_impossible=True,
        orig_answer_text_list=[],
        start_position_list=[],
        end_position_list=[],
    )


def is_null_doc_example(document):
    """If document has no positive paragraphs."""
    return all([example.is_impossible for example in document.example_list])


def read_doc_squad_examples_from_generator(input_file, is_training,
                                           keep_topk=None):
    """Read a SQuAD json file into a list of DocSquadExample."""
    examples = read_squad_examples_from_generator(input_file, is_training,
                                                  keep_topk=keep_topk)

    keyfunc = lambda x: str(x.qid)

    # Groups the example by qid.
    doc_examples = [
        _DocSquadExample(qid=qid, example_list=list(group))
        for qid, group in itertools.groupby(
            sorted(examples, key=keyfunc), key=keyfunc)
    ]

    if FLAGS.filter_null_doc and is_training:
        tf.logging.info("Filtering null documents for training.")
        tf.logging.info(
            "Before filtering, there are %d documents" % len(doc_examples))
        doc_examples = list(filter(lambda x: not is_null_doc_example(x), doc_examples))
        tf.logging.info(
            "After filtering, there are %d documents" % len(doc_examples))

    doc_size = [len(item.example_list) for item in doc_examples]
    tf.logging.info("Min of examples per doc %d" % np.min(doc_size))
    tf.logging.info("Max of examples per doc %d" % np.max(doc_size))
    tf.logging.info("Mean of examples per doc %d" % np.mean(doc_size))
    tf.logging.info("95 percentile %f" % np.percentile(doc_size, 95))

    num_raw_train_examples = sum(doc_size)
    return doc_examples, num_raw_train_examples


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can
    # match the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a
    # single token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def bert_doc_token_processor(doc_tokens, tokenizer):
    """Converts tokens into a list of BERT subtokens."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return (tok_to_orig_index, orig_to_tok_index, all_doc_tokens)


def convert_examples_to_feature_list_within_doc(
        examples, tokenizer, max_seq_length, doc_stride, max_query_length,
        is_training, unique_offset, answer_string_to_id,
        cls_tok="[CLS]", sep_tok="[SEP]", pad_token_id=0,
        doc_token_processor=bert_doc_token_processor):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000 + unique_offset

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            tf.logging.info("Query too long")
            tf.logging.info(" ".join([tokenization.printable_text(x) for x in query_tokens]))
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        (tok_to_orig_index, orig_to_tok_index, all_doc_tokens
         ) = doc_token_processor(example.doc_tokens, tokenizer)

        tok_start_position = None
        tok_end_position = None
        tok_start_position_list = []
        tok_end_position_list = []

        # Registers answer strings of the current examples.
        tok_answer_index_list = []

        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
            tok_start_position_list = [0]
            tok_end_position_list = [0]
            tok_answer_index_list = [0]

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]

            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

            # Iterates over all possible answer positions.
            for (start_index, end_index, orig_ans_txt) in zip(
                    example.start_position_list,
                    example.end_position_list,
                    example.orig_answer_text_list):
                tok_start_pos = orig_to_tok_index[start_index]
                tok_end_pos = len(all_doc_tokens) - 1
                if end_index < len(example.doc_tokens) - 1:
                    tok_end_pos = orig_to_tok_index[end_index + 1] - 1

                # Improves both start and end positions.
                (tok_start_pos_improved,
                 tok_end_pos_improved) = _improve_answer_span(
                     all_doc_tokens, tok_start_pos, tok_end_pos, tokenizer,
                     orig_ans_txt)

                ans_text_improved = " ".join(all_doc_tokens[
                    tok_start_pos_improved:(tok_end_pos_improved + 1)])

                # TODO(chenghao): Whether to use improved text or orig?
                # Previously, the improved one is used.
                index_ans_str = orig_ans_txt.lower()
                if index_ans_str not in answer_string_to_id:
                    # Only keeps answer strings up to the upperbound.
                    # TODO(chenghao): Changes this to frequency-based.
                    if len(answer_string_to_id) <= FLAGS.max_num_answer_strings:
                        answer_string_to_id[index_ans_str] = len(
                            answer_string_to_id)
                    else:
                        tf.logging.info(
                            "qid %s has more than %d short answers" % (
                                example.qid, FLAGS.max_num_answer_strings
                            ))

                ans_str_index = answer_string_to_id.get(index_ans_str, 0)

                if index_ans_str and ans_str_index == 0:
                    tf.logging.info("Drops answer %s for qid %s" % (
                        index_ans_str, example.qid
                    ))

                tok_start_position_list.append(tok_start_pos_improved)
                tok_end_position_list.append(tok_end_pos_improved)
                tok_answer_index_list.append(ans_str_index)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # tokens.append("[CLS]")
            tokens.append(cls_tok)
            segment_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            # tokens.append("[SEP]")
            tokens.append(sep_tok)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[
                    len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            # tokens.append("[SEP]")
            tokens.append(sep_tok)
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token_id)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            start_position_list = None
            end_position_list = None
            answer_index_list = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an
                # answer, we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                start_position_list = []
                end_position_list = []
                answer_index_list = []

                for (tok_start_pos, tok_end_pos, tok_ans_index) in zip(
                        tok_start_position_list, tok_end_position_list,
                        tok_answer_index_list):
                    # If the answer is out of range, skips this pair.
                    if not (tok_start_pos >= doc_start and
                            tok_end_pos <= doc_end and
                            tok_start_pos <= tok_end_pos):
                        continue

                    # Computes the start and end positions with the offset.
                    doc_offset = len(query_tokens) + 2
                    start_position_list.append(
                        tok_start_pos - doc_start + doc_offset
                    )
                    end_position_list.append(
                        tok_end_pos - doc_start + doc_offset
                    )

                    answer_index_list.append(tok_ans_index)

                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end and
                        tok_start_position <= tok_end_position):
                    out_of_span = True

                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # TODO(chenghao): This part needs to be cleaned up.
            if is_training and (example.is_impossible or (not start_position_list)):
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]
                answer_index_list = [0]

            # TODO(chenghao): This is kept for the purpose of making training
            # and evaluation consistent.
            if not is_training:
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]
                answer_index_list = [0]

            if unique_offset < 1 and example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in token_to_orig_map.items()]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in token_is_max_context.items()
                ]))
                tf.logging.info("input_ids: %s" % " ".join(
                    [str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training:
                    tf.logging.info("start_position_list: %s" % " ".join(
                        [str(x) for x in start_position_list]))
                    tf.logging.info("end_position_list: %s" % " ".join(
                        [str(x) for x in end_position_list]))
                    for start, end in zip(start_position_list,
                                          end_position_list):
                        answer_text = " ".join(
                            tokens[start:(end + 1)])
                        tf.logging.info("start_position: %d" % (start))
                        tf.logging.info("end_position: %d" % (end))
                        tf.logging.info("answer: %s" % (
                            tokenization.printable_text(answer_text)))
                    tf.logging.info("answer_string_to_id: %s" %
                                    json.dumps(answer_string_to_id))

                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                qid=example.qid,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                start_position_list=start_position_list,
                end_position_list=end_position_list,
                answer_index_list=answer_index_list,

            )

            unique_id += 1

            # This is originally inside the constructor of InputFeatures.
            # For each paragraph, there should at least one answer.
            if is_training and feature.num_answer < 1:
                raise ValueError(
                    "Each paragraph should have at least one answer!")

            yield feature


def convert_doc_examples_to_doc_features(
        doc_examples, tokenizer, max_seq_length, doc_stride, max_query_length,
        is_training, output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    start_id = 1000000000
    doc_id = start_id
    unique_offset = 0

    num_unique_str_stats = []

    null_doc_cnt = 0
    null_qid_list = []

    if is_training and FLAGS.filter_null_doc:
        tf.logging.info("Filtering null documents for training.")

    tf.logging.info("Converts data for BERT")
    doc_token_processor = bert_doc_token_processor
    cls_tok = "[CLS]"
    sep_tok = "[SEP]"
    pad_token_id = 0

    for doc_example in doc_examples:
        answer_string_to_id = collections.defaultdict(int)

        # Reserves the 0 index for no answer string.
        # Non-empty answers will have indices starting from 1.
        answer_string_to_id[FLAGS.no_answer_string] = 0

        feature_list = [
            feature for feature in convert_examples_to_feature_list_within_doc(
                doc_example.example_list, tokenizer, max_seq_length, doc_stride,
                max_query_length, is_training, unique_offset,
                answer_string_to_id,
                doc_token_processor=doc_token_processor,
                cls_tok=cls_tok,
                sep_tok=sep_tok,
                pad_token_id=pad_token_id,
            )
        ]

        if len(answer_string_to_id) < 2:
            null_doc_cnt += 1
            null_qid_list.append(feature_list[0].qid)

            # We throws out examples without any answer string in the document.
            if is_training and FLAGS.filter_null_doc:
                continue

        unique_offset += len(feature_list)
        doc_feature = DocInputFeatures(
            unique_id=doc_id,
            feature_list=list(feature_list),
            num_feature=len(feature_list),
            answer_string_to_id=dict(answer_string_to_id),
            qid=doc_example.qid
        )

        output_fn(doc_feature)
        num_unique_str_stats.append(len(answer_string_to_id))

        doc_id += 1

    tf.logging.info("Max num of unique answer string: {0}".format(
        np.max(num_unique_str_stats)))
    tf.logging.info("Min num of unique answer string: {0}".format(
        np.min(num_unique_str_stats)))
    tf.logging.info("Mean num of unique answer string: {0}".format(
        np.mean(num_unique_str_stats)))
    tf.logging.info("Number of null answer doc: {0}".format(null_doc_cnt))
    tf.logging.info("Null answer qids:{0}".format(null_qid_list))

    return unique_offset


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


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, unique_id_to_qid=None,
                                 cls_tok="[CLS]", sep_tok="[SEP]",
                                 pad_token_id=0,
                                 doc_token_processor=bert_doc_token_processor):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            tf.logging.info("Query too long")
            tf.logging.info(" ".join([tokenization.printable_text(x) for x in query_tokens]))
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        (tok_to_orig_index, orig_to_tok_index, all_doc_tokens
         ) = doc_token_processor(example.doc_tokens, tokenizer)

        tok_start_position = None
        tok_end_position = None
        tok_start_position_list = []
        tok_end_position_list = []
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
            tok_start_position_list = [-1]
            tok_end_position_list = [-1]

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]

            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

            # Iterates over all possible answer positions.
            for (start_index, end_index, orig_ans_txt) in zip(
                    example.start_position_list,
                    example.end_position_list,
                    example.orig_answer_text_list):
                tok_start_pos = orig_to_tok_index[start_index]
                tok_end_pos = len(all_doc_tokens) - 1
                if end_index < len(example.doc_tokens) - 1:
                    tok_end_pos = orig_to_tok_index[end_index + 1] - 1

                # Improves both start and end positions.
                (tok_start_pos_improved,
                 tok_end_pos_improved) = _improve_answer_span(
                     all_doc_tokens, tok_start_pos, tok_end_pos, tokenizer,
                     orig_ans_txt)

                tok_start_position_list.append(tok_start_pos_improved)
                tok_end_position_list.append(tok_end_pos_improved)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append(cls_tok)
            segment_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append(sep_tok)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[
                    len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            tokens.append(sep_tok)
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            start_position_list = None
            end_position_list = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an
                # answer, we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                start_position_list = []
                end_position_list = []

                for tok_start_pos, tok_end_pos in zip(tok_start_position_list,
                                                      tok_end_position_list):
                    # If the answer is out of range, skips this pair.
                    if not (tok_start_pos >= doc_start and
                            tok_end_pos <= doc_end and
                            tok_start_pos <= tok_end_pos):
                        continue

                    # Computes the start and end positions with the offset.
                    doc_offset = len(query_tokens) + 2
                    start_position_list.append(
                        tok_start_pos - doc_start + doc_offset
                    )
                    end_position_list.append(
                        tok_end_pos - doc_start + doc_offset
                    )

                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end and
                        tok_start_position <= tok_end_position):
                    out_of_span = True

                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and (example.is_impossible or (not start_position_list)):
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in token_to_orig_map.items()]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in token_is_max_context.items()
                ]))
                tf.logging.info("input_ids: %s" % " ".join(
                    [str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training:
                    tf.logging.info("start_position_list: %s" % " ".join(
                        [str(x) for x in start_position_list]))
                    tf.logging.info("end_position_list: %s" % " ".join(
                        [str(x) for x in end_position_list]))
                    for start, end in zip(start_position_list,
                                          end_position_list):
                        answer_text = " ".join(
                            tokens[start:(end + 1)])
                        tf.logging.info("start_position: %d" % (start))
                        tf.logging.info("end_position: %d" % (end))
                        tf.logging.info("answer: %s" % (
                            tokenization.printable_text(answer_text)))

                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                qid=example.qid,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                start_position_list=start_position_list,
                end_position_list=end_position_list,
            )

            # Run callback
            output_fn(feature)

            if unique_id_to_qid is not None:
                if example.qid:
                    unique_id_to_qid[unique_id] = example.qid
                else:
                    raise ValueError("When unique_id_to_qid is required,"
                                     "example.qid can not be None!")

            unique_id += 1
