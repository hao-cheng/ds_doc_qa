"""Extracting predictions from n-best file with multiple paragrpahs"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import tensorflow as tf
import six
import numpy as np


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "sum", True,
    "Whether to use the sum or max aggregation.")

flags.DEFINE_integer(
    "max_para", 8,
    "The lowest ranked paragraph to use -- ranks are 0 to m.")

flags.DEFINE_integer(
    "max_nbest_per_passage", 20,
    "The max nbest answer span candidates to use -- 1 to m.")

flags.DEFINE_string("nbest_file", "",
                    "A file with nbest predictions.")

flags.DEFINE_string("predictions_file", "",
                    "A file to write final predictions to.")

flags.DEFINE_bool("use_doc_score", False,
                  "Whether to use the doc score information.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case strings for aggregation.")


def make_predictions(nbest_file, predictions_file):
    # for each qid we record a map from answer string to occurences
    # an occurrence has a probability and a paragraph index
    RawOccurrence = collections.namedtuple("RawOccurrence", ["prob", "rank"])
    qid_to_answers = {}
    qid_to_aggregated_answers = {}
    result_dict = {}

    use_doc_score = FLAGS.use_doc_score
    sum_score = FLAGS.sum

    gp_func = lambda x: str(x[0].split("-")[0])

    # Only keeps those top ranked passages.
    filter_func = lambda x: int(x[0].split("-")[1]) < FLAGS.max_para

    # Only keeps certain topk span for each passage.
    map_func = lambda x: (x[0], x[1][0:FLAGS.max_nbest_per_passage])

    with open(nbest_file) as reader:
        input_data = list(json.load(reader).items())

    for qid, pred_group in itertools.groupby(
            sorted(input_data, key=gp_func), key=gp_func):
        predictions = list(map(map_func, filter(filter_func, list(pred_group))))

        if qid not in qid_to_answers:
            qid_to_answers[qid] = collections.defaultdict(list)
            qid_to_aggregated_answers[qid] = collections.defaultdict(float)

        for qid_w_rank, passage_preds in predictions:
            rank = int(qid_w_rank.split("-")[-1])
            for p in passage_preds:
                ans_string = p["text"]
                if FLAGS.do_lower_case:
                    ans_string = ans_string.lower()

                if FLAGS.use_doc_score:
                    prob = np.exp(
                        p["start_logit"] + p["end_logit"] - p["doc_score"])
                else:
                    prob = p["probability"]

                qid_to_answers[qid][ans_string].append(
                    RawOccurrence(prob=prob, rank=rank)
                    )
                score = prob

                if FLAGS.sum:
                    qid_to_aggregated_answers[qid][ans_string] += score
                else:
                  if qid_to_aggregated_answers[qid][ans_string] < score:
                      qid_to_aggregated_answers[qid][ans_string] = score

    # Chooses the best guess for each qid.
    result_with_score_dict = {}
    for qid, answer_dict in six.iteritems(qid_to_aggregated_answers):
        q_ans_ordered = collections.OrderedDict(sorted(answer_dict.items(),
                                                       key=lambda t: -t[1]))
        for ans, ans_score in six.iteritems(q_ans_ordered):
            if ans:
                result_dict[qid] = ans
                result_with_score_dict[qid] = {"answer": ans, "score": ans_score}
                break

    with open(predictions_file, mode="wt", encoding="utf8") as fout:
        fout.write(json.dumps(
            result_dict, sort_keys=True, indent=4, ensure_ascii=False))

    with open(predictions_file + "wo_score", mode="wt", encoding="utf8") as fout:
        fout.write(json.dumps(result_with_score_dict, sort_keys=True,
                              indent=4))


def main():
    make_predictions(FLAGS.nbest_file, FLAGS.predictions_file)


if __name__ == '__main__':
    main()
