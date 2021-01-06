#!/usr/bin/env python3
"""Loss help functions."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import modeling


def batch_gather(params, indices):
    """Performs batch_gather."""
    bs, nv = modeling.get_shape_list(indices, expected_rank=2)
    _, batch_offset = modeling.get_shape_list(params, expected_rank=2)

    # A [batch_size, num_answers]-sized tensor.
    offset = tf.tile(tf.reshape(
        tf.range(bs * batch_offset, delta=batch_offset, dtype=tf.int32),
        shape=[bs, 1]), [1, nv])

    f_indices = tf.reshape(indices + offset, shape=[-1])
    f_vals = tf.gather(tf.reshape(params, shape=[-1]), f_indices)

    return tf.reshape(f_vals, shape=[bs, nv])


def one_hot_batch_gather(params, indices):
    """Performs batch_gather using one_hot matrix multiplication."""
    batch_size, num_value = modeling.get_shape_list(indices, expected_rank=2)
    _, max_seq_len = modeling.get_shape_list(params, expected_rank=2)

    # This returns a tensor of shape [batch_size, num_value, max_seq_len].
    positional_one_hot = tf.one_hot(
        indices, depth=max_seq_len, dtype=tf.float32
    )

    # Expands the params tensor for selection.
    expand_params = tf.expand_dims(params, 2)

    gathered_values = tf.matmul(positional_one_hot, expand_params)

    # Reduces the last dimension.
    return tf.squeeze(gathered_values, 2)


def compute_span_log_score(start_log_scores, start_pos_list,
                           end_log_scores, end_pos_list, use_gather=False):
    """Computes the span log scores."""
    if use_gather:
        ans_span_start_log_scores = batch_gather(
            start_log_scores, start_pos_list
        )
        ans_span_end_log_scores = batch_gather(
            end_log_scores, end_pos_list
        )
    else:
        ans_span_start_log_scores = one_hot_batch_gather(
            start_log_scores, start_pos_list
        )
        ans_span_end_log_scores = one_hot_batch_gather(
            end_log_scores, end_pos_list
        )

    return (ans_span_start_log_scores + ans_span_end_log_scores)


def compute_logprob(logits, axis=-1, keepdims=None):
    """Computes the log prob based on logits."""
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=keepdims)


def doc_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, pos_par_mask, loss_type=None):
    """Computes document-level normalization span-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=None)
    end_log_prob = compute_logprob(end_logits, axis=None)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_prob = compute_span_log_score(
        start_log_prob, start_indices, end_log_prob, end_indices
    )

    log_score_mask = tf.cast(positions_mask, dtype=tf.float32)

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        span_loss = -tf.reduce_sum(pos_par_mask * tf.reduce_logsumexp(
            span_log_prob + tf.log(log_score_mask), axis=-1))
    elif loss_type == "h2_hard_em":
        span_loss = -tf.reduce_sum(pos_par_mask * tf.reduce_max(
            span_log_prob + tf.log(log_score_mask), axis=-1))
    elif loss_type == "h3_mml":
        # The whole document contains one correct span.
        span_loss = -tf.reduce_logsumexp(span_log_prob + tf.log(log_score_mask))
    elif loss_type == "h3_hard_em":
        span_loss = -tf.reduce_max(span_log_prob + tf.log(log_score_mask))
    else:
        raise ValueError("Unknwon loss_type %s for doc_span_loss!"
                         % loss_type)

    return span_loss


def one_hot_answer_positions(position_list, position_mask, depth):
    position_tensor = tf.one_hot(
        position_list, depth, dtype=tf.float32)
    position_masked = position_tensor * tf.cast(
        tf.expand_dims(position_mask, -1), dtype=tf.float32
    )
    onehot_positions = tf.reduce_max(position_masked, axis=1)
    return onehot_positions


def compute_masked_log_score(log_score, position_list, answer_masks, seq_length):
    position_tensor = one_hot_answer_positions(
        position_list, answer_masks, seq_length
    )
    return log_score + tf.log(position_tensor)


def group_answer_span_prob(ans_span_probs, group_ids, max_num_answer_strings,
                           group_all=True):
    """Sums all answer span probilities from the same group."""
    delta = max_num_answer_strings + 1

    if group_all:
        batch_size = 1
    else:
        batch_size, _ = modeling.get_shape_list(ans_span_probs, expected_rank=2)
        offset = tf.reshape(tf.range(0, batch_size * delta, delta),
                            shape=[batch_size, 1])
        group_ids += offset

    group_ans_probs = tf.math.unsorted_segment_sum(
        ans_span_probs, group_ids, batch_size * delta
    )

    return tf.reshape(
        group_ans_probs, [batch_size, delta]
    )


def compute_max_ans_str_mask(start_logits, start_positions_list, end_logits,
                             end_positions_list, positions_mask,
                             answer_index_list, batch_size, max_num_answers,
                             seq_length, max_num_answer_strings,
                             group_all=True):
    """Computes the max answer string mask."""
    # Here, we assume the full start_logits tensor comes from the same doc.
    start_log_prob = tf.nn.log_softmax(start_logits, axis=None)
    end_log_prob = tf.nn.log_softmax(end_logits, axis=None)

    span_log_prob = compute_span_log_score(
        start_log_prob, start_positions_list,
        end_log_prob, end_positions_list
    )
    span_log_prob -= tf.stop_gradient(tf.reduce_max(span_log_prob, axis=None))
    span_prob = tf.exp(span_log_prob) * tf.cast(positions_mask, dtype=tf.float32)


    str_prob = group_answer_span_prob(
        span_prob, answer_index_list, max_num_answer_strings,
        group_all=group_all
    )

    if group_all:
        max_ans_str_index = tf.tile(
            tf.reshape(tf.argmax(str_prob, axis=-1), shape=[1, 1]),
            [batch_size, max_num_answers]
        )
    else:
        max_ans_str_index = tf.tile(
            tf.reshape(tf.argmax(str_prob, axis=-1), shape=[batch_size, 1]),
            [1, max_num_answers]
        )

    max_ans_str_positions_mask = tf.stop_gradient(tf.cast(tf.equal(
        max_ans_str_index, answer_index_list), tf.int64))

    return max_ans_str_positions_mask


def get_max_mask(log_scores, reduce_all=True):
    """Creates the max value mask for 2D log scores."""
    bs, nv = modeling.get_shape_list(log_scores, expected_rank=2)
    if reduce_all:
        # Only keeps the max log score poistion for the all matrix.
        _, max_indices = tf.nn.top_k(
            tf.reshape(log_scores, [-1]), k=1
        )
        max_mask = tf.reshape(
            tf.one_hot(max_indices, depth=(bs*nv), dtype=tf.float32),
            [bs, nv]
        )
    else:
        # Keeps the max log score poistion for each row of the matrix.
        _, max_indices = tf.nn.top_k(log_scores, k=1)
        max_mask = tf.reduce_max(
            tf.one_hot(max_indices, depth=nv, dtype=tf.float32),
            axis=1
        )

    return tf.stop_gradient(max_mask)


def doc_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, pos_par_mask, seq_length, loss_type=None):
    """Computes document-level normalization position-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=None)
    end_log_prob = compute_logprob(end_logits, axis=None)

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask, seq_length
    )
    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask, seq_length
    )

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        start_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_logsumexp(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_logsumexp(masked_end_log_prob, axis=-1))
    elif loss_type == "h2_hard_em":
        # Each positive paragraph contains a correct span.
        start_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_max(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_max(masked_end_log_prob, axis=-1))
    elif loss_type == "h3_mml":
        # The whole document contains one correct span.
        start_loss = -tf.reduce_logsumexp(masked_start_log_prob, axis=None)
        end_loss = -tf.reduce_logsumexp(masked_end_log_prob, axis=None)
    elif loss_type == "h3_hard_em":
        start_loss = -tf.reduce_max(masked_start_log_prob, axis=None)
        end_loss = -tf.reduce_max(masked_end_log_prob, axis=None)
    elif loss_type == "h1":
        # The whole document contains one correct span.
        start_position_tensor = one_hot_answer_positions(
            start_indices, answer_positions_mask, seq_length
        )
        end_position_tensor = one_hot_answer_positions(
            end_indices, answer_positions_mask, seq_length
        )

        start_loss = -tf.reduce_sum(
            start_position_tensor * start_log_prob, axis=None)
        end_loss = -tf.reduce_sum(
            end_position_tensor * end_log_prob, axis=None)
    else:
        raise ValueError("Unknwon loss_type %s for doc_pos_loss!"
                         % loss_type)

    return (start_loss + end_loss) / 2.0


def par_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, loss_type=None):
    """Computes paragraph-level normalization span-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_prob = compute_span_log_score(
        start_log_prob, start_indices, end_log_prob, end_indices
    )

    masked_span_log_prob = span_log_prob + tf.log(
        tf.cast(positions_mask, dtype=tf.float32)
    )

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        span_loss = tf.reduce_mean(
            -tf.reduce_logsumexp(masked_span_log_prob, axis=-1))
    elif loss_type == "h2_hard_em":
        span_loss = tf.reduce_mean(
            -tf.reduce_max(masked_span_log_prob, axis=-1))
    elif loss_type == "h1":
        span_loss = tf.reduce_mean(
            -tf.reduce_sum(masked_span_log_prob, axis=-1)
        )
    else:
        raise ValueError("Unknwon loss_type %s for par_span_loss!"
                         % loss_type)

    return span_loss


def par_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, seq_length, loss_type=None):
    """Computes paragraph-level normalization position-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask, seq_length
    )

    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask, seq_length
    )

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        start_loss = tf.reduce_mean(
            -tf.reduce_logsumexp(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_mean(
            -tf.reduce_logsumexp(masked_end_log_prob, axis=-1))
    elif loss_type == "h2_hard_em":
        start_loss = tf.reduce_mean(
            -tf.reduce_max(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_mean(
            -tf.reduce_max(masked_end_log_prob, axis=-1))
    elif loss_type == "h1":
        # Each positive paragraph contains a correct span.
        start_loss = tf.reduce_mean(
            -tf.reduce_sum(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_mean(
            -tf.reduce_sum(masked_end_log_prob, axis=-1))
    else:
        raise ValueError("Unknwon loss_type %s for par_pos_loss!"
                         % loss_type)

    return (start_loss + end_loss) / 2.0
