import collections
import json
import os
import re
import tensorflow as tf
import time
import tokenization
import requests
from keras.backend.tensorflow_backend import set_session
import modeling
import optimization

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(tf.Session(config=config))

session = requests.session()
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_folder",
    "./tf_output",
    "Folder where the TFRecord files will be writen.")

flags.DEFINE_integer(
    "max_test_examples", 10,
    "Maximum number of test examples to be evaluated. If None, evaluate all "
    "examples in the test set.")

flags.DEFINE_string(
    "output_dir", "./data/output",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "vocab_file",
    "/nas/Datasets/msmarco/passage_reranking/uncased_L-24_H-1024_A-16/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "bert_config_file",
    "/nas/Datasets/msmarco/passage_reranking/uncased_L-24_H-1024_A-16/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_test_docs", 100,
    "The number of docs per query for the test set.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
    "init_checkpoint",
    "/nas/Datasets/msmarco/passage_reranking/BERT_Large_trained_on_TREC_CAR/model.ckpt-100000.index",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_float("learning_rate", 3e-6,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 400000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 40000,
    "Number of training steps to perform linear learning rate warmup.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


def convert_dataset(data, set_name, tokenizer):
    output_path = FLAGS.output_folder + '/dataset_' + set_name + '.tf'

    print('Converting {} to tfrecord'.format(set_name))

    #   random_title = list(corpus.keys())[0]

    with tf.python_io.TFRecordWriter(output_path) as writer:
        query, doc_contents = data
        print(doc_contents)
        query = query.replace('enwiki:', '')
        query = query.replace('%20', ' ')
        query = query.replace('/', ' ')
        query = tokenization.convert_to_unicode(query)
        query_ids = tokenization.convert_to_bert_input(
            text=query,
            max_seq_length=FLAGS.max_query_length,
            tokenizer=tokenizer,
            add_cls=True)

        query_ids_tf = tf.train.Feature(
            int64_list=tf.train.Int64List(value=query_ids))

        max_docs = FLAGS.num_test_docs
        doc_contents = doc_contents[:max_docs]

        # Add fake docs so we always have max_docs per query.
        # * [random_title] TODO check this
        # doc_contents += max(0, max_docs - len(doc_contents))

        labels = [0 for doc_title in doc_contents]

        doc_token_ids = [
            tokenization.convert_to_bert_input(
                text=tokenization.convert_to_unicode(doc_content),
                max_seq_length=FLAGS.max_seq_length - len(query_ids),
                tokenizer=tokenizer,
                add_cls=False)
            for doc_content in doc_contents
        ]

        for rank, (doc_token_id, label) in enumerate(zip(doc_token_ids, labels)):
            doc_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=doc_token_id))

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))

            len_gt_titles_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0]))

            features = tf.train.Features(feature={
                'query_ids': query_ids_tf,
                'doc_ids': doc_ids_tf,
                'label': labels_tf,
                'len_gt_titles': len_gt_titles_tf,
            })

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        len_gt_titles = features["len_gt_titles"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
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
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "logits": logits,
                    "label_ids": label_ids,
                    "len_gt_titles": len_gt_titles,
                },
                scaffold_fn=scaffold_fn)

        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(dataset_path, seq_length, is_training,
                     max_eval_examples=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""

        batch_size = params["batch_size"]
        output_buffer_size = batch_size * 1000

        def extract_fn(data_record):
            features = {
                "query_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "doc_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "label": tf.FixedLenFeature([], tf.int64),
                "len_gt_titles": tf.FixedLenFeature([], tf.int64),
            }
            sample = tf.parse_single_example(data_record, features)

            query_ids = tf.cast(sample["query_ids"], tf.int32)
            doc_ids = tf.cast(sample["doc_ids"], tf.int32)
            label_ids = tf.cast(sample["label"], tf.int32)
            # if "len_gt_titles" in sample:
            len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)
            # else:
            #  len_gt_titles = tf.constant(-1, shape=[1], dtype=tf.int32)
            input_ids = tf.concat((query_ids, doc_ids), 0)

            query_segment_id = tf.zeros_like(query_ids)
            doc_segment_id = tf.ones_like(doc_ids)
            segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "label_ids": label_ids,
                "len_gt_titles": len_gt_titles,
            }
            return features

        dataset = tf.data.TFRecordDataset([dataset_path])
        dataset = dataset.map(
            extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=1000)
        else:
            if max_eval_examples:
                dataset = dataset.take(max_eval_examples)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "input_ids": [seq_length],
                "segment_ids": [seq_length],
                "input_mask": [seq_length],
                "label_ids": [],
                "len_gt_titles": [],
            },
            padding_values={
                "input_ids": 0,
                "segment_ids": 0,
                "input_mask": 0,
                "label_ids": 0,
                "len_gt_titles": 0,
            },
            drop_remainder=True)

        return dataset

    return input_fn