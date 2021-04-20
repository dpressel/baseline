import tensorflow as tf
from eight_mile.tf.layers import set_tf_eager_debug
#set_tf_eager_debug(True)
import logging
import numpy as np
import argparse
import baseline
import sys
from baseline.embeddings import load_embeddings_overlay
from eight_mile.utils import read_config_stream, revlut
from baseline.tf.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
from mead.utils import convert_path, parse_extra_args, configure_logger, parse_and_merge_overrides, read_config_file_or_json, index_by_label
from eight_mile.confusion import ConfusionMatrix
from eight_mile.tf.train import TrainingTarget, Trainer, LogAllMetrics
from baseline.reader import TSVSeqLabelReader
from tensorflow.compat.v1 import count_nonzero

def cross_entropy_loss(pred, targets):
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=targets, logits=pred)


class ClassifyTarget(TrainingTarget):

    def __init__(self, model, loss=None, name=None):
        super().__init__(name=name)
        # For MEAD models, the loss is often available as part of the model
        if not loss:
            loss = model.create_loss()
        self._loss = loss
        self._model = model

    def train_step(self, batch):

        truth = batch['y']
        pred = self._model(batch)
        loss = self._loss(pred, truth)
        batchsz = truth.shape[0]
        report_loss = loss * batchsz
        metrics = {'loss': loss, 'report_loss': report_loss}
        return metrics

    def eval_step(self, batch):
        truth = batch['y']
        nc = self._model.num_classes
        pred = self._model(batch)
        loss = self._loss(pred, truth)
        y = tf.cast(truth, tf.int64)
        per_replica_cm = tf.zeros((nc, nc), dtype=tf.int64)
        y_ = tf.argmax(pred, axis=1, output_type=tf.int64)
        indices = tf.stack((y, y_), axis=-1)
        dense_shape = tf.cast(tf.shape(per_replica_cm), tf.int64)
        sparse_ups = tf.SparseTensor(indices=indices, values=tf.ones(get_shape_as_list(indices)[0], dtype=tf.int64),
                                     dense_shape=dense_shape)
        per_replica_cm = tf.compat.v1.sparse_add(per_replica_cm, sparse_ups)

        batchsz = truth.shape[0]
        report_loss = loss * batchsz
        metrics = {'loss': loss, 'report_loss': report_loss, 'confusion': per_replica_cm}
        return metrics

    @property
    def model(self):
        return self._model


def create_model(embeddings, model_type='finetune'):
    if model_type == 'finetune':
        classifier = FineTuneModel(len(labels), {'x': embeddings})
    elif model_type == 'kim':
        embeddings_stack = EmbeddingsStack({'x': embeddings})
        classifier = EmbedPoolStackModel(len(labels), embeddings_stack,
                                         WithoutLength(ParallelConv(embeddings.output_dim, 100, [3, 4, 5])))
    elif model_type == 'bilstm':
        embeddings_stack = EmbeddingsStack({'x': embeddings})
        classifier = EmbedPoolStackModel(len(labels), embeddings_stack,
                                         BiLSTMEncoderHidden(embeddings.output_dim, 300, nlayers=1))
    else:
        raise Exception(f"We dont support model {model_type} in this program yet")
    return classifier

logger = logging.getLogger(__file__)
DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_DATASETS_LOC = 'config/datasets.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
DEFAULT_EMBEDDINGS_LOC = 'config/embeddings.json'
DEFAULT_VECTORIZERS_LOC = 'config/vecs.json'

parser = argparse.ArgumentParser(description='Encode a sentence as an embedding')
parser.add_argument('--subword_model_file', help='Subword model file')
parser.add_argument('--nctx', default=256, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--vec_id', default='bert-base-uncased', help='Reference to a specific embedding type')
parser.add_argument('--embed_id', default='bert-base-uncased', help='What type of embeddings to use')
parser.add_argument('--train_file', required=True)
parser.add_argument('--valid_file', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--embeddings', help='index of embeddings: local file, remote URL or mead-ml/hub ref', type=convert_path)
parser.add_argument('--vecs', help='index of vectorizers: local file, remote URL or hub mead-ml/ref', type=convert_path)
parser.add_argument('--modules', default=[])
parser.add_argument('--optim', default='adam')
parser.add_argument('--max_steps_per_epoch', type=int)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--report_on', type=int, default=100_000)
parser.add_argument('--model_type', default='finetune', choices=['finetune', 'kim', 'bilstm'])
parser.add_argument('--early_stopping_metric', default='loss')
args = parser.parse_args()


for module in args.modules:
    import_user_module(module)

args.embeddings = convert_path(DEFAULT_EMBEDDINGS_LOC) if args.embeddings is None else args.embeddings
args.embeddings = read_config_stream(args.embeddings)

args.vecs = convert_path(DEFAULT_VECTORIZERS_LOC) if args.vecs is None else args.vecs

vecs_index = read_config_stream(args.vecs)
vecs_set = index_by_label(vecs_index)
vec_params = vecs_set[args.vec_id]
vec_params['mxlen'] = args.nctx

if 'transform' in vec_params:
    vec_params['transform_fn'] = vec_params['transform']

if 'transform_fn' in vec_params and isinstance(vec_params['transform_fn'], str):
    vec_params['transform_fn'] = eval(vec_params['transform_fn'])

vectorizer = create_vectorizer(**vec_params)
embeddings_index = read_config_stream(args.embeddings)
embeddings_set = index_by_label(embeddings_index)
embeddings_params = embeddings_set[args.embed_id]
embeddings_params['preserve_vocab_indices'] = True
embeddings = load_embeddings_overlay(embeddings_set, embeddings_params, vectorizer.vocab)

idx2word = revlut(embeddings['vocab'])
vocabs = {'x': embeddings['vocab']}

embeddings = embeddings['embeddings']

reader = TSVSeqLabelReader({'x': vectorizer}, clean_fn=TSVSeqLabelReader.do_clean)

# This builds a set of counters
vocabs, labels = reader.build_vocab([args.train_file,
                                     args.valid_file,
                                     args.test_file])

classifier = create_model(embeddings, args.model_type)
train_module = ClassifyTarget(classifier, cross_entropy_loss)


NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000
class Data:

    def __init__(self, ts, batchsz):
        self.x, self.y = self._to_tensors(ts)
        self.batchsz = batchsz
        self.length = len(ts) // self.batchsz

    def _to_tensors(self, ts):
        x = []
        y = []
        for sample in ts:
            x.append(sample['x'].squeeze())
            y.append(sample['y'].squeeze())
        return np.stack(x), np.stack(y)

    def __len__(self):
        return self.length

    def load(self, training=False):
        SET_TRAIN_FLAG(training)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        self.dataset = self.dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        self.dataset = self.dataset.batch(self.batchsz)
        self.dataset = self.dataset.map(lambda x, y: {'x': x, 'lengths': count_nonzero(x, axis=1), 'y': y})
        self.dataset = self.dataset.prefetch(NUM_PREFETCH)
        return self

    def __iter__(self):
        return iter(self.dataset)


train_set = Data(reader.load(args.train_file, vocabs=vocabs, batchsz=1), args.batch_size).load(True)
valid_set = Data(reader.load(args.valid_file, vocabs=vocabs, batchsz=1), args.batch_size).load()
test_set = Data(reader.load(args.test_file, vocabs=vocabs, batchsz=1), args.batch_size).load()
t = Trainer(train_module,
            lr=args.lr,
            train_metric_observers=LogAllMetrics("train"),
            valid_metric_observers=LogAllMetrics("valid"),
            test_metric_observers=LogAllMetrics("test"),
            optim=args.optim,
            )
t.run(train_set, valid_set, test_set,
      early_stopping_metric=args.early_stopping_metric,
      num_epochs=args.num_epochs,
      report_on=args.report_on,
      max_steps_per_epoch=args.max_steps_per_epoch,
      progress_bar='tqdm')
