import torch
import logging
import numpy as np
import argparse
import baseline
import sys
from baseline.embeddings import load_embeddings_overlay
from eight_mile.utils import read_config_stream, revlut
from baseline.pytorch.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
from mead.utils import convert_path, parse_extra_args, configure_logger, parse_and_merge_overrides, read_config_file_or_json, index_by_label
from eight_mile.confusion import ConfusionMatrix
from eight_mile.pytorch.train import TrainingTarget, Trainer, LogAllMetrics
from baseline.reader import TSVSeqLabelReader


class ClassifyTarget(TrainingTarget):

    def __init__(self, model, loss=None):
        super().__init__()
        # For MEAD models, the loss is often available as part of the model
        if not loss:
            loss = model.create_loss()
        self._loss = loss
        self._model = model

    def train_step(self, batch):

        truth = torch.tensor(batch['y'], device=self.device)
        batch['x'] = torch.tensor(batch['x'], device=self.device)
        batch['lengths'] = torch.tensor(batch['x_lengths'], device=self.device)
        pred = self._model(batch)
        loss = self._loss(pred, truth)
        batchsz = truth.shape[0]
        report_loss = loss.item() * batchsz
        metrics = {'loss': loss, 'report_loss': report_loss}
        return metrics

    def eval_step(self, batch):

        with torch.no_grad():
            cm = ConfusionMatrix(np.arange(self._model.num_classes))
            truth = torch.tensor(batch['y'], device=self.device)
            batch['x'] = torch.tensor(batch['x'], device=self.device)
            batch['lengths'] = torch.tensor(batch['x_lengths'], device=self.device)
            pred = self._model(batch)
            loss = self._loss(pred, truth)
            cm.add_batch(truth.cpu().numpy(), torch.argmax(pred, -1).cpu().numpy())
            batchsz = truth.shape[0]
            report_loss = loss.item() * batchsz
            metrics = {'loss': loss, 'report_loss': report_loss, 'confusion': cm}
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
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--report_on', type=int, default=100_000)
parser.add_argument('--model_type', default='finetune', choices=['finetune', 'kim', 'bilstm'])
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
parser.add_argument("--distributed",
                    type=str2bool,
                    default=False,
                    help="Are we doing distributed training?")
parser.add_argument('--early_stopping_metric', default='acc')
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="Local rank for distributed training (-1 means use the environment variables to find)")
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
train_module = ClassifyTarget(classifier, nn.NLLLoss())


train_set = reader.load(args.train_file, vocabs=vocabs, batchsz=args.batch_size)
valid_set = reader.load(args.valid_file, vocabs=vocabs, batchsz=args.batch_size)
test_set = reader.load(args.test_file, vocabs=vocabs, batchsz=2)
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
      device=args.device,
      local_rank=args.local_rank,
      distributed=args.distributed,
      report_on=args.report_on,
      grad_accum=args.grad_accum,
      max_steps_per_epoch=args.max_steps_per_epoch)
