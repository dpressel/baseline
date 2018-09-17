import os
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.model import Tagger, create_tagger_model, load_tagger_model
from baseline.tf.tfy import *
from baseline.utils import zip_model, unzip_model
from baseline.tf.embeddings import *


class RNNTaggerModel(Tagger):

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"crf": self.crf, "proj": self.proj, "crf_mask": self.crf_mask, 'span_type': self.span_type}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        # What it should do
        # vocab_suffixes = get_vocab_file_suffixes(basename)
        # for ty in vocab_suffixes:
        #    vocab_file = '{}-{}.vocab'.format(basename, ty)
        #    print('Reading {}', vocab_file)
        #    with open(vocab_file, 'r') as f:
        #        self.vocab[ty] = json.load(f)

        # Until we have backwards compat, figured out, do same as before...
        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(self.char_vocab, f)

    def make_input(self, batch_dict, do_dropout=False):
        y = batch_dict.get('y', None)

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    @staticmethod
    def load(basename, **kwargs):
        basename = unzip_model(basename)
        model = RNNTaggerModel()
        model.sess = kwargs.get('sess', tf.Session())
        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.state') as f:
            state = json.load(f)
            model.crf = bool(state.get('crf', False))
            model.crf_mask = bool(state.get('crf_mask', False))
            model.span_type = state.get('span_type')
            model.proj = bool(state.get('proj', False))

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            model.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

            model.sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0')
            model.probs = tf.get_default_graph().get_tensor_by_name('output/Reshape_1:0')  # TODO: rename
            try:
                model.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
                if not model.crf:
                    print('Warning: meta-data says no CRF but model contains transition matrix!')
                    model.crf = True
            except:
                if model.crf is True:
                    print('Warning: meta-data says there is a CRF but not transition matrix found!')
                model.A = None
                model.crf = False

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        model.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                model.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            model.char_vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        return model

    def save_using(self, saver):
        self.saver = saver

    def _compute_word_level_loss(self, mask):

        nc = len(self.labels)
        # Cross entropy loss
        cross_entropy = tf.one_hot(self.y, nc, axis=-1) * tf.log(tf.nn.softmax(self.probs))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def _compute_sentence_level_loss(self):

        if self.crf_mask:
            assert self.span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            A = tf.get_variable(
                "transitions_raw",
                shape=(len(self.labels), len(self.labels)),
                dtype=tf.float32,
                trainable=True
            )

            self.mask = crf_mask(self.labels, self.span_type, self.labels['<GO>'], self.labels['<EOS>'], self.labels.get('<PAD>'))
            self.inv_mask = tf.cast(tf.equal(self.mask, 0), tf.float32) * tf.constant(-1e4)

            self.A = tf.add(tf.multiply(A, self.mask), self.inv_mask, name="transitions")
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths, self.A)
        else:
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths)
        return tf.reduce_mean(-ll)

    def create_loss(self):

        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float32)
            mask = tf.sign(gold)

            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss()
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        return all_loss

    def __init__(self):
        super(RNNTaggerModel, self).__init__()
        pass

    def get_vocab(self, vocab_type='word'):
        return self.embeddings[vocab_type].vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):

        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict[self.lengths_key]
        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:

            probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)
            batch_sz, _, label_sz = probv.shape
            start = np.full((batch_sz, 1, label_sz), -1e4)
            start[:, 0, self.labels['<GO>']] = 0
            probv = np.concatenate([start, probv], 1)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl + 1]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                viterbi = viterbi[1:]
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = self.sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    @classmethod
    def create(cls, labels, embeddings, **kwargs):

        model = cls()
        model.embeddings = dict()
        for key in embeddings.keys():
            DefaultType = TensorFlowCharConvEmbeddings if key == 'char' else TensorFlowTokenEmbeddings
            model.embeddings[key] = tf_embeddings(embeddings[key], key, DefaultType=DefaultType, **kwargs)

        model.lengths_key = kwargs.get('lengths_key')
        if model.lengths_key is None:
            if 'word' in model.embeddings:
                model.lengths_key = 'word'
            elif 'x' in model.embeddings:
                model.lengths_key = 'x'

        if model.lengths_key is None:
            raise Exception("Require a `lengths_key`")
            # This allows user to short-hand the field to use
        if not model.lengths_key.endswith('_lengths'):
            model.lengths_key += '_lengths'
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))

        model.labels = labels
        nc = len(labels)
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)

        model.sess = kwargs.get('sess', tf.Session())

        hsz = int(kwargs['hsz'])
        model.pdrop_in = kwargs.get('dropin', 0.0)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        model.labels = labels
        model.crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.proj = bool(kwargs.get('proj', False))
        model.feed_input = bool(kwargs.get('feed_input', False))
        model.activation_type = kwargs.get('activation', 'tanh')

        all_embeddings_out = []
        for embedding in model.embeddings.values():
            embeddings_out = embedding.encode()
            all_embeddings_out += [embeddings_out]

        joint = tf.concat(values=all_embeddings_out, axis=2)
        embedseq = tf.nn.dropout(joint, model.pkeep)
        seed = np.random.randint(10e8)

        if rnntype == 'blstm':
            rnnfwd = stacked_lstm(hsz, model.pkeep, nlayers)
            rnnbwd = stacked_lstm(hsz, model.pkeep, nlayers)
            rnnout, _ = tf.nn.bidirectional_dynamic_rnn(rnnfwd, rnnbwd, embedseq, sequence_length=model.lengths, dtype=tf.float32)
            # The output of the BRNN function needs to be joined on the H axis
            rnnout = tf.concat(axis=2, values=rnnout)
        elif rnntype == 'cnn':
            filts = kwargs.get('wfiltsz', None)
            if filts is None:
                filts = [5]
            rnnout = stacked_cnn(embedseq, hsz, model.pkeep, nlayers, filts=filts)
        else:
            rnnfwd = stacked_lstm(hsz, model.pkeep, nlayers)
            rnnout, _ = tf.nn.dynamic_rnn(rnnfwd, embedseq, sequence_length=model.lengths, dtype=tf.float32)

        with tf.variable_scope("output"):
            if model.feed_input is True:
                rnnout = tf.concat(axis=2, values=[rnnout, embedseq])

            # Converts seq to tensor, back to (B,T,W)
            T = tf.shape(rnnout)[1]
            H = rnnout.get_shape()[2]
            # Flatten from [B x T x H] - > [BT x H]
            rnnout_bt_x_h = tf.reshape(rnnout, [-1, H])
            init = xavier_initializer(True, seed)

            with tf.contrib.slim.arg_scope([fully_connected], weights_initializer=init):
                if model.proj is True:
                    hidden = tf.nn.dropout(fully_connected(rnnout_bt_x_h, hsz,
                                                           activation_fn=tf_activation(model.activation_type)), model.pkeep)
                    preds = fully_connected(hidden, nc, activation_fn=None, weights_initializer=init)
                else:
                    preds = fully_connected(rnnout_bt_x_h, nc, activation_fn=None, weights_initializer=init)
            model.probs = tf.reshape(preds, [-1, T, nc])
            model.best = tf.argmax(model.probs, 2)
        return model

BASELINE_TAGGER_MODELS = {
    'default': RNNTaggerModel.create,
}

BASELINE_TAGGER_LOADERS = {
    'default': RNNTaggerModel.load
}


def create_model(labels, embeddings, **kwargs):
    return create_tagger_model(BASELINE_TAGGER_MODELS, labels, embeddings, **kwargs)


def load_model(modelname, **kwargs):
    return load_tagger_model(BASELINE_TAGGER_LOADERS, modelname, **kwargs)
