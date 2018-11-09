from six.moves import filter

import os
import re
import codecs
from collections import Counter
import numpy as np
import baseline.data
from baseline.vectorizers import Dict1DVectorizer, GOVectorizer
from baseline.utils import import_user_module, revlut, export, optional_params, Offsets

__all__ = []
exporter = export(__all__)


BASELINE_READERS = {}


@exporter
@optional_params
def register_reader(cls, task, name=None):
    """Register your own `Reader`

    Use this pattern if you want to provide an override to a `Reader` class.

    """
    if name is None:
        name = cls.__name__

    if task not in BASELINE_READERS:
        BASELINE_READERS[task] = {}

    if name in BASELINE_READERS[task]:
        raise Exception('Error: attempt to re-defined previously registered handler {} for task {} in registry'.format(name, task))

    BASELINE_READERS[task][name] = cls
    return cls


@exporter
def create_reader(task, vectorizers, trim, **kwargs):
    name = kwargs.get('reader_type', 'default')
    Constructor = BASELINE_READERS[task][name]
    return Constructor(vectorizers, trim, **kwargs)


@exporter
def num_lines(filename):
    """Counts the number of lines in a file.

    :param filename: `str` The name of the file to count the lines of.
    :returns: `int` The number of lines.
    """
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        return sum(1 for _ in f)


def _filter_vocab(vocab, min_fs):
    """Filter down the vocab based on rules in the vectorizers.

    :param vocab: `dict[Counter]`: A dict of vocabs.
    :param min_fs: `dict[int]: A dict of cutoffs.

    Note:
        Any key in the min_fs dict should appear in the vocab dict.

    :returns: `dict[dict]`: A dict of new filtered vocabs.
    """
    for k, min_f in min_fs.items():
        # If we don't filter then skip to save an iteration through the vocab
        if min_f == -1:
            continue
        vocab[k] = dict(filter(lambda x: x[1] >= min_f, vocab[k].items()))
    return vocab


def _build_vocab_for_col(col, files, vectorizers):
    vocabs = dict()

    for key in vectorizers.keys():
        vocabs[key] = Counter()

    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                cols = re.split("\t", line)
                text = re.split("\s", cols[col])
                for key, vectorizer in vectorizers.items():
                    vocabs[key].update(vectorizer.count(text))
    return vocabs


@exporter
class ParallelCorpusReader(object):

    def __init__(self, vectorizers, trim=False):

        self.src_vectorizers = {}
        self.tgt_vectorizer = None
        for k, vectorizer in vectorizers.items():
            if k == 'tgt':
                self.tgt_vectorizer = GOVectorizer(vectorizer)
            else:
                self.src_vectorizers[k] = vectorizer
        self.trim = trim

    def build_vocabs(self, files, **kwargs):
        pass

    def load_examples(self, tsfile, vocab1, vocab2, shuffle, sort_key):
        pass

    def load(self, tsfile, vocab1, vocab2, batchsz, shuffle=False, sort_key=None):
        examples = self.load_examples(tsfile, vocab1, vocab2, shuffle, sort_key)
        return baseline.data.ExampleDataFeed(examples, batchsz,
                                             shuffle=shuffle, trim=self.trim, sort_key=sort_key)


@register_reader(task='seq2seq', name='tsv')
class TSVParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, vectorizers,
                 trim=False, src_col_num=0, tgt_col_num=1, **kwargs):
        super(TSVParallelCorpusReader, self).__init__(vectorizers, trim)
        self.src_col_num = src_col_num
        self.tgt_col_num = tgt_col_num

    def build_vocabs(self, files, **kwargs):
        src_vocab = _build_vocab_for_col(self.src_col_num, files, self.src_vectorizers)
        tgt_vocab = _build_vocab_for_col(self.tgt_col_num, files, {'tgt': self.tgt_vectorizer})
        min_f = kwargs.get('min_f', {})
        tgt_min_f = {'tgt': min_f.pop('tgt', -1)}
        src_vocab = _filter_vocab(src_vocab, min_f)
        tgt_vocab = _filter_vocab(tgt_vocab, tgt_min_f)
        return src_vocab, tgt_vocab['tgt']

    def load_examples(self, tsfile, src_vocabs, tgt_vocab, do_shuffle, src_sort_key):
        ts = []
        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                splits = re.split("\t", line.strip())
                src = list(filter(lambda x: len(x) != 0, re.split("\s+", splits[0])))

                example = {}
                for k, vectorizer in self.src_vectorizers.items():
                    example[k], length = vectorizer.run(src, src_vocabs[k])
                    if length is not None:
                        example['{}_lengths'.format(k)] = length

                tgt = list(filter(lambda x: len(x) != 0, re.split("\s+", splits[1])))
                example['tgt'], example['tgt_lengths'] = self.tgt_vectorizer.run(tgt, tgt_vocab)
                ts.append(example)
        return baseline.data.Seq2SeqExamples(ts, do_shuffle=do_shuffle, src_sort_key=src_sort_key)


@exporter
@register_reader(task='seq2seq', name='default')
class MultiFileParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, vectorizers, trim=False, **kwargs):
        super(MultiFileParallelCorpusReader, self).__init__(vectorizers, trim)
        pair_suffix = kwargs['pair_suffix']

        self.src_suffix = pair_suffix[0]
        self.tgt_suffix = pair_suffix[1]
        if not self.src_suffix.startswith('.'):
            self.src_suffix = '.' + self.src_suffix
        if not self.tgt_suffix.startswith('.'):
            self.tgt_suffix = '.' + self.tgt_suffix

    # 2 possibilities here, either we have a vocab file, e.g. vocab.bpe.32000, or we are going to generate
    # from each column
    def build_vocabs(self, files, **kwargs):
        if len(files) == 1 and os.path.exists(files[0]):
            src_vocab = _build_vocab_for_col(0, files, self.src_vectorizers)
            tgt_vocab = _build_vocab_for_col(0, files, {'tgt': self.tgt_vectorizer})
        else:
            src_vocab = _build_vocab_for_col(0, [f + self.src_suffix for f in files], self.src_vectorizers)
            tgt_vocab = _build_vocab_for_col(0, [f + self.tgt_suffix for f in files], {'tgt': self.tgt_vectorizer})
        min_f = kwargs.get('min_f', {})
        tgt_min_f = {'tgt': min_f.pop('tgt', -1)}
        src_vocab = _filter_vocab(src_vocab, min_f)
        tgt_vocab = _filter_vocab(tgt_vocab, tgt_min_f)
        return src_vocab, tgt_vocab['tgt']

    def load_examples(self, tsfile, src_vocabs, tgt_vocab, do_shuffle, src_sort_key):
        ts = []

        with codecs.open(tsfile + self.src_suffix, encoding='utf-8', mode='r') as fsrc:
            with codecs.open(tsfile + self.tgt_suffix, encoding='utf-8', mode='r') as ftgt:
                for src, tgt in zip(fsrc, ftgt):
                    example = {}
                    src = re.split("\s+", src.strip())
                    for k, vectorizer in self.src_vectorizers.items():
                        example[k], length = vectorizer.run(src, src_vocabs[k])
                        if length is not None:
                            example['{}_lengths'.format(k)] = length
                    tgt = re.split("\s+", tgt.strip())
                    example['tgt'], example['tgt_lengths'] = self.tgt_vectorizer.run(tgt, tgt_vocab)
                    ts.append(example)
        return baseline.data.Seq2SeqExamples(ts, do_shuffle=do_shuffle, src_sort_key=src_sort_key)


@exporter
class SeqPredictReader(object):

    def __init__(self, vectorizers, trim=False, mxlen=-1, **kwargs):
        super(SeqPredictReader, self).__init__()
        self.vectorizers = vectorizers
        self.trim = trim
        self.label2index = {
            Offsets.VALUES[Offsets.PAD]: Offsets.PAD,
            Offsets.VALUES[Offsets.GO]: Offsets.GO,
            Offsets.VALUES[Offsets.EOS]: Offsets.EOS
        }
        self.label_vectorizer = Dict1DVectorizer(fields='y', mxlen=mxlen)

    def build_vocab(self, files, **kwargs):

        vocabs = {}
        for key in self.vectorizers.keys():
            vocabs[key] = Counter()

        labels = Counter()
        for file in files:
            if file is None:
                continue

            examples = self.read_examples(file)
            for example in examples:
                labels.update(self.label_vectorizer.count(example))
                for k, vectorizer in self.vectorizers.items():
                    vocab_example = vectorizer.count(example)
                    vocabs[k].update(vocab_example)

        vocabs = _filter_vocab(vocabs, kwargs.get('min_f', {}))
        base_offset = len(self.label2index)
        for i, k in enumerate(labels.keys()):
            self.label2index[k] = i + base_offset
        return vocabs

    def read_examples(self):
        pass

    def load(self, filename, vocabs, batchsz, shuffle=False, sort_key=None):

        ts = []
        texts = self.read_examples(filename)

        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        for i, example_tokens in enumerate(texts):
            example = {}
            for k, vectorizer in self.vectorizers.items():
                example[k], lengths = vectorizer.run(example_tokens, vocabs[k])
                if lengths is not None:
                    example['{}_lengths'.format(k)] = lengths
            example['y'], lengths = self.label_vectorizer.run(example_tokens, self.label2index)
            example['y_lengths'] = lengths
            example['ids'] = i
            ts.append(example)
        examples = baseline.data.DictExamples(ts, do_shuffle=shuffle, sort_key=sort_key)
        return baseline.data.ExampleDataFeed(examples, batchsz=batchsz, shuffle=shuffle, trim=self.trim), texts


@exporter
@register_reader(task='tagger', name='default')
class CONLLSeqReader(SeqPredictReader):

    def __init__(self, vectorizers, trim=False, **kwargs):
        super(CONLLSeqReader, self).__init__(vectorizers, trim, **kwargs)
        self.named_fields = kwargs.get('named_fields', {})

    def read_examples(self, tsfile):

        tokens = []
        examples = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                states = re.split("\s", line.strip())

                token = dict()
                if len(states) > 1:
                    for j in range(len(states)):
                        noff = j - len(states)
                        if noff >= 0:
                            noff = j
                        field_name = self.named_fields.get(str(j),
                                                           self.named_fields.get(str(noff), str(j)))
                        token[field_name] = states[j]
                    tokens.append(token)

                else:
                    examples.append(tokens)
                    tokens = []

        return examples


@exporter
class SeqLabelReader(object):

    def __init__(self):
        pass

    def build_vocab(self, files, **kwargs):
        pass

    def load(self, filename, index, batchsz, **kwargs):
        pass


@exporter
@register_reader(task='classify', name='default')
class TSVSeqLabelReader(SeqLabelReader):

    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }

    def __init__(self, vectorizers, trim=False, **kwargs):
        super(TSVSeqLabelReader, self).__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = kwargs.get('clean_fn')
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim

    SPLIT_ON = '[\t\s]+'

    @staticmethod
    def splits(text):
        return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

    @staticmethod
    def do_clean(l):
        l = l.lower()
        l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
        for k, v in TSVSeqLabelReader.REPLACE.items():
            l = l.replace(k, v)
        return l.strip()

    @staticmethod
    def label_and_sentence(line, clean_fn):
        label_text = re.split(TSVSeqLabelReader.SPLIT_ON, line)
        label = label_text[0]
        text = label_text[1:]
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text])))
        text = list(filter(lambda s: len(s) != 0, re.split('\s+', text)))
        return label, text

    def build_vocab(self, files, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary

        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.

        :param files: Either a directory (str), or an array of individual files
        :return:
        """
        label_idx = len(self.label2index)
        if type(files) == str:
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]
        vocab = dict()
        for k in self.vectorizers.keys():
            vocab[k] = Counter()

        for file in files:
            if file is None:
                continue
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for il, line in enumerate(f):
                    label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
                    if len(text) == 0:
                        continue

                    for k, vectorizer in self.vectorizers.items():
                        vocab_file = vectorizer.count(text)
                        vocab[k].update(vocab_file)

                    if label not in self.label2index:
                        self.label2index[label] = label_idx
                        label_idx += 1

        vocab = _filter_vocab(vocab, kwargs.get('min_f', {}))

        return vocab, self.get_labels()

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load(self, filename, vocabs, batchsz, **kwargs):

        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []

        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for il, line in enumerate(f):
                label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
                if len(text) == 0:
                    continue
                y = self.label2index[label]
                example_dict = dict()
                for k, vectorizer in self.vectorizers.items():
                    example_dict[k], lengths = vectorizer.run(text, vocabs[k])
                    if lengths is not None:
                        example_dict['{}_lengths'.format(k)] = lengths

                example_dict['y'] = y
                examples.append(example_dict)
        return baseline.data.ExampleDataFeed(baseline.data.DictExamples(examples,
                                                                        do_shuffle=shuffle,
                                                                        sort_key=sort_key),
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim)


@exporter
@register_reader(task='lm', name='default')
class LineSeqReader(object):

    def __init__(self, vectorizers, trim, **kwargs):
        self.nctx = kwargs['nctx']
        self.vectorizers = vectorizers

    def build_vocab(self, files, **kwargs):

        vocabs = {}
        for key in self.vectorizers.keys():
            vocabs[key] = Counter()
            vocabs[key] = Counter()

        for file in files:
            if file is None:
                continue

            with codecs.open(file, encoding='utf-8', mode='r') as f:
                sentences = []
                for line in f:
                    sentences += line.split() + ['<EOS>']
                for k, vectorizer in self.vectorizers.items():
                    vocabs[k].update(vectorizer.count(sentences))

        vocabs = _filter_vocab(vocabs, kwargs.get('min_f', {}))
        return vocabs

    def load(self, filename, vocabs, batchsz, tgt_key='x'):

        x = dict()
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            sentences = []
            for line in f:
                sentences += line.split() + ['<EOS>']
            for k, vectorizer in self.vectorizers.items():
                vec, valid_lengths = vectorizer.run(sentences, vocabs[k])
                x[k] = vec[:valid_lengths]
                shp = list(vectorizer.get_dims())
                shp[0] = valid_lengths
                x['{}_dims'.format(k)] = tuple(shp)

        return baseline.data.SeqWordCharDataFeed(x, self.nctx, batchsz, tgt_key=tgt_key)
