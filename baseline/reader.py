import os
import re
from typing import List, Dict
import codecs
import json
from itertools import chain
import logging
from collections import Counter
import numpy as np
import baseline.data
from baseline.vectorizers import Dict1DVectorizer, Token1DVectorizer, create_vectorizer, HasPredefinedVocab
from baseline.utils import import_user_module, revlut, exporter, optional_params, Offsets, listify, SingleFileDownloader
from eight_mile.progress import create_progress_bar
__all__ = []
export = exporter(__all__)

logger = logging.getLogger('baseline')


BASELINE_READERS = {}


@export
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
        logger.warning('Warning: attempting to re-defined previously registered handler {} for task {} in registry'.format(name, task))

    BASELINE_READERS[task][name] = cls
    return cls


@export
def create_reader(task, vectorizers, trim, **kwargs):
    name = kwargs.get('type', kwargs.get('reader_type', 'default'))
    Constructor = BASELINE_READERS[task][name]
    return Constructor(vectorizers, trim, **kwargs)


@export
def num_lines(filename):
    """Counts the number of lines in a file.

    :param filename: `str` The name of the file to count the lines of.
    :returns: `int` The number of lines.
    """
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        return sum(1 for _ in f)


def _all_predefined_vocabs(vectorizers):
    return all(isinstance(v, HasPredefinedVocab) for v in vectorizers.values())


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


def _read_from_col(col, files, col_splitter=r'\t', word_splitter=r'\s'):
    """Read from a single column of a file.

    :param col: `int`: The column to read from.
    :param files: List[str]: A list of files to read from.
    :param col_splitter: `str`: The regex that splits a line into columns.
    :param word_splitter: `str`: The regex that will split a column into words.

    :returns: List[str]: The text from the col of each file.
    """
    text = []
    for file_name in files:
        if file_name is None:
            continue
        with codecs.open(file_name, encoding='utf-8', mode='r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line == "":
                    continue
                cols = re.split(col_splitter, line)
                text.append(re.split(word_splitter, cols[col]))
    return text


def _build_vocab_for_col(col, files, vectorizers, text=None, col_splitter=r'\t', word_splitter=r'\s'):
    """Build vocab from a single column in file. (separated by `\t`).

    Used to read a vocab from a single conll column, read a vocab from the
    source or target of a seq2seq file, or reading from a vocab file.

    :param col: `int`: The column to read from.
    :param files: List[str]: A list of files to read from.
    :param vectorizers: dict[str] -> Vectorizer: The vectorizer to use to count the column
    :param text: List[str]: The text from the columns or None
    :param col_splitter: `str`: The regex that splits a line into columns.
    :param word_splitter: `str`: The regex that will split a column into words.

    :returns: dict[str] -> dict[str] -> int: The vocabs.
    """
    text = _read_from_col(col, files, col_splitter, word_splitter) if text is None else text
    vocab = {k: Counter() for k in vectorizers}
    for t in text:
        for k, vect in vectorizers.items():
            vocab[k].update(vect.count(t))
    return vocab


def _check_lens(vectorizers):
    failures = set()
    for k, vect in vectorizers.items():
        mxlen = getattr(vect, 'mxlen', None)
        if mxlen == -1:
            failures.add(k)
    return failures


def _vocab_allowed(vectorizers):
    fails = _check_lens(vectorizers)
    if fails:
        fail_str = "When using a vocab file mxlen for vectorizers must not be `-1`\n"
        vect_str = "\n".join("\t{}".format(fails))
        raise RuntimeError(fail_str + vect_str)


@export
class ParallelCorpusReader:

    def __init__(self, vectorizers, trim=False, truncate=False):
        super().__init__()

        self.src_vectorizers = {}
        self.tgt_vectorizer = None
        for k, vectorizer in vectorizers.items():
            if k == 'tgt':
                self.tgt_vectorizer = vectorizer
                if not self.tgt_vectorizer.emit_begin_tok:
                    self.tgt_vectorizer.emit_begin_tok.append(Offsets.VALUES[Offsets.GO])
                if not self.tgt_vectorizer.emit_end_tok:
                    self.tgt_vectorizer.emit_end_tok.append(Offsets.VALUES[Offsets.EOS])
            else:
                self.src_vectorizers[k] = vectorizer
        self.trim = trim
        self.truncate = truncate

    def build_vocabs(self, files, **kwargs):
        pass

    def load_examples(self, tsfile, vocab1, vocab2, shuffle, sort_key):
        pass

    def load(self, tsfile, vocab1, vocab2, batchsz, shuffle=False, sort_key=None):
        examples = self.load_examples(tsfile, vocab1, vocab2, shuffle, sort_key)
        return baseline.data.ExampleDataFeed(examples, batchsz,
                                             shuffle=shuffle, trim=self.trim, sort_key=sort_key, truncate=self.truncate)


@register_reader(task='seq2seq', name='tsv')
class TSVParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, vectorizers,
                 trim=False, truncate=False, src_col_num=0, tgt_col_num=1, **kwargs):
        super().__init__(vectorizers, trim, truncate)
        self.src_col_num = src_col_num
        self.tgt_col_num = tgt_col_num

    def build_vocabs(self, files, **kwargs):
        if _all_predefined_vocabs(self.src_vectorizers) and isinstance(self.tgt_vectorizer, HasPredefinedVocab):
            logger.info("Skipping building vocabulary.  All vectorizers have predefined vocabs!")
            return {k: v.vocab for k, v in self.src_vectorizers.items()}, self.tgt_vectorizer.vocab
        vocab_file = kwargs.get('vocab_file')
        if vocab_file is not None:
            all_vects = self.src_vectorizers.copy()
            all_vects['tgt'] = self.tgt_vectorizer
            _vocab_allowed(all_vects)
            # Only read the file once
            text = _read_from_col(0, listify(vocab_file))
            src_vocab = _build_vocab_for_col(None, None, self.src_vectorizers, text=text)
            tgt_vocab = _build_vocab_for_col(None, None, {'tgt': self.tgt_vectorizer}, text=text)
            return src_vocab, tgt_vocab['tgt']
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


@export
@register_reader(task='seq2seq', name='default')
class MultiFileParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__(vectorizers, trim, truncate)
        pair_suffix = kwargs['pair_suffix']

        self.src_suffix = pair_suffix[0]
        self.tgt_suffix = pair_suffix[1]
        if not self.src_suffix.startswith('.'):
            self.src_suffix = '.' + self.src_suffix
        if not self.tgt_suffix.startswith('.'):
            self.tgt_suffix = '.' + self.tgt_suffix

    def build_vocabs(self, files, **kwargs):
        if _all_predefined_vocabs(self.src_vectorizers) and isinstance(self.tgt_vectorizer, HasPredefinedVocab):
            logger.info("Skipping building vocabulary.  All vectorizers have predefined vocabs!")
            return {k: v.vocab for k, v in self.src_vectorizers.items()}, self.tgt_vectorizer.vocab
        vocab_file = kwargs.get('vocab_file')
        if vocab_file is not None:
            all_vects = self.src_vectorizers.copy()
            all_vects['tgt'] = self.tgt_vectorizer
            _vocab_allowed(all_vects)
            # Only read the file once.
            text = _read_from_col(0, listify(vocab_file))
            src_vocab = _build_vocab_for_col(None, None, self.src_vectorizers, text=text)
            tgt_vocab = _build_vocab_for_col(None, None, {'tgt': self.tgt_vectorizer}, text=text)
            return src_vocab, tgt_vocab['tgt']
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


def _try_read_labels(**kwargs):
    label_file = kwargs.get('label_file')
    label_list = kwargs.get('label_list')
    label2index = {}
    if label_file:
        pre_labels = Counter(chain(*_read_from_col(0, listify(label_file))))
        label2index = {l: i for i, l in enumerate(pre_labels)}
    elif label_list:
        label2index = {l: i for i, l in enumerate(label_list)}
    return label2index

@export
class SeqPredictReader:

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__()
        self.vectorizers = vectorizers
        self.trim = trim
        self.truncate = truncate
        label_vectorizer_spec = kwargs.get('label_vectorizer', None)
        if label_vectorizer_spec:
            cache = label_vectorizer_spec.get("data_download_cache", os.path.expanduser("~/.bl-data"))
            if 'model_file' in label_vectorizer_spec:
                label_vectorizer_spec['model_file'] = SingleFileDownloader(label_vectorizer_spec['model_file'], cache).download()
            if 'vocab_file' in label_vectorizer_spec:
                label_vectorizer_spec['vocab_file'] = SingleFileDownloader(label_vectorizer_spec['vocab_file'], cache).download()
            if 'transform' in label_vectorizer_spec:
                label_vectorizer_spec['transform_fn'] = label_vectorizer_spec['transform']
            if 'transform_fn' in label_vectorizer_spec and isinstance(label_vectorizer_spec['transform_fn'], str):
                label_vectorizer_spec['transform_fn'] = eval(label_vectorizer_spec['transform_fn'])
            self.label_vectorizer = create_vectorizer(**label_vectorizer_spec)
        else:
            self.label_vectorizer = Dict1DVectorizer(fields='y', mxlen=mxlen)
        self.label2index = {
            Offsets.VALUES[Offsets.PAD]: Offsets.PAD,
            Offsets.VALUES[Offsets.GO]: Offsets.GO,
            Offsets.VALUES[Offsets.EOS]: Offsets.EOS
        }

    def build_vocab(self, files, **kwargs):

        # TODO: Im not sure we should even support this option
        label2index = _try_read_labels(**kwargs)
        if label2index and _all_predefined_vocabs(self.vectorizers):
            offset = len(label2index)
            self.label2index.update({k: v + offset} for k, v in label2index.items())
            logger.info("Skipping building vocabulary.  All vectorizers have predefined vocabs and a label file was given!")
            return {k: v.vocab for k, v in self.vectorizers.items()}
        pre_vocabs = None
        vocabs = {k: Counter() for k in self.vectorizers.keys()}

        vocab_file = kwargs.get('vocab_file')

        if vocab_file:
            _vocab_allowed(self.vectorizers)
            pre_vocabs = _build_vocab_for_col(0, listify(vocab_file), self.vectorizers)

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

        if label2index and not pre_vocabs:
            return vocabs

        vocabs = _filter_vocab(vocabs, kwargs.get('min_f', {}))
        base_offset = len(self.label2index)
        labels.pop(Offsets.VALUES[Offsets.PAD], None)
        for i, k in enumerate(labels.keys()):
            self.label2index[k] = i + base_offset
        if pre_vocabs:
            vocabs = pre_vocabs
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
        return baseline.data.ExampleDataFeed(examples, batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate), texts


@export
class MultiLabelSeqPredictReader:

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__()
        self.vectorizers = vectorizers
        self.trim = trim
        self.truncate = truncate
        label_vectorizer_spec_dict = kwargs.get('label_vectorizers', {'y': Dict1DVectorizer(fields='y', mxlen=mxlen)})
        self.label_vectorizers = {}
        self.label2index = {}
        for k, label_vectorizer_spec in label_vectorizer_spec_dict.items():
            if 'label' not in label_vectorizer_spec:
                label_vectorizer_spec['label'] = k
            cache = label_vectorizer_spec.get("data_download_cache", os.path.expanduser("~/.bl-data"))
            if 'model_file' in label_vectorizer_spec:
                label_vectorizer_spec['model_file'] = SingleFileDownloader(label_vectorizer_spec['model_file'], cache).download()
            if 'vocab_file' in label_vectorizer_spec:
                label_vectorizer_spec['vocab_file'] = SingleFileDownloader(label_vectorizer_spec['vocab_file'], cache).download()
            label_vectorizer = create_vectorizer(**label_vectorizer_spec)

            self.label_vectorizers[k] = label_vectorizer
            self.label2index[k] = {
                Offsets.VALUES[Offsets.PAD]: Offsets.PAD,
                Offsets.VALUES[Offsets.GO]: Offsets.GO,
                Offsets.VALUES[Offsets.EOS]: Offsets.EOS
            }

    def build_vocab(self, files, **kwargs):
        """For this model, we are not going to support label pre-definitions

        :param files:
        :param kwargs:
        :return:
        """
        pre_vocabs = None
        vocabs = {k: Counter() for k in self.vectorizers.keys()}
        labels = {k: Counter() for k in self.label_vectorizers.keys()}

        vocab_file = kwargs.get('vocab_file')

        if vocab_file:
            _vocab_allowed(self.vectorizers)
            pre_vocabs = _build_vocab_for_col(0, listify(vocab_file), self.vectorizers)

        for file in files:
            if file is None:
                continue

            examples = self.read_examples(file)
            for example in examples:
                for k, vectorizer in self.label_vectorizers.items():
                    labels[k].update(vectorizer.count(example))
                for k, vectorizer in self.vectorizers.items():
                    vocab_example = vectorizer.count(example)
                    vocabs[k].update(vocab_example)

        vocabs = _filter_vocab(vocabs, kwargs.get('min_f', {}))
        base_offset = len(Offsets.VALUES) - 1  # Dont put UNK in labels

        for l in labels.keys():
            labels[l].pop(Offsets.VALUES[Offsets.PAD], None)
            for i, k in enumerate(labels[l]):
                self.label2index[l][k] = i + base_offset
        if pre_vocabs:
            vocabs = pre_vocabs
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
            for k, vectorizer in self.label_vectorizers.items():
                example[k], lengths = vectorizer.run(example_tokens, self.label2index[k])
                if lengths is not None:
                    example['{}_lengths'.format(k)] = lengths
            example['ids'] = i
            if np.max(example['heads']) >= example['word_lengths']:
                raise Exception("Invalid head", np.max(example['heads']), example['word_lengths'])
            ts.append(example)

        examples = baseline.data.DictExamples(ts, do_shuffle=shuffle, sort_key=sort_key)
        return baseline.data.ExampleDataFeed(examples, batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate), texts


@export
@register_reader(task='tagger', name='default')
class CONLLSeqReader(SeqPredictReader):

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.named_fields = kwargs.get('named_fields', {})

    def read_examples(self, tsfile):

        tokens = []
        examples = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for i, line in enumerate(f):
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
                    if len(tokens) == 0:
                        raise Exception("Unexpected empty line ({}) in {}".format(i, tsfile))
                    examples.append(tokens)
                    tokens = []
            if len(tokens) > 0:
                examples.append(tokens)
        return examples



@export
@register_reader(task='deps', name='default')
class CONLLParserSeqReader(MultiLabelSeqPredictReader):

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.named_fields = kwargs.get('named_fields', {})

    def read_examples(self, tsfile):

        tokens = []
        examples = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for i, line in enumerate(f):
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
                    if len(tokens) == 0:
                        raise Exception("Unexpected empty line ({}) in {}".format(i, tsfile))
                    examples.append(tokens)
                    tokens = []
            if len(tokens) > 0:
                examples.append(tokens)
        return examples


@export
@register_reader(task='tagger', name='srl')
class SRLSeqReader(SeqPredictReader):
    """Read data in the format described by https://github.com/luheng/deep_srl tested on CoNLL2012 SRL data

    """
    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.named_fields = kwargs.get('named_fields', {})

    def read_examples(self, tsfile):

        examples = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for i, line in enumerate(f):
                pred_surface, labels = line.strip().split("|||")
                pred_surface = pred_surface.strip().split()
                labels = labels.strip().split()
                pred = int(pred_surface[0])
                surface = pred_surface[1:]
                if len(surface) > len(labels):
                    logger.warning("No labels given for line [%s]", line)
                    labels = ['O'] * len(surface)

                tokens = [{'text': s, 'y': l, 'pred': 0} for s, l in zip(surface, labels)]
                tokens[pred]['pred'] = 1
                if tokens:
                    examples.append(tokens)
        return examples


def _norm_ext(ext):
    return ext if ext.startswith('.') else '.' + ext


@export
class SeqLabelReader:

    def __init__(self):
        pass

    def build_vocab(self, files, **kwargs):
        pass

    def load(self, filename, index, batchsz, **kwargs):
        pass


def _get_dir(files):
    if isinstance(files, str):
        if os.path.isdir(files):
            base = files
            files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
        else:
            files = [files]
    return files


@export
class LineSeqLabelReader(SeqLabelReader):

    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = kwargs.get('clean_fn')
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim
        self.truncate = truncate
        self.has_header = bool(kwargs.get('has_header', False))
        self.col_keys = kwargs.get('col_keys', [])
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

    def label_and_sentence(self, line, clean_fn, header):
        """This function produces the sample from a line

        :param line: The raw line
        :param clean_fn: A clean function
        :param header: An optional header
        :return: A tuple of the label and text sample
        """

    def build_vocab(self, files, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary

        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.

        :param files: Either a directory (str), or an array of individual files
        :return:
        """
        vocab_file = kwargs.get('vocab_file')
        self.label2index = _try_read_labels(**kwargs)

        if _all_predefined_vocabs(self.vectorizers):
            if not self.label2index:
                logger.warning("If you provide a label file or list, we can skip tabulating the labels from the data!")
                label_idx = 0
                for file_name in files:
                    with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                        header = next(f) if self.has_header else None
                        for il, line in enumerate(f):
                            label, _ = self.label_and_sentence(line, self.clean_fn, header)
                            if label not in self.label2index:
                                self.label2index[label] = label_idx
                                label_idx += 1
            else:
                logger.info("Skipping label file and vocabulary!")
            logger.info("Skipping building vocabulary!")
            return {k: v.vocab for k, v in self.vectorizers.items()}, self.get_labels()

        if vocab_file and self.label2index:
            _vocab_allowed(self.vectorizers)
            vocab = _build_vocab_for_col(0, listify(vocab_file), self.vectorizers)
            return vocab, self.get_labels()

        label_idx = len(self.label2index)
        if isinstance(files, str):
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]
        vocab = {k: Counter() for k in self.vectorizers.keys()}

        for file_name in files:
            if file_name is None:
                continue
            with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                header = next(f) if self.has_header else None
                for il, line in enumerate(f):
                    label, text = self.label_and_sentence(line, self.clean_fn, header)
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

    def load_text(self, filename, vocabs, batchsz, **kwargs):

        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []
        texts = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            header = next(f) if self.has_header else None
            for il, line in enumerate(f):
                label, text = self.label_and_sentence(line, self.clean_fn, header)
                texts.append(text)
                if len(text) == 0:
                    continue
                if label not in self.label2index:
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
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate), texts

    def load(self, filename, vocabs, batchsz, **kwargs):
    
        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'
    
        examples = []

        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            header = next(f) if self.has_header else None
            for il, line in enumerate(f):
                label, text = self.label_and_sentence(line, self.clean_fn, header)
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

        return self.to_data_loader(examples, shuffle, sort_key, batchsz)

    def to_data_loader(self, examples, shuffle, sort_key, batchsz):
        return baseline.data.ExampleDataFeed(baseline.data.DictExamples(examples,
                                                                        do_shuffle=shuffle,
                                                                        sort_key=sort_key),
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate)
@export
@register_reader(task='classify', name='default')
class TSVSeqLabelReader(LineSeqLabelReader):

    def label_and_sentence(self, line, clean_fn, header):
        if header is None:
            return self.label_and_sentence_no_header(line, clean_fn)
        cols = line.strip().split('\t')
        header = [h.strip() for h in header.split('\t')]
        col_indices = [header.index(c) for c in self.col_keys]
        label_sentence = [cols[c] for c in col_indices]
        label, sentence = label_sentence
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in sentence.split()])))
        return label, text

    def label_and_sentence_no_header(self, line, clean_fn):
        label_text = re.split(TSVSeqLabelReader.SPLIT_ON, line)
        label = label_text[0]
        text = label_text[1:]
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text])))
        text = TSVSeqLabelReader.splits(text)
        return label, text


class IndexPairLabelReader(SeqLabelReader):
    """Shared vocabulary paired label reader

    This is a TSV reader like TSVSeqLabelReader, but the vocabulary and vectorizer are used to produce a paired input.
    This would be useful, for example, if the problem is e.g. NLI classification.
    In this implementation, we assume that the user wishes to share the vocab and vectorizer for the 2 inputs, and
    we use them to generate a key that is slightly different from the user key defined.  So for instance, if the user
    defines the key as `x`, in the `TSVSeqLabelReader` this would produce a vector `x` and `x_lengths`.  In this reader,
    however, the keys will be generated as `x[0]` and `x[0]_lengths` for the first item in the pair, and index `1` for
    the second
    """
    REPLACE = {"'s": " 's ",
               "'ve": " 've ",
               "n't": " n't ",
               "'re": " 're ",
               "'d": " 'd ",
               "'ll": " 'll ",
               ",": " , ",
               "!": " ! ",
               }

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = kwargs.get('clean_fn')
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim
        self.truncate = truncate
        self.use_token_type = bool(kwargs.get('use_token_type', False))
        self.start_tokens_1 = kwargs.get('start_tokens_1', [])
        self.start_tokens_2 = kwargs.get('start_tokens_2', [])
        self.end_tokens_1 = kwargs.get('end_tokens_1', [])
        self.end_tokens_2 = kwargs.get('end_tokens_2', [])

        self.col_keys = kwargs.get('col_keys', ['pairID', 'sentence1', 'sentence2', 'gold_label'])
        example_type = kwargs.get('example_type', 'single')
        self.convert_to_example = self.convert_to_example_dual if example_type == 'dual' else self.convert_to_example_single
        self.has_header = True

    @staticmethod
    def splits(text):
        return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

    @staticmethod
    def do_clean(l):
        l = l.lower()
        l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
        for k, v in IndexPairLabelReader.REPLACE.items():
            l = l.replace(k, v)
        return l.strip()

    def index_pair_label(self, line, clean_fn, header):
        """Does the actual work of handling the file format abstractions.

        :param line: A line of text fromn the file
        :param clean_fn: A cleaning function to apply
        :return: A tuple of the shape (index, text1, text2, label)
        """

    def build_vocab(self, files, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary

        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.

        :param files: Either a directory (str), or an array of individual files
        :return:
        """
        vocab_file = kwargs.get('vocab_file')
        self.label2index = _try_read_labels(**kwargs)
        if _all_predefined_vocabs(self.vectorizers):
            if not self.label2index:
                logger.warning("If you provide a label file, we can skip tabulating the labels from the data!")
                label_idx = 0
                for file_name in files:
                    with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                        header = next(f) if self.has_header else None
                        for il, line in enumerate(f):
                            idx, text1, text2, label = self.index_pair_label(line, self.clean_fn, header)
                            if label not in self.label2index:
                                self.label2index[label] = label_idx
                                label_idx += 1
            else:
                logger.info("Skipping label file and vocabulary!")
            logger.info("Skipping building vocabulary!")
            return {k: v.vocab for k, v in self.vectorizers.items()}, self.get_labels()

        if vocab_file is not None and self.label2index:
            _vocab_allowed(self.vectorizers)
            vocab = _build_vocab_for_col(0, listify(vocab_file), self.vectorizers)
            return vocab, self.get_labels()

        label_idx = len(self.label2index)
        if isinstance(files, str):
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]
        vocab = {k: Counter() for k in self.vectorizers.keys()}

        for file_name in files:
            if file_name is None:
                continue
            with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                header = next(f) if self.has_header else None
                for il, line in enumerate(f):
                    idx, text1, text2, label = self.index_pair_label(line, self.clean_fn, header)
                    if len(text1) < 1 or len(text2) < 1:
                        raise Exception(f"Invalid line @ {il}: [{line}]")

                    for k, vectorizer in self.vectorizers.items():
                        vocab_file = vectorizer.count(text1) + vectorizer.count(text2)
                        vocab[k].update(vocab_file)

                    if label not in self.label2index:
                        self.label2index[label] = label_idx
                        label_idx += 1
                    #if il > 0 and il % 1000 == 0:
                    #    print(il)
        vocab = _filter_vocab(vocab, kwargs.get('min_f', {}))
        return vocab, self.get_labels()

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load_text(self, filename, vocabs, batchsz, **kwargs):

        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []
        texts = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            header = next(f) if self.has_header else None
            for il, line in enumerate(f):
                idx, text1, text2, label = self.index_pair_label(line, self.clean_fn, header)
                if len(text1) < 1 or len(text2) < 1:
                    raise Exception(f"Invalid line @ {il}: [{line}]")
                example_dict = self.convert_to_example(vocabs, text1, text2)
                example_dict['y'] = self.label2index[label]
                example_dict['idx'] = idx
                examples.append(example_dict)

        return baseline.data.ExampleDataFeed(baseline.data.DictExamples(examples,
                                                                        do_shuffle=shuffle,
                                                                        sort_key=sort_key),
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim,
                                             truncate=self.truncate), texts

    def convert_to_example_dual(self, vocabs: Dict[str, str], text1: List[str], text2: List[str]) -> Dict[str, object]:
        """Convert to 2 vectors, Useful for dual-encoder-style models

        :param vocabs: Vocabs to use
        :param text1: First text
        :param text2: Second text
        :return: An example dict
        """
        text = [text1, text2]
        example_dict = dict()
        for pair_idx in range(2):
            for vec_name, vectorizer in self.vectorizers.items():
                example_dict[f"{vec_name}[{pair_idx}]"], lengths = vectorizer.run(text[pair_idx], vocabs[vec_name])
                if lengths is not None:
                    example_dict[f'{vec_name}[{pair_idx}]_lengths'] = lengths
        return example_dict

    def convert_to_example_single(self, vocabs: Dict[str, str], text1: List[str], text2: List[str]) -> Dict[str, object]:
        """Convert to a concatenated vector.  Useful for BERT-style models

        :param vocabs: Vocabs to use
        :param text1: First text
        :param text2: Second text
        :return: An example dict
        """
        example_dict = dict()

        for key, vectorizer in self.vectorizers.items():
            v = vocabs[key]
            vec_1, lengths = vectorizer.run(text1, v)
            vec_1 = vec_1[:lengths]
            vec_2, lengths = vectorizer.run(text2, v)
            vec_2 = vec_2[:lengths]

            if self.start_tokens_1:
                vec_1 = np.concatenate([np.array([v[x] for x in self.start_tokens_1]), vec_1])
            if self.end_tokens_1:
                vec_1 = np.concatenate([vec_1, np.array([v[x] for x in self.end_tokens_1])])

            if self.start_tokens_2:
                vec_2 = np.concatenate([np.array([v[x] for x in self.start_tokens_2]), vec_2])
            if self.end_tokens_2:
                vec_2 = np.concatenate([vec_2, np.array([v[x] for x in self.end_tokens_2])])
            lengths_1 = vec_1.shape[-1]
            lengths_2 = vec_2.shape[-1]
            total_length = lengths_1 + lengths_2
            vec = np.concatenate([vec_1, vec_2])
            extend = vectorizer.mxlen - total_length

            if extend < 0:
                logger.warning(f"total length {total_length} exceeds allocated vectorizer width of {vectorizer.mxlen}")
                vec = vec[:vectorizer.mxlen]
            elif extend > 0:
                vec = np.concatenate([vec, np.zeros(extend, vec.dtype)])

            example_dict[key] = vec
            if self.use_token_type:
                token_type = np.zeros_like(vec)
                token_type[lengths_1:] = 1
                example_dict[f"{key}_tt"] = token_type
        return example_dict

    def load(self, filename, vocabs, batchsz, **kwargs):

        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []
        with open(filename, encoding='utf-8-sig', mode='r') as f:
            header = next(f) if self.has_header else None
            #f = list(f)
            ##pg = create_progress_bar(len(f), 'tqdm')
            for il, line in enumerate(f):
                try:
                    idx, text1, text2, label = self.index_pair_label(line, self.clean_fn, header)
                    if len(text1) < 1 or len(text2) < 1:
                        raise Exception(f"Invalid line @ {il}: [{line}]")
                    example_dict = self.convert_to_example(vocabs, text1.split(), text2.split())
                    if label not in self.label2index:
                        continue
                    example_dict['y'] = self.label2index[label]
                    example_dict['idx'] = idx
                    examples.append(example_dict)
                except Exception as e:
                    logger.warning(f'Skipping invalid line {il}')

        return baseline.data.ExampleDataFeed(baseline.data.DictExamples(examples,
                                                                        do_shuffle=shuffle,
                                                                        sort_key=sort_key),
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate)


@export
@register_reader(task='classify', name='tsv-paired-shared-vec')
class TSVIndexPairLabelReader(IndexPairLabelReader):

    def index_pair_label(self, line, clean_fn, header):
        cols = line.strip().split('\t')
        header = [h.strip() for h in header.split('\t')]
        col_indices = [header.index(c) for c in self.col_keys]
        # QQP files have some invalid samples
        #if max(col_indices) > len(cols):
        #    raise Exception(f"Error: {cols}")
        index_pair_label_list = [cols[c] for c in col_indices]
        index, text1, text2, label = index_pair_label_list
        text1 = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text1.split()])))
        text2 = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text2.split()])))
        return index, text1, text2, label


@export
@register_reader(task='classify', name='jsonl-paired-shared-vec')
class JSONLIndexPairLabelReader(IndexPairLabelReader):

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__(vectorizers, trim, truncate, **kwargs)
        self.has_header = False

    def index_pair_label(self, line, clean_fn, header):
        cols = json.loads(line)
        index_pair_label_list = [cols[c] for c in self.col_keys]
        index, text1, text2, label = index_pair_label_list
        text1 = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text1.split()])))
        text2 = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text2.split()])))
        return index, text1, text2, label


@export
@register_reader(task='lm', name='default')
class LineSeqReader:

    def __init__(self, vectorizers, trim=False, **kwargs):
        self.nctx = kwargs['nctx']
        self.vectorizers = vectorizers

    def build_vocab(self, files, **kwargs):
        if _all_predefined_vocabs(self.vectorizers):
            logger.info("Skipping building vocabulary.  All vectorizers have predefined vocabs!")
            return {k: v.vocab for k, v in self.vectorizers.items()}

        vocab_file = kwargs.get('vocab_file')
        if vocab_file is not None:
            _vocab_allowed(self.vectorizers)
            return _build_vocab_for_col(0, listify(vocab_file), self.vectorizers)

        vocabs = {k: Counter() for k in self.vectorizers.keys()}

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
