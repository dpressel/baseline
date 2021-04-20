import csv
from itertools import chain
import numpy as np
from eight_mile.utils import exporter
from collections import OrderedDict


__all__ = []
export = exporter(__all__)


@export
class ConfusionMatrix:
    """Confusion matrix with metrics

    This class accumulates classification output, and tracks it in a confusion matrix.
    Metrics are available that use the confusion matrix
    """

    def __init__(self, labels, cm = None):
        """Constructor with input labels

        :param labels: Either a dictionary (`k=int,v=str`) or an array of labels
        """
        if isinstance(labels, dict):
            self.labels = []
            for i in range(len(labels)):
                self.labels.append(labels[i])
        else:
            self.labels = labels
        nc = len(self.labels)
        if cm is not None:
            if len(cm) != len(self.labels):
                raise Exception("Unexpected shape mismatch")
            self._cm = cm
        else:
            self._cm = np.zeros((nc, nc), dtype=np.int)

    def update(self, cm):
        if hasattr(cm, 'labels'):
            if len(self.labels) != len(cm.labels):
                raise Exception("Cannot update, confusion matrices don't match")
            self._cm += cm._cm
        else:
            self._cm += cm

    def add(self, truth, guess):
        """Add a single value to the confusion matrix based off `truth` and `guess`

        :param truth: The real `y` value (or ground truth label)
        :param guess: The guess for `y` value (or assertion)
        """

        self._cm[truth, guess] += 1

    def __str__(self):
        values = []
        width = max(8, max(len(x) for x in self.labels) + 1)
        for i, label in enumerate([""] + self.labels):
            values += ["{:>{width}}".format(label, width=width + 1)]
        values += ["\n"]
        for i, label in enumerate(self.labels):
            values += ["{:>{width}}".format(label, width=width + 1)]
            for j in range(len(self.labels)):
                values += ["{:{width}d}".format(self._cm[i, j], width=width + 1)]
            values += ["\n"]
        values += ["\n"]
        return "".join(values)

    def save(self, outfile):
        ordered_fieldnames = OrderedDict([("labels", None)] + [(l, None) for l in self.labels])
        with open(outfile, "w") as f:
            dw = csv.DictWriter(f, delimiter=",", fieldnames=ordered_fieldnames)
            dw.writeheader()
            for index, row in enumerate(self._cm):
                row_dict = {l: row[i] for i, l in enumerate(self.labels)}
                row_dict.update({"labels": self.labels[index]})
                dw.writerow(row_dict)

    def reset(self):
        """Reset the matrix
        """
        self._cm *= 0

    def get_correct(self):
        """Get the diagonals of the confusion matrix

        :return: (``int``) Number of correct classifications
        """
        return self._cm.diagonal().sum()

    def get_total(self):
        """Get total classifications

        :return: (``int``) total classifications
        """
        return self._cm.sum()

    def get_acc(self):
        """Get the accuracy

        :return: (``float``) accuracy
        """
        return float(self.get_correct()) / self.get_total()

    def get_recall(self):
        """Get the recall

        :return: (``float``) recall
        """
        total = np.sum(self._cm, axis=1)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_support(self):
        return np.sum(self._cm, axis=1)

    def get_precision(self):
        """Get the precision
        :return: (``float``) precision
        """

        total = np.sum(self._cm, axis=0)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_mean_precision(self):
        """Get the mean precision across labels

        :return: (``float``) mean precision
        """
        return np.mean(self.get_precision())

    def get_weighted_precision(self):
        return np.sum(self.get_precision() * self.get_support()) / float(self.get_total())

    def get_mean_recall(self):
        """Get the mean recall across labels

        :return: (``float``) mean recall
        """
        return np.mean(self.get_recall())

    def get_weighted_recall(self):
        return np.sum(self.get_recall() * self.get_support()) / float(self.get_total())

    def get_weighted_f(self, beta=1):
        return np.sum(self.get_class_f(beta) * self.get_support()) / float(self.get_total())

    def get_macro_f(self, beta=1):
        """Get the macro F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) macro F_b
        """
        if beta < 0:
            raise Exception("Beta must be greater than 0")
        return np.mean(self.get_class_f(beta))

    def get_class_f(self, beta=1):
        p = self.get_precision()
        r = self.get_recall()

        b = beta * beta
        d = b * p + r
        d = (d == 0) + d

        return (b + 1) * p * r / d

    def get_f(self, beta=1):
        """Get 2 class F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) 2-class F_b
        """
        p = self.get_precision()[1]
        r = self.get_recall()[1]
        if beta < 0:
            raise Exception("Beta must be greater than 0")
        d = beta * beta * p + r
        if d == 0:
            return 0
        return (beta * beta + 1) * p * r / d

    def get_r_k(self):
        """Calculate the R k correlation coefficient from here
            https://www.sciencedirect.com/science/article/abs/pii/S1476927104000799?via%3Dihub
        See this blog post for an explanation of metric, how this calculation is equivalent and
        how it reduces to MCC for 2 classes.
            www.blester125.com/blog/rk.html
        """
        samples = np.sum(self._cm)
        correct = np.trace(self._cm)
        true = np.sum(self._cm, axis=1, dtype=np.float64)
        guess = np.sum(self._cm, axis=0, dtype=np.float64)

        cov_guess_true = correct * samples - np.dot(guess, true)
        cov_true_true = samples * samples - np.dot(true, true)
        cov_guess_guess = samples * samples - np.dot(guess, guess)

        denom = np.sqrt(cov_guess_guess * cov_true_true)
        denom = denom if denom != 0.0 else 1.0
        return cov_guess_true / denom

    def get_mcc(self):
        """Get the Mathews correlation coefficient.
        R k reduces to the Matthews Correlation Coefficient for two classes.
        People are often more familiar with the name MCC than R k so we provide this alias
        """
        return self.get_r_k()

    def get_bin_avg_f1_acc(self):
        return 0.5 * (self.get_acc() + self.get_f(1))

    def get_all_metrics(self):
        """Make a map of metrics suitable for reporting, keyed by metric name

        :return: (``dict``) Map of metrics keyed by metric names
        """
        metrics = {"acc": self.get_acc()}
        # If 2 class, assume second class is positive AKA 1
        if len(self.labels) == 2:
            metrics["precision"] = self.get_precision()[1]
            metrics["recall"] = self.get_recall()[1]
            metrics["f1"] = self.get_f(1)
            metrics["avg_f1_acc"] = self.get_bin_avg_f1_acc()
            metrics["mcc"] = self.get_mcc()
        else:
            metrics["mean_precision"] = self.get_mean_precision()
            metrics["mean_recall"] = self.get_mean_recall()
            metrics["macro_f1"] = self.get_macro_f(1)
            metrics["weighted_precision"] = self.get_weighted_precision()
            metrics["weighted_recall"] = self.get_weighted_recall()
            metrics["weighted_f1"] = self.get_weighted_f(1)
            metrics["r_k"] = self.get_r_k()
        return metrics

    def add_batch(self, truth, guess):
        """Add a batch of data to the confusion matrix

        :param truth: The truth tensor
        :param guess: The guess tensor
        :return:
        """
        for truth_i, guess_i in zip(truth, guess):
            self.add(truth_i, guess_i)

    @classmethod
    def create(cls, truth, guess):
        """Build a Confusion Matrix from truths and guesses.
        If there are classes missing from the union of golds and preds they will
        be missing from the confusion matrix
        The classes are added in sorted order, if they are ints already this means
        they were probably already encoded and we will hopefully respect the order
        they are in.
        :param truth: List[Union[int. str]] The correct labels
        :param guess: List[Union[int. str]] The predicted labels
        :returns: ConfusionMatrix
        """
        label_index = {i: k for i, k in enumerate(sorted(set(chain(truth, guess))))}
        rev_lut = {v: k for k, v in label_index.items()}
        cm = cls(label_index)
        for g, p in zip(truth, guess):
            cm.add(rev_lut[g], rev_lut[p])
        return cm
