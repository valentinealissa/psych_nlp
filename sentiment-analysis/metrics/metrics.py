from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import OrderedDict
import evaluate

# Loss function
ce = CrossEntropyLoss()


class LmMetrics(object):
    """
    Metrics to evaluate the fine-tuning of ClinicalBERT model
    on MLM, NSP, and LM tasks.
    """

    def __init__(self, sample_size=None):
        if not sample_size:
            print('Metrics are returned as sum over batches. '
                  'To get the average metrics divide by the number of samples.')
        self.model_metrics = {'mlm_loss': [],
                              'lm_loss': [],
                              'nsp_loss': [],
                              'mlm_accuracy': [],
                              'lm_accuracy': [],
                              'nsp_accuracy': [],
                              'lm_entropy': []}
        self.sample_size = sample_size
        self.batch_metric = {}

    def compute_batch_metrics(self,
                              mlm_logits,
                              nsp_logits,
                              mlm_labels,
                              nsp_labels,
                              vocab_size,
                              lm_labels=None,
                              dev=False):
        """Compute loss, accuracy, and LM Entropy as a sum over batch samples."""
        # Loss mlm, nsp
        batch_ce_mlm = ce(mlm_logits.view(-1, vocab_size),
                          mlm_labels.view(-1)) * mlm_labels.shape[0]
        batch_ce_nsp = ce(nsp_logits, nsp_labels.view(-1)) * nsp_labels.shape[0]
        # Loss sum
        self.batch_metric = {'mlm_loss': batch_ce_mlm.item(),
                             'nsp_loss': batch_ce_nsp.item()}

        if dev:
            # LM Entropy
            batch_ce_lm = ce(mlm_logits.view(-1, vocab_size),
                             lm_labels.view(-1)) * lm_labels.shape[0]
            # print('Computing CE for LM:')
            # print(mlm_logits.view(-1, vocab_size))
            # print(lm_labels.view(-1))
            # print(mlm_labels.view(-1))
            # print('\n')
            self.batch_metric['lm_entropy'] = batch_ce_lm
            self.batch_metric['lm_loss'] = batch_ce_lm.item()
            # Accuracy
            mlm_pred = np.argmax(mlm_logits.cpu(), axis=-1).numpy()
            mlm_ref = mlm_labels.detach().cpu().numpy()
            mlm_accuracy = self._padded_accuracy(pred=mlm_pred, ref=mlm_ref)

            lm_ref = lm_labels.detach().cpu().numpy()
            lm_accuracy = self._padded_accuracy(pred=mlm_pred, ref=lm_ref)

            nsp_pred = np.argmax(nsp_logits.cpu(), axis=-1).view(-1).numpy()
            nsp_ref = nsp_labels.view(-1).detach().cpu().numpy()
            nsp_accuracy = sum(i == j for i, j in zip(nsp_pred, nsp_ref))

            self.batch_metric['mlm_accuracy'] = mlm_accuracy.item()
            self.batch_metric['lm_accuracy'] = lm_accuracy.item()
            self.batch_metric['nsp_accuracy'] = nsp_accuracy.item()

    def add_batch(self):
        # Update model_metrics adding the batch metrics
        # LmMetrics.model_metrics update
        for k, val in self.batch_metric.items():
            self.model_metrics.setdefault(k, list()).append(val)

    def compute(self):
        """Compute performances of the training step, i.e., mean over samples for epoch."""
        if not self.sample_size:
            return {k: sum(val) for k, val in self.model_metrics.items()}

        epoch_metrics = {}
        for k, val in self.model_metrics.items():
            if len(val) > 0:
                if k == 'lm_entropy':
                    epoch_metrics['ppl'] = torch.exp(
                        torch.stack(self.model_metrics['lm_entropy']).sum() / self.sample_size).item()
                else:
                    epoch_metrics[k] = sum(val) / self.sample_size
        return epoch_metrics

    @staticmethod
    def _padded_accuracy(pred, ref):
        """Compute accuracy for the MLM"""
        acc = 0
        for p_vect, r_vect in zip(pred, ref):
            lab_len = len([rf for rf in r_vect if rf != -100])
            if lab_len > 0:
                acc += sum(i == j for i, j in zip(p_vect, r_vect) if j != -100) / lab_len
        return acc


class TaskMetrics(object):
    """
    Challenge metrics
    """

    def __init__(self, challenge):
        self.challenge = challenge
        self.model_labels = {'true_labels': [],
                             'pred_labels': []}
        # self.batch_metrics = {}
        # if self.challenge == 'smoking_challenge':
        #     self.model_metrics = {'f1_score': [],
        #                           'f1_micro': [],
        #                           'f1_macro': [],
        #                           'precision': [],
        #                           'precision_micro': [],
        #                           'precision_macro': [],
        #                           'recall': [],
        #                           'recall_micro': [],
        #                           'recall_macro': []}
        # else:
        #     self.model_metrics = {}

    def add_batch(self, true, pred):
        if isinstance(pred, list):
            self.model_labels['true_labels'].extend(true)
            self.model_labels['pred_labels'].extend(pred)
        else:
            self.model_labels['true_labels'].append(true)
            self.model_labels['pred_labels'].append(pred)

    def compute(self):
        return OrderedDict({'f1_score': self._f1_score(self.model_labels['true_labels'],
                                                       self.model_labels['pred_labels']),
                            'precision': self._precision(self.model_labels['true_labels'],
                                                         self.model_labels['pred_labels']),
                            'recall': self._recall(self.model_labels['true_labels'],
                                                   self.model_labels['pred_labels'])})

    @staticmethod
    def _f1_score(true, pred):
        single_score = f1_score(true, pred, average=None)
        scores = OrderedDict()
        scores['f1_micro'] = f1_score(true, pred, average='micro')
        scores['f1_macro'] = f1_score(true, pred, average='macro')
        for cl, s in enumerate(single_score):
            scores[f'f1_class{cl}'] = s
        return scores

    @staticmethod
    def _precision(true, pred):
        single_score = precision_score(true, pred, average=None)
        scores = OrderedDict()
        scores['p_micro'] = precision_score(true, pred, average='micro')
        scores['p_macro'] = precision_score(true, pred, average='macro')
        for cl, s in enumerate(single_score):
            scores[f'p_class{cl}'] = s
        return scores

    @staticmethod
    def _recall(true, pred):
        single_score = recall_score(true, pred, average=None)
        scores = OrderedDict()
        scores['r_micro'] = recall_score(true, pred, average='micro')
        scores['r_macro'] = recall_score(true, pred, average='macro')
        for cl, s in enumerate(single_score):
            scores[f'r_class{cl}'] = s
        return scores
