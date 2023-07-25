"""Metrics for duplicated content detection"""

import datasets
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import evaluate

_DESCRIPTION = """
This metrics script includes F1, Precision, Recall evaluations.
"""

_KWARGS_DESCRIPTION = """

"""

_CITATION = """
@article{scikit-learn,
    title={Scikit-learn: Machine Learning in {P}ython},
    author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
           and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
           and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
           Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
    journal={Journal of Machine Learning Research},
    volume={12},
    pages={2825--2830},
    year={2011}
}
"""


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + "\n\n" + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SCMetrics(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
                            "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html",
                            "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="weighted", sample_weight=None,
                 zero_division='warn'): # play with different average measures, i.e. macro vs micro
        f1 = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        p = precision_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight,
            zero_division=zero_division
        )
        r = recall_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight,
            zero_division=zero_division
        )
        c = classification_report(
            references, predictions, labels=labels
        )
        print(c)
        return {"f1": float(f1) if f1.size == 1 else f1,
                "precision": float(p) if p.size == 1 else p,
                "recall": float(r) if r.size == 1 else r}

