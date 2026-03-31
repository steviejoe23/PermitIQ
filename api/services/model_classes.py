"""
Custom model classes used by PermitIQ training pipeline.

These must be importable at pickle load time, so they live in a shared module
rather than inside the training script.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble: base models -> out-of-fold predictions -> meta-learner.

    The meta-learner learns which base model to trust in which probability range.
    """
    _estimator_type = "classifier"

    def __init__(self, base_models=None, meta_model=None, n_folds=5):
        self.base_models = base_models or []
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.fitted_base_models_ = []
        self.fitted_meta_model_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        from sklearn.model_selection import StratifiedKFold
        import copy

        n_samples = X.shape[0]
        n_base = len(self.base_models)

        oof_preds = np.zeros((n_samples, n_base))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        self.fitted_base_models_ = [[] for _ in range(n_base)]

        for fold_idx, (train_fold, val_fold) in enumerate(skf.split(X, y)):
            X_fold_train = X.iloc[train_fold] if hasattr(X, 'iloc') else X[train_fold]
            X_fold_val = X.iloc[val_fold] if hasattr(X, 'iloc') else X[val_fold]
            y_fold_train = y.iloc[train_fold] if hasattr(y, 'iloc') else y[train_fold]

            sw_fold = sample_weight[train_fold] if sample_weight is not None else None

            for i, (name, model) in enumerate(self.base_models):
                fold_model = copy.deepcopy(model)
                uses_sw = name in ('Gradient Boosting', 'XGBoost', 'XGBoost_Deep')
                if uses_sw and sw_fold is not None:
                    fold_model.fit(X_fold_train, y_fold_train, sample_weight=sw_fold)
                else:
                    fold_model.fit(X_fold_train, y_fold_train)
                oof_preds[val_fold, i] = fold_model.predict_proba(X_fold_val)[:, 1]
                self.fitted_base_models_[i].append(fold_model)

        self.fitted_meta_model_ = copy.deepcopy(self.meta_model)
        self.fitted_meta_model_.fit(oof_preds, y)

        return self

    def predict_proba(self, X):
        n_base = len(self.base_models)
        base_preds = np.zeros((X.shape[0], n_base))
        for i in range(n_base):
            fold_preds = []
            for fold_model in self.fitted_base_models_[i]:
                fold_preds.append(fold_model.predict_proba(X)[:, 1])
            base_preds[:, i] = np.mean(fold_preds, axis=0)

        return self.fitted_meta_model_.predict_proba(base_preds)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ManualCalibratedModel:
    """Wraps a model + Platt scaling for calibrated probabilities.

    Used when sklearn's CalibratedClassifierCV can't handle custom model classes.
    """
    _estimator_type = "classifier"

    def __init__(self, base=None, platt=None):
        self.base = base
        self.platt = platt
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw_proba = self.base.predict_proba(X)[:, 1]
        cal_proba = self.platt.predict_proba(raw_proba.reshape(-1, 1))
        return cal_proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
