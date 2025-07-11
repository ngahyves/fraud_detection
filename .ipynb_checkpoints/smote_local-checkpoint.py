#local smote

# Content for smote_local.py
# This code is a direct and simplified copy of the source code from imbalanced-learn v0.12.3
# licensed under MIT. (c) 2014-2024 The imbalanced-learn developers.

import numbers
import numpy as np
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, _safe_indexing

class BaseOverSampler:
    _sampling_type = "over-sampling"
    
    def fit(self, X, y):
        self._validate_params()
        self.sampling_strategy_ = self._check_sampling_strategy(self.sampling_strategy, y, self._sampling_type)
        return self

    def fit_resample(self, X, y):
        self._validate_params()
        X_resampled, y_resampled = self._fit_resample(X, y)
        return X_resampled, y_resampled

    def _check_sampling_strategy(self, sampling_strategy, y, sampling_type):
        if sampling_strategy == "auto":
            return {label: count for label, count in zip(*np.unique(y, return_counts=True)) if label != np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]}
        return sampling_strategy

    def _validate_params(self):
        pass

class SMOTE(BaseOverSampler):
    def __init__(self, sampling_strategy="auto", random_state=None, k_neighbors=5, n_jobs=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _fit_resample(self, X, y):
        super().fit(X, y)
        self.random_state_ = check_random_state(self.random_state)
        
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        
        for class_label, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            
            target_class_indices = np.flatnonzero(y == class_label)
            X_class = _safe_indexing(X, target_class_indices)
            
            self.nns_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1, n_jobs=self.n_jobs).fit(X_class)
            
            n_synthetic_samples = n_samples - X_class.shape[0]
            
            if n_synthetic_samples > 0:
                X_new = self._make_samples(X_class, y.dtype, class_label, n_synthetic_samples)
                X_resampled.append(X_new)
                y_resampled.append(np.full(n_synthetic_samples, fill_value=class_label, dtype=y.dtype))

        return np.vstack(X_resampled), np.hstack(y_resampled)

    def _make_samples(self, X, y_type, class_label, n_samples):
        n_features = X.shape[1]
        X_new = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            # pick a random sample
            sample_idx = self.random_state_.randint(0, X.shape[0])
            # find its k-nearest neighbors
            _, nn_indices = self.nns_.kneighbors(X[sample_idx].reshape(1, -1))
            nn_indices = nn_indices.flatten()
            # pick one of the k-nearest neighbors
            neighbor_idx = self.random_state_.choice(nn_indices)
            
            # create a synthetic sample
            diff = X[neighbor_idx] - X[sample_idx]
            step = self.random_state_.uniform()
            X_new[i] = X[sample_idx] + step * diff
            
        return X_new