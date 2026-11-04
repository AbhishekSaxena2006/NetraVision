import os
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_class_weights(labels, num_classes=5):
    """
    Computes inverse class frequencies to penalize majority class
    and give appropriate attention to rare severe DR classes.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Balanced weights formula: total / (num_classes * count)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)

def calculate_qwk(y_true, y_pred):
    """
    Computes Quadratic Weighted Kappa (QWK), the standard clinical metric
    for ordinal Diabetic Retinopathy grade assessment.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

class EarlyStopping:
    """
    Monitors validation QWK or loss and stops training early if no improvement.
    """
    def __init__(self, patience=5, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
