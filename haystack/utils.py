# haystack/utils.py
import torch
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_batch_images(samples, transform, device, batch_size=16):
    imgs, labels = [], []
    for sample in samples:
        try:
            if 'path' in sample:
                from PIL import Image
                img = Image.open(sample['path']).convert('RGB')
            elif 'img' in sample:
                img = sample['img']
            else:
                continue
            imgs.append(transform(img))
            labels.append(sample['label'])
        except:
            continue
    if not imgs:
        raise ValueError("No valid images")
    return torch.stack(imgs).to(device), np.array(labels)

def compute_metrics(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {'f1': f1, 'precision': p, 'recall': r, 'accuracy': acc}

def adaptive_threshold_roc(ref_scores, test_scores, test_labels):
    if len(np.unique(test_labels)) > 1:
        try:
            fpr, tpr, thresholds = roc_curve(test_labels, test_scores)
            youden = tpr - fpr
            idx = np.argmax(youden)
            tau = thresholds[idx]
            if not np.isnan(tau) and test_scores.min() <= tau <= test_scores.max():
                return float(tau)
        except:
            pass
    return float(np.percentile(test_scores, 60))