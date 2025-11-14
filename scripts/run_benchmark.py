# scripts/run_benchmark.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from haystack.model import HayStackMultiBackbone
from haystack.utils import load_batch_images, compute_metrics, adaptive_threshold_roc
from haystack.transforms import transform
from datasets.coco import COCODataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HayStackMultiBackbone().to(device).eval()
    dataset = COCODataset("/path/to/coco", max_samples=1000)

    K = 5
    ref = dataset.get_reference_images(K)
    test = dataset.get_test_images(100)

    ref_X, _ = load_batch_images(ref, transform, device)
    test_X, test_y = load_batch_images(test, transform, device)

    R_ref = model.extract_R(ref_X)
    Psi, dim, U = model.build_psi(R_ref, K)

    R_test = model.extract_R(test_X)
    R_proj = R_test @ U
    scores = (Psi * R_proj).sum(1).cpu().numpy()

    ref_self = (Psi * (R_ref @ U)).sum(1).cpu().numpy()
    tau = adaptive_threshold_roc(ref_self, scores, test_y)
    metrics = compute_metrics(scores, test_y, tau)

    print(f"HayStack | K={K} | F1: {metrics['f1']:.1%} | Acc: {metrics['accuracy']:.1%}")

if __name__ == "__main__":
    main()