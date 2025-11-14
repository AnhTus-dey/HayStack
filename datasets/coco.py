# datasets/coco.py
from pathlib import Path
from collections import defaultdict
import json
from PIL import Image

class COCODataset:
    def __init__(self, coco_dir, max_samples=20000):
        self.coco_dir = Path(coco_dir)
        self.samples = self._load_samples(max_samples)

    def _load_samples(self, max_samples):
        samples = []
        for split in ['val2017', 'train2017']:
            img_dir = self.coco_dir / split
            ann_file = self.coco_dir / 'annotations' / f'instances_{split}.json'
            if not ann_file.exists(): continue
            with open(ann_file) as f:
                data = json.load(f)
            person_id = next((c['id'] for c in data['categories'] if c['name'] == 'person'), None)
            img_to_anns = defaultdict(list)
            for ann in data['annotations']:
                img_to_anns[ann['image_id']].append(ann)
            for img_info in data['images']:
                if len(samples) >= max_samples: break
                img_path = img_dir / img_info['file_name']
                if not img_path.exists(): continue
                has_person = person_id and any(a['category_id'] == person_id for a in img_to_anns[img_info['id']])
                samples.append({'path': str(img_path), 'label': 1 if has_person else 0})
        return samples

    def get_reference_images(self, K=5):
        pos = [s for s in self.samples if s['label'] == 1]
        return pos[:K]

    def get_test_images(self, n=None):
        return self.samples[:n]