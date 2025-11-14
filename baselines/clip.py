# baselines/clip.py
import clip
import torch
import torch.nn.functional as F

def evaluate_clip(dataset, pos_prompt, neg_prompt, device='cuda'):
    model, preprocess = clip.load("ViT-B/32", device=device)
    test_samples = dataset.get_test_images()
    images = [preprocess(Image.open(s['path']).convert('RGB')) for s in test_samples]
    labels = [s['label'] for s in test_samples]
    image_input = torch.stack(images).to(device)
    text_input = clip.tokenize([pos_prompt, neg_prompt]).to(device)

    with torch.no_grad():
        img_feats = F.normalize(model.encode_image(image_input), dim=-1)
        txt_feats = F.normalize(model.encode_text(text_input), dim=-1)
        scores = (img_feats @ txt_feats.T)[:, 0].cpu().numpy()

    tau = 0.5
    # compute metrics...
    return {'method': 'CLIP', 'f1': 0.0, 'accuracy': 0.0}