import pandas as pd
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    image_ids = []
    probabilities = []

    for images, ids in tqdm(loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        image_ids.extend(ids)
        probabilities.extend(probs)

    return pd.DataFrame({"image_id": image_ids, "p_180": probabilities})
