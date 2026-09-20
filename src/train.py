from tqdm.auto import tqdm
from sklearn.metrics import brier_score_loss
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    """Обучает модель одну эпоху"""

    model.train()
    running_loss = 0

    progress = tqdm(loader, leave=False, desc=f"Epoch {epoch}/{epochs} [Train]")

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    progress.close()
    return running_loss / len(loader)

def validate(model, loader, criterion, device, epoch, epochs):
    """Вычисляет loss и Brier Score на валидационной выборке"""

    model.eval()

    running_loss = 0

    probabilities = []
    targets = []

    progress = tqdm(loader, leave=False, desc=f"Epoch {epoch}/{epochs} [Val]")

    with torch.no_grad():

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)

            probabilities.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    progress.close()
    val_loss = running_loss / len(loader)
    brier = brier_score_loss(targets, probabilities)

    return {
        "loss": val_loss,
        "score": 1 - brier
    }

def fit_model(model, train_loader, val_loader, optimizer, device, epochs, criterion=None, checkpoint_path=None):
    """Полный цикл обучения с сохранением лучшей модели (если указана папка)"""
    criterion = criterion or nn.BCEWithLogitsLoss()
    history = []
    best_score = -float("inf")
    best_epoch = None

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, epochs)
        metrics = validate(model, val_loader, criterion, device, epoch + 1, epochs)
        val_score = metrics["score"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_score": metrics["score"]
        })

        # Сохраняем модель с лучшим validation score
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)

    print(f"The best model from epoch {best_epoch} with score = {best_score:.4f}")
    history_df = pd.DataFrame(history)
    return model, history_df

def plot_results(history_df):
    """Визуализирует loss и validation score по эпохам"""

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Train"
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="o",
        label="Validation"
    )
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        history_df["epoch"],
        history_df["val_score"],
        marker="o"
    )
    plt.title("Score")
    plt.xlabel("Epoch")
    plt.ylabel("1 - Brier Score")

    plt.tight_layout()
    plt.show()