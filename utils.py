import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent / "outputs"
LOGS, FIGURES, MODELS = OUT / "logs", OUT / "figures", OUT / "models"
for folder in (LOGS, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)


def get_log(path):
    file = open(path, "w", encoding="utf-8")

    def log(text=""):
        print(text)
        file.write(text + "\n")
        file.flush()

    return log


def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(X), y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader)


def test_step(model, dataloader, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
    return test_loss / len(dataloader), correct / len(dataloader.dataset)


def train(model, train_dl, test_dl, loss_fn, optimizer, epochs, log):
    results = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(epochs):
        train_loss = train_step(model, train_dl, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_dl, loss_fn)
        log(f"  epoch {epoch + 1:2d} | train loss {train_loss:.4f} | val loss {test_loss:.4f} | test acc {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results


def binary_report(y_true, y_pred, log):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    log(f"  confusion matrix [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")
    log(f"  sensitivity {tp / (tp + fn):.4f} | specificity {tn / (tn + fp):.4f} | accuracy {(tp + tn) / len(y_true):.4f}")


def plot_loss_curves(results, title, path):
    epochs = range(1, len(results["train_loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, results["train_loss"], label="train loss")
    plt.plot(epochs, results["test_loss"], label="validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_images(images, titles, path):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), constrained_layout=True)
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_eeg(series, path):
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, (title, x) in zip(axes, series.items()):
        ax.plot(x, linewidth=0.7)
        ax.set_title(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
