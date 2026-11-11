import time

import joblib
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import nn
from torchinfo import summary

import data
import utils
from models import FCNet

EPOCHS = 20
CONFIGS = [([128], nn.ReLU), ([256, 128], nn.ReLU), ([512, 256, 128], nn.ReLU), ([256, 128], nn.Tanh)]


def run_classical(task, split, log):
    X_train, X_test, y_train, y_test = split
    svms = {f"SVM kernel={k}": SVC(kernel=k) for k in ["linear", "rbf", "poly", "sigmoid"]}
    knns = {f"kNN k={k}": KNeighborsClassifier(k) for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    for group, models in [("svm", svms), ("knn", knns)]:
        scores = {}
        for name, model in models.items():
            scores[name] = model.fit(X_train, y_train).score(X_test, y_test)
            log(f"  {name}: accuracy {scores[name]:.4f}")
        best = max(scores, key=scores.get)
        log(f"  best: {best}")
        joblib.dump((best, models[best]), utils.MODELS / f"{task}_{group}.joblib")


def run_images(task, name):
    log = utils.get_log(utils.LOGS / f"{task}_train.txt")
    log(f"{task}: {name} fully connected network on {utils.DEVICE}")
    train_dl, test_dl, classes = data.get_loaders(name)
    X, y = train_dl.dataset.tensors
    utils.plot_images([X[y == c][0] for c in range(10)], classes, utils.FIGURES / f"{task}_samples.png")
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    for hidden, activation in CONFIGS:
        log(f"hidden={hidden} activation={activation.__name__}")
        torch.manual_seed(42)
        model = FCNet(hidden, activation).to(utils.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start = time.time()
        results = utils.train(model, train_dl, test_dl, loss_fn, optimizer, EPOCHS, log)
        acc = results["test_acc"][-1]
        log(f"  training time {time.time() - start:.1f}s | test accuracy {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_summary = str(summary(model, input_size=(1, 28, 28), verbose=0))
            checkpoint ={"hidden": hidden, "activation": activation.__name__, "state_dict": model.state_dict()}
            torch.save(checkpoint, utils.MODELS / f"{task}.pth")
            utils.plot_loss_curves(results, f"{name}: hidden={hidden}, {activation.__name__}", utils.FIGURES / f"{task}_loss.png")
    log(f"best: hidden={checkpoint['hidden']} activation={checkpoint['activation']} | test accuracy {best_acc:.4f}")
    log(best_summary)
    (utils.LOGS / f"{task}_model_summary.txt").write_text(best_summary, encoding="utf-8")


log = utils.get_log(utils.LOGS / "problem1_train.txt")
log("problem1: Breast Cancer, SVM and kNN")
run_classical("problem1", data.get_breast_cancer(), log)

log = utils.get_log(utils.LOGS / "problem2_train.txt")
eeg = data.get_eeg()
utils.plot_eeg({f"Set {s} ({data.EEG_SETS[s]})": eeg[s][0] for s in eeg}, utils.FIGURES / "problem2_eeg_samples.png")
for a, b in data.EEG_PAIRS:
    log(f"problem2: {a} vs {b}, variance feature, SVM and kNN")
    run_classical(f"problem2_{a}_vs_{b}", data.get_eeg_pair(eeg, a, b), log)

run_images("problem3", "MNIST")
run_images("problem4", "FashionMNIST")
