import joblib
import torch
from torch import nn

import data
import utils
from models import FCNet


def eval_classical(task, split, log):
    _, X_test, _, y_test = split
    for group in ("svm", "knn"):
        name, model = joblib.load(utils.MODELS / f"{task}_{group}.joblib")
        log(f"{task} {name}")
        utils.binary_report(y_test, model.predict(X_test), log)


def eval_images(task, name):
    log = utils.get_log(utils.LOGS / f"{task}_eval.txt")
    _, _, test_dl, _ = data.get_loaders(name)
    checkpoint = torch.load(utils.MODELS / f"{task}.pth", map_location=utils.DEVICE)
    model = FCNet(checkpoint["hidden"], getattr(nn, checkpoint["activation"]), checkpoint["dropout"]).to(utils.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc = utils.test_step(model, test_dl, nn.CrossEntropyLoss())
    log(f"{task}: {name} test loss {test_loss:.4f} | test accuracy {test_acc:.4f}")


log = utils.get_log(utils.LOGS / "problem1_eval.txt")
eval_classical("problem1", data.get_breast_cancer(), log)

log = utils.get_log(utils.LOGS / "problem2_eval.txt")
eeg = data.get_eeg()
for a, b in data.EEG_PAIRS:
    eval_classical(f"problem2_{a}_vs_{b}", data.get_eeg_pair(eeg, a, b), log)

eval_images("problem3", "MNIST")
eval_images("problem4", "FashionMNIST")
