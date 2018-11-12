import os
import sys
import numpy as np
import numpy.linalg as la
import json

import torch
import torch.utils.data
from torch.autograd import Variable

from art.classifiers.pytorch import PyTorchClassifier
from art.attacks.fast_gradient import FastGradientMethod


class ThreeLayerCNN(torch.nn.Module):
    """
    Input: 128x128 face image (eye aligned).
    Output: 1-D tensor with 2 elements. Used for binary classification.
    Parameters:
        Number of conv layers: 3
        Number of fully connected layers: 2
    """
    def __init__(self):
        super(ThreeLayerCNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.conv3 = torch.nn.Conv2d(16,16,6)
        self.fc1 = torch.nn.Linear(16*4*4,120)
        self.fc2 = torch.nn.Linear(120,2)


    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1,16*4*4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_metrics(model, x_original, x_adv_samples, y):
    model_accuracy_on_non_adversarial_samples, y_pred = evaluate(model, x_original, y)
    model_accuracy_on_adversarial_samples, y_pred_adv = evaluate(model, x_adv_samples, y)

    pert_metric = get_perturbation_metric(x_original, x_adv_samples, y_pred, y_pred_adv, ord=2)
    conf_metric = get_confidence_metric(y_pred, y_pred_adv)

    data = {
        "model accuracy on test data": float(model_accuracy_on_non_adversarial_samples),
        "model accuracy on adversarial samples": float(model_accuracy_on_adversarial_samples),
        "confidence reduced on correctly classified adv_samples": float(conf_metric),
        "average perturbation on misclassified adv_samples": float(pert_metric)
    }
    return data, y_pred, y_pred_adv


def evaluate(model, X_test, y_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test = torch.utils.data.TensorDataset(Variable(torch.FloatTensor(X_test.astype('float32'))), Variable(torch.LongTensor(y_test.astype('float32'))))
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    accuracy = 0
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions = torch.softmax(outputs.data, dim=1).detach().numpy()
            correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
            y_pred += predictions.tolist()
        accuracy = 1. * correct / len(test_loader.dataset)
    y_pred = np.array(y_pred)
    return accuracy, y_pred


def get_perturbation_metric(x_original, x_adv, y_pred, y_pred_adv, ord=2):
    idxs = (np.argmax(y_pred_adv, axis=1) != np.argmax(y_pred, axis=1))

    if np.sum(idxs) == 0.0:
        return 0

    perts_norm = la.norm((x_adv - x_original).reshape(x_original.shape[0], -1), ord, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x_original[idxs].reshape(np.sum(idxs), -1), ord, axis=1))


# This computes the change in confidence for all images in the test set
def get_confidence_metric(y_pred, y_pred_adv):
    y_classidx = np.argmax(y_pred, axis=1)
    y_classconf = y_pred[np.arange(y_pred.shape[0]), y_classidx]

    y_adv_classidx = np.argmax(y_pred_adv, axis=1)
    y_adv_classconf = y_pred_adv[np.arange(y_pred_adv.shape[0]), y_adv_classidx]

    idxs = (y_classidx == y_adv_classidx)

    if np.sum(idxs) == 0.0:
        return 0

    idxnonzero = y_classconf != 0
    idxs = idxs & idxnonzero

    return np.mean((y_classconf[idxs] - y_adv_classconf[idxs]) / y_classconf[idxs])


def main(argv):
    if len(argv) < 2:
        sys.exit("Not enough arguments provided.")

    global network_definition_filename, weights_filename, dataset_filename

    i = 1
    while i <= 8:
        arg = str(argv[i])
        print(arg)
        if arg == "--datax":
            dataset_filenamex = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--datay":
            dataset_filenamey = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--weights":
            weights_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--epsilon":
            epsilon = float(argv[i + 1])

        i += 2

    print("dataset_x:", dataset_filenamex)
    print("dataset_y:", dataset_filenamey)
    print("weights:", weights_filename)

    # load & compile model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ThreeLayerCNN().to(device)
    model.load_state_dict(torch.load(weights_filename))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create pytorch classifier
    classifier = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1,3,64,64), 2)

    # load data set
    x = np.load(dataset_filenamex)
    y = np.loadtxt(dataset_filenamey)

    # craft adversarial samples using FGSM
    crafter = FastGradientMethod(classifier, eps=epsilon)
    x_samples = crafter.generate(x)

    # obtain all metrics (robustness score, perturbation metric, reduction in confidence)
    metrics, y_pred_orig, y_pred_adv = get_metrics(model, x, x_samples, y)

    print("metrics:", metrics)

    report_file = os.path.join(os.environ["RESULT_DIR"], "report.txt")

    with open(report_file, "w") as report:
        report.write(json.dumps(metrics))

    adv_samples_file = os.path.join(os.environ["RESULT_DIR"], "adv_samples")
    print("adversarial samples saved to: ", adv_samples_file)
    np.savez(adv_samples_file, x_original=x, x_adversarial=x_samples, y=y)


if __name__ == "__main__":
    main(sys.argv)
