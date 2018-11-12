import os
import sys
import json
import numpy as np
import numpy.linalg as la
from keras.models import model_from_json
from keras.utils import np_utils
from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.keras import KerasClassifier


def get_metrics(model, x_original, x_adv_samples, y):
    scores = model.evaluate(x_original, y, verbose=0)
    model_accuracy_on_non_adversarial_samples = scores[1]

    y_pred = model.predict(x_original, verbose=0)
    y_pred_adv = model.predict(x_adv_samples, verbose=0)

    scores = model.evaluate(x_adv_samples, y, verbose=0)
    model_accuracy_on_adversarial_samples = scores[1]

    pert_metric = get_perturbation_metric(x_original, x_adv_samples, y_pred, y_pred_adv, ord=2)
    conf_metric = get_confidence_metric(y_pred, y_pred_adv)

    data = {
        "model accuracy on test data": float(model_accuracy_on_non_adversarial_samples),
        "model accuracy on adversarial samples": float(model_accuracy_on_adversarial_samples),
        "reduction in confidence": float(conf_metric),
        "average perturbation": float(pert_metric)
    }
    return data


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
        if arg == "--data":
            dataset_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--networkdefinition":
            network_definition_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--weights":
            weights_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i + 1]))
        if arg == "--epsilon":
            epsilon = float(argv[i + 1])

        i += 2

    print("dataset: ", dataset_filename)
    print("network definition: ", network_definition_filename)
    print("weights: ", weights_filename)

    # load & compile model
    json_file = open(network_definition_filename, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weights_filename)
    comp_params = {'loss': 'categorical_crossentropy',
                   'optimizer': 'adam',
                   'metrics': ['accuracy']}
    model.compile(**comp_params)

    # create keras classifier
    classifier = KerasClassifier((0, 1), model)

    # load data set
    pf = np.load(dataset_filename)

    x = pf['x_test']
    y = pf['y_test']

    # pre-process numpy array

    x = np.expand_dims(x, axis=3)
    x = x.astype('float32') / 255

    y = np_utils.to_categorical(y, 10)

    # craft adversarial samples using FGSM
    crafter = FastGradientMethod(classifier, eps=epsilon)
    x_samples = crafter.generate(x)

    # obtain all metrics (robustness score, perturbation metric, reduction in confidence)
    metrics = get_metrics(model, x, x_samples, y)

    print("metrics: ", metrics)

    report_file = os.path.join(os.environ["RESULT_DIR"], "report.txt")

    with open(report_file, "w") as report:
        report.write(json.dumps(metrics))

    adv_samples_file = os.path.join(os.environ["RESULT_DIR"], 'adv_samples')
    print("adversarial samples saved to: ", adv_samples_file)
    np.savez(adv_samples_file, x_original=x, x_adversarial=x_samples, y=y)


if __name__ == "__main__":
    main(sys.argv)
