from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import numpy as np
import argparse
import pandas as pd


def dataset_wrapper(outcome, protected, unprivileged_groups, privileged_groups, favorable_label, unfavorable_label):
    """ A wrapper function to create aif360 dataset from outcome and protected in numpy array format.
    """
    df = pd.DataFrame(data=outcome,
                      columns=['outcome'])
    df['race'] = protected

    dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                 unfavorable_label=unfavorable_label,
                                 df=df,
                                 label_names=['outcome'],
                                 protected_attribute_names=['race'],
                                 unprivileged_protected_attributes=unprivileged_groups)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, default=".", help='Label directory path')
    parser.add_argument('--model_dir', type=str, default=".", help='Dataset directory path')
    args = parser.parse_args()

    label_dir = args.label_dir
    model_dir = args.model_dir

    print("metrics: %s" % fairness_check(label_dir, model_dir))


def fairness_check(label_dir, model_dir):
    """Need to generalize the protected features"""

    # races_to_consider = [0,4]
    unprivileged_groups = [{'race': 4.0}]
    privileged_groups = [{'race': 0.0}]
    favorable_label = 0.0
    unfavorable_label = 1.0

    """Load the necessary labels and protected features for fairness check"""

    # y_train = np.loadtxt(label_dir + '/y_train.out')
    # p_train = np.loadtxt(label_dir + '/p_train.out')
    y_test = np.loadtxt(label_dir + '/y_test.out')
    p_test = np.loadtxt(label_dir + '/p_test.out')
    y_pred = np.loadtxt(label_dir + '/y_pred.out')

    """Calculate the fairness metrics"""

    # original_traning_dataset = dataset_wrapper(outcome=y_train, protected=p_train,
    #                                            unprivileged_groups=unprivileged_groups,
    #                                            privileged_groups=privileged_groups,
    #                                            favorable_label=favorable_label,
    #                                            unfavorable_label=unfavorable_label)
    original_test_dataset = dataset_wrapper(outcome=y_test, protected=p_test,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups,
                                            favorable_label=favorable_label,
                                            unfavorable_label=unfavorable_label)
    plain_predictions_test_dataset = dataset_wrapper(outcome=y_pred, protected=p_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups,
                                                     favorable_label=favorable_label,
                                                     unfavorable_label=unfavorable_label)

    classified_metric_nodebiasing_test = ClassificationMetric(original_test_dataset,
                                                              plain_predictions_test_dataset,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)

    print("#### Plain model - without debiasing - classification metrics on test set")
    # print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
    # print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    # print("Test set: Statistical parity difference = %f" % classified_metric_nodebiasing_test.statistical_parity_difference())
    # print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
    # print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
    # print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
    # print("Test set: Theil index = %f" % classified_metric_nodebiasing_test.theil_index())
    # print("Test set: False negative rate difference = %f" % classified_metric_nodebiasing_test.false_negative_rate_difference())

    metrics = {
        "Classification accuracy": classified_metric_nodebiasing_test.accuracy(),
        "Balanced classification accuracy": bal_acc_nodebiasing_test,
        "Statistical parity difference": classified_metric_nodebiasing_test.statistical_parity_difference(),
        "Disparate impact": classified_metric_nodebiasing_test.disparate_impact(),
        "Equal opportunity difference": classified_metric_nodebiasing_test.equal_opportunity_difference(),
        "Average odds difference": classified_metric_nodebiasing_test.average_odds_difference(),
        "Theil index": classified_metric_nodebiasing_test.theil_index(),
        "False negative rate difference": classified_metric_nodebiasing_test.false_negative_rate_difference()
    }
    return {"metrics": metrics}


if __name__ == "__main__":
    main()
