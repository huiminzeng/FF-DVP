import numpy as np 
import pdb

def demographic_parity(predictions, attributes, targets):
    # |p(y_hat=1|a=0) - P(y_hat=1|a=1)|
    # a is for demographic attributes
    p_joint_att_positive = np.mean((predictions == 1) * (attributes==1))
    p_joint_att_negative = np.mean((predictions == 1) * (attributes==0))

    p_margin_att_positive = np.mean((attributes==1))
    p_margin_att_negative = np.mean((attributes==0))

    p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
    p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
    demo_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

    return demo_parity

def true_positive_parity(predictions, attributes, targets):
    # p(y_hat=1|y=1,a=0) - P(y_hat=1|y=1,a=1)
    # positive attribute ids
    p_joint_att_positive = np.mean((predictions == targets) * (targets == 1) * (attributes==1))
    p_joint_att_negative = np.mean((predictions == targets) * (targets == 1) * (attributes==0))

    p_margin_att_positive = np.mean((targets == 1) * (attributes==1))
    p_margin_att_negative = np.mean((targets == 1) * (attributes==0))

    p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
    p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
    # print("true positive parity p(y_hat=1|y=1,a=1): ", p_condition_att_positive)
    # print("true positive parity P(y_hat=1|y=1,a=0): ", p_condition_att_negative)
    tp_parity = np.abs(p_condition_att_positive - p_condition_att_negative)
    # tp_parity = p_condition_att_negative - p_condition_att_positive

    return tp_parity

def false_positive_parity(predictions, attributes, targets):
    # |p(y_hat=1|y=0,a=0) - P(y_hat=1|y=0,a=1)|
    # positive attribute ids
    p_joint_att_positive = np.mean((predictions != targets) * (targets == 0) * (attributes==1))
    p_joint_att_negative = np.mean((predictions != targets) * (targets == 0) * (attributes==0))

    p_margin_att_positive = np.mean((targets == 0) * (attributes==1))
    p_margin_att_negative = np.mean((targets == 0) * (attributes==0))

    p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
    p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
    fp_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

    return fp_parity

def equal_odds(predictions, attributes, targets):

    tp_parity = true_positive_parity(predictions, attributes, targets)
    fp_parity = false_positive_parity(predictions, attributes, targets)

    eq_odds = tp_parity + fp_parity

    return tp_parity, fp_parity, eq_odds

def compute_fairness_metrics(predictions, attributes, targets):
    demo_parity = demographic_parity(predictions, attributes, targets)
    tp_parity, fp_parity, eq_odds = equal_odds(predictions, attributes, targets)

    # print('demo parity: {:.4f} tp parity: {:.4f} fp parity: {:.4f} equ odds: {:.4f}'.format(demo_parity, tp_parity, fp_parity, eq_odds))
    # print('demo parity: {:.4f}, equ odds: {:.4f}'.format(demo_parity, eq_odds))
    return demo_parity, eq_odds

def compute_accurate_metrics(predictions, attributes, targets):
    group_1_acc = np.sum((predictions == targets) * (targets == 1) * (attributes==1)) / np.sum((targets == 1) * (attributes==1) + 1e-8)
    group_2_acc = np.sum((predictions == targets) * (targets == 1) * (attributes==0)) / np.sum((targets == 1) * (attributes==0) + 1e-8)
    group_3_acc = np.sum((predictions == targets) * (targets == 0) * (attributes==1)) / np.sum((targets == 0) * (attributes==1) + 1e-8)
    group_4_acc = np.sum((predictions == targets) * (targets == 0) * (attributes==0)) / np.sum((targets == 0) * (attributes==0) + 1e-8)
    acc_mean = (group_1_acc + group_2_acc + group_3_acc + group_4_acc) / 4
    acc_std = np.abs(group_1_acc - group_2_acc) + np.abs(group_3_acc - group_4_acc)
    acc_std += np.abs(group_1_acc - group_3_acc) + np.abs(group_2_acc - group_4_acc)

    acc = np.mean(predictions == targets)
    # print("attractive male acc: {:.4f}".format(attractive_male_acc * 100))
    # print("attractive female acc: {:.4f}".format(attractive_female_acc * 100))
    # print("unattractive male acc: {:.4f}".format(unattractive_male_acc * 100))
    # print("unattractive female acc: {:.4f}".format(unattractive_female_acc * 100))
    # print("acc mean: {:.4f}, acc std: {:.4f}".format(acc_mean, acc_std))
    # print("overall acc: {:.4f}".format(np.mean(predictions == targets)))

    # return attractive_male_acc, attractive_female_acc, unattractive_male_acc, unattractive_female_acc, acc_mean, acc_std, acc_overall
    return acc_mean, acc, acc_std