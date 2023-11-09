
import sys
sys.path.extend([
    "./"
])
import argparse
import os

from src.data_util import Dataset_
import yaml
import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.general_utils import util_data

def compute_metrics(y_test_real, y_pred, aggregation='mean'):
    cm = confusion_matrix(y_test_real, y_pred)
    num_classes = cm.shape[0]

    accuracy = np.trace(cm) / float(np.sum(cm))

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        g_mean = np.sqrt(recall * specificity)
        return accuracy, precision, recall, specificity, f1, g_mean

    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    g_mean_list = []

    for j in range(num_classes):
        tp = cm[j, j]
        fp = np.sum(cm[:, j]) - tp
        fn = np.sum(cm[j, :]) - tp
        tn = np.sum(cm) - (fp + fn + tp)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        g_mean = np.sqrt(recall * specificity)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1)
        g_mean_list.append(g_mean)

    if aggregation == 'mean':
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        specificity = np.mean(specificity_list)
        f1 = np.mean(f1_list)
        g_mean = np.mean(g_mean_list)
    else:
        raise NotImplementedError

    return accuracy, precision, recall, specificity, f1, g_mean

def get_parser():

    parser = argparse.ArgumentParser(description='Test Metric')
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_ev/")
    return parser

# Main
if __name__ == '__main__':

    # Configuration file
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    report_dir = './reports/'
    dataset_name = 'pneumoniamnist'

    cfg_file =  './src/configs_eval_backbone/pneumoniamnist/inceptionv3_torch__pneumonia.yaml'  # './src/configs_eval_backbone/pneumoniamnist/resnet50_torch__pneumonia.yaml'
    with open(cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    n_classes = cfg['DATA']['num_classes']

    #model = torch.hub.load("pytorch/vision:v0.10.0", 'resnet50', weights='ResNet50_Weights.DEFAULT')
    model = torch.hub.load("pytorch/vision:v0.10.0", 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

    #model_path = os.path.join(report_dir, dataset_name, 'backbone', 'pneumoniamnist-ResNet50_torch-truefake-train-2023_10_25_11_38_03', 'model_best_epoch_21.pt')
    model_path = os.path.join(report_dir, dataset_name, 'backbone', 'pneumoniamnist-InceptionV3_torch-pneumonia-train-2023_10_25_11_36_59', 'model_best_epoch_22.pt')
    model.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
    model.eval()

    # Preparing data.
    test_dataset = Dataset_(
        data_name=dataset_name,
        data_dir=cfg['DATA']['data_dir'],
        train=False,
        split='test',
        crop_long_edge=cfg['PRE']['crop_long_edge'],
        resize_size=cfg['PRE']['resize_size'],
        resizer=cfg['PRE']['pre_resizer'],
        random_flip=cfg['PRE']['apply_rflip'],
        normalize=cfg['PRE']['normalize'],
        cfgs=cfg
    )
    #test_dataset = util_data.ImageNetDataset(copy.deepcopy(test_dataset), model_name='ResNet50_torch')
    test_dataset = util_data.ImageNetDataset(copy.deepcopy(test_dataset), model_name='InceptionV3_torch')

    # Test
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg['TRAINER']['batch_size'],
        shuffle=False
    )

    # Predict.
    labels_list = []
    preds_list = []
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')
        labels_list.append(labels.cpu().numpy())
        # Prediction
        outputs = model(inputs.float())
        _, preds = torch.max(outputs, 1)
        preds_list.append(preds.cpu().numpy())

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)

    acc, prec, rec, spec, f1, gmean = compute_metrics(labels_list, preds_list)
    print('Accuracy: {:.4f}'.format(acc))
    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('Specificity: {:.4f}'.format(spec))
    print('F1: {:.4f}'.format(f1))
    print('G-Mean: {:.4f}'.format(gmean))

    print('May the force be with you!')