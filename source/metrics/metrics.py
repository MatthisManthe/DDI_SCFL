import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
from data.cityscapes_labels import CsLabels
from pprintpp import pprint


class CumulativeClassificationMetrics():
    
    def __init__(self, num_classes, classes):
        self.pred = []
        self.truth = []
        self.num_classes = num_classes
        self.classes = classes
        
    def __call__(self, y_pred=None, y=None):
        self.pred.extend(list(y_pred.cpu().numpy()))
        self.truth.extend(list(y.cpu().numpy()))

    def aggregate(self):
        return classification_report(self.truth, self.pred, target_names=self.classes, output_dict=True)
        
    def reset(self):
        self.pred = []
        self.truth = []

        
class CumulativeIoUCityscapes():
    
    def __init__(self, num_classes=20, label_type="trainId", ignore_background=True, per_sample=False):
        self.num_classes = num_classes - ignore_background
        self.labels = list(range(self.num_classes))
        self.label_type = label_type
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.per_sample = per_sample
        if self.per_sample:
            self.per_sample_conf_matrix = []
            self.per_sample_report = []
            
    def __call__(self, y_pred=None, y=None):
        # y_pred is the prediction logits of a model
        # y is the label
        for element in range(y_pred.shape[0]):
            current_y_pred = y_pred[element, ...].unsqueeze(0)
            current_y = y[element, ...].unsqueeze(0)
            pred = current_y_pred.flatten().cpu().numpy()
            truth = current_y.flatten().cpu().numpy()

            conf = confusion_matrix(pred, truth, labels=self.labels)
            if self.per_sample:
                self.per_sample_conf_matrix.append(conf.copy())
            self.conf_matrix += conf

    def aggregate(self):
        IoUs = (np.diag(self.conf_matrix) / (np.sum(self.conf_matrix, axis=0)+np.sum(self.conf_matrix, axis=1)-np.diag(self.conf_matrix)))
        IoUs[np.isnan(IoUs)] = -0.0001
        
        report = {"IoUs":{}}
        
        if self.label_type == "trainId":
            for train_id in range(len(IoUs)):
                label = CsLabels.trainId2label[train_id].name
                report["IoUs"][label] = IoUs[train_id]
                
        elif self.label_type == "categories":
            for cat_id in range(len(IoUs)):
                label = CsLabels.category2label[cat_id].name
                report["IoUs"][label] = IoUs[cat_id]
                
        report["mIoU"] = np.mean(IoUs)
        
        if self.per_sample:
            for sample_conf_matrix in self.per_sample_conf_matrix:
                
                IoUs = (np.diag(sample_conf_matrix) / (np.sum(sample_conf_matrix, axis=0)+np.sum(sample_conf_matrix, axis=1)-np.diag(sample_conf_matrix)))
                
                sample_report = {"IoUs":{}}
                if self.label_type == "trainId":
                    for train_id in range(len(IoUs)):
                        label = CsLabels.trainId2label[train_id].name
                        sample_report["IoUs"][label] = IoUs[train_id]
                        
                elif self.label_type == "categories":
                    for cat_id in range(len(IoUs)):
                        label = CsLabels.category2label[cat_id].name
                        sample_report["IoUs"][label] = IoUs[cat_id]
                        
                sample_report["mIoU"] = np.nanmean(IoUs)
                self.per_sample_report.append(sample_report)
        
        return report
        
    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        if self.per_sample:
            self.per_sample_conf_matrix = []
            self.per_sample_report = []
            
            
def sample_mIoU_Cityscapes(y_pred, y, labels, label_type):
    
    pred = y_pred.flatten().cpu().numpy()
    truth = y.flatten().cpu().numpy()
    
    conf = confusion_matrix(pred, truth, labels=labels)
    
    IoUs = (np.diag(conf) / (np.sum(conf, axis=0)+np.sum(conf, axis=1)-np.diag(conf)))
    
    return np.nanmean(IoUs)



class CumulativeIoUTripleMNISTSegmentation():
    
    def __init__(self, num_classes=11, classes=["Background"]+list("0123456789"), ignore_background=False):
        self.num_classes = num_classes - ignore_background
        self.labels = list(range(self.num_classes))
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.classes = classes
            
    def __call__(self, y_pred=None, y=None):
        # y_pred is the prediction logits of a model
        # y is the label
        for element in range(y_pred.shape[0]):
            current_y_pred = y_pred[element, ...].unsqueeze(0)
            current_y = y[element, ...].unsqueeze(0)
            pred = current_y_pred.flatten().cpu().numpy()
            truth = current_y.flatten().cpu().numpy()
            conf = confusion_matrix(pred, truth, labels=self.labels)
            self.conf_matrix += conf

    def aggregate(self):
        IoUs = (np.diag(self.conf_matrix) / (np.sum(self.conf_matrix, axis=0)+np.sum(self.conf_matrix, axis=1)-np.diag(self.conf_matrix)))
        IoUs[np.isnan(IoUs)] = -0.0001
        
        report = {"IoUs":{}}

        for class_id in range(len(IoUs)):
            report["IoUs"][self.classes[class_id]] = IoUs[class_id]

        report["mIoU"] = np.mean(IoUs)

        return report
        
    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))