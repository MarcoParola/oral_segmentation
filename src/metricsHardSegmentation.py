import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps) # 
        recall = (tp + self.eps) / (tp + fn + self.eps) # also called sensitivity or TPR
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        jaccard = (tp + self.eps) / (fp + tp + fn + self.eps)

        return pixel_acc, dice, precision, specificity, recall, jaccard

    def __call__(self, y_true, y_pred, lbl_true): #lbl_true = label atttuale
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        lbl_true = lbl_true.to("cpu")
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                            ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall, jaccard = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                        activated_pred)
    
        dict_metrics = {'pixel_acc': pixel_acc, 'dice': dice, 'precision': precision, 'specificity': specificity,
                        'recall': recall, 'jaccard': jaccard}

        # get unique labels
        unique_lbls = np.unique(lbl_true)
        # calculate metrics for each label
        for lbl in unique_lbls:
            mask = lbl_true == lbl
            pixel_acc, dice, precision, specificity, recall, jaccard = self._calculate_overlap_metrics(y_true[mask],
                                                                                                    activated_pred[mask])
        
            # append to dictionary
            dict_metrics[f'pixel_acc_{lbl}'] = pixel_acc
            dict_metrics[f'dice_{lbl}'] = dice
            dict_metrics[f'precision_{lbl}'] = precision
            dict_metrics[f'specificity_{lbl}'] = specificity
            dict_metrics[f'recall_{lbl}'] = recall
            dict_metrics[f'jaccard_{lbl}'] = jaccard

        return dict_metrics


'''
Args:
    eps: float, a value added to the denominator for numerical stability.
        Default: 1e-5

    average: bool. Default: ``True``
        When set to ``True``, average Dice Coeff, precision and recall are
        returned. Otherwise Dice Coeff, precision and recall of each class
        will be returned as a numpy array.

    ignore_background: bool. Default: ``True``
        When set to ``True``, the class will not calculate related metrics on
        background pixels. When the segmentation of background pixels is not
        important, set this value to ``True``.

    activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
        This parameter determines what kind of activation function that will be
        applied on model output.

Input:
    y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
    to denote every class, where ``0`` denotes background class.
    y_pred: :math:`(N, C, H, W)`, torch tensor.

Examples::
    >>> metric_calculator = MultiClassMetrics(average=True, ignore_background=True)
    >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
'''

class MultiClassMetrics(object):
    def __init__(self, eps=1e-5, average=True, ignore_background=True):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))
        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:] # permette di non cosiderare il background
            

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps ) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]) + self.eps)
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred, lbl_true):
        lbl_true = lbl_true.to("cpu")
        class_num = y_pred.size(1)

        # Applicata activation 0-1
        pred_argmax = torch.argmax(y_pred, dim=1)
        activated_pred = self._one_hot(pred_argmax, y_pred, class_num)

        #gt_onehot = self._one_hot(y_true, y_pred, class_num)
        gt_onehot = y_true
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)

        dict_metrics = {'pixel_acc': pixel_acc, 'dice': dice, 'precision': precision,
                        'recall': recall}

        return dict_metrics

class MultiClassMetrics_manual_v2(object):
    def __init__(self, eps=1e-5, ignore_background=True):
        self.eps = eps
        self.ignore = ignore_background
    
    def __call__(self, y_true, y_pred, version_number):

        idx_class0 = 0
        idx_class1 = 1
        idx_class2 = 2
        idx_class3 = 3

        precision, recall, f1_score, support = precision_recall_fscore_support(y_true,y_pred,average=None, labels=[0,1,2,3])
        pixel_acc = accuracy_score(y_true,y_pred)

        cartella_destinazione = f"confusion_matrix/version_{version_number}"
        if os.path.exists(cartella_destinazione):
            for root, dirs, files in os.walk(cartella_destinazione):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            print(f"Cartella '{cartella_destinazione}' svuotata con successo.")
        else:
            os.makedirs(cartella_destinazione)
            print(f"Cartella '{cartella_destinazione}' creata con successo.")

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{cartella_destinazione}/confusion_matrix_multiclass.png")
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{cartella_destinazione}/confusion_matrix_multiclass_pred.png")


        y_true = [float(y > 0) for y in y_true]
        y_pred = [float(y > 0) for y in y_pred]
        precision_bin, recall_bin, f1_score_bin, support_bin = precision_recall_fscore_support(y_true,y_pred)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{cartella_destinazione}/confusion_matrix_binary.png")
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{cartella_destinazione}/confusion_matrix_binary_pred.png")


        weighted_precision = sum(precision * support) / sum(support)
        weighted_recall = sum(recall * support) / sum(support)
        weighted_fscore = sum(f1_score * support) / sum(support)

        dict_metrics = {
            'pixel_acc': pixel_acc,

            'precision_weighted': weighted_precision, 
            'precision_a': precision[idx_class0], 
            'precision_b': precision[idx_class1], 
            'precision_c': precision[idx_class2], 
            'precision_d': precision[idx_class3],
            'precision_binary_b':precision_bin[1],

            'recall_weighted': weighted_recall, 
            'recall_a': recall[idx_class0],
            'recall_b': recall[idx_class1], 
            'recall_c': recall[idx_class2], 
            'recall_d': recall[idx_class3],
            'recall_binary_b':recall_bin[1],

            'F_score_weighted': weighted_fscore,
            'F_score_a': f1_score[idx_class0],
            'F_score_b': f1_score[idx_class1], 
            'F_score_c': f1_score[idx_class2], 
            'F_score_d': f1_score[idx_class3],
            'F_socre_binary_b':f1_score_bin[1]
            }

        return dict_metrics 

class BinaryMetrics_manual(object):
    def __init__(self, eps=1e-5, ignore_background=True):
        self.eps = eps
        self.ignore = ignore_background
    
    def __call__(self, y_true, y_pred, version_number):

        pixel_acc = accuracy_score(y_true,y_pred)

        # By setting zero_division=1, it is assumed that the precision for these classes is perfect (1.0), 
        # whereas setting zero_division=0 assumes that it does not contribute to the score (0.0). 
        # The choice of how to handle these situations depends on the specific context and the goals of your analysis.
        precision, recall, f1_score, support = precision_recall_fscore_support(y_true,y_pred,average='binary', zero_division = 1)
        precision_weighted, recall_weighted, f1_score_weighted, support_micro = precision_recall_fscore_support(y_true,y_pred, average='weighted', zero_division = 0)
        
        cartella_destinazione = f"confusion_matrix/version_{version_number}"
        if os.path.exists(cartella_destinazione):
            for root, dirs, files in os.walk(cartella_destinazione):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            print(f"Cartella '{cartella_destinazione}' svuotata con successo.")
        else:
            os.makedirs(cartella_destinazione)
            print(f"Cartella '{cartella_destinazione}' creata con successo.")

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"confusion_matrix/version_{version_number}/confusion_matrix_true.png")

        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"confusion_matrix/version_{version_number}/confusion_matrix_pred.png")

        dict_metrics = {
            'Precision_binary': precision,
            'Recall_binary': recall,
            'F1_Score_binary': f1_score,
            'Precision_weighted': precision_weighted,
            'Recall_weighted': recall_weighted,
            'F1_Score_weighted': f1_score_weighted,
            'Pixel_accuracy': pixel_acc
        }

        return dict_metrics