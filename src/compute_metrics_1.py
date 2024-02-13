#compute IoU e Accuracy

from utils import AverageMeter
from tqdm import tqdm
import torch
import time
import numpy as np
import torch.cuda as cuda
import collections


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj

def evaluate_IoU_Acc(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        # as_numpy definita sopra
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                del feed_dict['name']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp, ft = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            tmp_scores = scores
            if cfg.metrics.exclude_back:
                tmp_scores = tmp_scores[:,1:]

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        # seg_label[seg_label >= 4] = seg_label[seg_label >= 4] + 1
        acc, pix = accuracy(pred, seg_label)
        # print(np.unique(pred, False, False, True))
        # print(np.unique(seg_label, False, False, True))

        # pred[pred != 12] = -1
        # seg_label[seg_label != 12] = -1

        # plt.figure()
        # plt.imshow(pred)
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(seg_label)
        # plt.show()

        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        print(intersection, union)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        """
        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.TEST.result),
                as_numpy(scores.squeeze(0).cpu())
            )

        """
        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))