import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np
from sklearn import metrics

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples,_ , targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if max_norm != 0:
                #print('start')
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=1.0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_f1score(probability, truth, threshold):
    if threshold is None:
        predict = [probability]
    else:
        predict = [
            (probability > t).astype(np.float32) for t in threshold
        ]
        
    f1score = []
    for p in predict:
        tp = ((p >= 0.5) & (truth >= 0.5)).sum()
        fp = ((p >= 0.5) & (truth < 0.5)).sum()
        fn = ((p < 0.5) & (truth >= 0.5)).sum()
        recall = tp / (tp + fn + 1e-3)
        precision = tp / (tp + fp + 1e-3)
        f1 = 2 * recall * precision / (recall + precision + 1e-3)
        f1score.append(f1)
    f1score = np.array(f1score)
    return f1score

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    ground_truths_multiclass = []
    ground_truths_multilabel = []
    predictions_class = []
    scores = []
    total = 0
    # switch to evaluation mode
    model.eval()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        labels_onehot = batch[1]
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        
        _, predicted_class = torch.max(output.data, 1)
        #output_class = utils.softmax()
        
        outputs_class = utils.softmax(output.data.cpu().numpy())
        ground_truths_multiclass.extend(target.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)
        total += target.size(0)
        metric_logger.update(loss=loss.item())
        
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #batch_size = images.shape[0]
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    """Mesure the prediction performance on valid set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    auc_score = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    

    specificity = metrics.recall_score(gts, preds,average='micro')
    
    #thresh  = np.linspace(0, 1, 50)
    #f1score = get_f1score(preds, gts, thresh)
    
    f1score = metrics.f1_score(gts, preds, average='micro')
    #wF1 = metrics.f1_score(gts, preds,average='weighted')

    #metric_logger.add_meter(name='auc',meter=auc_score)
    metric_logger.update(auc=auc_score)
    metric_logger.update(kappa=wKappa)
    metric_logger.update(spec=specificity)
    metric_logger.update(f1=f1score)
    metric_logger.update(accuracy_score=accuracy)
    #metric_logger.add_meter(name='kappa',meter=wKappa)
    #metric_logger.add_meter(name='f1',meter=wF1)
    #print(metric_logger.kappa)
    #print(metric_logger.auc)
    #print(metric_logger.f1)
    #print(metric_logger.loss)
    print(' Accuracy:{accuracy_score.global_avg:.4f}==============  SPEC:{spec.global_avg:.4f}===========   kappa:{kappa.global_avg:.4f} ============= F1:{f1.global_avg:.4f} ============== Loss:{loss.global_avg:.4f}'
          .format(accuracy_score=metric_logger.accuracy_score, spec=metric_logger.spec, kappa=metric_logger.kappa, f1=metric_logger.f1, loss=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}