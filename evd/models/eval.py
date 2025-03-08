import time
import torch
import wandb

import evd.utils

def validate(val_loader, model, criterion, epoch, device_id=None, wandb_run=None, logger=None) :
    batch_time = evd.utils.AverageMeter("Time", ":6.3f")
    losses = evd.utils.AverageMeter("Loss", ":.4e")
    top1 = evd.utils.AverageMeter("Acc@1", ":6.2f")
    top5 = evd.utils.AverageMeter("Acc@5", ":6.2f")
    
    # ProgressMeter object to display 
    progress = evd.utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix=f"Test epoch {epoch}: ",
    )
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            images = images.cuda(device_id, non_blocking=True) if device_id is not None else (images.cuda(0, non_blocking=True) if torch.cuda.is_available() else images)
            if torch.cuda.is_available():
                target = target.cuda(device_id, non_blocking=True)

            # compute output
            try:
                with torch.cuda.amp.autocast(True):
                    output = model(images)
                    loss = criterion(output, target)
            except:
                if torch.cuda.is_available():
                    model = model.cuda(device_id)
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = evd.utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if evd.utils.is_main_process():
                progress.display(i)

        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

        if evd.utils.is_main_process() and wandb_run is not None:
            wandb.log(
                {
                    "val/top1": acc1[0].item(),
                    "val/top5": acc5[0].item(),
                    "val/loss": loss.item(),
                    "epoch": epoch,
                }
            )
            metrics = progress.display(i)
            logger.info(metrics)
        

    return top1.avg, top5.avg



# Using example from evd_gpus/scripts/analysis.py
# to call validate(val_loader, model, criterion, epoch, device_id, wandb_run=None, logger=None), need to pass in the following arguments:
# val_loader, model, criterion, epoch, device_id, wandb_run=None, logger=None