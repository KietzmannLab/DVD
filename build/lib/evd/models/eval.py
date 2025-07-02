import torch
import wandb
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import evd.utils



@torch.no_grad()
def validate(
    val_loader,
    model,
    criterion,
    epoch,
    device_id=0,
    wandb_run=None,
    logger=None,
    image_size=None,
    log_freq=10,
    subset_class_ids=None,  # NEW ARGUMENT
    compute_per_class_acc = False,
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    model.eval()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    batch_time = evd.utils.AverageMeter("Time", ":6.3f")

    total_loss = torch.zeros((), device=device)
    total_samples = 0
    outputs_accum = []
    targets_accum = []

    start_evt.record()

    for i, (images, targets) in enumerate(val_loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if image_size and (images.shape[2] != image_size or images.shape[3] != image_size):
            images = torch.nn.functional.interpolate(
                images, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
            if evd.utils.is_main_process() and i == 0:
                print(f"Notice: resizing to {image_size}×{image_size} on GPU")

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, targets)

        bs = images.size(0)
        total_loss   += loss * bs
        total_samples += bs
        outputs_accum.append(outputs)
        targets_accum.append(targets)

        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0
        batch_time.update(elapsed)
        start_evt.record()

        if evd.utils.is_main_process() and (i % log_freq == 0 or i + 1 == len(val_loader)):
            evd.utils.ProgressMeter(
                len(val_loader), [batch_time], prefix=f"Test epoch {epoch}: "
            ).display(i)

    all_out = torch.cat(outputs_accum, dim=0)
    all_tgt = torch.cat(targets_accum, dim=0)

    if subset_class_ids is not None:
        subset_class_ids = torch.tensor(subset_class_ids, device=all_tgt.device)
        mask = torch.isin(all_tgt, subset_class_ids)
        all_out = all_out[mask]
        all_tgt = all_tgt[mask]
        if all_tgt.numel() == 0:
            print("Warning: no samples found for the specified subset class IDs.")
            acc1 = acc5 = torch.tensor(0.0, device=device)
        else:
            acc1, acc5 = evd.utils.accuracy(all_out, all_tgt, topk=(1, 5))
    else:
        acc1, acc5 = evd.utils.accuracy(all_out, all_tgt, topk=(1, 5))

    avg_loss = (total_loss / total_samples).item()

    if evd.utils.is_main_process():
        print(f"epoch {epoch} | Val Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}\n")

    if evd.utils.is_main_process() and wandb_run:
        wandb_run.log({
            "val/top1": acc1.item(),
            "val/top5": acc5.item(),
            "val/loss": avg_loss,
            "epoch": epoch,
        })
        if logger:
            logger.info(f"Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}")
    
    if compute_per_class_acc:
        if evd.utils.is_main_process():
            # Compute per-class accuracy
            top1_preds = all_out.argmax(dim=1)
            correct_mask = top1_preds == all_tgt

            class_correct = {}
            class_total = {}

            for label in all_tgt.unique():
                label = label.item()
                mask = all_tgt == label
                correct = (top1_preds[mask] == label).sum().item()
                total = mask.sum().item()
                class_correct[label] = correct
                class_total[label] = total

            # Compute accuracy
            class_acc = {
                k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0.0
                for k in class_total
            }

            # Sort and print
            sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
            print("Per-class Top-1 Accuracy (sorted):")
            for class_id, acc in sorted_acc:
                print(f"Class {class_id:4d} | Accuracy: {acc:.3f} ({class_correct[class_id]}/{class_total[class_id]})")

    return acc1.item(), acc5.item()

@torch.no_grad()
def validate_in_limited(
    val_loader,
    model,
    criterion,
    epoch,
    device_id=0,
    wandb_run=None,
    logger=None,
    image_size=None,
    log_freq=10,
    subset_class_ids=None,  # LIMIT DECISION SPACE TO THESE
    compute_per_class_acc=False,
):
    import torch.nn.functional as F
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    model.eval()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    batch_time = evd.utils.AverageMeter("Time", ":6.3f")

    total_loss = torch.zeros((), device=device)
    total_samples = 0
    outputs_accum = []
    targets_accum = []

    start_evt.record()

    for i, (images, targets) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if image_size and (images.shape[2] != image_size or images.shape[3] != image_size):
            images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
            if evd.utils.is_main_process() and i == 0:
                print(f"Notice: resizing to {image_size}×{image_size} on GPU")

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        bs = images.size(0)
        total_loss += loss * bs
        total_samples += bs
        outputs_accum.append(outputs)
        targets_accum.append(targets)

        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0
        batch_time.update(elapsed)
        start_evt.record()

        if evd.utils.is_main_process() and (i % log_freq == 0 or i + 1 == len(val_loader)):
            evd.utils.ProgressMeter(
                len(val_loader), [batch_time], prefix=f"Test epoch {epoch}: "
            ).display(i)

    all_out = torch.cat(outputs_accum, dim=0)
    all_tgt = torch.cat(targets_accum, dim=0)

    if subset_class_ids is not None:
        # Restrict prediction logits to only the subset
        subset_class_ids = torch.tensor(subset_class_ids, device=all_out.device)
        # Create a logits tensor only for the subset
        all_out_subset = all_out[:, subset_class_ids]  # (N, subset_size)
        # Map original targets to their index within subset_class_ids
        id_to_index = {cls_id.item(): i for i, cls_id in enumerate(subset_class_ids)}
        tgt_mapped = torch.tensor([id_to_index.get(t.item(), -1) for t in all_tgt], device=all_out.device)

        # Filter out targets not in the subset
        valid_mask = tgt_mapped != -1
        all_out_subset = all_out_subset[valid_mask]
        tgt_mapped = tgt_mapped[valid_mask]

        if tgt_mapped.numel() == 0:
            print("Warning: No targets match subset_class_ids")
            acc1 = acc5 = torch.tensor(0.0, device=device)
        else:
            acc1, acc5 = evd.utils.accuracy(all_out_subset, tgt_mapped, topk=(1, 5))
    else:
        acc1, acc5 = evd.utils.accuracy(all_out, all_tgt, topk=(1, 5))

    avg_loss = (total_loss / total_samples).item()

    if evd.utils.is_main_process():
        print(f"epoch {epoch} | Val Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}\n")

    if evd.utils.is_main_process() and wandb_run:
        wandb_run.log({
            "val/top1": acc1.item(),
            "val/top5": acc5.item(),
            "val/loss": avg_loss,
            "epoch": epoch,
        })
        if logger:
            logger.info(f"Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}")

    if compute_per_class_acc and evd.utils.is_main_process():
        top1_preds = all_out.argmax(dim=1)
        correct_mask = top1_preds == all_tgt
        class_correct = {}
        class_total = {}

        for label in all_tgt.unique():
            label = label.item()
            mask = all_tgt == label
            correct = (top1_preds[mask] == label).sum().item()
            total = mask.sum().item()
            class_correct[label] = correct
            class_total[label] = total

        class_acc = {
            k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0.0
            for k in class_total
        }

        sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
        print("Per-class Top-1 Accuracy (sorted):")
        for class_id, acc in sorted_acc:
            print(f"Class {class_id:4d} | Accuracy: {acc:.3f} ({class_correct[class_id]}/{class_total[class_id]})")

    return acc1.item(), acc5.item()

# @torch.no_grad()
# def validate(
#     val_loader,
#     model,
#     criterion,
#     epoch,
#     device_id=0,
#     wandb_run=None,
#     logger=None,
#     image_size=None,
#     log_freq=10,
# ):
#     # — Setup device & move model/criterion once —
#     device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     criterion = criterion.to(device)
#     try:
#         model = torch.compile(model)   # PyTorch 2.0+ JIT-style compiler
#     except Exception:
#         pass
#     model.eval()

#     # — Prepare timing (CUDA events) and meters —
#     start_evt = torch.cuda.Event(enable_timing=True)
#     end_evt   = torch.cuda.Event(enable_timing=True)
#     batch_time = evd.utils.AverageMeter("Time", ":6.3f")

#     total_loss = torch.zeros((), device=device)
#     total_samples = 0
#     outputs_accum = []
#     targets_accum = []

#     # — Warm up timing —
#     start_evt.record()

#     for i, (images, targets) in enumerate(val_loader):
#         # 1) Transfer to GPU (async)
#         images  = images.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         # 2) (Optional) GPU resize if needed
#         if image_size and (images.shape[2] != image_size or images.shape[3] != image_size):
#             images = torch.nn.functional.interpolate(
#                 images,
#                 size=(image_size, image_size),
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             if evd.utils.is_main_process() and i == 0:
#                 print(f"Notice: resizing to {image_size}×{image_size} on GPU")

#         # 3) Inference + loss under AMP
#         with autocast():
#             outputs = model(images)
#             loss    = criterion(outputs, targets)

#         # 4) Accumulate on GPU
#         bs = images.size(0)
#         total_loss   += loss * bs
#         total_samples += bs
#         outputs_accum.append(outputs)
#         targets_accum.append(targets)

#         # 5) Measure time with CUDA events
#         end_evt.record()
#         torch.cuda.synchronize()
#         elapsed = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
#         batch_time.update(elapsed)
#         start_evt.record()

#         # 6) Occasional progress print
#         if evd.utils.is_main_process() and (i % log_freq == 0 or i + 1 == len(val_loader)):
#             evd.utils.ProgressMeter(
#                 len(val_loader),
#                 [batch_time],
#                 prefix=f"Test epoch {epoch}: "
#             ).display(i)

#     # — End of loop: compute metrics once on GPU —
#     # import pdb;pdb.set_trace()
#     all_out = torch.cat(outputs_accum, dim=0)
#     all_tgt = torch.cat(targets_accum, dim=0)
#     acc1, acc5 = evd.utils.accuracy(all_out, all_tgt, topk=(1, 5))
#     avg_loss = (total_loss / total_samples).item()

#     if evd.utils.is_main_process():
#         print(f"epoch {epoch} | Val Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}\n")

#     # — Log to W&B and your logger if desired —
#     if evd.utils.is_main_process() and wandb_run:
#         wandb_run.log({
#             "val/top1": acc1.item(),
#             "val/top5": acc5.item(),
#             "val/loss": avg_loss,
#             "epoch": epoch,
#         })
#         if logger:
#             logger.info(f"Acc@1 {acc1.item():.3f}  Acc@5 {acc5.item():.3f}  Loss {avg_loss:.4f}")

#     return acc1.item(), acc5.item()


# Using example from evd_gpus/scripts/analysis.py
# to call validate(val_loader, model, criterion, epoch, device_id, wandb_run=None, logger=None), need to pass in the following arguments:
# val_loader, model, criterion, epoch, device_id, wandb_run=None, logger=None



