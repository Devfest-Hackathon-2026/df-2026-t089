"""YOLO26 training and validation with ignored, unlabeled product regions.

The model and trainer classes live at module scope because PyTorch checkpoints
serialize model class references by their importable module path.
"""
from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

IGNORE_BY_STEM = {}


class IgnoreLoss(v8DetectionLoss):
    def get_assigned_targets_and_loss(self, preds, batch, **assign_kwargs):
        """Standard YOLO assignment and box loss; zero BCE on unknown anchors."""
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, **assign_kwargs)
        target_scores_sum = target_scores.sum().clamp_(min=1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        unknown = torch.zeros((batch_size, anchor_points.shape[0]), dtype=torch.bool, device=self.device)
        centers = anchor_points * stride_tensor
        for image_idx, box in zip(batch.get("ignore_idx", []), batch.get("ignore_boxes", [])):
            x0, y0, x1, y1 = box * imgsz[[1, 0, 1, 0]]
            unknown[image_idx] |= ((centers[:, 0] >= x0) & (centers[:, 0] <= x1)
                & (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
        bce_loss *= (~(unknown & ~fg_mask)).unsqueeze(-1)
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss, dict(zip(self.loss_names, loss.detach())))


class IgnoreModel(DetectionModel):
    def init_criterion(self):
        return E2ELoss(self, loss_fn=IgnoreLoss) if getattr(self.model[-1], "one2one_cv2", None) is not None else IgnoreLoss(self)


class IgnoreValidator(DetectionValidator):
    def update_metrics(self, preds, batch):
        """Do not score predictions inside unknown regions as false positives."""
        height, width = batch["img"].shape[-2:]
        for i, pred in enumerate(preds):
            if not pred["conf"].numel():
                continue
            centers = (pred["bboxes"][:, :2] + pred["bboxes"][:, 2:]) / 2
            centers = centers / centers.new_tensor([width, height])
            unknown = torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
            for box in IGNORE_BY_STEM.get(Path(batch["im_file"][i]).stem, []):
                x0, y0, x1, y1 = box
                unknown |= ((centers[:, 0] >= x0) & (centers[:, 0] <= x1)
                            & (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
            labels = batch["bboxes"][batch["batch_idx"].view(-1) == i]
            for x, y, w, h in labels:
                unknown &= ~((centers[:, 0] >= x-w/2) & (centers[:, 0] <= x+w/2)
                             & (centers[:, 1] >= y-h/2) & (centers[:, 1] <= y+h/2))
            keep = ~unknown
            preds[i] = {key: value[keep] for key, value in pred.items()}
        return super().update_metrics(preds, batch)


class IgnoreTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(IgnoreModel(cfg, nc=self.data["nc"],
                                                         ch=self.data["channels"], verbose=verbose))
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        pairs = [(i, box) for i, filename in enumerate(batch["im_file"])
                 for box in IGNORE_BY_STEM.get(Path(filename).stem, [])]
        batch["ignore_idx"] = torch.tensor([i for i, _ in pairs], dtype=torch.long, device=self.device)
        batch["ignore_boxes"] = torch.tensor([box for _, box in pairs], dtype=torch.float32,
                                              device=self.device).reshape(-1, 4)
        return batch

    def get_validator(self):
        return IgnoreValidator(self.test_loader, save_dir=self.save_dir,
                               args=copy(self.args), _callbacks=self.callbacks)
