# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import os
import matplotlib.pyplot as plt
import numpy as np

#import nibabel as nib

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
#from __future__ import print_function

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class MaxSizeQueue:
    def __init__(self, max_size, device="cpu"):
        self.max_size = max_size
        self.queue = torch.Tensor()
    
    def push(self, element):
        """
        element is expected of size [batch_size, dim]
        Prepend the new elements to the provided list and only keep the first x elements
        """
        assert element.shape
        if element.device != self.queue.device:
            self.to_device(element.device)
        self.queue = torch.cat((element, self.queue),dim=0)[:self.max_size,:]
    
    def getsize(self):
        return self.queue.shape[0]
    
    def isful(self):
        return self.queue.shape[0]==self.max_size
    
    def getqueue(self):
        return self.queue

    def to_device(self,device):
        self.queue = self.queue.to(device=device)
        
class MemoryBank:
    def __init__(self,nof_classes,size, dimensionality) -> None:
        self.nof_classes = nof_classes
        self.size = size
        self.dimensionality = dimensionality
        self.bank = dict(zip(range(nof_classes),[MaxSizeQueue(self.size) for _ in range(nof_classes)]))
        self.device = "cpu"
        
    def isful(self):
        """Returns if the current bank is full
        """
        return all(x.isful() for x in self.bank.values())
    
    def push(self, values, class_id):
        """Push new elements into the queue"""
        assert len(values.shape) == 2, f"Expect two dimensional input [batchsize, features], but got {len(values.shape)}"
        self.bank[class_id].push(values)
    
    def get_bank(self, return_labels=True):
        assert self.isful(), "Can only return memory bank if it it is full"
        if return_labels:
            values = torch.stack([x.getqueue() for x in self.bank.values()],dim=0).flatten(start_dim=0,end_dim=1)
            labels = torch.cat([torch.tensor(i).repeat(self.size) for i in range(self.nof_classes)]).to(self.device)
            return values,labels
        else:
            return torch.stack([x.getqueue() for x in self.bank.values()],dim=0).flatten(start_dim=0,end_dim=1)
    
    def to_device(self,device):
        """Send memory bank to passed device"""
        self.device = device
        for key, val in self.bank.items():
            val.to_device(device=device)
        


@META_ARCH_REGISTRY.register()
class MaskFormer_dual_transformer_decoder_pixelcontrast(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        anatomy_criterion: nn.Module,
        pathology_criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.anatomy_criterion = anatomy_criterion
        self.pathology_criterion = pathology_criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        #Inference arg
        self._inference_mode = "anatomy"

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self.memory_bank = MaxSizeQueue(256)
        self.memory_bank.to_device(self.pixel_mean.device)
        self.cont_loss = SupConLoss()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        anatomy_criterion = SetCriterion(
            145,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        pathology_criterion = SetCriterion(
            2,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "anatomy_criterion": anatomy_criterion,
            "pathology_criterion": pathology_criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        anatomy_outputs, pathology_outputs, mask_features = self.sem_seg_head(features)
        #assert len(batched_inputs)==1
        #os.environ["CANCER_ID"]=batched_inputs[0]["file_name"].split("/")[-1].split("_")[2]
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            if "instances_path" in batched_inputs[0]:
                gt_instances_path = [x["instances_path"].to(self.device) for x in batched_inputs]
                targets_pathology = self.prepare_targets(gt_instances_path, images)
            else:
                targets_pathology = None

            assert all([x["labels"][0] == 0 for x in targets])
            assert all([x["labels"][0] == 0 for x in targets_pathology])
            features_representing_anatomy = torch.stack([x["masks"][1:].sum(dim=0) for x in targets])
            features_representing_pathology = torch.stack([x["masks"][1:].sum(dim=0) for x in targets_pathology])

            #TODO: Alex continue here later. Interploate the masks downwards and apply the pixel contrast loss
            sce_loss = None
            target_size = mask_features.shape[-2:]
            features_representing_anatomy = F.interpolate(features_representing_anatomy.to(torch.uint8).unsqueeze(1),target_size,mode="nearest").bool()
            features_representing_pathology = F.interpolate(features_representing_pathology.to(torch.uint8).unsqueeze(1),target_size,mode="nearest").bool()
            
            bs = mask_features.shape[0]
            collected_anatomy_features = []
            collected_pathology_features = []
            for i in range(bs):
                selected_elements_anatomy = mask_features[i][:, features_representing_anatomy[i,0]] 
                selected_elements_pathology = mask_features[i][:, features_representing_pathology[i,0]]
                collected_anatomy_features.append(selected_elements_anatomy)
                collected_pathology_features.append(selected_elements_pathology)
            collected_anatomy_features = torch.concat(collected_anatomy_features,dim=1).transpose(0,1)
            collected_pathology_features = torch.concat(collected_pathology_features,dim=1).transpose(0,1)
            
            

            #Apply contrastive loss
            if self.memory_bank.isful() and collected_anatomy_features.shape[0]>256:
                positives = collected_anatomy_features[torch.randperm(collected_anatomy_features.shape[0])][:256]
                negatives = torch.concat((self.memory_bank.getqueue(),collected_pathology_features))
                loss_in = F.normalize(torch.concat((positives,negatives)),dim=1)
                label_in = torch.concat((torch.Tensor([0]).to(torch.uint8).repeat(256), torch.Tensor([1]).to(torch.uint8).repeat(256+collected_pathology_features.shape[0]))).to(loss_in.device)
                sce_loss = self.cont_loss(loss_in.unsqueeze(1),label_in)
            
            if collected_pathology_features.shape[0]>0:
                self.memory_bank.push(collected_pathology_features.detach())

            # bipartite matching-based loss
            losses_anatomy = self.anatomy_criterion(anatomy_outputs, targets)
            losses_pathology = self.pathology_criterion(pathology_outputs, targets_pathology)


            for k in list(losses_anatomy.keys()):
                if k in self.anatomy_criterion.weight_dict:
                    losses_anatomy[k] *= self.anatomy_criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_anatomy.pop(k)
            
            for k in list(losses_pathology.keys()):
                if k in self.pathology_criterion.weight_dict:
                    losses_pathology[k] *= self.pathology_criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_pathology.pop(k)
            
            #Merge the two loss dicts
            pathology_anatomy_weight = 0.5
            sce_weight = 0.25
            #Scale the losses to reflect the desired output
            merged_losses = {
                k:pathology_anatomy_weight*losses_anatomy[k] + (1-pathology_anatomy_weight)*losses_pathology[k] for k in set(losses_anatomy.keys()).intersection(losses_pathology.keys()) 
            }
            if sce_loss is not None:
                merged_losses.update({"sup_con":sce_loss*sce_weight})
            return merged_losses
        else:
            if self.inference_mode == "anatomy":

                #Take care of anatomy
                mask_cls_results_anatomy = anatomy_outputs["pred_logits"]
                mask_pred_results_anatomy = anatomy_outputs["pred_masks"]

                mask_pred_results_anatomy = F.interpolate(
                    mask_pred_results_anatomy,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del anatomy_outputs
                del pathology_outputs

                ## Process the results for the anatomy
                
                processed_results_anatomy = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results_anatomy, mask_pred_results_anatomy, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results_anatomy.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results_anatomy[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results_anatomy[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results_anatomy[-1]["instances"] = instance_r
                
                return processed_results_anatomy
            

            elif self.inference_mode == "pathology":
                
                mask_cls_results_pathology = pathology_outputs["pred_logits"]
                mask_pred_results_pathology = pathology_outputs["pred_masks"]
                # upsample pathology masks
                mask_pred_results_pathology = F.interpolate(
                    mask_pred_results_pathology,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                
                del anatomy_outputs
                del pathology_outputs
                
                processed_results_pathology = []
                
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results_pathology, mask_pred_results_pathology, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results_pathology.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results_pathology[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results_pathology[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results_pathology[-1]["instances"] = instance_r

                if "RESULTS_PATH" in os.environ:
                    qual_results = [x["sem_seg"].argmax(dim=0) for x  in processed_results_pathology]
                    target_path = os.environ["RESULTS_PATH"]
                    for i,r in enumerate(qual_results):
                        if True:
                        #if 1 in np.unique(r.cpu()) and 1 in batched_inputs[i]["sem_seg"]:
                            #fig = plt.figure(figsize=(10, 5))
                            #fig.add_subplot(2, 2, 1) 
                            #plt.imshow(images[i].cpu().permute(1,2,0)[:,:,0])
                            nifti_ct = nib.Nifti1Image(images[i].cpu().permute(1,2,0)[:,:,0].numpy(), affine=np.eye(4))
                            nifti_ct.to_filename(os.path.join(target_path,"ct.nii"))
                            nifti_pet = nib.Nifti1Image(images[i].cpu().permute(1,2,0)[:,:,1].numpy(), affine=np.eye(4))
                            nifti_pet.to_filename(os.path.join(target_path,"pet.nii"))
                            nifti_gt = nib.Nifti1Image(batched_inputs[i]["sem_seg"].numpy().astype(np.uint8), affine=np.eye(4))
                            nifti_gt.to_filename(os.path.join(target_path,"gt.nii"))
                            nifty_pred = nib.Nifti1Image(r.cpu().numpy().astype(np.uint8), affine=np.eye(4))
                            nifty_pred.to_filename(os.path.join(target_path,"pred.nii"))
                            
                            #Select based on entire volume

                            if "_".join(batched_inputs[i]["file_name"].split("/")[-1].split("_")[2:6]) in ["2e97a9e5c2_2003_6_23","ae96f738c0_2003_5_1","ba717a1f22_2003_3_28","2f7200f771_2005_11_5","d69c3fceba_2003_4_5"]:
                                cur_file_name = batched_inputs[i]["file_name"].split("/")[-1]
                                cur_file_name = cur_file_name.split(".")[0]
                                nifti_ct.to_filename(os.path.join(target_path,cur_file_name+"_ct.nii"))
                                nifti_pet.to_filename(os.path.join(target_path,cur_file_name+"_pet.nii"))
                                nifti_gt.to_filename(os.path.join(target_path,cur_file_name+"_gt.nii"))
                                nifty_pred.to_filename(os.path.join(target_path,cur_file_name+"_pred.nii"))

                            # #Skip the rest for now    
                            continue

                            #Calc Iou for this image
                            intersection = np.logical_and(batched_inputs[i]["sem_seg"].numpy().astype(np.uint8) == 1, r.cpu().numpy().astype(np.uint8) == 1).sum()
                            union = np.logical_or(batched_inputs[i]["sem_seg"].numpy().astype(np.uint8) == 1, r.cpu().numpy().astype(np.uint8) == 1).sum()
                            our_iou = intersection/union

                            try:#Try to load the array of baseline
                                baseline = nib.load("/local/baseline_results/"+batched_inputs[0]["file_name"].split("/")[-1]+".nii").get_fdata()
                                intersection = np.logical_and(batched_inputs[i]["sem_seg"].numpy().astype(np.uint8) == 1, baseline == 1).sum()
                                union = np.logical_or(batched_inputs[i]["sem_seg"].numpy().astype(np.uint8) == 1, baseline == 1).sum()
                                baseline_iou = intersection/union

                                if our_iou - baseline_iou>0.2 and baseline_iou>0.1:
                                    print("YES!")
                                    print("/local/baseline_results/"+batched_inputs[0]["file_name"].split("/")[-1]+".nii")

                            except: 
                                pass
                            
                            #fig.add_subplot(2, 2, 2) 
                            #plt.imshow(images[i].cpu().permute(1,2,0)[:,:,1])
                            #fig.add_subplot(2, 2, 3) 
                            #plt.imshow(r.cpu())
                            #fig.add_subplot(2, 2, 4)
                            #plt.imshow(batched_inputs[i]["sem_seg"].cpu()) 
                            #plt.savefig(os.path.join(target_path,batched_inputs[i]["file_name"].split("/")[-1]))
                            #plt.close('all')
                
                return processed_results_pathology
            else:
                raise ValueError(f"Inference Mode of Model whould only be possible to be either 'anatomy' or 'pathology', but got {self.inference_mode}")


    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    

    @property
    def inference_mode(self):
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, mode):
        assert mode in ["anatomy", "pathology"], f"Expected inference mode to be either anatomy or pathology. Got {mode}"
        self._inference_mode = mode