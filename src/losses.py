import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from src.matcher import HungarianMatcher, BoxUtils

class OWLVitLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses_dict: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses_dict = losses_dict
        self.background_label = num_classes
        empty_weight = torch.ones(self.num_classes)
        # empty_weight[0] = eos_coef
        self.class_criterion = nn.BCELoss(reduction='none', weight = empty_weight)

    def _get_target_classes(self, outputs, targets, indices):
        assert 'logits' in outputs
        batch_idx, src_idx = self._get_src_permutation_idx(indices=indices)
        target_classes_o = torch.cat(
            [t['class_labels'][J].to(torch.int64) for t,(_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            outputs['logits'].shape[:2], # batch, num_queries
            self.num_classes,
            dtype =  torch.int64,
            device =outputs['logits'].device
        )
        target_classes[batch_idx, src_idx] = target_classes_o
        return target_classes

    def loss_labels(self, outputs, targets, indices, num_boxes):
        target_classes = self._get_target_classes(outputs=outputs, targets=targets, indices=indices)
        torch.save(target_classes, f'output/logs/target_classes.pt')
        source_logits = outputs['logits'].transpose(1,2) # batch_size, num_queries,  num_patches
        source_logits = torch.nn.Sigmoid()(source_logits)
        # batch_size = 1
        target_classes = target_classes.squeeze(0)
        source_logits = source_logits.squeeze(0)

        pred_logits = source_logits[:, target_classes != self.background_label].t()
        background_logits = source_logits[:, target_classes == self.background_label].t()
        target_classes = target_classes[target_classes != self.background_label]

        pos_targets = F.one_hot(target_classes, self.background_label).float()
        neg_targets = torch.zeros(background_logits.shape).to(background_logits.device)

        pos_loss = self.class_criterion(pred_logits, pos_targets)
        neg_loss = self.class_criterion(background_logits, neg_targets)

        pos_loss = (torch.pow(1- torch.exp(-pos_loss), 2) * pos_loss).sum(dim=1).mean()
        neg_loss = (torch.pow(1- torch.exp(-neg_loss), 2) * neg_loss).sum(dim=1).mean()

        losses= {'loss_ce': pos_loss,'loss_bg' : neg_loss}
        
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(src_boxes),
            BoxUtils.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss_name, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss_name in loss_map, f'do you really want to compute {loss_name} loss?'
        return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)
    

    def check_format(self, outputs, targets):
        assert isinstance(outputs, dict), "Outputs should be a dictionary"
        assert "logits" in outputs.keys(), "Outputs should contain 'logits' key"
        assert "pred_boxes" in outputs.keys(), "Outputs should contain 'pred_boxes' key"
        assert outputs["logits"].dim() == 3, "Logits tensor should have shape [batch_size, num_queries, num_classes]"
        assert outputs["pred_boxes"].dim() == 3, "Pred_boxes tensor should have shape [batch_size, num_queries, 4]"
        assert outputs["pred_boxes"].size(-1) == 4, "Pred_boxes tensor should have last dimension of size 4"
        assert len(targets) == outputs["logits"].size(0), "Number of targets should match batch size"
        for target in targets:
            assert isinstance(target, dict), "Each target should be a dictionary"
            assert "class_labels" in target.keys(), "Target should contain 'class_labels' key"
            assert "boxes" in target.keys(), "Target should contain 'boxes' key"
            assert target["class_labels"].dim() == 1, "Class_labels tensor should have shape [num_target_boxes]"
            assert target["boxes"].dim() == 2, "Boxes tensor should have shape [num_target_boxes, 4]"
            assert target["boxes"].size(-1) == 4, "Boxes tensor should have last dimension of size 4"

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        self.check_format(outputs_without_aux, targets)
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['logits'].device)

        # Compute all the requested losses
        losses = {}
        for loss_name in self.losses_dict:
            losses.update(self.get_loss(loss_name, outputs, targets, indices, num_boxes))
        
        return losses