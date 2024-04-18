import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area
import numpy as np
from torch import nn
class BoxUtils(object):
    @staticmethod
    def box_cxcywh_to_xyxy(x: torch.Tensor): # [ N, 4]
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def box_xyxy_to_cxcywh(x : torch.Tensor):# [ N, 4]
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):# [ N, 4]
        img_h, img_w = size
        b = BoxUtils.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b


    @staticmethod
    def box_area(boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.
        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format
        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
    @staticmethod
    # modified from torchvision to also return the union
    def box_iou(boxes1, boxes2):
        area1 = BoxUtils.box_area(boxes1)
        area2 = BoxUtils.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = BoxUtils.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(
        self, 
        cost_class: float = 1, 
        cost_bbox: float = 1, 
        cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                           The boxes form is [cx, cy, width, height]
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                            The boxes form is [cx, cy, width, height]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]
        all_num_target_boxes = sum([len(v["boxes"]) for v in targets])

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        # Normalized boxes, represented as (center_x, center_y, width, height)
        tgt_ids = torch.cat([v["class_labels"] for v in targets]).int()
        # [num_target_boxes] to [all_num_target_boxes]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) 

        # [num_target_boxes, 4] to [all_num_target_boxes, 4]
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.        
        cost_class = torch.ones(batch_size*num_queries, all_num_target_boxes).to(out_prob.device) - out_prob[:, tgt_ids] # [batch_size * num_queries, all_num_target_boxes] -> the issof target classes

        # Compute the L1 cost between boxes
        # out_bbox [batch_size*num_queries, 4]
        # tgt_bbox [all_num_target_boxes, 4]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # cost_bbox [batch_size*num_queries, all_num_target_boxes]
        
        # Compute the giou cost betwen boxes
        cost_giou = -BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(out_bbox),
            BoxUtils.box_cxcywh_to_xyxy(tgt_bbox)
        )
        # a [batch_size * num_queries, all_num_target_boxes] pairwise matrix
        # cost_giou = -BoxUtils.generalized_box_iou(out_bbox,tgt_bbox)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox 
            + self.cost_class * cost_class 
            + self.cost_giou * cost_giou
        )# [batch_size * num_queries, all_num_target_boxes]
        C = C.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            # C.split(sizes, -1)，在最后一个维度进行拆分，每个拆分的分量按照每个targets里面每个boxes的个数
            # c [batch_size , num_queries, num_target_boxes]
            # i batch_num - 1
        ]
        # indices list [batch_size] 
        return [
            (
                torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64)
            ) 
            for i, j in indices
            # row_ind,col_ind 开销矩阵对应的行索引和最优指派的列索引
        ]

