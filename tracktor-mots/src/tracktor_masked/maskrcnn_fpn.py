from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.roi_heads import paste_masks_in_image

# Modified from FasterRCNN_FPN
class MaskRCNN_FPN(MaskRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(MaskRCNN_FPN, self).__init__(backbone, num_classes)

        # Cache for feature use
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)
        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['masks'].detach()

    # Needs modification for num_class != 2. If you decide to work with higher class number, modify it
    def predict(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        # Boxes and scores
        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()

        # Masks
        proposals = [pred_boxes]
        labels = [torch.ones((len(proposals[0]),), dtype=torch.int64)]
        mask_features = self.roi_heads.mask_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        mask_features = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_features)
        mask_probs = maskrcnn_inference(mask_logits, labels)

        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_masks = paste_masks_in_image(mask_probs[0], pred_boxes, self.original_image_sizes[0])

        return pred_boxes, pred_scores, pred_masks

    def predict_masks(self, boxes, return_roi_masks=False):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        # Get masks
        labels = [torch.ones((len(proposals[0]),), dtype=torch.int64)] # Set person label for coco
        mask_features = self.roi_heads.mask_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        mask_features = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_features)
        mask_probs = maskrcnn_inference(mask_logits, labels)

        if return_roi_masks:
            return mask_probs[0]

        boxes = resize_boxes(boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_masks = paste_masks_in_image(mask_probs[0], boxes, self.original_image_sizes[0])
        return pred_masks

    def load_image(self, images, pos_embeddings=False):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])

        # Naive positional embeddings
        if pos_embeddings:
            for key, value in self.features.items():
                pos_embedding_col = torch.arange(0, value.shape[3], dtype=torch.float32, device=value.device)
                pos_embedding_col /= value.shape[3]  # Normalize between (0, 1)
                pos_embedding_col = pos_embedding_col.view(1, 1, 1, -1)
                pos_embedding_col = pos_embedding_col.repeat(1, 1, value.shape[2], 1)

                pos_embedding_row = torch.arange(0, value.shape[2], dtype=torch.float32, device=value.device)
                pos_embedding_row /= value.shape[2]  # Normalize between (0, 1)
                pos_embedding_row = pos_embedding_row.view(1, 1, -1, 1)
                pos_embedding_row = pos_embedding_row.repeat(1, 1, 1, value.shape[3])
                self.features[key] = torch.cat((value, pos_embedding_col, pos_embedding_row), dim=1)

    def get_node_embeddings(self, boxes, feature_pooler):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]
        node_embeddings = feature_pooler(self.features, proposals, self.preprocessed_images.image_sizes)
        return node_embeddings
