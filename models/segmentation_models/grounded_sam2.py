import torch, os
import torch.nn as nn

from grounding_dino.groundingdino.util.inference import load_model
from sam2.build_sam import build_sam2
from torchvision.ops import box_convert

from Object_Centric_Local_Navigation.models.modules.gdino_batch_image_predictor import GDinoBatchImagePredictor
from Object_Centric_Local_Navigation.models.modules.sam2_batch_image_predictor import SAM2BatchImagePredictor

GROUNDING_DINO_CONFIG = os.path.expanduser("/opt/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.expanduser("/opt/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.expanduser("/opt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")

class GroundedSAM2(nn.Module):
    BOX_THRESHOLD = 0.1
    TEXT_THRESHOLD = 0.4

    def __init__(self):
        super().__init__()

        # Build Grounding DINO
        self.gdino_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT
        )
        self.gdino_predictor = GDinoBatchImagePredictor(self.gdino_model)

        # Build SAM-2
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
        self.sam2_predictor = SAM2BatchImagePredictor(self.sam2_model)

    @staticmethod
    def calculate_iou(box1, box2):
        # Intersection coordinates
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        # Compute intersection area
        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Compute areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Union area
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / union_area
        return iou
    
    @staticmethod
    def mask_to_bounding_box(mask):
        """
        Compute the bounding box (xyxy) from a binary mask.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask of shape (1, H, W), with values 0 or 1.

        Returns
        -------
        bbox : torch.Tensor
            Bounding box in (x1, y1, x2, y2) format. Returns a zero box (0, 0, 0, 0) if mask is empty.
        """
        nonzero = mask.squeeze(0).nonzero(as_tuple=False)  # shape (N, 2) where each row is [y, x]

        if nonzero.numel() == 0:
            return torch.zeros([4]).to(mask)

        y_min = nonzero[:, 0].min()
        x_min = nonzero[:, 1].min()
        y_max = nonzero[:, 0].max()
        x_max = nonzero[:, 1].max()

        return torch.tensor([x_min, y_min, x_max, y_max]).to(mask)
    
    def _merge_overlapping_boxes_keep_largest(self, boxes, scale):

        keep = [True] * len(boxes)

        for i in range(len(boxes)):
            if not keep[i]:
                continue
            
            for j in range(i + 1, len(boxes)):
                if not keep[j]:
                    continue

                box1 = box_convert(boxes[i] * scale, in_fmt="cxcywh", out_fmt="xyxy")
                box2 = box_convert(boxes[j] * scale, in_fmt="cxcywh", out_fmt="xyxy")

                if self.calculate_iou(box1, box2) > 0:
                    area_i = (box1[2]-box1[0]) * (box1[3]-box1[1])
                    area_j = (box2[2]-box2[0]) * (box2[3]-box2[1])

                    if area_i >= area_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        filtered_boxes = [b for k, b in zip(keep, boxes) if k]
        return filtered_boxes
    
    def _get_best_box(self, boxes, scale, confidences):

        boxes = self._merge_overlapping_boxes_keep_largest(boxes, scale)
        best_box = boxes[confidences.argmax()]

        return best_box

    @torch.no_grad() 
    def forward(self, batch_images, prompts, previous_bounding_box=None):
        """
        Run the GroundingDINO + SAM-2 pipeline on a batch of images and text prompts, 
        with optional temporal matching based on previous bounding box.

        This method uses GroundingDINO to detect candidate bounding boxes for each image 
        conditioned on the given text prompt. If `previous_bounding_box` is provided, the 
        detection with the highest IoU against it is selected; otherwise, the box with the 
        highest confidence score is chosen. SAM-2 then predicts a binary segmentation mask 
        for the selected box. The final bounding box is refined from the predicted mask 
        region.

        Parameters
        ----------
        batch_images : torch.Tensor
            Input batch of RGB images of shape (B, 3, H, W).
        prompts : List[str]
            A list of text prompts corresponding to the input images.
        previous_bounding_box : torch.Tensor or None, optional
            A single reference bounding box (shape: (4,)) in (x1, y1, x2, y2) format, used to select the most 
            temporally consistent detection via highest IoU. If None, the box with highest confidence is used.

        Returns
        -------
        bounding_boxes : torch.Tensor
            Tensor of shape (B, 4) with (x1, y1, x2, y2) pixel coordinates for each image representing 
            the predicted bounding box. If no detection was found for an image, the row is zeros.
        masks : torch.Tensor
            Tensor of shape (B, 1, H, W) with binary masks (values 0 or 1).  
            If no detection was found for an image, the mask is all zeros.
        """
        with torch.autocast(device_type=batch_images.device.type, enabled=False):
            batch_images = batch_images.float()
            batch_size, _, H, W = batch_images.shape

            # Grounding‑DINO
            boxes_list, confidences_list, labels_list = self.gdino_predictor.predict(batch_images, prompts, self.BOX_THRESHOLD, self.TEXT_THRESHOLD)

            # Extract SAM2 Embeddings
            batch_image_embed, batch_high_res_feats_split = self.sam2_predictor.extract_features(batch_images)
            
            empty_box = torch.zeros([4]).to(batch_images)
            empty_mask = torch.zeros([1, H, W]).to(batch_images)
            bounding_boxes = []
            masks = []
            scale = torch.tensor([W, H, W, H], device=self.device, dtype=self.dtype)
            for i in range(batch_size):
                if boxes_list[i].numel() == 0:
                    bounding_boxes.append(empty_box)
                    masks.append(empty_mask)
                else:
                    # Get the best box
                    best_box = self._get_best_box(boxes_list[i], scale, confidences_list[i])
                    box_xyxy = box_convert(best_box * scale, in_fmt="cxcywh", out_fmt="xyxy")

                    # Predict the mask from the best box
                    image_mask, _, _ = self.sam2_predictor.predict_once(
                        batch_image_embed[i].unsqueeze(0), 
                        batch_high_res_feats_split[i],
                        (H, W),
                        boxes=box_xyxy.unsqueeze(0), 
                        multimask_output=False)
                    
                    image_mask = image_mask.float().squeeze(0)
                    masks.append(image_mask)
                    best_box = self.mask_to_bounding_box(image_mask)
                    bounding_boxes.append(best_box)

            bounding_boxes = torch.stack(bounding_boxes)
            masks = torch.stack(masks)
        return bounding_boxes, masks
    
    @property
    def device(self) -> torch.device:
        return self.gdino_predictor.device
    @property
    def dtype(self) -> torch.dtype:
        return self.gdino_predictor.dtype
    
if __name__ == '__main__':

    from PIL import Image, ImageDraw
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    images_dir = ''
    images = []
    for i in range(4):
        image_path = os.path.join(images_dir, f'{i}.jpg')
        image = Image.open(image_path)
        image_tensor = transform(image)
        images.append(image_tensor)

    images = torch.stack(images)
    N, C, H, W = images.shape
    images = images.permute(1, 2, 0, 3).reshape(C, H, N * W).unsqueeze(0)
    print(images.shape)
    
    prompts = ['']
    
    segmentation_model = GroundedSAM2()
    segmentation_model.cuda()
    
    images = images.to(segmentation_model.device)
    bounding_boxes, masks = segmentation_model(images, prompts)
    print(bounding_boxes)

    for i, (bounding_box, mask) in enumerate(zip(bounding_boxes, masks)):
        masked_image = images[0] * mask
        masked_image = transforms.ToPILImage()(masked_image)
        box = bounding_box.detach().cpu().tolist()
        
        draw = ImageDraw.Draw(masked_image)
        draw.rectangle(box, outline="green", width=2)
        masked_image.save(f'masked_image_{i}.jpg')
