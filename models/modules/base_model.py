import torch
from collections import OrderedDict
from Object_Centric_Local_Navigation.models.modules.utils import get_masked_region

class BaseModel(torch.nn.Module):
    IOU_THRESHOLD = 0.6
    COM_DIFF_X_THRESHOLD = 25
    COM_DIFF_Y_THRESHOLD = 20   # currently unused: robot height is fixed, so Y stays roughly constant

    def __init__(self, vision_encoder, segmentation_model, action_decoder, use_embeddings=False, auxiliary_stopping=True):
        super().__init__()

        self.use_embeddings = use_embeddings
        self.auxiliary_stopping = auxiliary_stopping
        if not use_embeddings:
            self.vision_encoder = vision_encoder
            self.segmentation_model = segmentation_model
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((16, 16))
        
        self.action_decoder = action_decoder

    def load_weights(self, weight_path):
        """
        Load model weights from a file, with support for DataParallel-trained checkpoints.

        Parameters
        ----------
        weight_path : str
            Path to the weight (.pth) file.
        """
        state_dict = torch.load(weight_path, map_location=next(self.parameters()).device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            # Trained with DataParallel, strip "module."
            new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        else:
            new_state_dict = state_dict
        self.load_state_dict(new_state_dict, strict=False)

    @staticmethod
    def get_com(mask):

        _, y_indices, x_indices = torch.nonzero(mask, as_tuple=True)

        if len(y_indices) == 0:
            return (0, 0)
        
        cy = y_indices.float().mean()
        cx = x_indices.float().mean()

        return (cx, cy)

    @classmethod
    def calculate_com_diff(cls, mask, goal_mask):

        com = cls.get_com(mask)
        com_goal = cls.get_com(goal_mask)

        com_diff_x = abs(com[0] - com_goal[0])
        com_diff_y = abs(com[1] - com_goal[1])

        return com_diff_x, com_diff_y
    
    def extract_embeddings(self, batch_images, prompts, previous_bounding_box=None):
        """
        Encode images, segment them, and produce pooled object centric embeddings.
        
        Parameters
        ----------
        batch_images : torch.Tensor
            Batch of multi-view RGB images of shape `(B, N, C, H, W)`, where
            `B` is the batch size and `N` is the number of camera views.
        prompts : List[str]
            Text prompts describing the target object, one per batch element.
            Length must equal `B`.
        previous_bounding_box : torch.Tensor or None, optional
            A single reference bounding box of shape `(4,)` in `(x1, y1, x2, y2)`
            format, used by the segmentation model to select the most temporally
            consistent detection via highest IoU. If None, the highest confidence
            detection is used. Defaults to None.

        Returns
        -------
        boxes : torch.Tensor
            Bounding boxes of shape `(B, 4)` in `(x1, y1, x2, y2)` pixel
            coordinates within the panoramic image. Rows are zeros for batch
            elements where no detection was found.
        masked_embeddings : torch.Tensor
            Pooled object centric embeddings of shape `(B, C_out, 16, 16)`,
            where `C_out` is the encoder embedding dimension. Computed by
            masking the encoder features with the segmentation mask and
            applying adaptive average pooling.
        masks : torch.Tensor
            Binary segmentation masks of shape `(B, 1, H, N*W)` in panoramic
            layout. All zeros for batch elements where no detection was found.
        panoramic : torch.Tensor
            The panoramic RGB images of shape `(B, C, H, N*W)`, formed by
            concatenating the `N` views horizontally. Returned for convenience
            so callers can visualize masks against the original imagery without
            recomputing the layout.
        """
        B, N, C, H, W = batch_images.shape
        panoramic = batch_images.permute(0, 2, 3, 1, 4).reshape(B, C, H, N*W)

        # Encode Images
        batch_images = batch_images.reshape(B*N, C, H, W)
        embeds = self.vision_encoder(batch_images)

        _, C_out, H_out, W_out = embeds.shape
        embeds = embeds.reshape(B, N, C_out, H_out, W_out)
        embeds = embeds.permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N*W_out)

        # Segment Images
        boxes, masks = self.segmentation_model(panoramic, prompts, previous_bounding_box)

        embedding_masks = torch.nn.functional.interpolate(masks, [H_out, N*W_out], mode="nearest")
        embedding_boxes = get_masked_region(embedding_masks)

        masked_embeddings = []
        for i in range(B):
            x1, y1, x2, y2 = embedding_boxes[i]
            masked_embeds = embedding_masks[i][:, y1:y2+1, x1:x2+1] * embeds[i][:, y1:y2+1, x1:x2+1]
            masked_embeds = self.avg_pool(masked_embeds)
            masked_embeddings.append(masked_embeds)
        masked_embeddings = torch.stack(masked_embeddings)

        return boxes, masked_embeddings, masks, panoramic

    def forward(self, current_images, goal_images, target_prompt=None, previous_bounding_box=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        current_images : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The current state images.
            If `self.use_embeddings` is False:
                A `torch.Tensor` of shape `(B, N, C, H, W)` containing raw RGB images from multiple camera views.
            If `self.use_embeddings` is True:
                A `Tuple` containing precomputed embeddings and bounding boxes. The format is `(current_boxes, current_embeddings)`.
        goal_images : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The goal state images.
            If `self.use_embeddings` is False:
                A `torch.Tensor` of shape `(B, N, C, H, W)` containing raw RGB images for the goal state.
            If `self.use_embeddings` is True:
                A `Tuple` containing precomputed embeddings and bounding boxes. The format is `(goal_boxes, goal_embeddings)`.
        target_prompt : str, optional
            A text prompt describing the target object. This parameter is not used
            when `use_embeddings` is True, as the segmentation model is bypassed. Defaults to None.
        previous_bounding_box : torch.Tensor or None, optional
            A single reference bounding box (shape: (4,)) in (x1, y1, x2, y2) format, used to select the most 
            temporally consistent detection via highest IoU. If None, the box with highest confidence is used.

        Returns
        -------
        actions : torch.Tensor
            Predicted actions of shape `(B, 3, 3)`, containing 3-class logits over x translation, y translation, and rotation.
        current_boxes : torch.Tensor
            The bounding boxes detected in the current images.
        debug_info : Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], Any]
            Auxiliary outputs for debugging and visualization.
            - The first element is `(current_masks, goal_masks)`, each a tensor of
            shape `(B, 1, H, N*W)` in panoramic layout, or `None` when `use_embeddings=True` (since segmentation is bypassed).
            - The second element is debugging information returned by the action
            decoder; its structure depends on the decoder implementation.
        """
        current_masks, goal_masks = None, None
        if not self.use_embeddings:
            B = current_images.shape[0]
            prompts = [target_prompt] * B

            # Process goal images
            goal_boxes, goal_embeddings, goal_masks, _ = self.extract_embeddings(goal_images, prompts)

            # Process current images
            current_boxes, current_embeddings, current_masks, _ = self.extract_embeddings(current_images, prompts, previous_bounding_box)
            
            # Action Decoder
            actions, decoder_debug_info = self.action_decoder(current_boxes, current_embeddings, goal_boxes, goal_embeddings)

            if self.auxiliary_stopping:
                for i in range(B):
                    iou = self.segmentation_model.calculate_iou(goal_boxes[i], current_boxes[i])
                    com_diff_x, com_diff_y = self.calculate_com_diff(current_masks[i], goal_masks[i])

                    print(f'iou: {iou}, com diff: {com_diff_x}, {com_diff_y}')

                    if iou > self.IOU_THRESHOLD and com_diff_x < self.COM_DIFF_X_THRESHOLD:
                        print('auxiliary succeed!')
                        actions[i] = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        
        else:
            current_boxes = current_images[0]
            current_embeddings = current_images[1]
            goal_boxes = goal_images[0]
            goal_embeddings = goal_images[1]
        
            actions, decoder_debug_info = self.action_decoder(current_boxes, current_embeddings, goal_boxes, goal_embeddings)

        return actions, current_boxes, ((current_masks, goal_masks), decoder_debug_info)

if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
    from Object_Centric_Local_Navigation.models.action_decoders.score_mlp5 import ScoreMlp5

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    goal_images_dir = ''
    current_image_dir = ''
    prompt = ''

    goal_images = []
    current_images = []
    for i in range(4):
        current_image = Image.open(os.path.join(current_image_dir, f'{i}.jpg'))
        current_image = transform(current_image)
        current_images.append(current_image)

        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    current_images = torch.stack(current_images).to(device='cuda')
    goal_images = torch.stack(goal_images).to(device='cuda')

    vision_encoder=DinoV2()
    segmentation_model=OwlV2Sam2()
    action_decoder=ScoreMlp5()
    model = BaseModel(vision_encoder, segmentation_model, action_decoder).to(device='cuda')
    # weight_path = ''
    # model.load_weights(weight_path)

    output, current_boxes, debug_info = model(current_images.unsqueeze(0), goal_images.unsqueeze(0), prompt)
    output = torch.argmax(output, dim=2)
    print(f'Output: {output}, Box:{current_boxes}')
