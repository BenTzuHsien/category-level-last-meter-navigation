import torch
from Object_Centric_Local_Navigation.models.modules.utils import resize_and_normalize_tensor

class DinoV2(torch.nn.Module):
    TRANSFORM_SIZE = (476, 476)
    TRANSFORM_MEAN = [0.485, 0.456, 0.406]
    TRANSFORM_STD = [0.229, 0.224, 0.225]
    PATCH_NUM = 34
    EMBED_DIM = 1024

    def __init__(self):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self.dinov2.eval()

    @torch.no_grad() 
    def forward(self, batch_images):
        """
        Extract DINOv2 embeddings

        Parameters
        ----------
        batch_images : torch.Tensor
            Batch of RGB images of shape (B, 3, H, W)
        
        Returns
        -------
        batch_embeddings : torch.Tensor
            Patch embeddings of shape (B, C, P, P), where:
                - C = `EMBED_DIM` (embedding dimension per patch token)
                - P = `PATCH_NUM` (number of patches per spatial axis after resizing)
            Each embedding corresponds to a specific spatial patch in the input image.
        """
        with torch.autocast(device_type=batch_images.device.type, enabled=False):
            batch_images = batch_images.float()

            batch_images = resize_and_normalize_tensor(batch_images, self.TRANSFORM_SIZE, self.TRANSFORM_MEAN, self.TRANSFORM_STD)
            dino_output = self.dinov2.forward_features(batch_images)
            batch_embeddings = torch.reshape(dino_output['x_norm_patchtokens'], [-1, self.PATCH_NUM, self.PATCH_NUM, self.EMBED_DIM]).permute(0, 3, 1, 2)

        return batch_embeddings
        
if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    image_path = ''
    image = Image.open(image_path)
    image_tensor = transform(image).to('cuda')

    vision_encoder = DinoV2().to('cuda')
    embedding = vision_encoder(image_tensor.unsqueeze(0))
    print(embedding.shape)