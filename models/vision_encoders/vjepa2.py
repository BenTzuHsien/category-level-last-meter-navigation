import torch
from Object_Centric_Local_Navigation.models.modules.utils import resize_and_normalize_tensor

class Vjepa2(torch.nn.Module):
    # TRANSFORM_SIZE = (476, 476)
    # TRANSFORM_MEAN = [0.485, 0.456, 0.406]
    # TRANSFORM_STD = [0.229, 0.224, 0.225]
    PATCH_NUM = 16
    EMBED_DIM = 1024

    def __init__(self):
        super().__init__()
        self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
        self.vjepa2, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
        for param in self.vjepa2.parameters():
            param.requires_grad = False
        self.vjepa2.eval()

    @torch.no_grad() 
    def forward(self, batch_images):
        """
        Extract Vjepa2 ebeddings

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
            batch_size = batch_images.shape[0]
            # batch_images = resize_and_normalize_tensor(batch_images, self.TRANSFORM_SIZE, self.TRANSFORM_MEAN, self.TRANSFORM_STD)

            # Preprocess images 
            batch_images_processed = []
            for image in batch_images:
                image = image.unsqueeze(0).expand(2, -1, -1, -1)
                image = self.processor(image)[0]
                batch_images_processed.append(image)
            
            batch_images_processed = torch.stack(batch_images_processed)
            vjepa_output = self.vjepa2(batch_images_processed)
            batch_embeddings = vjepa_output.reshape(batch_size, self.PATCH_NUM, self.PATCH_NUM, self.EMBED_DIM).permute(0, 3, 1, 2)

        return batch_embeddings
        
if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    images_dir = ''
    images = []
    for i in range(4):
        image = Image.open(os.path.join(images_dir, f'{i}.jpg'))
        image = transform(image)
        images.append(image)

    images = torch.stack(images).to(device='cuda')

    vision_encoder = Vjepa2().to(device='cuda')
    embedding = vision_encoder(images)
    print(embedding.shape)