import torch
from torchvision.models import resnet18, ResNet18_Weights
from Object_Centric_Local_Navigation.models.modules.utils import resize_and_normalize_tensor

class Resnet18(torch.nn.Module):
    TRANSFORMS = ResNet18_Weights.IMAGENET1K_V1.transforms()
    PATCH_NUM = 7
    EMBED_DIM = 512

    def __init__(self):
        super().__init__()
        base_resnet = resnet18(weights='DEFAULT')
        self.resnet18 = torch.nn.Sequential(*list(base_resnet.children())[:-2])
    
    def forward(self, batch_images):
        with torch.autocast(device_type=batch_images.device.type, enabled=False):
            batch_images = batch_images.float()

            batch_images = self.TRANSFORMS(batch_images)
            batch_embeddings= self.resnet18(batch_images)

        return batch_embeddings

if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    image_path = ''
    image = Image.open(image_path)
    image_tensor = transform(image)

    vision_encoder = Resnet18()
    embedding = vision_encoder(image_tensor.unsqueeze(0))
    print(embedding.shape)