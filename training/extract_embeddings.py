import os, torch, shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel

def extract_embeddings(dataset_dir, vision_encoder, segmentation_model):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    base_model = BaseModel(vision_encoder, segmentation_model, None).to('cuda')
    base_model.eval()

    embeddings_dir_name = f'{os.path.basename(dataset_dir)}_{vision_encoder.__class__.__name__.lower()}_embeddings'
    embeddings_dir = os.path.join(os.path.dirname(dataset_dir), embeddings_dir_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompt = [prompt]

    # Copy target txt
    target_txt_save_path = os.path.join(embeddings_dir, 'target_object.txt')
    shutil.copy(target_txt_path, target_txt_save_path)

    # ----- Process Goal Images -----
    goal_images_dir = os.path.join(dataset_dir, 'Goal_Images')
    goal_images = []
    for i in range(4):
        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    goal_images = torch.stack(goal_images).unsqueeze(0).to(device='cuda')
    
    ## Extract embeddings
    with torch.no_grad():
        goal_box, goal_embedding, goal_mask, goal_panoramic = base_model.extract_embeddings(goal_images, prompt)
    goal_box = goal_box.squeeze(0)
    goal_embedding = goal_embedding.squeeze(0)
    goal_mask = goal_mask.squeeze(0)
    goal_panoramic = goal_panoramic.squeeze(0)

    ## Save embeddings
    goal_box_path = os.path.join(embeddings_dir, 'goal_box.pt')
    torch.save(goal_box, goal_box_path)
    goal_embedding_path = os.path.join(embeddings_dir, 'goal_embedding.pt')
    torch.save(goal_embedding, goal_embedding_path)

    ## Save segmented images
    masked_goal_image = goal_mask * goal_panoramic
    masked_goal_image_path = os.path.join(embeddings_dir, 'masked_goal_image.jpg')
    save_image(masked_goal_image, masked_goal_image_path)
    
    # Release memory
    del goal_images, goal_box, goal_embedding, goal_mask, goal_panoramic, masked_goal_image
    torch.cuda.empty_cache()

    # ----- Process Current Images -----
    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

    for trajectory in tqdm(trajectories, desc='Extracting'):
        trajectory_dir = os.path.join(dataset_dir, trajectory)
        
        trajectory_embeddings_dir = os.path.join(embeddings_dir, trajectory)
        os.makedirs(trajectory_embeddings_dir, exist_ok=True)

        # Copy labels
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        label_save_path = os.path.join(trajectory_embeddings_dir, 'actions.csv')
        shutil.copy(label_path, label_save_path)

        # Extract Steps
        traj_images = []
        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())
        for step in steps:
            step_dir = os.path.join(trajectory_dir, step)
            
            current_images = []
            for i in range(4):
                img_path = os.path.join(step_dir, f'{i}.jpg')
                image = Image.open(img_path)
                image_tensor = transform(image)
                current_images.append(image_tensor)
            current_images = torch.stack(current_images)
            traj_images.append(current_images)
        traj_images = torch.stack(traj_images).to(device='cuda')

        ## Extract embeddings
        prompts = prompt * len(steps)
        with torch.no_grad():
            traj_boxes, traj_embeddings, traj_masks, traj_panoramics = base_model.extract_embeddings(traj_images, prompts)

        ## Save embeddings
        traj_box_path = os.path.join(trajectory_embeddings_dir, 'boxes.pt')
        torch.save(traj_boxes, traj_box_path)
        traj_embedding_path = os.path.join(trajectory_embeddings_dir, 'embeddings.pt')
        torch.save(traj_embeddings, traj_embedding_path)

        ## Save segmented images
        masked_traj_images = traj_masks * traj_panoramics
        masked_traj_image_path = os.path.join(trajectory_embeddings_dir, 'masked.jpg')
        save_image(masked_traj_images, masked_traj_image_path)

        # Release memory
        del traj_images, traj_boxes, traj_embeddings, traj_masks, traj_panoramics, masked_traj_images
        torch.cuda.empty_cache()

if __name__ == '__main__':

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2

    dataset_dir = ''
    vision_encoder = DinoV2().to(device='cuda')
    segmentation_model = OwlV2Sam2().to(device='cuda')

    extract_embeddings(dataset_dir, vision_encoder, segmentation_model)