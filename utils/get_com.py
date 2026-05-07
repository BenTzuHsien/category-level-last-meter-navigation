import os, torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

def masked_center_of_mass(masked_rgb):
    # Convert to grayscale mask (True where non-zero pixel)
    # mask = torch.any(masked_rgb != 0, dim=-1).astype(numpy.uint8)

    # Get indices of masked pixels
    y_indices, x_indices = torch.nonzero(masked_rgb, as_tuple=True)

    if len(y_indices) == 0:
        return (0, 0)  # no masked region

    # Compute mean of coordinates = center of mass
    cy = y_indices.float().mean()
    cx = x_indices.float().mean()

    return (cx, cy)

def get_com(dataset_dir, segmentation_model):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    # Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompt = [prompt]

    # ----- Process Goal Images -----
    goal_images_dir = os.path.join(dataset_dir, 'Goal_Images')
    goal_images = []
    for i in range(4):
        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    goal_images = torch.stack(goal_images).to(device='cuda')
    N, C, H, W = goal_images.shape
    goal_panoramic = goal_images.permute(1, 2, 0, 3).reshape(C, H, N * W)

    ## Segment goal imgaes
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        goal_box, goal_masks = segmentation_model(goal_panoramic.unsqueeze(0), prompt)
    
    masked_goal_image = goal_masks * goal_panoramic
    masked_goal_image = masked_goal_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    com_goal = masked_center_of_mass(goal_masks.squeeze(0, 1))

    # ----- Process Current Images -----
    iou_list = []
    com_list = []
    box_list = []
    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

    for trajectory in tqdm(trajectories, desc='Calculating CoM'):
        trajectory_dir = os.path.join(dataset_dir, trajectory)

        # Extract Steps
        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())

        current_images = []
        for i in range(4):
            img_path = os.path.join(trajectory_dir, steps[-1], f'{i}.jpg')
            image = Image.open(img_path)
            image_tensor = transform(image)
            current_images.append(image_tensor)
        current_images = torch.stack(current_images).to(device='cuda')
        N, C, H, W = current_images.shape
        current_panoramic = current_images.permute(1, 2, 0, 3).reshape(C, H, N * W)

        ## Segment current images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            current_box, current_masks = segmentation_model(current_panoramic.unsqueeze(0), prompt)

        if not current_box.any():
            continue

        box_list.append(current_box.squeeze(0))

        # IOU
        iou = segmentation_model.calculate_iou(goal_box.squeeze(0), current_box.squeeze(0))
        iou_list.append(iou)
        
        masked_current_image = current_masks * current_panoramic
        masked_current_image = masked_current_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        com_current = masked_center_of_mass(current_masks.squeeze(0, 1))

        com_list.append(com_current)

    return iou_list, com_list, com_goal, goal_panoramic.cpu(), box_list

if __name__ == '__main__':

    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2

    dataset_dirs = ['/data/SPOT_Real_World_Dataset/gray_basket']
    segmentation_model = OwlV2Sam2().to(device='cuda')

    transform = ToPILImage()

    for i, dataset_dir in enumerate(dataset_dirs):
        iou_list, com_list, com_goal, goal_panoramic, box_list = get_com(dataset_dir, segmentation_model)

        iou_tensor = torch.stack(iou_list)
        print(f'iou avg: {iou_tensor.mean()}, min: {iou_tensor.min()}')

        com_tensor = torch.stack([torch.stack(c) for c in com_list])
        com_goal_tensor = torch.stack(list(com_goal))

        x_max = com_tensor[:, 0].max() - com_goal_tensor[0]
        x_min = com_tensor[:, 0].min() - com_goal_tensor[0]
        y_max = com_tensor[:, 1].max() - com_goal_tensor[1]
        y_min = com_tensor[:, 1].min() - com_goal_tensor[1]

        print(f'x_max: {x_max}, x_min: {x_min}, y_max: {y_max}, y_min: {y_min}')
