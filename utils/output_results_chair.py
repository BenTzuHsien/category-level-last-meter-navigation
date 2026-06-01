import os, csv, yaml, numpy, cv2, torch
from PIL import Image
from torchvision import transforms

MULTIPLIER_0 = 1
MULTIPLIER_1 = 1.5
MULTIPLIER_2 = 2
MULTIPLIER_3 = 2.5

def masked_center_of_mass(masked_rgb):
    # Convert to grayscale mask (True where non-zero pixel)
    mask = numpy.any(masked_rgb != 0, axis=-1).astype(numpy.uint8)

    # Get indices of masked pixels
    y_indices, x_indices = numpy.nonzero(mask)

    if len(y_indices) == 0:
        return (None, None)  # no masked region

    # Compute mean of coordinates = center of mass
    cy = y_indices.mean()
    cx = x_indices.mean()

    return (cx, cy)

def evaluate_file(traj_dir, com_goal):

    COM_BOX_DIS = [(-74.78, -29.18), (60.34, 4.09)]     # Use the get_com.py; [(x_min, y_min), (x_max, y_max)]

    # Check Manully Stop
    actions_file_path = os.path.join(traj_dir, 'actions.csv')
    with open(actions_file_path, 'r') as file:
        actions = list(csv.reader(file))
    
    if not (actions[-2] == ['1', '1', '1'] and actions[-1] == ['1', '1', '1']):
        return 'F', 'F', 'F', 'F'
    
    evaluation_file_path = os.path.join(traj_dir, 'evaluation.yaml')
    with open(evaluation_file_path, 'r') as f:
        error = yaml.load(f, Loader=yaml.Loader)

    translation_error = error['translation_error']
    rotation_error_rad = error['rotation_error']

    succ_strict = 'F'
    succ_relaxed = 'F'
    com_strict = 'F'
    com_relaxed = 'F'

    if (translation_error < 0.2 * MULTIPLIER_1) and (rotation_error_rad < 0.1 * MULTIPLIER_1):
        succ_strict = 'T'
    if (translation_error < 0.2 * MULTIPLIER_2) and (rotation_error_rad < 0.1 * MULTIPLIER_2):
        succ_relaxed = 'T'

    # Center of Mass
    steps = sorted(x for x in os.listdir(traj_dir) if x.isdigit())
    last_masked_image_path = os.path.join(traj_dir, steps[-1], 'current_masked_panoramic.jpg')
    last_masked_image = cv2.imread(last_masked_image_path)
    com = masked_center_of_mass(last_masked_image)
    
    com_succ = (com[0] > (com_goal[0] + COM_BOX_DIS[0][0] * MULTIPLIER_1)) and (com[0] < (com_goal[0] + COM_BOX_DIS[1][0] * MULTIPLIER_1))
    com_succ_relaxed = (com[0] > (com_goal[0] + COM_BOX_DIS[0][0] * MULTIPLIER_2)) and (com[0] < (com_goal[0] + COM_BOX_DIS[1][0] * MULTIPLIER_2))

    if (translation_error < 0.2 * MULTIPLIER_1) and com_succ:
        com_strict = 'T'
    if (translation_error < 0.2 * MULTIPLIER_2) and com_succ_relaxed:
        com_relaxed = 'T'

    return succ_strict, succ_relaxed, com_strict, com_relaxed

def main():
    dataset_dir = ''
    output_csv = ''
    results = []

    # Get Goal Center of Mass
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
    
    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    ## Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompt = [prompt]
    
    segmentation_model = OwlV2Sam2().to(device='cuda')

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
        _, goal_masks = segmentation_model(goal_panoramic.unsqueeze(0), prompt)

    masked_goal_image = goal_masks * goal_panoramic
    masked_goal_image = masked_goal_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    com_goal = masked_center_of_mass(masked_goal_image)

    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())
    print(len(trajectories))
    for traj in trajectories:
        traj_dir = os.path.join(dataset_dir, traj)
        succ_stict, succ_relaxed, com_strict, com_relaxed = evaluate_file(traj_dir, com_goal)
        results.append({
            'strict_success': succ_stict,
            'relaxed_success': succ_relaxed,
            'com_strict': com_strict,
            'com_relaxed': com_relaxed
        })

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['strict_success', 'relaxed_success', 'com_strict', 'com_relaxed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved results to {output_csv}")

if __name__ == '__main__':
    main()
