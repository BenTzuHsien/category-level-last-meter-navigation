import torch, os, numpy, time, yaml, select, termios, tty
from collections import OrderedDict
from torchvision.utils import save_image
from SpotStack import GraphCore
from Object_Centric_Local_Navigation.object_centric_local_navigation import ObjectCentricLocalNavigation

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr != []

class Rollout(ObjectCentricLocalNavigation):
    TRANSLATION_TOLERANCE = 0.3
    ROTATION_TOLERANCE = 0.15
    TIME_LIMIT = 100
    EVALUATION_DICT_ORDER = ["success", "duration", "translation_error", "rotation_error", "pose_error"]

    def __init__(self, architecture, weight_name, robot, eval_graph_path, auxiliary_stopping=True):
        super().__init__(architecture, weight_name, robot, auxiliary_stopping)
        
        self._graph_core = GraphCore(robot, eval_graph_path)
        self._graph_core.load_graph()
    
    def _predict(self, observation):

        with torch.no_grad():
            output_logist, current_box, debug_info = self._model(observation, self._goal_images, self._prompt)
            prediction = torch.argmax(output_logist, dim=2).flatten()

        return prediction, current_box.squeeze(0), debug_info[0][0], debug_info[0][1]

    def _save_observation(self, step_dir, observation, current_mask, goal_mask):

        os.makedirs(step_dir, exist_ok=True)
        observation = observation.squeeze(0)

        for index, image in enumerate(observation):
            image_path = os.path.join(step_dir, f'{index}.jpg')
            save_image(image, image_path)
        
        # Bounding Box
        N, C, H, W = observation.shape
        panoramic = observation.permute(1, 2, 0, 3).reshape(C, H, N * W)
        masked_panoramic = current_mask * panoramic
        image_path = os.path.join(step_dir, 'current_masked_panoramic.jpg')
        save_image(masked_panoramic, image_path)

        # Goal Mask
        panoramic = self._goal_images.squeeze(0).permute(1, 2, 0, 3).reshape(C, H, N * W)
        masked_panoramic = goal_mask * panoramic
        image_path = os.path.join(step_dir, 'goal_masked_panoramic.jpg')
        save_image(masked_panoramic, image_path)
    
    def run(self, goal_images, target_object_prompt, traj_dir):

        os.makedirs(traj_dir, exist_ok=True)
        self._prompt = target_object_prompt

        # Get Goal Image
        goal_images_transformed = []
        for image in goal_images:
            image = self.data_transforms(image)
            goal_images_transformed.append(image)
        self._goal_images = torch.stack(goal_images_transformed).unsqueeze(0).to('cuda')

        success = False
        step = 0
        actions = []

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            start_time = time.time()
            while not success:

                if key_pressed() or (time.time() - start_time) > self.TIME_LIMIT:
                    print('Manully Stop Robot !')
                    self._stop()
                    break

                observation = self._get_observation()
                prediction, self._current_box, current_mask, goal_mask = self._predict(observation)
                
                step_dir = os.path.join(traj_dir, f'{step:02}')
                self._save_observation(step_dir, observation, current_mask, goal_mask)
                actions.append(prediction)
                print(prediction)
                
                success = self._move(prediction)
                step += 1

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)

        duration = time.time() - start_time

        # Save actions
        actions = torch.stack(actions)
        actions_path = os.path.join(traj_dir, 'actions.csv')
        numpy.savetxt(actions_path, actions.cpu().numpy(), delimiter=',', fmt='%d')

        # Evaluation
        evaluation_dict = {}
        evaluation_dict['duration'] = duration
        
        evaluation_dict['pose_error'] = self._graph_core.get_relative_pose_from_waypoint('Goal_Pose')
        evaluation_dict['translation_error'] = abs(((evaluation_dict['pose_error'].x ** 2) + (evaluation_dict['pose_error'].y ** 2)) ** 0.5)
        evaluation_dict['rotation_error'] = abs(evaluation_dict['pose_error'].rotation.to_yaw())

        if evaluation_dict['translation_error'] < self.TRANSLATION_TOLERANCE and evaluation_dict['rotation_error'] < self.ROTATION_TOLERANCE:
            print(f'Succeeded !!, duration: {duration}')
            evaluation_dict['success'] = True
        else:
            print(f'Failed !, duration: {duration}')
            evaluation_dict['success'] = False

        # Reorder evaluation_dict
        evaluation_dict = OrderedDict((k, evaluation_dict[k]) for k in self.EVALUATION_DICT_ORDER if k in evaluation_dict)
        evaluation_path = os.path.join(traj_dir, 'evaluation.yaml')
        with open(evaluation_path, 'w') as file:
            yaml.dump(evaluation_dict, file)

if __name__ == '__main__':

    MODEL = ''
    WEIGHT = ''
    AUXILIARY_STOPPING = True

    # radii 1.0, 0.5
    TRAJ_START_NUM = 0
    radii = [1.0]
    angles = [80, 50, 25, 0]
    # angles = [-25, -50, -80]
    orientations = [135, 90, 45, 0, -45, -90, -135]

    import argparse, bosdyn.client.util, sys
    from PIL import Image
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    from bosdyn.api.geometry_pb2 import Vec2, SE2Pose
    from SpotStack import GraphNavigator
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('Rollout')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    
    rollout_graph_path = options.graph_path
    goal_image_dir = os.path.join(rollout_graph_path, 'Goal_Images')
    goal_images = []
    for image_name in sorted(os.listdir(goal_image_dir)):
        image_path = os.path.join(goal_image_dir, image_name)
        image = Image.open(image_path)
        goal_images.append(image)
    
    target_txt_path = os.path.join(rollout_graph_path, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    print(f'Target Object: {prompt}')

    try:
        lease_client.take()
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                rollout_model = Rollout(MODEL, WEIGHT, robot, rollout_graph_path, auxiliary_stopping=AUXILIARY_STOPPING)
                graph_navigator = GraphNavigator(robot, rollout_graph_path)
                
                traj_num = TRAJ_START_NUM
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:
                            
                            traj_dir = os.path.join(rollout_graph_path, f'{traj_num:03}')
                            os.makedirs(traj_dir, exist_ok=True)

                            # Save the Starting Point Configuration
                            starting_point_config = {}
                            starting_point_config['radius'] = rad
                            starting_point_config['angle'] = ang
                            starting_point_config['orientation'] = ori
                            stating_point_config_path = os.path.join(traj_dir, 'stating_point_config.yaml')
                            with open(stating_point_config_path, 'w') as file:
                                yaml.dump(starting_point_config, file, sort_keys=False)

                            # Starting Point
                            orientation_in_radius = (ori / 180) * numpy.pi
                            starting_pose = SE2Pose(position=Vec2(x=x, y=y), angle=orientation_in_radius)
                            graph_navigator.navigate_to(f'Goal_Pose', starting_pose)

                            print(f'Starting Rollout {traj_num}')
                            rollout_model.run(goal_images, prompt, traj_dir)
                            traj_num += 1
                            time.sleep(1.5)

            except Exception as exc:
                print("Rollout threw an error.")
                print(exc)
            finally:
                rollout_model.on_quit()

    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )