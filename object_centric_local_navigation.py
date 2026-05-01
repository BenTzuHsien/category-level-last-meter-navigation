import importlib, re, os, torch, time
from torchvision import transforms
from SpotStack import ImageFetcher, MotionController

class ObjectCentricLocalNavigation:
    data_transforms = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    STOP_THRESHOLD = 2
    ACTION_LOOKUP = {0: -0.2, 1: 0.0, 2: 0.2}

    def __init__(self, architecture, weight_name, robot, auxiliary_stopping=True):

        # Setup Model
        module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', architecture).lower()
        module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
        module = importlib.import_module(module_path)
        self._model = getattr(module, architecture)(auxiliary_stopping=auxiliary_stopping)
        
        # Load Weight
        weight_path = os.path.join(os.path.dirname(__file__), 'weights', weight_name)
        self._model.load_weights(weight_path)
        
        self._model.cuda()
        self._model.eval()

        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)
        self._motion_controller = MotionController(robot)
        self._stop_prediction_count = 0
        self._current_box = None

    def on_quit(self):
        self._motion_controller.on_quit()

    def _stop(self):
        self._motion_controller.send_velocity_command(0, 0, 0, duration=2)
        time.sleep(1)

    def _get_observation(self):
        observation = self._image_fetcher.get_images(self.data_transforms)
        observation = torch.stack(observation).unsqueeze(0).to('cuda')

        return observation

    def _predict(self, observation):

        with torch.no_grad():
            output_logist, current_box, _ = self._model(observation, self._goal_images, self._prompt)
            prediction = torch.argmax(output_logist, dim=2).flatten()

        return prediction, current_box.squeeze(0)
    
    def _move(self, prediction):

        if (prediction[0] == 1) and (prediction[1] == 1) and (prediction[2] == 1):
            self._stop()
            self._stop_prediction_count += 1
            if self._stop_prediction_count >= self.STOP_THRESHOLD:
                return True
            
        else:
            self._stop_prediction_count = 0

            d_x, d_y, d_yaw = [self.ACTION_LOOKUP[p.item()] for p in prediction]
            self._motion_controller.send_displacement_command(d_x, d_y, d_yaw)
            time.sleep(1)

            return False
    
    def run(self, goal_images, target_object_prompt):
        """
        Executes the object-centric local navigation loop.

        This function initiates the navigation process by first preparing the goal images.
        It then enters a continuous loop where it:
        1. Acquires a live observation from the robot's cameras.
        2. Uses the model to predict the next best action based on the observation, goal images, and target object prompt.
        3. Commands the robot to move according to the prediction.
        4. Repeats the process until a success condition (reaching the target) is met.

        Parameters
        ----------
        goal_images : list
            A list of PIL Image objects representing the goal state. These images
            are used by the model to identify the target object and location.
        target_object_prompt : str
            A text prompt (e.g., 'chair', 'table') that specifies the object to
            navigate to. This prompt is used by the segmentation model.
        """
        self._prompt = target_object_prompt
        
        # Get Goal Image
        goal_images_transformed = []
        for image in goal_images:
            image = self.data_transforms(image)
            goal_images_transformed.append(image)
        self._goal_images = torch.stack(goal_images_transformed).unsqueeze(0).to('cuda')

        success = False
        while not success:
            observation = self._get_observation()
            prediction, self._current_box = self._predict(observation)
            success = self._move(prediction)

# Example Usage
if __name__ == '__main__':

    MODEL = ''
    WEIGHT = ''

    import argparse, bosdyn.client.util, sys
    from PIL import Image
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('ObjectCentricLocalNavigation')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    graph_path = options.graph_path
    goal_image_dir = os.path.join(graph_path, 'Goal_Images')
    goal_images = []
    for image_name in sorted(os.listdir(goal_image_dir)):
        image_path = os.path.join(goal_image_dir, image_name)
        image = Image.open(image_path)
        goal_images.append(image)

    target_txt_path = os.path.join(graph_path, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    print(f'Target Object: {prompt}')

    try:
        lease_client.take()
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                rollout_model = ObjectCentricLocalNavigation(MODEL, WEIGHT, robot)
                rollout_model.run(goal_images, prompt)

            except Exception as exc:
                print("ObjectCentricLocalNavigation threw an error.")
                print(exc)
            finally:
                rollout_model.on_quit()


    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )