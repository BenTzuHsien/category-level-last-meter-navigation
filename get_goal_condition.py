import os
from SpotStack import GraphRecorder, ImageFetcher

class GetGoalCondition:

    def __init__(self, robot, graph_path, target_object_name):
        self._graph_recorder = GraphRecorder(robot, graph_path)
        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)

        if not target_object_name.endswith("."):
            target_object_name += "."
        self._prompt = target_object_name
        self._graph_path = graph_path
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

    def record_goal(self):

        goal_image_dir = os.path.join(self._graph_path, 'Goal_Images')
        if not os.path.exists(goal_image_dir):
            os.mkdir(goal_image_dir)

        self._graph_recorder.start_recording()
        self._graph_recorder.record_waypoint(f'Goal_Pose')

        target_txt_path = os.path.join(self._graph_path, 'target_object.txt')
        with open(target_txt_path, "w") as f:
            f.write(self._prompt)
        
        current_images = self._image_fetcher.get_images()
        for index, image in enumerate(current_images):
            image_path = os.path.join(goal_image_dir, f'{index}.jpg')
            image.save(image_path)

        self._graph_recorder.stop_recording()
        self._graph_recorder.download_full_graph()
        print('Goal Pose Recorded !')

if __name__ == '__main__':

    import argparse, bosdyn.client.util, sys
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    parser.add_argument('--target-object',
                        help='Name of the target object.')
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('GetGoalCondition')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    get_goal_condition = GetGoalCondition(robot, options.graph_path, options.target_object)

    try:
        get_goal_condition.record_goal()

    except Exception as exc:
        print("GetGoalCondition threw an error.")
        print(exc)