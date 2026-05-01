import os, numpy, yaml
from SpotStack import GraphCore, MotionController, ImageFetcher
from Object_Centric_Local_Navigation.object_centric_local_navigation import ObjectCentricLocalNavigation

class TrajectoryCollector(ObjectCentricLocalNavigation):
    DISCRETIZED_TOLERANCE = 0.1

    def __init__(self, robot, eval_graph_path):

        self._motion_controller = MotionController(robot)
        self._graph_core = GraphCore(robot, eval_graph_path)
        self._graph_core.load_graph()
        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)

    def _record_images(self, step_dir):

        if not os.path.exists(step_dir):
            os.mkdir(step_dir)
        
        current_images = self._image_fetcher.get_images()
        
        for index, image in enumerate(current_images):
            image_path = os.path.join(step_dir, f'{index}.jpg')
            image.save(image_path)

    def _predict(self):

        action = [1, 1, 1]
        pose_error = self._graph_core.get_relative_pose_from_waypoint('Goal_Pose')
        print(f'error x: {pose_error.x}, y: {pose_error.y}, yaw: {pose_error.rotation.to_yaw()}')

        error_yaw = pose_error.rotation.to_yaw()
        error_x = numpy.cos(error_yaw) * pose_error.x + numpy.sin(error_yaw) * pose_error.y
        error_y = -numpy.sin(error_yaw) * pose_error.x + numpy.cos(error_yaw) * pose_error.y

        if error_x > self.DISCRETIZED_TOLERANCE:
            action[0] = 0
        elif error_x < -self.DISCRETIZED_TOLERANCE:
            action[0] = 2
        
        if error_y > self.DISCRETIZED_TOLERANCE:
            action[1] = 0
        elif error_y < -self.DISCRETIZED_TOLERANCE:
            action[1] = 2
        
        if error_yaw > self.DISCRETIZED_TOLERANCE:
            action[2] = 0
        elif error_yaw < -self.DISCRETIZED_TOLERANCE:
            action[2] = 2

        return numpy.array(action)
    
    def move_and_record_to_goal(self, traj_dir):

        if not os.path.exists(traj_dir):
            os.mkdir(traj_dir)
        step_num = 0
        actions = numpy.empty([0, 3])

        while True:
            step_dir = os.path.join(traj_dir, f'{step_num:02}')
            self._record_images(step_dir)
            
            action = self._predict()
            actions = numpy.vstack([actions, action])
            
            self._move(action)
            if numpy.array_equal(action, [1, 1, 1]):
                break
            else:
                step_num += 1

        actions_path = os.path.join(traj_dir, 'actions.csv')
        numpy.savetxt(actions_path, actions, fmt='%d')

if __name__ == '__main__':

    # Map 1
    # radii 1.2, 0.9, 0.6, 0.45, 0.3
    radii = [1.2]
    angles = [90, 75, 60, 45, 30, 15, 0]
    # angles = [-15, -30, -45, -60, -75, -90]
    orientations = [150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150]

    # Map 2
    # radii 1.0, 0.8, 0.5
    # radii = [0.5]
    # angles = [80, 50, 25, 0, -25, -50, -80]
    # orientations = [135, 90, 45, 0, -45, -90, -135]

    import argparse, bosdyn.client.util, sys, time
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
    sdk = bosdyn.client.create_standard_sdk('TrajectoryCollector')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    graph_path = options.graph_path

    try:
        lease_client.take()
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                trajectory_collector = TrajectoryCollector(robot, graph_path)
                graph_navigator = GraphNavigator(robot, options.graph_path)
                
                traj_num = 0
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:
                            
                            traj_dir = os.path.join(graph_path, f'{traj_num:03}')
                            if not os.path.exists(traj_dir):
                                os.mkdir(traj_dir)
                            
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

                            # Record
                            print(f'Starting Trajectory {traj_num}')
                            trajectory_collector.move_and_record_to_goal(traj_dir)
                            print(f'Finished Trajectory {traj_num}')
                            traj_num += 1
                            time.sleep(1.5)

            except Exception as exc:
                print("TrajectoryCollector threw an error.")
                print(exc)
            finally:
                trajectory_collector.on_quit()

    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )