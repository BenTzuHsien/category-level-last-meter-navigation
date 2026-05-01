import numpy
from SpotStack import MotionController, GraphCore
from Object_Centric_Local_Navigation.rollout import Rollout

class TrajectoryReplayer(Rollout):

    def __init__(self, robot, eval_graph_path):
        self._motion_controller = MotionController(robot)
        self._graph_core = GraphCore(robot, eval_graph_path)
        self._graph_core.load_graph()

    def run(self, actions_path):

        actions = numpy.loadtxt(actions_path, delimiter=' ')
        for action in actions:
            self._move(action)

    def evaluate(self):
        pose_error = self._graph_core.get_relative_pose_from_waypoint('Goal_Pose')
        print(f'error x: {pose_error.x}, y: {pose_error.y}, yaw: {pose_error.rotation.to_yaw()}')

        translation_error = abs(((pose_error.x ** 2) + (pose_error.y ** 2)) ** 0.5)
        rotation_error = abs(pose_error.rotation.to_yaw())
        if translation_error < self.TRANSLATION_TOLERANCE and rotation_error < self.ROTATION_TOLERANCE:
            print('Succeeded !!')
        else:
            print('Failed !')

# Example Usage
if __name__ == '__main__':

    # radii 1.2, 0.9, 0.6, 0.45, 0.3
    radii = [1.2]
    angles = [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90]
    orientations = [150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150]

    import argparse, bosdyn.client.util, os, sys, time
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
    sdk = bosdyn.client.create_standard_sdk('TrajectoryReplayer')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    try:
        lease_client.take()
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                
                trajectory_replayer = TrajectoryReplayer(robot, options.graph_path)
                graph_navigator = GraphNavigator(robot, options.graph_path)

                traj_num = 0
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:

                            actions_path = os.path.join(options.graph_path, f'{traj_num:03}', 'actions.csv')

                            # Starting Point
                            orientation_in_radius = (ori / 180) * numpy.pi
                            starting_pose = SE2Pose(position=Vec2(x=x, y=y), angle=orientation_in_radius)
                            graph_navigator.navigate_to(f'Goal_Pose', starting_pose)

                            # Replay
                            print(f'Starting Trajectory {traj_num}')
                            trajectory_replayer.run(actions_path)
                            traj_num += 1
                            time.sleep(1.5)

                            # Evaluate
                            trajectory_replayer.evaluate()

            except Exception as exc:
                print("TrajectoryReplayer threw an error.")
                print(exc)
            finally:
                trajectory_replayer.on_quit()


    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )