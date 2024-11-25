import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
import traceback

import tf2_ros
import tf2_geometry_msgs

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')

        self.fixed_height = 0.3  # Fixed z-height for the robot's target
        self.move_group_client = ActionClient(
            self, MoveGroup, 'move_action')

        # Initialize variables
        self.desired_position_received = False
        self.desired_position = None
        self.home_pose = None  # Will store the robot's home pose

        # Subscription to the object position topic
        self.position_subscriber = self.create_subscription(
            Point,
            'object_position',
            self.position_callback,
            QoSProfile(depth=10)
        )

        # Initialize tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Robot Mover initialized")

    def position_callback(self, msg):
        """Callback to handle received target positions."""
        try:
            if not self.desired_position_received:
                self.desired_position_received = True
                self.desired_position = [msg.x, msg.y, self.fixed_height]
                self.get_logger().info(
                    f"Received target position: x={msg.x:.3f}, y={msg.y:.3f}, z={self.fixed_height:.3f}")

                # Proceed to move the robot
                self.move_robot()
            else:
                # Ignore subsequent messages
                self.get_logger().info(
                    "Desired position already received. Ignoring additional messages.")
        except Exception as e:
            self.get_logger().error(f"Error in position_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def move_robot(self):
        """Plan and execute trajectory to the target position and back to home."""
        try:
            # Get and save the current pose (home pose)
            self.home_pose = self.get_current_pose()
            if self.home_pose is None:
                self.get_logger().error(
                    "Could not retrieve the current pose of the robot.")
                return

            self.get_logger().info(f"Home pose: {self.home_pose.pose}")

            # Move to the desired position
            target_pose = PoseStamped()
            target_pose.header.frame_id = "base"  # Ensure this matches your base frame
            target_pose.pose.position.x = self.desired_position[0]
            target_pose.pose.position.y = self.desired_position[1]
            target_pose.pose.position.z = self.desired_position[2]

            # Set orientation to point downwards (adjust as needed)
            target_pose.pose.orientation = Quaternion(
                x=1.0,
                y=0.0,
                z=0.0,
                w=0.0
            )

            self.get_logger().info("Moving to target position...")
            self.send_moveit_goal(target_pose, use_orientation_constraint=True)

            # After reaching the target, return to home pose
            self.get_logger().info("Returning to home position...")
            self.send_moveit_goal(self.home_pose, use_orientation_constraint=False)

        except Exception as e:
            self.get_logger().error(f"Error in move_robot: {e}")
            self.get_logger().error(traceback.format_exc())

    def get_current_pose(self):
        """Get the current pose of the robot's end effector using tf2."""
        try:
            target_frame = 'base'  # Adjust if different
            source_frame = 'tool0'  # Adjust if different

            timeout = 5.0  # seconds
            start_time = self.get_clock().now()
            while not self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)):
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout:
                    self.get_logger().error(
                        f"Transform from {source_frame} to {target_frame} not available after {timeout} seconds")
                    return None
                rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks to receive TF data

            # Look up the transform
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            # Convert TransformStamped to PoseStamped
            current_pose = PoseStamped()
            current_pose.header.frame_id = target_frame
            current_pose.header.stamp = self.get_clock().now().to_msg()
            current_pose.pose.position.x = trans.transform.translation.x
            current_pose.pose.position.y = trans.transform.translation.y
            current_pose.pose.position.z = trans.transform.translation.z
            current_pose.pose.orientation = trans.transform.rotation

            return current_pose

        except Exception as e:
            self.get_logger().error(f"Error in get_current_pose: {e}")
            self.get_logger().error(traceback.format_exc())
            return None

    def send_moveit_goal(self, target_pose, use_orientation_constraint=True):
        """Create and send a goal to the MoveGroup action server."""
        # Ensure the action server is available
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                "MoveIt2 action server ('/move_action') not available.")
            return

        # Create a new MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"  # Adjust if different
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 10.0

        # Set maximum velocity and acceleration scaling factors
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # Prepare Constraints object
        constraints = Constraints()

        # Add Position Constraint
        position_constraint = PositionConstraint()
        position_constraint.header = target_pose.header
        position_constraint.link_name = 'tool0'  # Adjust if different
        position_constraint.constraint_region.primitives.append(self.create_bounding_box())
        position_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)

        # Add Orientation Constraint if needed
        if use_orientation_constraint:
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = target_pose.header.frame_id
            orientation_constraint.link_name = 'tool0'  # Adjust if different
            orientation_constraint.orientation = target_pose.pose.orientation
            # Relax tolerances to allow flexibility in planning
            orientation_constraint.absolute_x_axis_tolerance = 0.3  # Adjust as needed
            orientation_constraint.absolute_y_axis_tolerance = 0.3
            orientation_constraint.absolute_z_axis_tolerance = 3.14  # Allow rotation around Z-axis
            orientation_constraint.weight = 1.0
            constraints.orientation_constraints.append(orientation_constraint)

        # Add constraints to the goal
        goal_msg.request.goal_constraints.append(constraints)

        # Log the target pose for debugging
        self.get_logger().info(f"Planning target pose: {target_pose.pose}")

        # Send goal message to MoveIt2 Action server
        self.get_logger().info("Sending goal to MoveIt2...")
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)

        # Wait for the goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "Goal was rejected by the MoveIt2 action server.")
            return

        self.get_logger().info("Goal accepted. Executing...")

        # Wait for the result
        result_future = goal_handle.get_result_async()
        self.get_logger().info("Waiting for result...")
        rclpy.spin_until_future_complete(self, result_future)
        
        # Check if result is available
        if result_future.result() is None:
            self.get_logger().error("Failed to receive result from action server.")
            return

        result = result_future.result().result

        self.get_logger().info(f"Result error code: {result.error_code.val}")
        if result.error_code.val == 1:  # SUCCESS
            self.get_logger().info("Motion executed successfully!")
        else:
            self.get_logger().warn(
                f"Motion failed with error code: {result.error_code.val}")

    def create_bounding_box(self):
        """Create a small bounding box for position constraint."""
        from shape_msgs.msg import SolidPrimitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.005, 0.005, 0.005]  # Small box around the target
        return box

def main():
    """Entry point for the `robot_mover` node."""
    rclpy.init()

    try:
        node = RobotMover()
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()