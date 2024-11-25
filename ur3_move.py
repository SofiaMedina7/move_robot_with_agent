#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import traceback


class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')

        self.fixed_height = 0.3  # Fixed z-height for the robot's target
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')  # Action server: /move_action

        # Subscription to the object position topic
        self.position_subscriber = self.create_subscription(
            Point,
            'object_position',  # Topic publishing target positions
            self.position_callback,
            10  # Queue size
        )

        self.get_logger().info("Robot Mover initialized")

    def position_callback(self, msg):
        """Callback to handle received target positions."""
        try:
            self.get_logger().info(f"Received target position: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")
            # Send the received position to MoveIt2
            self.send_moveit_goal(msg.x, msg.y, self.fixed_height)
        except Exception as e:
            self.get_logger().error(f"Error in position_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def send_moveit_goal(self, x, y, z):
        """Create and send a goal to the MoveGroup action server."""
        # Ensure the action server is available
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveIt2 action server ('/move_action') not available.")
            return

        # Create a new MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"  # Replace "ur_manipulator" if your group name is different
        goal_msg.request.num_planning_attempts = 20  # Increase attempts for reliability
        goal_msg.request.allowed_planning_time = 15.0  # Allow more time for planning complex paths

        # Define target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"  # Frame of target pose
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z

        # Set fixed orientation (upright)
        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0

        # Log the target pose for debugging
        self.get_logger().info(f"Planning target pose: {target_pose}")

        # Create constraints
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        # Define PositionConstraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "base_link"
        position_constraint.link_name = "tool0"  # Use the robot's end-effector link
        region_primitive = SolidPrimitive()
        region_primitive.type = SolidPrimitive.BOX
        region_primitive.dimensions = [0.05, 0.05, 0.05]  # Tolerance region/box around the target pose
        position_constraint.constraint_region.primitives.append(region_primitive)
        position_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        # Define OrientationConstraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.link_name = "tool0"
        orientation_constraint.orientation.x = 1.0
        orientation_constraint.orientation.y = 0.0
        orientation_constraint.orientation.z = 0.0
        orientation_constraint.orientation.w = 0.0  # Upright orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1  # Tolerances for orientation
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        # Combine constraints
        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)

        # Add constraints to the goal message
        goal_msg.request.goal_constraints.append(constraints)

        # Send goal message to MoveIt2 Action server
        self.get_logger().info("Sending goal to MoveIt2...")
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the response after sending a goal to MoveIt2."""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Goal was rejected by the MoveIt2 action server.")
                return

            self.get_logger().info("Goal accepted. Waiting for execution result...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.get_result_callback)
        except Exception as e:
            self.get_logger().error(f"Error in goal_response_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def get_result_callback(self, future):
        """Process the result of the motion execution."""
        try:
            result = future.result().result
            error_code = result.error_code
            self.get_logger().info(f"Received result. Error code: {error_code}")
            if error_code == 1:  # SUCCESS
                self.get_logger().info("Motion executed successfully!")
            else:
                self.get_logger().warn(f"Motion failed with error code: {error_code}")
        except Exception as e:
            self.get_logger().error(f"Error in get_result_callback: {e}")
            self.get_logger().error(traceback.format_exc())


def main():
    """Entry point for the `robot_mover` node."""
    rclpy.init()

    try:
        node = RobotMover()
        rclpy.spin(node)  # Run until interrupted
    except Exception as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    