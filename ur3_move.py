#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import RobotState, Constraints, JointConstraint, PositionConstraint
from geometry_msgs.msg import Point, Pose, PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from std_msgs.msg import Header

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')
        
        self.fixed_height = 0.3
        self.callback_group = ReentrantCallbackGroup()
        
        # Create action client
        self.move_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.callback_group
        )
        
        # Subscribe to position topic
        self.position_sub = self.create_subscription(
            Point,
            'object_position',
            self.position_callback,
            10
        )
        
        self.get_logger().info('Robot Mover initialized')

    def create_pose_goal(self, x, y, z):
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        
        return pose

    def position_callback(self, msg):
        try:
            self.get_logger().info(f'Received position: X={msg.x:.3f}, Y={msg.y:.3f}')
            
            # Create goal message
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = "manipulator"
            goal_msg.request.num_planning_attempts = 10
            goal_msg.request.allowed_planning_time = 5.0
            
            # Create constraints
            constraint = Constraints()
            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = "base_link"
            pos_constraint.link_name = "tool0"  # Adjust this to your robot's end effector link name
            
            # Set target position
            pos_constraint.target_point_offset.x = msg.x
            pos_constraint.target_point_offset.y = msg.y
            pos_constraint.target_point_offset.z = self.fixed_height
            
            constraint.position_constraints.append(pos_constraint)
            goal_msg.request.goal_constraints.append(constraint)
            
            # Send goal
            self.move_client.wait_for_server()
            self.get_logger().info('Sending goal request...')
            
            future = self.move_client.send_goal_async(goal_msg)
            future.add_done_callback(self.goal_response_callback)
            
        except Exception as e:
            self.get_logger().error(f'Error in position callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        
        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:  # SUCCESS
            self.get_logger().info('Movement succeeded')
        else:
            self.get_logger().warn(f'Movement failed with error code: {result.error_code.val}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotMover()
        rclpy.spin(node)
    except Exception as e:
        print(f'Error in main: {str(e)}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()