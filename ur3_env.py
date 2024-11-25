import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np
import gymnasium as gym
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class ReachingUR3(gym.Env):
    def __init__(self):
        # Define spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(25,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # Set to (6,)

        # Initialize ROS2 node
        rclpy.init()
        self.node = Node('ur3_env')

        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # QoS profile
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Publishers and subscribers
        self.joint_pub = self.node.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory',
            self.qos
        )
        
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_states_callback,
            self.qos
        )

        # State variables
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.current_joints = np.zeros(6)
        self.current_joint_vel = np.zeros(6)
        self.last_action = np.zeros(6)  # Set to size 6

        # Environment parameters
        self.dt = 1 / 120.0  # As per training
        self.action_scale = 2.5  # As per training
        self.dof_vel_scale = 1.0
        self.max_episode_length = 100
        self.progress_buf = 0
        
        self.target_pos = np.array([-0.055,  0.178,  0.351])
        self.target_orientation = np.zeros(4)

        # Joint limits
        self.joint_limits = {
            'lower': np.array([-2*np.pi] * 6),
            'upper': np.array([2*np.pi] * 6)
        }

        # Start ROS2 spinner in separate thread
        import threading
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        # Wait for connections
        time.sleep(1.0)
        print("UR3 Environment Initialized")

    def _get_end_effector_pos(self):
        try:
            # Get the transform
            transform = self.tf_buffer.lookup_transform(
                'base',
                'tool0',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
        except Exception as e:
            self.node.get_logger().warn(f"Could not get end-effector position: {e}")
            # Return approximate position based on current joint values
            return np.array([0.5, 0.0, 0.5])

    def _get_observation(self):
        # Create observation vector with correct dimensions
        obs = np.zeros(25, dtype=np.float32)
        
        # Normalize joint positions
        dof_pos_scaled = 2.0 * (self.current_joints - self.joint_limits['lower']) / \
                         (self.joint_limits['upper'] - self.joint_limits['lower']) - 1.0
        
        # Scale joint velocities
        dof_vel_scaled = self.current_joint_vel * self.dof_vel_scale
        
        # Fill observation vector
        obs[0] = self.progress_buf / self.max_episode_length
        obs[1:7] = dof_pos_scaled
        obs[7:13] = dof_vel_scaled
        obs[13:16] = self.target_pos
        obs[16:20] = self.target_orientation
        obs[20:] = self.last_action[:5]  # Keep as per your training setup
        
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.node.get_logger().info("Resetting UR3...")
        self.progress_buf = 0
        
        # Get current end effector position
        current_pos = self._get_end_effector_pos()
        
        # Set target position (adjust as needed)
        self.target_pos = current_pos.copy()
        self.target_pos[2] += 0.1  # For example, move 0.1m upwards
        
        # Set target orientation (fixed upright)
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Reset last action
        self.last_action = np.zeros(6)  # Reset to zeros
        
        # Get observation
        observation = self._get_observation()
        
        # Log target information
        self.node.get_logger().info(f"Current position: {current_pos}")
        self.node.get_logger().info(f"Target position: {self.target_pos}")
        self.node.get_logger().info(f"Distance: {np.linalg.norm(self.target_pos - current_pos):.3f}m")
        
        return observation, {}

    def step(self, action):
        try:
            self.progress_buf += 1
            
            # Store action for observation
            self.last_action = action.copy()
            
            # Scale action and apply
            scaled_action = action * self.action_scale * self.dt
            new_joint_positions = self.current_joints + scaled_action  # Apply to all 6 joints
            
            # Clip to joint limits
            new_joint_positions = np.clip(
                new_joint_positions,
                self.joint_limits['lower'],
                self.joint_limits['upper']
            )
            
            # Send command to robot
            self._send_joint_trajectory(new_joint_positions)
            
            # Small sleep to allow for robot movement
            time.sleep(1 / 30.0)
            
            # Get new observation
            observation = self._get_observation()
            
            # Calculate reward
            reward = self._compute_reward()
            
            # Check if episode is done
            terminated = self._check_termination()
            truncated = False
            
            return observation, reward, terminated, truncated, {}
                
        except Exception as e:
            self.node.get_logger().error(f"Error during step: {e}")
            return self._get_observation(), 0.0, True, False, {"error": str(e)}

    def _send_joint_trajectory(self, joint_positions):
        try:
            msg = JointTrajectory()
            msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = joint_positions.tolist()
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int((1 / 30.0) * 1e9)
            
            msg.points = [point]
            self.joint_pub.publish(msg)
            
        except Exception as e:
            self.node.get_logger().error(f"Error sending trajectory: {e}")

    def _compute_reward(self):
        try:
            ee_pos = self._get_end_effector_pos()
            distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
            reward = -distance_to_target  # Negative distance as reward
            return reward
        except Exception as e:
            self.node.get_logger().warn(f"Error computing reward: {e}")
            return 0.0

    def _check_termination(self):
        try:
            # Max episode length reached
            if self.progress_buf >= self.max_episode_length:
                return True
                
            # Target reached
            ee_pos = self._get_end_effector_pos()
            distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
            if distance_to_target < 0.05:  # 5cm threshold
                return True
                
            return False
        except Exception as e:
            self.node.get_logger().warn(f"Error checking termination: {e}")
            return True

    def _joint_states_callback(self, msg):
        try:
            for i, name in enumerate(msg.name):
                if name in self.joint_names:
                    idx = self.joint_names.index(name)
                    self.current_joints[idx] = msg.position[i]
                    self.current_joint_vel[idx] = msg.velocity[i]
        except Exception as e:
            self.node.get_logger().error(f"Error in joint states callback: {e}")

    def _spin(self):
        try:
            rclpy.spin(self.node)
        except Exception as e:
            self.node.get_logger().error(f"Error in ROS2 spin: {e}")

    def close(self):
        try:
            # Move to safe position
            self.node.get_logger().info("Moving to safe position...")
            home_position = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
            self._send_joint_trajectory(home_position)
            time.sleep(2.0)
            
            # Cleanup
            self.node.get_logger().info("Cleaning up...")
            if hasattr(self, 'spin_thread'):
                rclpy.shutdown()
                self.spin_thread.join(timeout=1.0)
            
            if self.node:
                self.node.destroy_node()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def render(self):
        pass