#!/usr/bin/env python3
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'


class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('crane_plus_act_infer')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)

        # 状態・画像サブスクライバ
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # モデルのロードとデバイス割り当て
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Loading model from: {model_path} onto {self.device}')
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.policy.to(self.device)

        self.latest_joint_state = None
        self.latest_image = None
        self.last_action = np.zeros(5, dtype=np.float32)

        # 20Hz 周期でアクション推論・パブリッシュ
        self.timer = self.create_timer(0.05, self.publish_action)

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        gripper_pos = name_to_pos.get(GRIPPER_JOINT_NAME, 0.0)
        self.latest_joint_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)

    def image_callback(self, msg):
        # sensor_msgs/Image (H, W, step) の生バイト列を NumPy 配列へ変換
        channels = 4 if 'rgba' in msg.encoding.lower() or 'bgra' in msg.encoding.lower() else 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))

        # RGBA/BGRA の場合は先頭3ch (RGB) を取得
        if img.shape[2] >= 3:
            if 'bgr' in msg.encoding.lower():
                img = img[:, :, [2, 1, 0]]  # BGR -> RGB
            else:
                img = img[:, :, :3]

        self.latest_image = img

    def publish_action(self):
        # 画像と関節状態の双方を受信するまで推論を待機
        if self.latest_joint_state is None or self.latest_image is None:
            return

        # 1. 関節状態テンソル化: (1, 5)
        state_tensor = torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. 画像テンソル化: (H, W, C) -> (1, C, H, W)、正規化 [0, 1]
        img_tensor = torch.from_numpy(self.latest_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 3. 観測辞書の構築（学習時と同一のキー名）
        obs = {
            'observation.state': state_tensor,
            'observation.environment_state': state_tensor,
            'observation.images.cam_external': img_tensor,
        }

        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()

        self.last_action = action

        # 4. アーム制御指令のパブリッシュ
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = action[:4].tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        # 5. グリッパ制御指令のパブリッシュ
        jt_g = JointTrajectory()
        jt_g.joint_names = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(action[4])]
        pt_g.time_from_start.sec = 0
        pt_g.time_from_start.nanosec = int(0.1 * 1e9)
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)

        self.get_logger().info(f"Arm: {[round(p, 3) for p in action[:4]]}, Gripper: {round(float(action[4]), 3)}")


def main():
    parser = argparse.ArgumentParser(description='ACTPolicy inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained ACTPolicy model directory')
    args = parser.parse_args()
    rclpy.init()
    node = InferenceNode(args.model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()