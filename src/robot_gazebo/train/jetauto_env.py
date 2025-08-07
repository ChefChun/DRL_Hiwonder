# jetauto_env.py
import os
import json
import math
import random
import subprocess
import shlex
import re
import numpy as np
import gymnasium 
from gymnasium import spaces


import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from transforms3d import euler
from geometry_msgs.msg import Twist, Pose

from shapely.geometry import Polygon
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional





import subprocess
import shlex
import re









# 超参
R_MAX  = 20
R_MIN  =  5
THRESH = 0.9




def ign_check_collision(topic: str, timeout: float = 0.5) -> bool:
    """
    调用 `ign topic` 抓一条 contact 消息（JSON 格式），
    如果 collision1/collision2 字段存在，则认为发生了碰撞。
    """
    cmd = [
        "ign", "topic",
        "-e",
        "-n", "1",                  # 只抓一条就退出
        "-t", topic,
        "-m","ignition.msgs.Contacts",
        "--json-output"
    ]
    try:
        # 捕获 stdout，忽略 stderr
        res = subprocess.run(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,
                             text=True,
                             timeout=timeout)
        if res.returncode != 0 or not res.stdout:
            return False
        msg = json.loads(res.stdout)
        # print(msg)
        # Ignition Contacts 消息里，collision1.name / collision2.name 存在时，说明有接触
        # contact = msg.get("contact", {})
        # name1 = contact.get("collision1", {}).get("name", "")
        # name2 = contact.get("collision2", {}).get("name", "")
        # return bool(name1 and name2)
        if (msg ):
            return True
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return False




import time
import subprocess, shlex, re

def ign_set_pose(entity_name: str,
                 x: float, y: float, z: float,
                 qx: float, qy: float, qz: str, qw: float,
                 world: str = "/world/all_training",
                 timeout_ms: int = 2000,
                 retries: int = 3,
                 retry_delay: float = 0.1) -> bool:
    """
    Calls `ign service -s {world}/set_pose` to teleport `entity_name` and returns True on success.
    If the service returns data: false, retry up to `retries` times (with delay).
    Raises RuntimeError on subprocess failure, ValueError on parse failure.
    """
    # 构造请求体
    req = f"""
name: "{entity_name}"
position {{
  x: {x}
  y: {y}
  z: {z}
}}
orientation {{
  x: {qx}
  y: {qy}
  z: {qz}
  w: {qw}
}}
""".strip()

    cmd = [
        "ign", "service", "-s", f"{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req
    ]

    for attempt in range(1, retries+1):
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode != 0:
            # 进程异常退出，直接报错
            raise RuntimeError(
                f"`{' '.join(shlex.quote(c) for c in cmd)}` failed:\n"
                f"{result.stderr}"
            )

        # 解析返回的 data: true/false
        m = re.search(r"data:\s*(true|false)", result.stdout)
        if not m:
            # raise ValueError(f"无法解析 ign 返回值：\n{result.stdout!r}")
            print(f"无法解析 ign 返回值：\n{result.stdout!r}")   
            time.sleep(retry_delay)         
            continue
        ok = (m.group(1) == "true")
        if ok:
            return True

        # 返回 false，准备重试
        if attempt < retries:
            time.sleep(retry_delay)

    # 连续 retries 次都失败
    return False




def compute_ackermann(v: float, delta: float, wheel_base: float) -> Tuple[float, float]:
    """给定油门 v 和前轮转角 delta，计算线速度 & 角速度 ω。"""
    if abs(wheel_base) < 1e-6:
        raise ValueError("wheel_base must be non-zero")
    ω = v * math.tan(delta) / wheel_base if abs(delta) > 1e-6 else 0.0
    return v, ω


class JetAutoEnv(gymnasium.Env):
    """Gym 环境：JetAuto 在 Ignition Gazebo 中的搬运 + SAC 训练接口。"""

    metadata = {"render.modes": []}

    def __init__(self, config_path: str,
                 wheel_base: float = 0.213,
                 max_v: float = 0.5,
                 max_delta_deg: float = 23.0,
                 max_steps: int = 300):
        super().__init__()

        # 1) 初始化 ROS 2 节点（假设 rclpy.init() 已在外部调用）
        self._node = Node("jetauto_env_node")

        # 2) 加载所有 configuration
        with open(config_path) as f:
            raw = json.load(f)



        # 我们用一个小结构来追踪每个 config 的试验次数和成功次数
        self._configs       = raw
        self._cfg_trials    = [0] * len(raw)
        self._cfg_successes = [0] * len(raw)
        self._pool          = list(range(len(raw)))  # 活跃的 config 索引

        # 供 reset / step 调用
        self._current_cfg_idx = None

        # 每条 episode 的累计回报
        self._episode_reward = 0.0
        self._prev_dist      = None

        # if collision then train the robot in the same scenario
        # this is an indicator
        self._new_scenario = True

        # 3) 发布 / 订阅
        self._cmd_pub   = self._node.create_publisher(Twist,     '/controller/cmd_vel', 10)
        self._odom_sub  = self._node.create_subscription(Odometry, '/odom',            self._odom_cb,    10)
        self._scan_sub  = self._node.create_subscription(LaserScan, '/scan',            self._scan_cb,    10)
        self._contact_topic = "/world/all_training/model/all_walls_and_cylinders/link/single_link/sensor/sensor_contact/contact"
   

        # 内部状态
        self._odom     = None
        self._scan     = None
        self._collided = False

        # 目标 direction
        self.exit_direction = None

        # 等待第一条激光消息到达，以便确定 observation 大小
        while self._scan is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        n_rays = len(self._scan.ranges)

        # 4) 定义 Gym 的 action_space & observation_space
        self.wheel_base   = wheel_base
        max_delta = math.radians(max_delta_deg)
        self.action_space = spaces.Box(
            low=np.array([-max_v, -max_delta], dtype=np.float32),
            high=np.array([+max_v, +max_delta], dtype=np.float32),
            dtype=np.float32
        )
        # 观测：n_rays 激光 + 碰撞标志 + 目标位置（x,y）相对坐标
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_rays + 1 + 2,), dtype=np.float32
        )

        self._max_steps = max_steps
        self._step_cnt = 0

        self._target_poly = None

        self.car_length = 0.316  # or 从 config 里读
        self.car_width  = 0.259

    def _odom_cb(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    # def _touch_cb(self, msg: Bool):
        
    #     if msg.data:
    #         self._collided = True

        # else:
        #     self._collided = False  


    def _robot_poly(self) -> Polygon:
        """
        返回当前 odom.pose 下，机器人底盘在 XY 平面上的多边形轮廓。
        """
        if self._odom is None:
            # 还没收到任何里程计
            return Polygon()

        # 1) 读取位置
        x = self._odom.position.x
        y = self._odom.position.y

        # 2) 读取四元数，计算 yaw
        qx = self._odom.orientation.x
        qy = self._odom.orientation.y
        qz = self._odom.orientation.z
        qw = self._odom.orientation.w

        # 标准的 yaw 提取公式：
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # 3) 调用已有工具，算出四个角点
        corners = _robot_corners(
            x, y, yaw,
            self.car_length,
            self.car_width
        )

        # 4) 构造并返回 Polygon
        return Polygon(corners)

    # def reset(self):
    #     # 随机取一个配置
    #     cfg = random.choice(self._configs)
    #     sx, sy, syaw = cfg['start_pose']
    #     tx, ty      = cfg['target_position']
    #     self._target = (tx, ty)

    #     # 瞬移小车到起始位姿
    #     q = euler.euler2quat(0, 0, syaw)
    #     ign_set_pose("jetauto", sx, sy, 0.0, q[1], q[2], q[3], q[0])

    #     # 清空状态
    #     self._collided = False
    #     self._step_cnt = 0

    #     coords = cfg['target_poly']  # [[x1,y1], [x2,y2], …]
    #     self._target_poly = Polygon(coords)

    #     # 让订阅回调刷新一轮数据
    #     rclpy.spin_once(self._node, timeout_sec=0.1)
    #     return self._get_obs()

    def reset(self,
                *,               # 这样可以强制把 seed 当关键字参数
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        # 1) 如果外部指定了种子，就用它来初始化 Python 随机
        if seed is not None:
            random.seed(seed)
        # 1) 从活跃池里随机选一个配置
        if self._new_scenario:
            cfg_idx = random.choice(self._pool)
            self._current_cfg_idx = cfg_idx
            cfg = self._configs[cfg_idx]

            # 2) 统计它的 trials
            self._cfg_trials[cfg_idx] += 1
        else:
            cfg_idx = self._current_cfg_idx
            cfg = self._configs[cfg_idx]


        print("cfg_idx:", cfg_idx, "trials:", self._cfg_trials[cfg_idx])

        # 3) 清零 episode reward 和 prev_dist
        self._episode_reward = 0.0
        self._prev_dist      = None
        self._collided       = False
        self._step_cnt       = 0

        # 4) teleport 到起点
        sx, sy, syaw = cfg['start_pose']
        q = euler.euler2quat(0, 0, syaw)
        ign_set_pose("jetauto", sx, sy, 0.0,
                     q[1], q[2], q[3], q[0])
        
        
        
                # wait for a valid odom message
        start = time.time()
        while self._odom is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() - start > 2.0:
                raise RuntimeError("Timeout waiting for odom in reset()")
            
        time.sleep(1.5)   
        rclpy.spin_once(self._node, timeout_sec=0.1) 
        print(self._odom.position.x," ",self._odom.position.y)

        # 5) 记录目标
        tx, ty = cfg['target_position']
        self._target = (tx, ty)
        self._target_poly = Polygon(cfg['target_poly'])

        # calculating proper angle exit indicator:
        centroids = cfg['walls_centroids'] or cfg['cylinders_centroids']
        cen_quantity = len(centroids)
        point_1=centroids[0]
        point_2=centroids[math.ceil(cen_quantity/4)]
        point_3=centroids[cen_quantity - math.ceil(cen_quantity/4)]
        point_4=centroids[-1]
        cen_1 = ((point_1[0]+point_4[0])/2,(point_1[1]+point_4[1])/2)
        cen_2 = ((point_2[0]+point_3[0])/2,(point_2[1]+point_3[1])/2)
        direction_vector = [(cen_1[0] - cen_2[0]), (cen_1[1] - cen_2[1])]
        self.exit_direction = math.atan2(direction_vector[1],direction_vector[0])

        # print(tx," ",ty)

        # 刷新一次传感器数据
        rclpy.spin_once(self._node, timeout_sec=0.1)
        obs = self._get_obs()
        return obs, {}

    

    def _get_obs(self) -> np.ndarray:
        # raw ranges may contain inf/nan
        raw = np.array(self._scan.ranges, dtype=np.float32)

        # replace inf with max_range, nan with max_range (or some large finite value)
        # You can read range_max from the LaserScan message if you want.
        max_r = getattr(self._scan, 'range_max', 12.0)
        min_r = getattr(self._scan, 'range_min', 0.1)
        ranges = np.nan_to_num(raw,
                               nan=max_r,
                               posinf=max_r,
                               neginf=min_r)
        # print('start to check col')
        self._collided = ign_check_collision(self._contact_topic)
        # print('checked is',self._collided)
        col = np.array([1.0 if self._collided else 0.0], dtype=np.float32)
        dx = self._target[0] - self._odom.position.x
        dy = self._target[1] - self._odom.position.y
        tgt = np.array([dx, dy], dtype=np.float32)
        obs = np.concatenate([ranges, col, tgt])

        # sanity check for debugging:
        if not np.isfinite(obs).all():
            raise ValueError(f"Non-finite observation: {obs}")

        return obs


    def step(self, action):

        obs, reward, done, info = None, 0.0, False, {}    

        v, delta = action
        # 转换成 (linear, angular)
        lin, ang = compute_ackermann(v, delta, self.wheel_base)


        # 发布速度
        twist = Twist()
        twist.linear.x  = float(lin)
        twist.angular.z = float(ang)
        self._cmd_pub.publish(twist)

        # 等待一次仿真
        rclpy.spin_once(self._node, timeout_sec=0.1)
        self._step_cnt += 1


        obs = self._get_obs()
        #done = False
        #reward = 0.0

        print(self._odom.position.x," ",self._odom.position.y)
        
        dx   = self._target[0] - self._odom.position.x
        dy   = self._target[1] - self._odom.position.y
        dist = math.hypot(dx, dy)


        self._new_scenario = True  # default: 触发新场景
        # 在 step() 或者你计算 reward 的地方
        # --------------------------------------------------------
        # 1) 撞墙或圆柱立即终止
        if self._collided:
            done = True
            self._new_scenario = False  # no 触发新场景
            # print('contact happen')
            reward = -50.0

        

            # 2) 计算到目标的欧氏距离



            # 目标到达
        elif dist < 0.3:
            done   = True
            reward += +100.0

            # 超时
        elif self._step_cnt > self._max_steps:
            done   = True
            reward += -20.0

            # 常规 step，累加三部分 reward
        else:
            done = False

                # —— 1. IoU Reward —— 
                # robot_poly: 当前机器人底盘多边形



            robot_poly = self._robot_poly()
            target_poly = self._target_poly
            inter = robot_poly.intersection(target_poly).area
            union = robot_poly.union(target_poly).area
            iou = inter / union if union > 0 else 0.0
            w_iou = 30.0   # 你可以调这个权重
            r_iou = w_iou * iou

                # —— 2. 差分距离 Reward —— 
                # 上一步到目标的距离保存在 self._prev_dist
            if self._prev_dist is None:
                    # 第一步差分距离用 0
                r_dist = 0.0
            else:
                w_dist = 200.0   # 距离差分的权重
                r_dist = w_dist * (self._prev_dist - dist)
                # 更新 prev_dist
            self._prev_dist = dist

                # —— 3. 时间惩罚 —— 
                # 随 step_cnt 增加，由 tanh 有界地增加惩罚
            alpha = 1.0    # 最大惩罚幅度
            beta  = 0.02   # 增长速率
            r_time = - alpha * math.tanh(beta * self._step_cnt)
                # direction reward
            x1, y1 = robot_poly.exterior.coords[0]
            x2, y2 = robot_poly.exterior.coords[1]

            dx = x2 - x1
            dy = y2 - y1
            yaw = math.atan2(dy, dx)  # 得到 yaw（弧度）
            if self.exit_direction is not None:
                w_dir = 40
                vec = math.cos(yaw)*math.cos(self.exit_direction) + math.sin(yaw)*math.sin(self.exit_direction) 
                r_dir = w_dir * vec

                # 总 reward
            reward += (r_iou + r_dist + r_time + r_dir)

            print("\nstep:", self._step_cnt,
              "\ncollided:", self._collided,
              "\ntarget_dist:", dist,
              "\nr_iou:", r_iou,
              "\nr_dist:", r_dist,
              "\nr_time",r_time,
              "\n----------",

              "\nreward",reward,
              "\nepisode_reward",self._episode_reward)



        self._episode_reward += reward

        # print("coll:", self._collided)

        

        # 2) 如果本 step 结束了
        if done:
            idx = self._current_cfg_idx
            # 如果是成功到达
            dx, dy = self._target
            dist = math.hypot(self._odom.position.x - dx,
                              self._odom.position.y - dy)
            if dist < 0.2 and not self._collided:
                self._cfg_successes[idx] += 1

            # 3) 检查是否要移除
            trials = self._cfg_trials[idx]
            succ   = self._cfg_successes[idx]
            rate   = succ / trials
            if (trials > R_MAX) or (trials >= R_MIN and rate >= THRESH):
                self._pool.remove(idx)

            # 4) 在 info 里带上统计数据
            info['config_idx']      = idx
            info['episode_reward']  = self._episode_reward
            info['config_trials']   = trials
            info['config_successes']= succ
            info['config_rate']     = rate

        self._collided = False  # 重置碰撞标志
        # return self._get_obs(), reward, done, info
        # Gymnasium expects: obs, reward, terminated, truncated, info
        terminated = done
        truncated  = False
        return obs, reward, terminated, truncated, info
# --------------------------------------------------------





    def render(self, mode='human'):
        pass

    def close(self):
        try:
            self._node.destroy_node()
        except:
            pass

def _robot_corners(x: float, y: float, yaw: float, L: float, W: float) -> List[Tuple[float, float]]:
    hl, hw = L/2.0, W/2.0
    corners_local = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
    corners = []
    c = math.cos(yaw)
    s = math.sin(yaw)
    for lx, ly in corners_local:
        gx = x + lx * c - ly * s
        gy = y + lx * s + ly * c
        corners.append((gx, gy))
    return corners