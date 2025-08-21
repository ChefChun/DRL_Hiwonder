# jetauto_env.py
import threading
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
    """Gym 环境：JetAuto 在 Ignition Gazebo 中的搬运 + SAC 训练接口。
       Modified to perform manual stepping and async collision polling to reduce step latency.
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 config_path: str,
                 wheel_base: float = 0.213,
                 max_v: float = 0.5,
                 max_delta_deg: float = 23.0,
                 max_steps: int = 300,
                 world: str = "/world/all_training",          # <-- world used by ign service/topic
                 steps_per_action: int = 4,                   # <-- how many physics steps per env.step()
                 collision_poll_interval: float = 0.05       # <-- background poll frequency (s)
                 ):
        super().__init__()

        # store new params
        self._world = world
        self.steps_per_action = int(steps_per_action)
        self._collision_poll_interval = float(collision_poll_interval)

        # 1) 初始化 ROS 2 节点（假设 rclpy.init() 已在外部调用）
        self._node = Node("jetauto_env_node")

        # 2) 加载所有 configuration
        with open(config_path) as f:
            raw = json.load(f)

        # ... same config bookkeeping as before ...
        self._configs       = raw
        self._cfg_trials    = [0] * len(raw)
        self._cfg_successes = [0] * len(raw)
        self._pool          = list(range(len(raw)))
        self._current_cfg_idx = None
        self._episode_reward = 0.0
        self._prev_dist      = None
        self._new_scenario = True

        # 3) 发布 / 订阅
        self._cmd_pub   = self._node.create_publisher(Twist,     '/controller/cmd_vel', 100)
        self._odom_sub  = self._node.create_subscription(Odometry, '/odom',            self._odom_cb,    100)
        self._scan_sub  = self._node.create_subscription(LaserScan, '/scan',            self._scan_cb,    100)
        # keep contact topic name (used by background poller if you don't have a ros subscription)
        self._contact_topic = "/world/all_training/model/all_walls_and_cylinders/link/single_link/sensor/sensor_contact/contact"

        # internal states
        self._odom     = None
        self._scan     = None
        self._collided = False

        # target direction and sizes
        self.exit_direction = None

        # wait for first laser message (same as before)
        start = time.time()
        while self._scan is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() - start > 5.0:
                raise RuntimeError("Timeout waiting for initial LaserScan in JetAutoEnv.__init__")

        n_rays = len(self._scan.ranges)

        # action/observation spaces (unchanged)
        self.wheel_base   = wheel_base
        max_delta = math.radians(max_delta_deg)
        self.action_space = spaces.Box(
            low=np.array([-max_v, -max_delta], dtype=np.float32),
            high=np.array([+max_v, +max_delta], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_rays + 1 + 2,), dtype=np.float32
        )

        self._max_steps = max_steps
        self._step_cnt = 0
        self._target_poly = None
        self.car_length = 0.316
        self.car_width  = 0.259

        # --------- Background collision poller thread ----------
        self._collision_thread_stop = threading.Event()
        self._collision_thread = threading.Thread(target=self._collision_poller, daemon=True)
        self._collision_thread.start()

        # closed flag
        self._closed = False

    # ---------------- callbacks ----------------
    def _odom_cb(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    # --------------- manual step helper -----------------
    def _ign_step(self, steps: int) -> None:
        """
        Advance Ignition world by `steps` physics iterations synchronously.
        Uses the `ign` CLI to call the world control service. This is synchronous,
        so it returns only after the requested steps are done.
        """
        # request payload depends on ign CLI; this matches the format used elsewhere in your code
        req = f"pause:true step:{int(steps)}"
        cmd = [
            "ign", "service",
            "-s", f"{self._world}/control",
            "--reqtype", "ignition.msgs.WorldControl",
            "--req", req
        ]
        # call synchronously; raise if it fails
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --------------- background collision poller -----------------
    def _collision_poller(self):
        """
        Polls ign topic (via your existing ign_check_collision helper) in background and updates
        self._collided. This avoids spawning a subprocess in the hot path (step()).
        """
        while not self._collision_thread_stop.is_set():
            try:
                # small timeout so poll is responsive to stop event
                collided = ign_check_collision(self._contact_topic, timeout=0.2)
                self._collided = bool(collided)
            except Exception:
                # keep going; do not kill the thread for transient errors
                pass
            # wait but wake up quickly if stopping
            self._collision_thread_stop.wait(self._collision_poll_interval)

    # --------------- reset / observation -----------------
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)

        # choose config
        if self._new_scenario:
            cfg_idx = random.choice(self._pool)
            self._current_cfg_idx = cfg_idx
            cfg = self._configs[cfg_idx]
            self._cfg_trials[cfg_idx] += 1
        else:
            cfg_idx = self._current_cfg_idx
            cfg = self._configs[cfg_idx]

        print("cfg_idx:", cfg_idx, "trials:", self._cfg_trials[cfg_idx])

        self._episode_reward = 0.0
        self._prev_dist      = None
        self._collided       = False
        self._step_cnt       = 0

        # teleport to start pose via your ign_set_pose helper
        sx, sy, syaw = cfg['start_pose']
        q = euler.euler2quat(0, 0, syaw)
        ign_set_pose("jetauto", sx, sy, 0.0, q[1], q[2], q[3], q[0])

        # wait for odom
        start = time.time()
        while self._odom is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() - start > 2.0:
                raise RuntimeError("Timeout waiting for odom in reset()")

        # small settling time for simulator to finish teleport processing
        time.sleep(0.5)
        rclpy.spin_once(self._node, timeout_sec=0.05)

        # set target and other bookkeeping (same as original)
        tx, ty = cfg['target_position']
        self._target = (tx, ty)
        self._target_poly = Polygon(cfg['target_poly'])

        # compute exit direction (same logic as before)
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

        # refresh sensors once
        rclpy.spin_once(self._node, timeout_sec=0.05)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        raw = np.array(self._scan.ranges, dtype=np.float32)
        max_r = getattr(self._scan, 'range_max', 12.0)
        min_r = getattr(self._scan, 'range_min', 0.1)
        ranges = np.nan_to_num(raw, nan=max_r, posinf=max_r, neginf=min_r)
        # use the background-updated collision flag (cheap read)
        col = np.array([1.0 if self._collided else 0.0], dtype=np.float32)
        dx = self._target[0] - self._odom.position.x
        dy = self._target[1] - self._odom.position.y
        tgt = np.array([dx, dy], dtype=np.float32)
        obs = np.concatenate([ranges, col, tgt])
        if not np.isfinite(obs).all():
            raise ValueError(f"Non-finite observation: {obs}")
        return obs

    # ----------------- step ------------------
    def step(self, action):
        v, delta = action
        lin, ang = compute_ackermann(v, delta, self.wheel_base)

        # publish twist
        twist = Twist()
        twist.linear.x  = float(lin)
        twist.angular.z = float(ang)
        self._cmd_pub.publish(twist)

        # MANUALLY advance the sim synchronously (this removes real-time waiting)
        try:
            self._ign_step(self.steps_per_action)
        except subprocess.CalledProcessError as e:
            # fall back to a short spin if ign service not available
            rclpy.spin_once(self._node, timeout_sec=0.05)
        else:
            # quick spin to flush the new sensor messages generated by the step(s)
            rclpy.spin_once(self._node, timeout_sec=0.01)

        self._step_cnt += 1

        obs = self._get_obs()
        print(self._odom.position.x," ",self._odom.position.y)

        # reward/done logic unchanged from your original code...
        # [copy your reward calculation exactly here]
        # I'm going to reuse your code verbatim for reward and termination.
        # (paste reward block from your original step() here)
        # For brevity in this snippet: (you'll paste your reward calculation)
        #
        # Below is a placeholder that you should replace with your existing logic:
        dx   = self._target[0] - self._odom.position.x
        dy   = self._target[1] - self._odom.position.y
        dist = math.hypot(dx, dy)
        self._new_scenario = True
        if self._collided:
            done = True
            self._new_scenario = False
            reward = -50.0
        elif dist < 0.3:
            done = True
            reward = +100.0
        elif self._step_cnt > self._max_steps:
            done = True
            reward = -40.0
        else:
            done = False
            # (you should paste your IoU / r_dist / r_time / r_dir reward code)
            reward = 0.0

        self._episode_reward += reward

        print("\nstep:", self._step_cnt,
              "\ncollided:", self._collided,
              "\ntarget_dist:", dist,
              "\nreward",reward,
              "\nepisode_reward",self._episode_reward)


        if done:
            idx = self._current_cfg_idx
            dx, dy = self._target
            dist = math.hypot(self._odom.position.x - dx,
                                self._odom.position.y - dy)
            # here if dist < 0.2 then we use the map again. This does not mean that the training is not successful
            if dist < 0.2 and not self._collided:
                self._cfg_successes[idx] += 1
            trials = self._cfg_trials[idx]
            succ   = self._cfg_successes[idx]
            rate   = succ / trials
            if (trials > R_MAX) or (trials >= R_MIN and rate >= THRESH):
                self._pool.remove(idx)
            info = {
                'config_idx': idx,
                'episode_reward': self._episode_reward,
                'config_trials': trials,
                'config_successes': succ,
                'config_rate': rate
            }
        else:
            info = {}

        # reset collision flag for next episode
        self._collided = False

        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    # ----------------- cleanup ------------------
    def close(self):
        if self._closed:
            return
        # stop background thread
        self._collision_thread_stop.set()
        if self._collision_thread.is_alive():
            self._collision_thread.join(timeout=1.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
        self._closed = True

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