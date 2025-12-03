import os
import socket
import threading
import time

import numpy as np
from functions import angle, point2world, precompute_projections
from is_msgs.common_pb2 import Orientation, Pose, Position
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, Message, Status, StatusCode, Subscription

from .Connection import Connection
from .KalmanFilter2D import KalmanFilter2D
from .StreamChannel import StreamChannel

CONFIDENCE_KP = float(os.environ.get("confidence_kp", 0.99))
CONFIDENCE_BB = float(os.environ.get("confidence_bb", 0.5))


class Threading:
    def __init__(self, connection: Connection, parameters: dict):
        self.connection = connection
        self.log = connection.log
        self.parameters = parameters
        self.proj_cache = precompute_projections(self.parameters)
        self._last_keypoints = {}
        self.keypoints_event = {
            cam_id: threading.Event() for cam_id in parameters.keys()
        }
        self._end_time = 0.0
        self.pose_event = threading.Event()

        self.new_keypoints_event = threading.Event()
        self.keypoints_lock = threading.Lock()

    def get_keypoints_by_camera(self, camera_id: int) -> None:
        self.keypoints_event[camera_id].set()

        kf_center = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0, gate=9.21)
        kf_front = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0, gate=9.21)

        channel = StreamChannel(self.connection.broker_uri)
        Subscription(channel).subscribe(f"Tiffany.{camera_id}.Keypoints")

        last_keypoints_time = time.time()

        while time.time() < self._end_time:
            try:
                reply = channel.consume_last(timeout=0.1)

                curr_time = time.time()
                dt = curr_time - last_keypoints_time

                if dt > 0.2:
                    kf_center.reset()
                    kf_front.reset()

                if not reply:
                    continue

                kp = reply.unpack(ObjectAnnotations)

                objs = kp.objects
                if not objs:
                    continue

                obj = objs[0]
                kps = obj.keypoints

                if (
                    len(kps) < 2
                    or obj.score <= CONFIDENCE_BB
                    or kps[0].score <= CONFIDENCE_KP
                    or kps[1].score <= CONFIDENCE_KP
                ):
                    continue

                center_raw = (int(kps[0].position.x), int(kps[0].position.y))
                front_raw = (int(kps[1].position.x), int(kps[1].position.y))

                real_dt = float(np.clip(dt, 1e-2, 0.5))
                kf_center.set_dt(real_dt)
                kf_front.set_dt(real_dt)

                kf_center.predict()
                kf_front.predict()

                center_filtered = kf_center.update(center_raw)
                front_filtered = kf_front.update(front_raw)

                kps[0].position.x = int(center_filtered[0])
                kps[0].position.y = int(center_filtered[1])
                kps[1].position.x = int(front_filtered[0])
                kps[1].position.y = int(front_filtered[1])

                last_keypoints_time = curr_time

                with self.keypoints_lock:
                    self._last_keypoints[camera_id] = kp

                self.new_keypoints_event.set()

            except socket.timeout:
                continue
            except Exception as e:
                self.log.warn(f"Error cam {camera_id}: {e}")
                continue

        self.log.info(f"Thread cam {camera_id} finished.")
        with self.keypoints_lock:
            if camera_id in self._last_keypoints:
                del self._last_keypoints[camera_id]

        self.keypoints_event[camera_id].clear()
        channel.close()

    def define_pose(self) -> None:
        self.pose_event.set()
        channel = Channel(self.connection.broker_uri)

        while time.time() < self._end_time:
            if not self.new_keypoints_event.wait(timeout=0.1):
                continue

            self.new_keypoints_event.clear()

            with self.keypoints_lock:
                keypoints = self._last_keypoints.copy()

            if len(keypoints) < 2:
                continue

            try:
                kp_center = {}
                kp_front = {}

                for cid, k in keypoints.items():
                    pts = k.objects[0].keypoints
                    kp_center[cid] = (pts[0].position.x, pts[0].position.y)
                    kp_front[cid] = (pts[1].position.x, pts[1].position.y)

                world_center = point2world(self.proj_cache, kp_center)
                world_front = point2world(self.proj_cache, kp_front)

                yaw_deg, pitch_deg = angle(world_center, world_front)

                pose = Pose(
                    position=Position(
                        x=world_center[0], y=world_center[1], z=world_center[2]
                    ),
                    orientation=Orientation(yaw=yaw_deg, pitch=pitch_deg),
                )

                msg = Message(content=pose)
                msg.topic = "Tiffany.Pose"
                channel.publish(msg)

            except Exception as e:
                self.log.error(f"Pose calc error: {e}")

        self.log.info("Pose calculation finished.")
        self.pose_event.clear()
        channel.close()

    def start_detections(self, seconds, ctx) -> Status:
        now = time.time()
        if self._end_time < now:
            self._end_time = now + seconds.seconds
        else:
            self._end_time += seconds.seconds
        events = [ev.is_set() for ev in self.keypoints_event.values()]
        if not all(events):
            channel = StreamChannel(self.connection.broker_uri)
            subscription = Subscription(channel)
            request = Message(content=seconds, reply_to=subscription)
            for cam_id in self.parameters.keys():
                channel.publish(
                    request,
                    topic=f"Tiffany.Keypoints.{cam_id}.StartDetection",
                )
                if not self.keypoints_event[cam_id].is_set():
                    t = threading.Thread(
                        target=self.get_keypoints_by_camera, args=(cam_id,)
                    )
                    t.start()
            channel.close()
            if not self.pose_event.is_set():
                t = threading.Thread(target=self.define_pose)
                t.start()

            return Status(StatusCode.OK, "Started.")

        return Status(StatusCode.ALREADY_EXISTS, "Extended time.")

    def stop(self, *args) -> Status:
        self._end_time = 0.0
        self.pose_event.clear()
        self.new_keypoints_event.set()
        return Status(StatusCode.OK, "Stopping.")
