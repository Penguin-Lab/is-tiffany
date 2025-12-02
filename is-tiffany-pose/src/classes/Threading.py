import os
import threading
import time
from typing import Union
import socket
import numpy as np
from google.protobuf import json_format
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.common_pb2 import Orientation, Pose, Position
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, Message, Status, StatusCode, Subscription
from .StreamChannel import StreamChannel
from functions import angle, point2world

from .Connection import Connection
from .KalmanFilter2D import KalmanFilter2D

CONFIDENCE_KP = os.environ.get("confidence_kp", 0.99)
CONFIDENCE_BB = os.environ.get("confidence_bb", 0.5)


class Threading:
    """
    Manages background threads for object keypoints detection and pose estimation.

    This class encapsulates the logic to run two main types of threads:
    1. Continuous detection threads that consume camera images and execute the model.
    2. On-demand streaming threads that annotate images with detection results
       and publish them for a specified duration.

    A lock is used to ensure thread-safe access to the latest detection data.
    """

    def __init__(self, connection: Connection, parameters: dict):
        """
        Initializes the thread manager.

        Args:
            connection (Connection): Object managing the broker connection.
            parameters (dict): Camera calibration parameters.
            log (Logger): Logger instance for recording messages.
        """
        self.connection = connection
        self.log = connection.log
        self.parameters = parameters
        self._last_keypoints = {}
        self.keypoints_event = {
            cam_id: threading.Event() for cam_id in parameters.keys()
        }
        self._end_time = 0.0
        self.pose_event = threading.Event()
        self.new_keypoints_event = threading.Event()
        self.keypoints_lock = threading.Lock()

    def get_keypoints_by_camera(self, camera_id: int) -> None:
        """
        Continuously fetches keypoints from a specific camera for a defined duration.

        Args:
            seconds (Duration): Duration in seconds to fetch keypoints.
            camera_id (int): ID of the camera to fetch keypoints from.
        """
        if camera_id == 4:
            return
        self.keypoints_event[camera_id].set()
        kf_center = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0)
        kf_front = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0)

        channel = StreamChannel(self.connection.broker_uri)
        Subscription(channel).subscribe(f"Tiffany.{camera_id}.Keypoints")

        last_keypoints_time = 0.0

        last_end_time = self._end_time
        while time.time() < last_end_time:
            last_end_time = self._end_time
            if time.time() - last_keypoints_time > 0.1:
                kf_center = KalmanFilter2D(
                    dt=0.1,
                    process_var=1e-5,
                    meas_var=4.0,
                )
                kf_front = KalmanFilter2D(
                    dt=0.1,
                    process_var=1e-5,
                    meas_var=4.0,
                )
                self.set_last_keypoints(None, camera_id)
                last_keypoints_time = time.time()
                continue

            try:
                reply = channel.consume_last(0.1)
                if not reply:
                    continue

                kp = reply.unpack(ObjectAnnotations)

                if (
                    not kp.objects
                    or len(kp.objects[0].keypoints) < 2
                    or kp.objects[0].keypoints[0].score <= CONFIDENCE_KP
                    or kp.objects[0].keypoints[1].score <= CONFIDENCE_KP
                    or kp.objects[0].score <= CONFIDENCE_BB
                ):
                    continue

                center = (
                    int(kp.objects[0].keypoints[0].position.x),
                    int(kp.objects[0].keypoints[0].position.y),
                )
                front = (
                    int(kp.objects[0].keypoints[1].position.x),
                    int(kp.objects[0].keypoints[1].position.y),
                )

                # Optionally update filter dt from detection timestamps to handle variable frame rates
                if last_keypoints_time != 0.0:
                    dt = time.time() - last_keypoints_time
                    dt = float(np.clip(dt, 1e-2, 1.0))
                    kf_center.set_dt(dt)
                    kf_front.set_dt(dt)
                last_keypoints_time = time.time()

                kf_center.predict()
                kf_front.predict()

                try:
                    center_filtered = kf_center.update(center)
                    front_filtered = kf_front.update(front)
                except Exception as e:
                    self.log.warn(f"Kalman update failed for camera {camera_id}: {e}")
                    continue

                kp.objects[0].keypoints[0].position.x = int(center_filtered[0])
                kp.objects[0].keypoints[0].position.y = int(center_filtered[1])
                kp.objects[0].keypoints[1].position.x = int(front_filtered[0])
                kp.objects[0].keypoints[1].position.y = int(front_filtered[1])

                self.set_last_keypoints(kp, camera_id)
                self.new_keypoints_event.set()
            except Exception as exc:
                self.log.debug(
                    f"Error while fetching keypoints for camera {camera_id}: {exc}"
                )
                continue

        self.log.info("Thread finished.")
        self.set_last_keypoints(None, camera_id)
        self.keypoints_event[camera_id].clear()
        channel.close()

    def set_last_keypoints(
        self, keypoints: ObjectAnnotations, camera_id: int
    ) -> None:
        """
        Updates the last keypoints safely for a specific camera.

        Args:
            keypoints (ObjectAnnotations): Last keypoints result.
            camera_id (int): Camera ID that provided the keypoints.
        """
        with self.keypoints_lock:
            if keypoints is None:
                if camera_id in self._last_keypoints:
                    del self._last_keypoints[camera_id]
            else:
                self._last_keypoints[camera_id] = keypoints


    def get_last_keypoints(self) -> dict:
        """
        Retrieves the last keypoints results safely.

        Returns:
            Dict[int, ObjectAnnotations]: Last stored keypoints for each camera.
        """
        with self.keypoints_lock:
            return self._last_keypoints

    def define_pose(self) -> None:
        """Continuously calculates and updates Tiffany's pose using keypoints from multiple cameras.

        Args:
            seconds (Duration): Duration to compute the pose in seconds.
        """

        self.pose_event.set()
        channel = Channel(self.connection.broker_uri)

        last_end_time = self._end_time
        while time.time() < last_end_time:
            last_end_time = self._end_time
            if not self.new_keypoints_event.is_set():
                continue
            timestamp = time.time()

            keypoints = self.get_last_keypoints().copy()

            if len(keypoints) < 2:
                continue

            kp_center = {
                cam_id: (
                    kp.objects[0].keypoints[0].position.x,
                    kp.objects[0].keypoints[0].position.y,
                )
                for cam_id, kp in keypoints.items()
            }
            kp_front = {
                cam_id: (
                    kp.objects[0].keypoints[1].position.x,
                    kp.objects[0].keypoints[1].position.y,
                )
                for cam_id, kp in keypoints.items()
            }

            if len(kp_center) < 2 or len(kp_front) < 2:
                continue

            world_center = point2world(self.parameters, kp_center)
            world_front = point2world(self.parameters, kp_front)

            Xw_center = world_center
            Xw_front = world_front

            Xw_center = np.round(Xw_center, 2)
            Xw_front = np.round(Xw_front, 2)

            yaw_deg, pitch_deg = angle(Xw_center, Xw_front)

            pose = Pose(
                position=Position(x=Xw_center[0], y=Xw_center[1], z=Xw_center[2]),
                orientation=Orientation(yaw=yaw_deg, pitch=pitch_deg),
            )

            msg = Message()
            msg.topic = f"Tiffany.Pose"
            msg.pack(pose)
            channel.publish(msg)

            self.new_keypoints_event.clear()
        self.log.info("Pose calculation finished.")
        self.pose_event.clear()
        channel.close()

    def start_detections(self, seconds: Duration, ctx) -> Status:
        """
        Starts the detection thread if not already running.

        Args:
            seconds (Duration): Duration of detection in seconds.
            ctx: RPC service context provided by is-wire.

        Returns:
            Status: OK if detection started, ALREADY_EXISTS if detection is
                    already running, DEADLINE_EXCEEDED if the detection service
                    did not respond.
        """
        self._end_time = time.time() + seconds.seconds if self._end_time < time.time() else self._end_time + seconds.seconds
        events = [boolean.is_set() for boolean in self.keypoints_event.values()]

        if not all(events):
            channel = Channel(self.connection.broker_uri)
            subscription = Subscription(channel)
            request = Message(content=seconds, reply_to=subscription)
            for cam_id in self.parameters.keys():
                channel.publish(
                    request,
                    topic=f"Tiffany.Keypoints.{cam_id}.StartDetection",
                )
            try:
                reply = channel.consume(timeout=5.0)
            except socket.timeout:
                return Status(
                    StatusCode.DEADLINE_EXCEEDED, "No response from detection service"
                )
            if (
                reply.status.code == StatusCode.OK
                or reply.status.code == StatusCode.ALREADY_EXISTS
            ):
                for cam_id in self.parameters.keys():
                    if not self.keypoints_event[cam_id].is_set():
                            threading.Thread(
                                target=self.get_keypoints_by_camera,
                                name=f"Keypoints.{cam_id}.Thread",
                                args=(cam_id,),
                            ).start()
                threading.Thread(
                    target=self.define_pose,
                    name="PoseThread",
                    args=(),
                ).start()
            channel.close()
            return Status(
                StatusCode.OK,
                f"Detection started with a duration of {seconds.seconds / 60:.2f} minutes.",
            )
        else:
            return Status(
                StatusCode.ALREADY_EXISTS,
                f"Detection already running. Added +{seconds.seconds / 60:.2f} minutes.",
            )
    
    def stop(self, *args) -> Status:
            self.pose_event.clear()
            for event in self.keypoints_event.values():
                event.clear()
            self.new_keypoints_event.clear()
            self._end_time = 0.0
            return Status(StatusCode.OK, "Stopping detection.")