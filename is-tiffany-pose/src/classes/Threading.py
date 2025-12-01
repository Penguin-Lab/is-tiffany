import os
import threading
import time
from typing import Union

import numpy as np
from google.protobuf import json_format
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.common_pb2 import Orientation, Pose, Position
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, Message, Status, StatusCode, Subscription

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
        self._last_pose = (Pose(), 0, 0)
        self.keypoints_event = {
            cam_id: threading.Event() for cam_id in parameters.keys()
        }
        self.pose_event = threading.Event()
        self.keypoints_lock = threading.Lock()
        self.pose_lock = threading.Lock()

    def get_keypoints_by_camera(self, seconds: Duration, camera_id: int) -> None:
        """
        Continuously fetches keypoints from a specific camera for a defined duration.

        Args:
            seconds (Duration): Duration in seconds to fetch keypoints.
            camera_id (int): ID of the camera to fetch keypoints from.
        """

        self.keypoints_event[camera_id].set()
        start_time = time.time()
        self.log.info(
            f"Starting keypoints acquisition. Duration: {seconds.seconds / 60:.2f} minutes."
        )
        kf_center = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0)
        kf_front = KalmanFilter2D(dt=0.1, process_var=1e-5, meas_var=4.0)

        channel = Channel(self.connection.broker_uri)
        subscription = Subscription(channel)

        last_keypoints_time = 0.0
        last_detection_timestamp = None

        while time.time() - start_time < seconds.seconds:
            time.sleep(1 / 15)
            if time.time() - last_keypoints_time > 1.0:
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
                last_keypoints_time = time.time()
                continue

            msg = Struct()
            msg.fields["timestamp"].bool_value = True
            request = Message(reply_to=subscription)
            request.pack(msg)

            try:
                channel.publish(
                    request, topic=f"Tiffany.Keypoints.{camera_id}.GetDetection"
                )
                reply = channel.consume(timeout=0.5)

                if not reply or reply.status.code != StatusCode.OK:
                    continue

                data = json_format.MessageToDict(reply.unpack(Struct))
                timestamp = float(data.get("timestamp", 0))
                kp = json_format.ParseDict(
                    data.get("detection", {}), ObjectAnnotations()
                )

                if timestamp == 0 or (time.time() - timestamp) >= 1.0:
                    self.set_last_keypoints(None, camera_id, 0)
                    continue

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
                if last_detection_timestamp is not None:
                    dt = timestamp - last_detection_timestamp
                    dt = float(np.clip(dt, 1e-2, 1.0))
                    kf_center.set_dt(dt)
                    kf_front.set_dt(dt)
                last_detection_timestamp = timestamp

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

                self.set_last_keypoints(kp, camera_id, timestamp)
                last_keypoints_time = time.time()

            except Exception as exc:
                self.log.debug(
                    f"Error while fetching keypoints for camera {camera_id}: {exc}"
                )
                continue

        self.log.info("Thread finished.")
        self.set_last_keypoints(None, camera_id, 0)
        self.keypoints_event[camera_id].clear()
        channel.close()

    def set_last_keypoints(
        self, keypoints: ObjectAnnotations, camera_id: int, timestamp
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
                self._last_keypoints[camera_id] = (keypoints, timestamp)

    def set_last_pose(self, pose: Pose, num_cameras: int, timestamp: float) -> None:
        """
        Updates the last detected pose safely.
        Args:
            pose (Pose): The latest detected pose.
            num_cameras (int): Number of cameras that contributed to the pose.
            timestamp (float): Timestamp of the pose detection.
        """

        with self.pose_lock:
            self._last_pose = (pose, num_cameras, timestamp)

    def get_last_keypoints(self) -> dict:
        """
        Retrieves the last keypoints results safely.

        Returns:
            Dict[int, ObjectAnnotations]: Last stored keypoints for each camera.
        """
        with self.keypoints_lock:
            return self._last_keypoints

    def get_last_message(self, *args) -> Union[Struct, Pose]:
        """Return the latest detection either as a Struct (if request fields are present) or as ObjectAnnotations.

        If a Struct-like request is provided as the first positional argument, the method builds and
        returns a Struct containing:
          - "pose": the pose converted to a dict
          - "timestamp":
        If no request fields are provided, the raw ObjectAnnotations object is returned.

        The method reads shared state under a lock to ensure thread safety.
        """
        with self.pose_lock:
            pose, num_cameras, timestamp = self._last_pose
        if time.time() - timestamp > 5.0:
            self.set_last_pose(Pose(), 0, 0)

        request = (
            json_format.MessageToDict(args[0], preserving_proto_field_name=True)
            if args
            else None
        )
        if request:
            js = {
                "pose": json_format.MessageToDict(
                    pose, preserving_proto_field_name=True
                )
                if pose != Pose()
                else {},
            }
            if request.get("timestamp", False):
                js["timestamp"] = timestamp
            if request.get("num_cameras", False):
                js["num_cameras"] = num_cameras
            struct = Struct()
            struct.update(js)
            return struct
        return pose

    def define_pose(self, seconds: Duration) -> None:
        """Continuously calculates and updates Tiffany's pose using keypoints from multiple cameras.

        Args:
            seconds (Duration): Duration to compute the pose in seconds.
        """

        self.pose_event.set()
        start_time = time.time()
        self.log.info(
            f"Starting pose calculation for {seconds.seconds / 60:.2f} minutes."
        )

        while time.time() - start_time < seconds.seconds:
            keypoints = self.get_last_keypoints().copy()

            if len(keypoints) < 2:
                continue

            recent = max(ts for kp, ts in keypoints.values())
            if time.time() - recent > 2.0:
                self.set_last_pose(Pose(), 0, 0)
                continue

            kp_center = {
                cam_id: (
                    kp.objects[0].keypoints[0].position.x,
                    kp.objects[0].keypoints[0].position.y,
                )
                for cam_id, (kp, ts) in keypoints.items()
                if time.time() - ts < 1.0
            }
            kp_front = {
                cam_id: (
                    kp.objects[0].keypoints[1].position.x,
                    kp.objects[0].keypoints[1].position.y,
                )
                for cam_id, (kp, ts) in keypoints.items()
                if time.time() - ts < 1.0
            }

            if len(kp_center) < 2 or len(kp_front) < 2:
                continue

            try:
                timestamp = min(
                    ts
                    for kp, ts in keypoints.values()
                    if kp.objects[0].keypoints[0].position.x
                    in [v[0] for v in kp_center.values()]
                )
            except ValueError:
                timestamp = time.time()

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
            self.set_last_pose(pose, len(kp_center), timestamp)

        self.log.info("Pose calculation finished.")
        self.set_last_pose(Pose(), 0, 0)
        self.pose_event.clear()

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
        if (
            any(event.is_set() for event in self.keypoints_event.values())
            or self.pose_event.is_set()
        ):
            return Status(StatusCode.ALREADY_EXISTS, "Detection already in progress")
        for cam_id in self.parameters.keys():
            channel = Channel(self.connection.broker_uri)
            subscription = Subscription(channel)
            request = Message(content=seconds, reply_to=subscription)
            try:
                channel.publish(
                    request, topic=f"Tiffany.Keypoints.{cam_id}.StartDetection"
                )
                reply = channel.consume(timeout=1.0)
                time.sleep(0.5)
                if reply.status.code in [StatusCode.OK, StatusCode.ALREADY_EXISTS]:
                    if not self.keypoints_event[cam_id].is_set():
                        threading.Thread(
                            target=self.get_keypoints_by_camera,
                            name=f"Keypoints.{cam_id}.Thread",
                            args=(seconds, cam_id),
                        ).start()
            except Exception:
                return Status(
                    StatusCode.DEADLINE_EXCEEDED, "No response from detection service"
                )
        if not self.pose_event.is_set():
            threading.Thread(
                target=self.define_pose,
                name="PoseThread",
                args=(seconds,),
            ).start()
        return Status(StatusCode.OK, "Detections started successfully")
