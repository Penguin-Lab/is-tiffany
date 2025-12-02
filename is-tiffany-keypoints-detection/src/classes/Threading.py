import socket
import threading
import time
from typing import Union

import cv2
import numpy as np
from amqp.exceptions import UnexpectedFrame
from functions import get_images_from_camera, to_image
from google.protobuf import json_format
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from is_wire.core import Channel, Message, Status, StatusCode, Subscription
from opencensus.trace.blank_span import BlankSpan
from opencensus.trace.span import Span

from .Connection import Connection
from .Detector import Detector
from .StreamChannel import StreamChannel


class Threading:
    """Manages background threads for object detection and result streaming.

    This class encapsulates the logic to run two main threads:
    1. A continuous detection thread that consumes images and runs the model.
    2. An on-demand streaming thread that annotates images with detections
       and publishes them for a specified duration.

    A lock is used to ensure thread-safe access to the last detection data.
    """

    def __init__(self, connection: Connection, detector: Detector):
        """Initializes the threading manager.

        Args:
            connection (Connection): An object that manages the broker connection.
            detector (Detector): An object responsible for running predictions.
        """
        self.connection = connection
        self.log = connection.log
        self.detector = detector
        self._end_time = 0.0
        self.stream_event = threading.Event()
        self.detection_event = threading.Event()

    def detection_thread(self) -> None:
        """Runs for a defined duration to fetch images and perform detection.

        Designed to run a thread in the background. It consumes images
        from the camera feed, passes them to the detector, and safely stores
        the most recent results. Also handles connection errors, attempting
        to reset the connection when needed.
        """
        self.detection_event.set()
        def _init_channels():
            channel_stream = StreamChannel(self.connection.broker_uri)
            channel_camera = StreamChannel(self.connection.broker_uri)
            Subscription(channel_camera).subscribe(
                f"CameraGateway.{self.connection.camera_id}.Frame"
            )
            Subscription(channel_stream).subscribe(
                f"Tiffany.{self.connection.camera_id}.Detection"
            )
            self.connection.create_exporter(
                self, self.connection.service_name, self.connection.zipkin_uri, self.log
                )
            return channel_stream, channel_camera
        
        channel_stream, channel_camera = _init_channels()
        
        self.log.info("Detection started.")
                      
        last_end_time = self._end_time
        while time.time() < last_end_time:
            last_end_time = self._end_time
            try:
                img, tracer, span, offset, original_img = (
                    get_images_from_camera(channel_camera, channel_stream, self.connection.exporter, last_end_time)
                )
            except KeyboardInterrupt:
                self.log.error("Shutting down...")
                raise KeyboardInterrupt

            except (ConnectionResetError, IndexError, UnexpectedFrame, TypeError):
                # self.log.warn("Skipping frame due to temporary issue.")
                continue

            except OSError:
                self.log.warn("Restarting server connection due to OSError...")
                time.sleep(2.5)
                channel_stream, channel_camera = _init_channels()
                continue
            
            if img.size == 0:
                continue

            with tracer.span(name="predict_tiffany"):
                results = self.detector.predict(img)
                result_dict = self.detector.results_to_dict(results, offset)

            with tracer.span(name="pack_and_publish_detection"):
                if len(result_dict["boxes"]):
                    obj = ObjectAnnotations(
                        objects=[self.detector.dict_to_obj_annot(result_dict)],
                        resolution=Resolution(height=720, width=1280),
                        frame_id=self.connection.camera_id,
                    )
                    msg = Message()
                    msg.inject_tracing(span)
                    msg.topic = f"Tiffany.{self.connection.camera_id}.Keypoints"
                    msg.pack(obj)
                    channel_stream.publish(msg)

                    if self.stream_event.is_set():
                        threading.Thread(
                            target=self.stream_detection,
                            name="StreamDetection",
                            args=(channel_stream, obj, original_img, span),
                        ).start()

            tracer.end_span()
        self.detection_event.clear()
        self.stream_event.clear()
        self._end_time = 0.0
        self.log.info("Detection finished.")
        channel_camera.close()
        channel_stream.close()

    def stream_detection(
        self,
        channel: StreamChannel,
        last_detection: ObjectAnnotations,
        last_image: np.ndarray,
        last_span: Union[Span, BlankSpan],
    ) -> None:
        """Draws detections on images and streams.

        Args:
            channel (StreamChannel): The channel used to publish annotated images.
            last_detection (ObjectAnnotations): Function to get the latest detection.
            last_image (np.ndarray): Function to get the latest image.
            last_span (Span | BlankSpan): Function to get the latest tracing span.
        """
        channel = StreamChannel(self.connection.broker_uri)

        det = last_detection
        img = last_image
        span = last_span

        if img is None:
            return

        img_to_draw = img.copy()

        if det.objects:
            kp = det.objects[0].keypoints
            kp1 = [kp[0].position.x, kp[0].position.y]
            kp2 = [kp[1].position.x, kp[1].position.y]
            box = det.objects[0].region.vertices
            bb1 = (int(box[0].x), int(box[0].y))
            bb2 = (int(box[1].x), int(box[1].y))

            cv2.rectangle(img_to_draw, bb1, bb2, (255, 255, 0), 2)
            cv2.circle(img_to_draw, (int(kp1[0]), int(kp1[1])), 3, (0, 255, 0), -1)
            cv2.circle(img_to_draw, (int(kp2[0]), int(kp2[1])), 3, (0, 0, 255), -1)
            cv2.putText(
                img_to_draw,
                f"{kp[0].score:.2f} | {(kp[0].score - 0.99) * 100}",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_to_draw,
                f"{kp[1].score:.2f} | {(kp[0].score - 0.99) * 100}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img_to_draw,
                f"{det.objects[0].score:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

            try:
                msg = Message()
                msg.inject_tracing(span)
                msg.topic = f"Tiffany.Keypoints.{self.connection.camera_id}.Frame"
                msg.pack(to_image(img_to_draw, compression_level=0.8))
                channel.publish(msg)
            except (UnexpectedFrame, ConnectionResetError, OSError):
                return
            except Exception as e:
                self.log.error(f"Unexpected error while publishing: {e}")
                return

    def init_stream(self, seconds: Duration, ctx) -> Status:
        """Starts the detection streaming if not already running.

        Exposed as an RPC method. Checks if a stream is active using an event flag,
        and active a flag if none is running.

        Args:
            seconds (Duration): Desired duration of the stream in seconds.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if the stream started, or `ALREADY_EXISTS` if one is already running.
        """
        if not self.stream_event.is_set():
            if not self.detection_event.is_set():
                self.init_detection(seconds, ctx)
            if not self.stream_event.is_set():
                self.stream_event.set()
            return Status(StatusCode.OK, "Stream started")
        else:
            return Status(StatusCode.ALREADY_EXISTS, "Stream already running")

    def init_detection(self, seconds: Duration, ctx) -> Status:
        """Starts the detection thread if not already running.

        Exposed as an RPC method. Checks if the detection thread is active using
        an event flag, and starts a new `detection_thread` if none is running.

        Args:
            seconds (Duration): Desired duration of detection in seconds.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if detection started, or `ALREADY_EXISTS` if already running.
        """
        self._end_time = time.time() + seconds.seconds if self._end_time < time.time() else self._end_time + seconds.seconds
        channel = Channel(self.connection.broker_uri)
        subscription = Subscription(channel)
        request = Message(content=seconds, reply_to=subscription)
        channel.publish(
            request,
            topic=f"Tiffany.Detection.{self.connection.camera_id}.StartDetection",
        )
        try:
            reply = channel.consume(timeout=5.0)
            channel.close()
        except socket.timeout:
            return Status(
                StatusCode.DEADLINE_EXCEEDED, "No response from detection service"
            )

        if not self.detection_event.is_set() and (reply.status.code == StatusCode.OK or reply.status.code == StatusCode.ALREADY_EXISTS):
            threading.Thread(
                target=self.detection_thread,
                name="DetectionThread"
            ).start()
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
        self.stream_event.clear()
        self.detection_event.clear()
        self._end_time = 0.0
        return Status(StatusCode.OK, "Stopping detection.")
