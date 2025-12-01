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
from is_wire.core import Message, Status, StatusCode, Subscription
from opencensus.trace.blank_span import BlankSpan
from opencensus.trace.span import Span

from .Connection import Connection
from .Detector import Detector
from .StreamChannel import StreamChannel


class Threading:
    """Manages background threads for object detection and streaming results.

    This class encapsulates logic to run two main threads:

    1. A continuous detection thread that consumes images and runs the model.
    2. An on-demand streaming thread that annotates images with detections and publishes them for a specified duration.

    A lock is used to ensure thread-safe access to the last detection data.
    """

    def __init__(self, connection: Connection, detector: Detector):
        """Initializes the threading manager.

        Args:
            connection (Connection): Manages the broker connection.
            detector (Detector): Responsible for running predictions.
        """
        self.connection = connection
        self.log = connection.log
        self.detector = detector
        self._end_time = 0.0
        self.stream_event = threading.Event()
        self.detection_event = threading.Event()
        self.lock = threading.Lock()

    def detection_thread(self) -> None:
        """Runs for a defined duration to fetch images and perform detection.

        Designed to run a thread in the background. It consumes images
        from the camera feed, runs detection, and stores the latest results safely.
        Handles connection errors by attempting to reset when needed.
        """
        self.detection_event.set()
        def _init_channels():
            channel_stream = StreamChannel(self.connection.broker_uri)
            channel_camera = StreamChannel(self.connection.broker_uri)
            Subscription(channel_camera).subscribe(
                f"CameraGateway.{self.connection.camera_id}.Frame"
            )
            return channel_stream, channel_camera

        def _check_time() -> bool:
            with self.lock
                end_time = self._end_time
            return time.time() < end_time

        channel_stream, channel_camera = _init_channels()
        exporter = self.connection.exporter

        self.log.info("Detection started.")

        while _check_time():
            try:
                img, tracer, span = get_images_from_camera(
                    channel_camera, exporter, _check_time
                )
                timestamp = time.time()
            except KeyboardInterrupt:
                self.log.error("Shutting down...")
                raise
            except (ConnectionResetError, IndexError, UnexpectedFrame, TypeError):
                # self.log.warn("Skipping frame due to temporary issue.")
                continue
            except OSError:
                self.log.warn("Resetting server connection due to OSError...")
                time.sleep(2.5)
                channel_stream, channel_camera = _init_channels()
                continue
            except Exception:
                continue

            with tracer.span(name="predict_tiffany"):
                results = self.detector.predict(img)
                result_dict = self.detector.results_to_dict(results)

            with tracer.span(name="pack_and_publish_detection"):
                if len(result_dict["boxes"]):
                    obj = ObjectAnnotations(
                        objects=[self.detector.dict_to_obj_annot(result_dict)],
                        resolution=Resolution(height=720, width=1280),
                        frame_id=self.connection.camera_id,
                    )
                    msg = Message()
                    msg.inject_tracing(span)
                    msg.topic = f"Tiffany.{self.connection.camera_id}.Detection"
                    msg.pack(obj)
                    channel_stream.publish(msg)

                    if self.stream_event.is_set():
                        threading.Thread(
                            target=self.stream_detection,
                            name="StreamDetection",
                            args=(channel_stream, obj, img, span),
                        ).start()

            tracer.end_span()

        self.detection_event.clear()
        self.stream_event.clear()
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
            channel (StreamChannel): Channel to publish annotated images.
            last_detection (ObjectAnnotations): The detection results to draw.
            last_image (np.ndarray): The image to annotate.
            last_span (Span | BlankSpan): The tracing span to attach.
        """
        det = last_detection
        img = last_image
        span = last_span

        if img is None:
            return

        img_to_draw = img.copy()

        if det.objects:
            box = det.objects[0].region.vertices
            bb1 = (int(box[0].x), int(box[0].y))
            bb2 = (int(box[1].x), int(box[1].y))

            cv2.rectangle(img_to_draw, bb1, bb2, (255, 255, 0), 2)
            cv2.putText(
                img_to_draw,
                f"Score: {det.objects[0].score:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        try:
            msg = Message()
            msg.inject_tracing(span)
            msg.topic = f"Tiffany.{self.connection.camera_id}.Frame"
            msg.pack(to_image(img_to_draw, compression_level=0.8))
            channel.publish(msg)
        except ConnectionResetError:
            return
        except OSError:
            return
        except Exception as e:
            self.log.error(f"Unexpected error while publishing: {e}")
            return

    def init_stream(self, seconds: Duration, ctx) -> Status:
        """Starts the detection streaming if not already running.

        Exposed as an RPC method. Checks if a stream is active using an event flag,
        and active a flag if none is running.

        Args:
            seconds (Duration): Desired duration of the stream in minutes.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if the stream started, or `ALREADY_EXISTS` if one is already running.
        """
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
        an event flag, and starts a new `DetectionThread` if none is running.

        Args:
            seconds (Duration): Desired duration of detection.
            ctx: Service context provided by is-wire RPC.

        Returns:
            Status: `OK` if detection started, or `ALREADY_EXISTS` if already running.
        """
        with self.lock
            self._end_time += seconds.seconds

        if not self.detection_event.is_set():
            threading.Thread(
                target=self.detection_thread,
                name="DetectionThread",
            ).start()
            return Status(StatusCode.OK, f"Detection started with a duration of {seconds.seconds / 60:.2f} minutes.")
        else:
            return Status(StatusCode.ALREADY_EXISTS, f"Detection already running. Added +{seconds.seconds / 60:.2f} minutes.")
