import queue
import threading
import time

import cv2
import numpy as np
from functions import get_images_from_camera, to_image
from google.protobuf.duration_pb2 import Duration
from is_msgs.image_pb2 import ObjectAnnotations, Resolution
from is_wire.core import Message, Status, StatusCode, Subscription

from .Connection import Connection
from .Detector import Detector
from .StreamChannel import StreamChannel


class Threading:
    def __init__(self, connection: Connection, detector: Detector):
        self.connection = connection
        self.log = connection.log
        self.detector = detector
        self._end_time = 0.0

        self.detection_event = threading.Event()
        self.stream_active = False

        self.vis_queue = queue.Queue(maxsize=1)

        self.vis_thread = threading.Thread(
            target=self._vis_worker, name="VisWorker", daemon=True
        )
        self.vis_thread.start()

    def detection_thread(self) -> None:
        self.detection_event.set()

        channel_stream = StreamChannel(self.connection.broker_uri)
        channel_camera = StreamChannel(self.connection.broker_uri)

        Subscription(channel_camera).subscribe(
            f"CameraGateway.{self.connection.camera_id}.Frame"
        )

        self.connection.create_exporter(
            self, self.connection.service_name, self.connection.zipkin_uri, self.log
        )

        self.log.info("Detection started.")

        while time.time() < self._end_time:
            try:
                ret = get_images_from_camera(
                    channel_camera, self.connection.exporter, self._end_time
                )
                if not ret:
                    continue
                img, tracer, span = ret

            except Exception:
                continue

            if img is None or img.size == 0:
                continue

            with tracer.span(name="predict_tiffany") as s:
                results = self.detector.predict(img)
                obj_annot = self.detector.to_annotation(results)

            if obj_annot:
                obj = ObjectAnnotations(
                    objects=[obj_annot],
                    resolution=Resolution(height=720, width=1280),
                    frame_id=self.connection.camera_id,
                )

                msg = Message()
                msg.pack(obj)
                msg.topic = f"Tiffany.{self.connection.camera_id}.Detection"
                msg.inject_tracing(span)
                channel_stream.publish(msg)

                if self.stream_active:
                    try:
                        self.vis_queue.put_nowait(
                            {
                                "img": img,
                                "annot": obj,
                                "span": span,
                            }
                        )
                    except queue.Full:
                        pass

            tracer.end_span()

        self.detection_event.clear()
        self.stream_active = False
        channel_camera.close()
        channel_stream.close()
        self.log.info("Detection finished.")

    def _vis_worker(self):
        """Worker persistente que desenha e envia imagens."""
        channel = StreamChannel(self.connection.broker_uri)

        while True:
            try:
                item = self.vis_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            img = item["img"]
            annot = item["annot"]
            span = item["span"]

            img_drawn = self._draw(img, annot)

            jpeg_img = to_image(img_drawn, compression_level=0.8)

            msg = Message()
            msg.pack(jpeg_img)
            msg.topic = f"Tiffany.{self.connection.camera_id}.Frame"
            msg.inject_tracing(span)
            try:
                channel.publish(msg)
            except Exception:
                try:
                    channel = StreamChannel(self.connection.broker_uri)
                except Exception:
                    time.sleep(1)

    def _draw(self, img, annot):
        """Helper para limpar a classe."""
        img_c = img.copy()
        if annot.objects:
            obj = annot.objects[0]
            if obj.region and len(obj.region.vertices) >= 2:
                v = obj.region.vertices
                cv2.rectangle(
                    img_c,
                    (int(v[0].x), int(v[0].y)),
                    (int(v[1].x), int(v[1].y)),
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img_c,
                    f"Score: {obj.score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
        return img_c

    def init_detection(self, seconds: Duration, ctx) -> Status:
        self._end_time = max(self._end_time, time.time() + seconds.seconds)

        if not self.detection_event.is_set():
            t = threading.Thread(target=self.detection_thread, name="DetectionThread")
            t.start()
            return Status(StatusCode.OK, "Started.")
        return Status(StatusCode.OK, "Extended time.")

    def init_stream(self, seconds: Duration, ctx) -> Status:
        if not self.detection_event.is_set():
            self.init_detection(seconds, ctx)

        if not self.stream_active:
            self.stream_active = True
            return Status(StatusCode.OK, "Stream Started.")
        return Status(StatusCode.ALREADY_EXISTS, "Stream Already Active.")

    def stop(self, *args) -> Status:
        self.stream_active = False
        self._end_time = 0.0
        return Status(StatusCode.OK, "Stopping.")
