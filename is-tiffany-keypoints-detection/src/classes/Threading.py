import queue
import threading
import time

import cv2
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
            target=self._visualization_worker, name="VisWorker", daemon=True
        )
        self.vis_thread.start()

    def detection_thread(self) -> None:
        self.detection_event.set()

        channel_stream = StreamChannel(self.connection.broker_uri)
        channel_camera = StreamChannel(self.connection.broker_uri)

        sub_cam = Subscription(channel_camera)
        sub_cam.subscribe(f"CameraGateway.{self.connection.camera_id}.Frame")
        sub_stream = Subscription(channel_stream)
        sub_stream.subscribe(f"Tiffany.{self.connection.camera_id}.Detection")

        self.connection.create_exporter(
            self, self.connection.service_name, self.connection.zipkin_uri, self.log
        )

        self.log.info("Detection started.")

        while time.time() < self._end_time:
            try:
                data = get_images_from_camera(
                    channel_camera,
                    channel_stream,
                    self.connection.exporter,
                    self._end_time,
                )
                if not data:
                    continue
                img, tracer, span, offset, original_img = data
            except Exception:
                continue

            if img.size == 0:
                continue

            with tracer.span(name="predict_tiffany") as s:
                results = self.detector.predict(img)
                obj_annot = self.detector.to_annotation(results, offset)

            if obj_annot is not None:
                annot_msg = ObjectAnnotations(
                    objects=[obj_annot],
                    resolution=Resolution(height=720, width=1280),
                    frame_id=self.connection.camera_id,
                )

                msg = Message()
                msg.pack(annot_msg)
                msg.topic = f"Tiffany.{self.connection.camera_id}.Keypoints"
                msg.inject_tracing(span)

                channel_stream.publish(msg)

                if self.stream_active:
                    try:
                        self.vis_queue.put_nowait(
                            {
                                "img": original_img,
                                "annot": annot_msg,
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

    def _visualization_worker(self):
        """Thread persistente que consome da fila e desenha."""
        channel = StreamChannel(self.connection.broker_uri)

        while True:
            try:
                item = self.vis_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            img = item["img"]
            annot = item["annot"]
            span = item["span"]

            img_drawn = self._draw_annotations(img, annot)

            jpeg_img = to_image(img_drawn, compression_level=0.8)

            msg = Message()
            msg.pack(jpeg_img)
            msg.topic = f"Tiffany.Keypoints.{self.connection.camera_id}.Frame"
            msg.inject_tracing(span)
            try:
                channel.publish(msg)
            except Exception as e:
                self.log.error(f"Stream error: {e}")
                try:
                    channel = StreamChannel(self.connection.broker_uri)
                except Exception:
                    time.sleep(1)

    def _draw_annotations(self, img, annot):
        """Função auxiliar pura para desenhar (movemos a lógica pra cá para limpar a classe)."""
        img_to_draw = img.copy()
        obj = annot.objects[0]
        kp = obj.keypoints

        if len(kp) >= 2:
            p1 = (int(kp[0].position.x), int(kp[0].position.y))
            p2 = (int(kp[1].position.x), int(kp[1].position.y))

            if obj.region and len(obj.region.vertices) >= 2:
                bb1 = (int(obj.region.vertices[0].x), int(obj.region.vertices[0].y))
                bb2 = (int(obj.region.vertices[1].x), int(obj.region.vertices[1].y))
                cv2.rectangle(img_to_draw, bb1, bb2, (255, 255, 0), 2)

            cv2.circle(img_to_draw, p1, 3, (0, 255, 0), -1)
            cv2.circle(img_to_draw, p2, 3, (0, 0, 255), -1)

            cv2.putText(
                img_to_draw,
                f"Original: {kp[0].score:.2f} | Shift: {(kp[0].score - 0.99) * 100:.2f}",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_to_draw,
                f"Original: {kp[1].score:.2f} | Shift: {(kp[0].score - 0.99) * 100:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img_to_draw,
                f"{obj.score:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        return img_to_draw

    def init_stream(self, seconds: Duration, ctx) -> Status:
        if not self.stream_active:
            if not self.detection_event.is_set():
                self.init_detection(seconds, ctx)

            self.stream_active = True
            return Status(StatusCode.OK, "Stream started.")
        return Status(StatusCode.ALREADY_EXISTS, "Stream active.")

    def init_detection(self, seconds: Duration, ctx) -> Status:
        self._end_time = max(self._end_time, time.time() + seconds.seconds)
        channel = StreamChannel(self.connection.broker_uri)
        subscription = Subscription(channel)
        request = Message(content=seconds, reply_to=subscription)
        channel.publish(
            request,
            topic=f"Tiffany.Detection.{self.connection.camera_id}.StartDetection",
        )
        channel.close()
        if not self.detection_event.is_set():
            t = threading.Thread(target=self.detection_thread, name="DetectionThread")
            t.start()
            return Status(StatusCode.OK, "Started.")

        return Status(StatusCode.ALREADY_EXISTS, "Extended time.")

    def stop(self, *args) -> Status:
        self.stream_active = False
        self._end_time = 0.0
        return Status(StatusCode.OK, "Stopping.")
