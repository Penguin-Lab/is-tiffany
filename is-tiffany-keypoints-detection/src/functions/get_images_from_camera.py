import os
import time

import numpy as np
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Channel, Message, Subscription, Tracer

from classes import Connection

from .to_np import to_np

CONFIDENCE = float(os.environ.get("confidence", 0.5))


def get_images_from_camera(connection: Connection, end_time: float, channel_camera):
    """
    Obtains the cropped image (ROI) from the camera detection.

    Args:
        connection (Connection): Connection object containing the channels and the exporter.
        end_time (float): The time at which the function should stop trying to get images.
        channel_camera (StreamChannel): Channel to get images
    Returns:
        Tuple containing:
            - crop (np.ndarray): Cropped region of interest (ROI) image from the camera.
            - tracer (Tracer): Tracer object for distributed monitoring.
            - span (BlankSpan): Active trace span for the operation.
            - roi_offset (np.ndarray): Coordinates (x1, y1) of the top-left corner of the ROI in the original image.
            - original_img (np.ndarray): Full original image from the camera.
    """
    channel_detection = Channel(connection.broker_uri)
    exporter = connection.exporter
    camera_id = connection.camera_id

    subscription = Subscription(channel_detection)
    request = Struct()
    request.fields["timestamp"].bool_value = True
    msg = Message(reply_to=subscription)
    msg.pack(request)

    while time.time() < end_time:
        obj = channel_camera.consume_last()
        if not obj:
            continue

        img_msg = obj.unpack(Image)
        original_img = to_np(img_msg)

        try:
            channel_detection.publish(
                msg, topic=f"Tiffany.Detection.{camera_id}.GetDetection"
            )
            reply = channel_detection.consume(timeout=1.0)
            dict_reply = json_format.MessageToDict(reply.unpack(Struct))
            det = json_format.ParseDict(
                dict_reply.get("detection", {}), ObjectAnnotations()
            )
            timestamp = dict_reply.get("timestamp", time.time())

        except Exception as e:
            print(f"[WARN] Error getting detection: {e}")
            continue

        if det.objects:
            score: float = det.objects[0].score
            if score < CONFIDENCE:
                continue

            box = det.objects[0].region.vertices
            x1, y1 = int(box[0].x), int(box[0].y)
            x2, y2 = int(box[1].x), int(box[1].y)

            tracer = Tracer(exporter=exporter, span_context=reply.extract_tracing())
            span = tracer.start_span(name="tiffany_keypoints_detection")

            crop = original_img[y1:y2, x1:x2]
            roi_offset = np.array([x1, y1])

            channel_detection.close()
            return crop, tracer, span, roi_offset, original_img, timestamp

    channel_detection.close()
    return (
        np.ndarray(0),
        Tracer(),
        Tracer().start_span(),
        np.ndarray(0),
        np.ndarray(0),
        0,
    )
