import os
import time

import numpy as np
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Tracer

from .to_np import to_np

CONFIDENCE = float(os.environ.get("confidence", 0.5))
_EMPTY_RET = (
    np.ndarray(0),
    Tracer(),
    Tracer().start_span(),
    np.ndarray(0),
    np.ndarray(0),
)


def get_images_from_camera(
    channel_camera, channel_detection, exporter, end_time: float
):
    """
    Obtains the cropped image (ROI) optimized for latency.
    """

    while time.time() < end_time:
        obj = channel_camera.consume_last(0.1)
        if not obj:
            continue

        reply = channel_detection.consume_last(0.1)
        if not reply:
            continue

        det = reply.unpack(ObjectAnnotations)

        if not det.objects or det.objects[0].score < CONFIDENCE:
            continue

        box = det.objects[0].region.vertices
        x1, y1 = int(box[0].x), int(box[0].y)
        x2, y2 = int(box[1].x), int(box[1].y)

        img_msg = obj.unpack(Image)
        original_img = to_np(img_msg)

        h, w = original_img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        tracer = Tracer(exporter=exporter, span_context=reply.extract_tracing())
        span = tracer.start_span(name="tiffany_keypoints_crop")

        crop = original_img[y1:y2, x1:x2]
        roi_offset = np.array([x1, y1])

        return crop, tracer, span, roi_offset, original_img

    return _EMPTY_RET
