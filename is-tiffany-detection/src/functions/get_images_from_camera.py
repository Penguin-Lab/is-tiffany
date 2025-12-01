import time

import numpy as np
from classes import StreamChannel
from is_msgs.image_pb2 import Image
from is_wire.core import Tracer
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter

from .to_np import to_np


def get_images_from_camera(
    channel_camera: StreamChannel,
    exporter: ZipkinExporter,
    end_time: float,
):
    """Consumes the most recent image from a channel and prepares distributed tracing.

    Args:
        channel_camera (StreamChannel): The channel from which the image will be consumed.
        exporter (ZipkinExporter): The Zipkin exporter used to create the tracer.
        end_time (float): The time at which the function should stop trying to get images.

    Returns:
        image, tracer, span: The image as a NumPy array, the Tracer object, and the Span.
    """
    while time.time() < end_time:
        message = channel_camera.consume_last(timeout=1.0)

        if isinstance(message, bool) or isinstance(message, tuple):
            continue

        tracer: Tracer = Tracer(
            exporter=exporter, span_context=message.extract_tracing()
        )
        span = tracer.start_span(name="tiffany_detection")
        with tracer.span(name="get_and_unpack_image_from_camera"):
            image_proto = message.unpack(Image)
            image_np = to_np(image_proto)
            return image_np, tracer, span
    return np.ndarray(0), Tracer(), Tracer().start_span()
