import time

import numpy as np
from is_msgs.image_pb2 import Image
from is_wire.core import Message, Tracer
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter

from .to_np import to_np

_EMPTY_RET = (np.array([], dtype=np.uint8), None, None)


def get_images_from_camera(
    channel_camera,
    exporter: ZipkinExporter,
    end_time: float,
):
    """
    Versão otimizada com timeout curto e tratamento robusto de mensagens.
    """
    while time.time() < end_time:
        ret = channel_camera.consume_last(timeout=0.1)

        if isinstance(ret, bool):
            continue
        else:
            message = ret

        if not message:
            continue

        tracer = Tracer(exporter=exporter, span_context=message.extract_tracing())

        span = tracer.start_span(name="detection_process")

        with tracer.span(name="unpack_image"):
            image_proto = message.unpack(Image)
            image_np = to_np(image_proto)

            return image_np, tracer, span

    return _EMPTY_RET
