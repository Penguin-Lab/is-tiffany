import cv2
import numpy as np
from is_msgs.image_pb2 import Image

_DEFAULT_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]


def to_image(
    image: np.ndarray,
    encode_format: str = ".jpeg",
    compression_level: float = 0.8,
) -> Image:
    """
    Versão otimizada para streaming.
    AVISO: Removemos suporte a PNG propositalmente para evitar latência.
    """

    if image is None or image.size == 0:
        return Image()

    if compression_level == 0.8:
        params = _DEFAULT_JPEG_PARAMS
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * 100)]

    success, encoded_image = cv2.imencode(".jpg", image, params)

    if not success:
        return Image()

    return Image(data=encoded_image.tobytes())
