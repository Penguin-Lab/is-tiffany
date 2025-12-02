from typing import Union

import cv2
import numpy as np
from is_msgs.image_pb2 import Image

_EMPTY_ARRAY = np.array([], dtype=np.uint8)


def to_np(input_image: Union[np.ndarray, Image]) -> np.ndarray:
    """
    Versão otimizada: Remove verificação redundante de canais.
    cv2.IMREAD_COLOR já garante BGR.
    """
    if isinstance(input_image, np.ndarray):
        return input_image

    if isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)

        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        if output_image is None:
            return _EMPTY_ARRAY

        return output_image

    return _EMPTY_ARRAY
