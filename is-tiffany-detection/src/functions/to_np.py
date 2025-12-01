from typing import Union

import cv2
import numpy as np
from is_msgs.image_pb2 import Image


def to_np(input_image: Union[np.ndarray, Image]) -> np.ndarray:
    """Converts an image to an OpenCV-compatible NumPy array.

    This utility function ensures that the input image, whether already a
    NumPy array or a Protobuf `Image` message, is returned as a decoded
    NumPy array ready for OpenCV processing.

    Args:
        input_image (Union[np.ndarray, Image]): The input image, which can be
            either a Protobuf `Image` or a NumPy array.

    Returns:
        np.ndarray: The image in NumPy format (BGR). Returns the input array
            unchanged if it's already a NumPy array, or an empty array if the
            input type is unsupported or decoding fails.
    """
    if isinstance(input_image, np.ndarray):
        return input_image

    if isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
        if output_image is None:
            return np.array([], dtype=np.uint8)
        # Convert grayscale to BGR if necessary
        if len(output_image.shape) == 2:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        return output_image

    return np.array([], dtype=np.uint8)
