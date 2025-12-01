import cv2
import numpy as np
from is_msgs.image_pb2 import Image


def to_image(
    image: np.ndarray, encode_format: str = ".jpeg", compression_level: float = 1.0
) -> Image:
    """Encodes a NumPy image array into a Protobuf `Image` message.

    This function compresses an OpenCV image (NumPy array) into either JPEG or PNG format,
    then encapsulates the resulting bytes into an `Image` protobuf message compatible
    with the `is-msgs` ecosystem.

    Args:
        image (np.ndarray): Input image in OpenCV format (BGR or grayscale).
        encode_format (str): Desired encoding format. Supported values are ".jpeg" and ".png".
                             Defaults to ".jpeg".
        compression_level (float | int): Compression level from 0.0 (max compression, lowest quality)
                                         to 1.0 (min compression, highest quality). Defaults to 0.8.
                                         For PNG, this value is inverted internally
                                         since OpenCV uses 0 (no compression) to 9 (max compression).

    Returns:
        Image: A Protobuf `Image` message containing the compressed image data.
               Returns an empty `Image` message if the format is unsupported or encoding fails.
    """
    if encode_format in [".jpeg", ".jpg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * 100)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int((1.0 - compression_level) * 9)]
    else:
        return Image()

    success, encoded_image = cv2.imencode(encode_format, image, params)

    if not success:
        return Image()

    return Image(data=encoded_image.tobytes())
