import os

from classes import Connection, Detector, Threading
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Status


def main() -> None:
    broker_uri = os.environ.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = os.environ.get("zipkin_uri", "http://10.10.2.211:30200")

    camera_id = int(os.getenv("CAMERA_ID", 1))

    service_name = f"Tiffany.{camera_id}.Keypoints"

    c = Connection(broker_uri, zipkin_uri, camera_id, service_name)
    provider = c.provider

    detector = Detector("models/orientation_model.pt", device="cuda")
    threading_instance = Threading(c, detector)
    provider.delegate(
        topic=f"Tiffany.Keypoints.{camera_id}.GetDetection",
        function=threading_instance.get_last_message,
        request_type=Struct,
        reply_type=(Struct, ObjectAnnotations),
    )
    provider.delegate(
        topic=f"Tiffany.Keypoints.{camera_id}.StartStream",
        function=threading_instance.init_stream,
        request_type=Duration,
        reply_type=Status,
    )
    provider.delegate(
        topic=f"Tiffany.Keypoints.{camera_id}.StartDetection",
        function=threading_instance.init_detection,
        request_type=Duration,
        reply_type=Status,
    )

    provider.run()


if __name__ == "__main__":
    main()
