import os

import numpy as np
from classes import Connection, Threading
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.common_pb2 import Pose
from is_wire.core import Status


def main() -> None:
    broker_uri = os.environ.get("broker_uri", "amqp://guest:guest@10.10.2.211:30000")
    zipkin_uri = os.environ.get("zipkin_uri", "http://10.10.2.211:30200")

    service_name = "Tiffany.Pose"

    c = Connection(broker_uri, zipkin_uri, service_name)
    provider = c.provider

    parameters = {
        1: dict(np.load("calibrations/calib_rt1.npz")),
        2: dict(np.load("calibrations/calib_rt2.npz")),
        3: dict(np.load("calibrations/calib_rt3.npz")),
        4: dict(np.load("calibrations/calib_rt4.npz")),
    }
    threading_instance = Threading(c, parameters)
    provider.delegate(
        topic="Tiffany.GetPose",
        function=threading_instance.get_last_message,
        request_type=Struct,
        reply_type=(Pose, Struct),
    )
    provider.delegate(
        topic="Tiffany.StartDetections",
        function=threading_instance.start_detections,
        request_type=Duration,
        reply_type=Status,
    )
    provider.run()


if __name__ == "__main__":
    main()
