import socket
import time
import json

from google.protobuf import json_format 
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.common_pb2 import Pose
from is_wire.core import Channel, Message, Subscription

# Connect to broker
channel = Channel("amqp://guest:guest@10.10.2.211:30000")
subscription = Subscription(channel)

# Start detections
request = Message(content=Duration(seconds=3600), reply_to=subscription)
channel.publish(request, topic="Tiffany.StartDetections")

try:
    reply = channel.consume(timeout=5.0)
    print("[OK] Started. Status:", reply.status)
except socket.timeout:
    print("[WARN] No reply for StartDetections")
    exit()

# Wait 30 seconds before requesting detections
time.sleep(30)

struct = Struct()
struct.fields["timestamp"].bool_value = True
struct.fields["num_cameras"].bool_value = True
request = Message(reply_to=subscription)
request.pack(struct)

# Collect 20 successful measurements and save to a file (JSON lines)
output_file = "measurements.jsonl"
collected = 0
with open(output_file, "w") as f:
    while collected < 20:
        channel.publish(request, topic="Tiffany.GetPose")
        try:
            reply = channel.consume(timeout=1.0)
            obj = reply.unpack(Struct)
            data = json_format.MessageToDict(obj)
            # substituir o timestamp pelo tempo de ida e volta (latência)
            data["timestamp"] = time.time() - data["timestamp"]
            f.write(json.dumps(data) + "\n")
            collected += 1
            print(f"[OK] Saved measurement {collected}")
            time.sleep(0.1)
        except socket.timeout:
            print("[WARN] No reply")
