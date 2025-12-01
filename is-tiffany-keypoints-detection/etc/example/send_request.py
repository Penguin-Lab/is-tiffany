import socket
import time

from google.protobuf import json_format
from google.protobuf.duration_pb2 import Duration
from google.protobuf.struct_pb2 import Struct
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, Message, Subscription

# Connect to broker
channel = Channel("amqp://guest:guest@10.10.2.211:30000")
subscription = Subscription(channel)

# Start detection stream
camera_id = 3
for id in range(1, 5):
    request = Message(content=Duration(seconds=3600), reply_to=subscription)
    channel.publish(request, topic=f"Tiffany.Keypoints.{id}.StartStream")

try:
    reply = channel.consume(timeout=5.0)
    print("[OK] Stream started. Status:", reply.status)
except socket.timeout:
    print("[WARN] No reply for StartStream")
    exit()

time.sleep(5)  # Wait a bit before requesting detections

# Request detection with Struct
struct = Struct()
struct.fields["timestamp"].bool_value = True
request = Message(reply_to=subscription)
request.pack(struct)
channel.publish(request, topic=f"Tiffany.Keypoints.{camera_id}.GetDetection")

try:
    reply = channel.consume(timeout=5.0)
    data = json_format.MessageToDict(reply.unpack(Struct))
    print("\n[OK] Detection with Struct:")
    print("Full reply as dict:", data)
    # Convert detection field to ObjectAnnotations
    detection = json_format.ParseDict(data.get("detection", {}), ObjectAnnotations())
    print("→ Parsed detection:", detection)

except socket.timeout:
    print("[WARN] No reply for Struct request")

# Request detection directly (ObjectAnnotations only)
request = Message(reply_to=subscription)
channel.publish(request, topic=f"Tiffany.Keypoints.{camera_id}.GetDetection")

try:
    reply = channel.consume(timeout=5.0)
    obj = reply.unpack(ObjectAnnotations)
    print("\n[OK] Detection without Struct (ObjectAnnotations):")
    print(obj)
except socket.timeout:
    print("[WARN] No reply for ObjectAnnotations request")
