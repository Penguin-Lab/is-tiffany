# is-tiffany-detection

> Real-time object detection service for Tiffany, the hexapod robot, part of LabSEA's the Intelligent Space ('is') ecosystem.


## Overview

**is-tiffany-detection** is a detection microservice designed to process video feeds from external cameras (e.g., CCTV) and provide real-time detection results from **Tiffany**, a hexapod robot operating within the **Intelligent Space (is)** framework.

While Tiffany itself is a mobile hexapod robot, the detection service processes images streamed from fixed cameras deployed in the environment, allowing us to know where Tiffany is.

This project combines PyTorch-based object detection models with the `is-wire` messaging middleware to provide:

- Continuous object detection from camera feeds
- RPC interfaces to start/stop detection and streaming
- Distributed tracing with Zipkin support
- Scalable deployment with Docker and Kubernetes

---

## Architecture


- The camera gateway captures images from fixed CCTV cameras and publishes them on an `is-wire` stream.
- The detection service consumes these images, runs detection models, and publishes annotated detection results.

---

## Features

- GPU-accelerated detection using PyTorch (`ultralytics`/YOLO-based model)
- Consumes images from camera gateways in real-time
- Thread-safe detection and streaming management
- RPC interface to control detection and stream durations dynamically
- Supports distributed tracing with Zipkin to monitor performance
- Easily deployable with Docker and Kubernetes manifests

---

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU
- RabbitMQ or AMQP broker for `is-wire` messaging
- Zipkin server for tracing (optional)

Install dependencies with:

```bash
pip install -r requirements.txt
```
## Usage
### Running Locally
Set environment variables:
```bash
export BROKER_URI="amqp://guest:guest@10.10.2.211:30000"
export ZIPKIN_URI="http://10.10.2.211:30200"
export CAMERA_ID=1
```

Run the service:
```bash
python src/main.py
```

### RPC Endpoints

`Tiffany.Detection.{camera_id}.GetDetection`
Returns the latest object detection result from the specified camera as an `ObjectAnnotations` protobuf.

`Tiffany.Detection.{camera_id}.StartStream`
Starts streaming object detections from the specified camera for a given duration (in minutes, `FloatValue`). Returns a `Status` message indicating success or failure.

`Tiffany.Detection.{camera_id}.StartDetection`
Starts continuous object detection on the specified camera for a given duration (in minutes, `FloatValue`). Returns a `Status` message indicating success or failure.

#### Example: Sending RPC Requests
Use the example script to start a stream and fetch detections:
```bash
python etc/example/send_request.py
```
This script:

1. Sends a request to start a detection stream.

2. Waits for the stream to initialize.

3. Requests the latest detection result.

4. Prints the detection status and annotations.

#### Expected Output:
```json
RPC Status: { code=StatusCode.OK why='Stream started' }

RPC Status: { code=StatusCode.OK why='' }
Detection: objects {
  label: "Tiffany"
  score: 0.79606918
  region {
    vertices {
      x: 157.02948
      y: 593.1647
    }
    vertices {
      x: 184.01685
      y: 625.41174
    }
  }
}
resolution {
  height: 720
  width: 1280
}
frame_id: 1
```

## Deployment
### Docker
Pull the Docker image:
```bash
docker pull miguelgrigorio27/is-tiffany-detection:latest
```
Or build your own Docker image:
```bash
docker build -t yourusername/is-tiffany-detection:latest -f etc/docker/Dockerfile .
```
Run the container:
```bash
docker run --gpus all -e BROKER_URI=$BROKER_URI -e ZIPKIN_URI=$ZIPKIN_URI -e CAMERA_ID=$CAMERA_ID yourusername/is-tiffany-detection:latest
```

### Kubernetes
Apply the Kubernetes manifests in `etc/k8s`:
```bash
kubectl apply -f etc/k8s/ConfigMap.yaml
kubectl apply -f etc/k8s/Service.yaml
```
The StatefulSet deploys 4 replicas, each assigned a unique camera ID from the pod hostname ordinal index.


## About Tiffany and Intelligent Space ('is')

Tiffany is a hexapod robot designed for autonomous navigation and interaction within an intelligent environment.

The Intelligent Space (is) ecosystem provides middleware and infrastructure to integrate multiple robotic and sensor components, enabling a distributed intelligent system.

The `is-tiffany-detection` service fits into this ecosystem as the vision perception module, interpreting environmental camera data to inform Tiffany's actions.
