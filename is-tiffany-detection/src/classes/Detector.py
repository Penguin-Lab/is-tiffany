from typing import Optional

import numpy as np
from is_msgs.image_pb2 import BoundingPoly, ObjectAnnotation, Vertex
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Detector:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.model = YOLO(model_path)
        self.model.to(device)

    def predict(self, img: np.ndarray) -> Results:
        """
        Runs prediction with Half Precision (FP16) enabled if on GPU.
        """
        results = self.model.predict(source=img, imgsz=640, verbose=False, half=False)
        return results[0]

    def to_annotation(self, result: Results) -> Optional[ObjectAnnotation]:
        """
        Converts YOLO results directly to IS-Msgs ObjectAnnotation.
        Optimized for Bulk Data Transfer (GPU -> CPU).
        """
        if len(result.boxes) == 0:
            return None

        box_data = result.boxes.data.cpu().numpy()[0]

        x1, y1, x2, y2 = box_data[0:4]
        conf = float(box_data[4])

        return ObjectAnnotation(
            label="Tiffany",
            id=0,
            score=conf,
            region=BoundingPoly(
                vertices=[
                    Vertex(x=x1, y=y1),
                    Vertex(x=x2, y=y2),
                ]
            ),
        )
