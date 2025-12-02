from typing import Optional

import numpy as np
from is_msgs.image_pb2 import BoundingPoly, ObjectAnnotation, PointAnnotation, Vertex
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Detector:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model = YOLO(model_path, task="pose")
        self.model.to(device)

    def predict(self, img: np.ndarray) -> Results:
        results = self.model.predict(source=img, imgsz=96, verbose=False, half=False)
        return results[0]

    def to_annotation(
        self, result: Results, offset: np.ndarray
    ) -> Optional[ObjectAnnotation]:
        """
        Converts YOLO results directly to IS-Msgs ObjectAnnotation.
        Optimized to minimize CPU-GPU sync points.
        """
        if len(result.boxes) == 0:
            return None

        boxes_data = result.boxes.data.cpu().numpy()

        kpts_data = result.keypoints.data.cpu().numpy()

        box = boxes_data[0]
        kps = kpts_data[0]

        x1 = box[0] + offset[0]
        y1 = box[1] + offset[1]
        x2 = box[2] + offset[0]
        y2 = box[3] + offset[1]
        score_box = float(box[4])

        kps[:, 0] += offset[0]
        kps[:, 1] += offset[1]

        return ObjectAnnotation(
            label="Tiffany",
            id=0,
            score=score_box,
            region=BoundingPoly(
                vertices=[
                    Vertex(x=x1, y=y1),
                    Vertex(x=x2, y=y2),
                ]
            ),
            keypoints=[
                PointAnnotation(
                    id=i,
                    score=float(kps[i, 2]),
                    position=Vertex(x=kps[i, 0], y=kps[i, 1]),
                )
                for i in range(len(kps))
            ],
        )
