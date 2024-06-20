from collections import Counter
from typing import Dict

from PIL import Image
import matplotlib.pyplot as plt
import torch


class ImageAnalysor:
    def __init__(self, model_name: str = "yolov5s") -> None:
        self.available_models = [f"yolov5{i}" for i in ["l", "m", "n", "s", "x"]]
        self.errors = ""
        self.results = ""
        self.model = self._set_model(model_name)

    def _set_model(self, model_name):
        if model_name in self.available_models:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name)
        else:
            self.errors = "Model not found. :("
            model = ""
        return model

    def analyse_image(self, image_path: str) -> None:
        try:
            img = Image.open(image_path)
            if self.model and not self.errors:
                self.results = self.model(img)
        except FileNotFoundError:
            self.errors = "Image not found. :("
        except Exception as e:
            self.errors = str(e)

    def get_result(self) -> str:
        if self.errors:
            return self.errors
        else:
            object_recognized = self._prepare_output()
            objects = [
                f"- {object_recognized[obj] }, {obj}\n\t"
                for obj in object_recognized.keys()
            ]
            message = f"""
Image analysis results:
    * Number of objects detected: {len(object_recognized)}
    * Objects detected:
        {"".join(objects)}
""".strip()
            return message

    def _prepare_output(self) -> Dict:
        detections = self.results.pred[0]
        recognized_objects = []
        for *_, _, cls in detections:
            recognized_objects.append(self.results.names[int(cls)])

        class_counts = Counter(recognized_objects)
        return class_counts
