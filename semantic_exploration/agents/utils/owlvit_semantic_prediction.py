# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import sys
from pathlib import Path
ROOT_DETIC = str(Path(__file__).resolve().parent).split("third_party")[0]+"third_party/"
sys.path.insert(0, ROOT_DETIC + "Detic/third_party/CenterNet2")
sys.path.insert(0, ROOT_DETIC + "Detic")

import argparse  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from third_party.semantic_exploration.constants import coco_categories, coco_categories_mapping  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import OwlViTForObjectDetection, OwlViTProcessor  # noqa: E402


class SemanticPredOwlvit:
    def __init__(self, config):
        self.config = config
        # Get the device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Get the owlvit model
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model.eval()
        self.model.to(self.device)
        # Define the prefix
        self.prefix="an image of "
        # Get the pretrained model
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        # Get the meta info
        labels = []
        for _key in coco_categories:
            labels.append(self.prefix+_key)
        self.labels = [labels]
        self.score_threshold = 0.15

    def get_prediction(self, img):
        img = img[:, :, ::-1]
        # Process inputs
        inputs = self.processor(text=self.labels, images=img, return_tensors="pt")
        target_sizes = torch.Tensor([img.shape[:2]])

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )

        # Process the image
        img_i = 0
        boxes, scores, labels = (
            results[img_i]["boxes"],
            results[img_i]["scores"],
            results[img_i]["labels"],
        )
        semantic_input = np.zeros((img.shape[0], img.shape[1], 16 + 1))
        for box, score, label in zip(boxes, scores, labels):
            # Get the location of the bounding box
            if score >= self.score_threshold:
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = [
                    int(round(i, 0)) for i in box.tolist()
                ]
                semantic_input[
                    top_left_x:bottom_right_x, top_left_y:bottom_right_y, int(label)
                ] = 1
            if self.config.VISUALIZE is True and score >= self.score_threshold:
                # Use this line code to add bounding box to the image
                img = np.ascontiguousarray(img, dtype=np.uint8)
                cv2.rectangle(
                    img,
                    (top_left_x, top_left_y),
                    (bottom_right_x, bottom_right_y),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img,
                    self.labels[0][int(label)],
                    (top_left_x, top_left_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        return semantic_input, img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.0] = i + 1
    return c_map
