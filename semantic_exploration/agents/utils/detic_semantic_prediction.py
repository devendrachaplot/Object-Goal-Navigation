# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import pathlib
import sys
import time
from pathlib import Path

import detectron2.data.transforms as T
import numpy as np
import torch

ROOT_DETIC = str(Path(__file__).resolve().parent).split("third_party")[0]+"third_party/"
sys.path.insert(0, ROOT_DETIC + "Detic/third_party/CenterNet2")
sys.path.insert(0, ROOT_DETIC + "Detic")
from centernet.config import add_centernet_config  # noqa: E402
from third_party.semantic_exploration.constants import coco_categories_mapping  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data.catalog import MetadataCatalog  # noqa: E402
from detectron2.engine.defaults import DefaultPredictor  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402
from detectron2.utils.visualizer import ColorMode, Visualizer  # noqa: E402
from detic.config import add_detic_config  # noqa: E402
from detic.modeling.text.text_encoder import build_text_encoder  # noqa: E402
from detic.modeling.utils import reset_cls_test  # noqa: E402

BUILDIN_CLASSIFIER = {
    "lvis": ROOT_DETIC + "Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": ROOT_DETIC + "Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": ROOT_DETIC + "Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": ROOT_DETIC + "Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}


class SemanticPredDetic:
    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.args = args

    def get_prediction(self, img):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2
        )

        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros(
            (img.shape[0], img.shape[1], 16 + 1)
        )  # self.args.num_sem_categories )) #15 + 1))

        for j, class_idx in enumerate(
            seg_predictions[0]["instances"].pred_classes.cpu().numpy()
        ):
            if class_idx in list(coco_categories_mapping.keys()):
                idx = coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]["instances"].pred_masks[j] * 1.0
                semantic_input[:, :, idx] += obj_mask.cpu().numpy()
        # The shape of the semantic input is (480, 640, 17)
        return semantic_input, img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.0] = i + 1
    return c_map


class ImageSegmentation:
    def __init__(self, args):
        string_args = """
            --config-file {}
            --input input1.jpeg
            --vocabulary coco
            --confidence-threshold {}
            --opts MODEL.WEIGHTS {}
            """.format(
            ROOT_DETIC + "/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            args.sem_pred_prob_thr,
            ROOT_DETIC + "/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        )

        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        assert args.vocabulary in ["coco", "custom"]
        if args.vocabulary == "custom":
            raise NotImplementedError
        elif args.vocabulary == "coco":
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]
            self.categories_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
                60: 6,  # table
                69: 7,  # oven
                71: 8,  # sink
                72: 9,  # refrigerator
                73: 10,  # book
                74: 11,  # clock
                75: 12,  # vase
                41: 13,  # cup
                39: 14,  # bottle
            }

        self.num_sem_categories = len(self.categories_mapping)
        num_classes = len(self.metadata.thing_classes)
        self.instance_mode = ColorMode.IMAGE
        self.demo = VisualizationDemo(cfg, classifier, num_classes)

    def get_predictions(self, img, visualize=0):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    cfg = get_cfg()
    # We forcefully use cpu here
    cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
        ROOT_DETIC + "Detic/" + cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    # Fix cfg paths given we're not running from the Detic folder
    cfg.MODEL.TEST_CLASSIFIERS[0] = (
        ROOT_DETIC + "Detic/" + cfg.MODEL.TEST_CLASSIFIERS[0]
    )
    cfg.MODEL.TEST_CLASSIFIERS[1] = (
        ROOT_DETIC + "Detic/" + cfg.MODEL.TEST_CLASSIFIERS[1]
    )
    cfg.freeze()
    return cfg


class VisualizationDemo(object):
    def __init__(self, cfg, classifier, num_classes, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = BatchPredictor(cfg)

        if type(classifier) == pathlib.PosixPath:
            classifier = str(classifier)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode
            )
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances
                    )

        return all_predictions, vis_output


def get_seg_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
