#!/usr/bin/python3


from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.ros import create_detection_msg
from visualizer import draw_detections


import os
from typing import Tuple, Union, List


import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rclpy
from rclpy.node import Node

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes


def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    """Rescale the output to the original image shape
    :param ori_shape: original width and height [width, height].
    :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
        [num_boxes, >=4], where the first 4 entries of each element have to be
        [x1, y1, x2, y2].
    :param target_shape: target width and height [width, height].
    """
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]

    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale

    return boxes


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()


    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections




class Yolov7Publisher(Node):
    def __init__(self, img_topic: str, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, pub_topic: str = "yolov7_detections",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 queue_size: int = 1, visualize: bool = False,
                 class_labels: Union[List, None] = None):
        """
        :param img_topic: name of the image topic to listen to
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param pub_topic: name of the output topic (will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        :param class_labels: List of length num_classes, containing the class
            labels. The i-th element in this list corresponds to the i-th
            class id. Only for visualization. If it is None, then no class
            labels are visualized.
        """
        super().__init__('yolov7_node')


        self.img_size = img_size
        self.device = device
        self.class_labels = class_labels


        vis_topic = pub_topic + "visualization" if pub_topic.endswith("/") else \
            pub_topic + "/visualization"
        self.visualization_publisher = self.create_publisher(
            Image, vis_topic, queue_size
        ) if visualize else None


        self.bridge = CvBridge()


        self.tensorize = ToTensor()
        self.model = YoloV7(
            weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
            device=device
        )
        self.img_subscriber = self.create_subscription(
            Image, img_topic, self.process_img_msg, 10
        )
        self.detection_publisher = self.create_publisher(
            Detection2DArray, pub_topic, queue_size
        )
        self.get_logger().info("YOLOv7 initialization complete. Ready to start inference")
        self.counter = 0


    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )


        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)


        h_orig, w_orig, c = np_img_orig.shape


        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))


        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)


        # inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections[:, :4] = rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()


        # publishing
        detection_msg = create_detection_msg(img_msg, detections, self.get_clock().now().to_msg())
        self.detection_publisher.publish(detection_msg)

        # self.get_logger().info(str(detections))


        # visualizing if required
        # bboxes = [[int(x1), int(y1), int(x2), int(y2)]
        #             for x1, y1, x2, y2 in detections[:, :4].tolist()]
        # classes = [int(c) for c in detections[:, 5].tolist()]
        # vis_img = draw_detections(np_img_orig, bboxes, classes,
        #                             self.class_labels)
        # image_name = os.path.join("images", f"image_{self.counter:04}.png")
        # cv2.imwrite(image_name, vis_img)
        # self.counter += 1
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes,
                                      self.class_labels)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            self.visualization_publisher.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)


    node = rclpy.create_node('yolov7_node')


    weights_path = node.get_parameter_or('weights_path', 'weights/best.pt')
    classes_path = node.get_parameter_or('classes_path', 'class_labels/test.txt')
    img_topic = node.get_parameter_or('img_topic', '/camera')
    out_topic = node.get_parameter_or('out_topic', '/yolov7_detections')
    conf_thresh = node.get_parameter_or('conf_thresh', 0.5)
    iou_thresh = node.get_parameter_or('iou_thresh', 0.45)
    queue_size = node.get_parameter_or('queue_size', 1)
    img_size = node.get_parameter_or('img_size', 640)
    visualize = node.get_parameter_or('visualize', True)
    device = node.get_parameter_or('device', 'cuda')

    node.get_logger().info(weights_path)
    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError(f"Weights not found ({weights_path}).")


    if classes_path:
        if not os.path.isfile(classes_path):
            raise FileExistsError(f"Classes file not found ({classes_path}).")
        classes = parse_classes_file(classes_path)
    else:
        node.get_logger().info("No class file provided. Class labels will not be visualized.")
        classes = None


    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")


    yolov7_publisher = Yolov7Publisher(
        img_topic=img_topic,
        pub_topic=out_topic,
        weights=weights_path,
        device=device,
        visualize=visualize,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=(img_size, img_size),
        queue_size=queue_size,
        class_labels=classes
    )


    rclpy.spin(yolov7_publisher)
    yolov7_publisher.destroy_node()
    rclpy.shutdown()




if __name__ == "__main__":
    main()
