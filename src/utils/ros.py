import rclpy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Pose2D, Point2D, ObjectHypothesis
import torch


def create_header(clock) -> Header:
    """
    Creates a ROS 2 Header with the current timestamp.
    
    :param clock: rclpy.clock.Clock instance to get the current time
    :returns: Header with the current timestamp
    """
    header = Header()
    header.stamp = clock
    return header


def create_detection_msg(img_msg: Image, detections: torch.Tensor, clock) -> Detection2DArray:
    """
    Converts detection results to a ROS 2 Detection2DArray message.
    
    :param img_msg: Original ROS 2 image message
    :param detections: Torch tensor of shape [num_boxes, 6] with each element being
                        [x1, y1, x2, y2, confidence, class_id]
    :param clock: rclpy.clock.Clock instance to get the current time
    :returns: ROS 2 Detection2DArray message containing the detections
    """
    detection_array_msg = Detection2DArray()


    # Header
    header = create_header(clock)
    detection_array_msg.header = header


    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        single_detection_msg = Detection2D()
        single_detection_msg.header = header


        # Source image
        # single_detection_msg.source_img = img_msg


        # Bounding box
        bbox = BoundingBox2D()
        w = float(round(x2 - x1))
        h = float(round(y2 - y1))
        cx = float(round(x1 + w / 2))
        cy = float(round(y1 + h / 2))
        bbox.size_x = w
        bbox.size_y = h


        bbox.center = Pose2D()
        bbox.center.position = Point2D()
        bbox.center.position.x = cx
        bbox.center.position.y = cy


        single_detection_msg.bbox = bbox


        # Class id & confidence
        obj_hyp = ObjectHypothesisWithPose()
        obj_hyp.hypothesis = ObjectHypothesis()
        obj_hyp.hypothesis.class_id = str(cls)
        obj_hyp.hypothesis.score = float(conf)
        single_detection_msg.results = [obj_hyp]


        detection_array_msg.detections.append(single_detection_msg)


    return detection_array_msg
