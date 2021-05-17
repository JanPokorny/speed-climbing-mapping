import gc
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import sys
import argparse
import sklearn
import sklearn.base
import sklearn.cluster
import sklearn.preprocessing
import sklearn.linear_model
import pycpd
import scipy
import scipy.spatial.distance
import scipy.optimize
import scipy.interpolate
import scipy.signal
import csaps
from dataclasses import dataclass
from object_detection.utils import label_map_util

np.set_printoptions(precision=3)
np.seterr(invalid='ignore')

parser = argparse.ArgumentParser(
    description="Map a speed climbing video to a reference wall."
)

parser.add_argument(
    "-o",
    "--out_dir",
    help="output directory",
    type=str,
    required=True,
)

parser.add_argument(
    "-v",
    "--save_video",
    help="enable saving of evaluation video",
    action="store_true",
    default=False
)

parser.add_argument(
    "-d",
    "--save_data",
    help="enable saving of absolute transformations -- for further use, load by numpy.loadtxt('file.txt.gz').reshape((-1,3,3))",
    action="store_true",
    default=False
)

parser.add_argument(
    "-p",
    "--save_preview",
    help="enable saving of evaluation video preview",
    action="store_true",
    default=False
)

parser.add_argument(
    "-l",
    "--save_log",
    help="enable saving of log.jsonl",
    action="store_true",
    default=False
)

parser.add_argument(
    "--detection_threshold",
    help="detection threshold, from interval <0.0, 1.0)",
    type=float,
    default=0.5,
)

parser.add_argument(
    "--cleaning_eps",
    help="eps for the cleaning step (removing outliers)",
    type=float,
    default=0.3,
)

parser.add_argument(
    "--saved_model_dir",
    help="saved model directory (containing saved_model.pb)",
    type=str,
    default="models/20k/export/saved_model",
)

parser.add_argument(
    "--min_track_length",
    help="remove tracks shorter than this",
    type=int,
    default=3,
)

parser.add_argument(
    "--csaps_smoothing",
    help="smoothing coefficient for cubic splines",
    type=float,
    default=0.005,
)

parser.add_argument(
    "--degrees_of_freedom",
    help="degrees of freedom for the final transformation",
    type=int,
    default=8,
    choices=[2, 4, 6, 8]
)

parser.add_argument(
    "input_video", help="video to detect", type=str, default=None, nargs="+"
)

args = parser.parse_args()


def time_fn(label, fn):
    print(label + "... ", end="", flush=True, file=sys.stderr)
    start_time = time.time()
    value = fn()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(" [{:.2f}s]".format(elapsed_time), flush=True, file=sys.stderr)
    return value


def compute_transform_matrix(src_points, dst_points, points_of_freedom):
    points_of_freedom = min(len(src_points), points_of_freedom)
    if points_of_freedom >= 4:
        transformation, inliers = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
        )
        if transformation is None:
            return compute_transform_matrix(src_points, dst_points, points_of_freedom=3)
    elif points_of_freedom == 3:
        transformation, inliers = cv2.estimateAffine2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
        )
        if transformation is None:
            return compute_transform_matrix(src_points, dst_points, points_of_freedom=2)
        transformation = np.vstack((transformation, np.float32([[0, 0, 1]])))
    elif points_of_freedom == 2:
        transformation, inliers = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
        )
        if transformation is None:
            return compute_transform_matrix(src_points, dst_points, points_of_freedom=1)
        transformation = np.vstack((transformation, np.float32([[0, 0, 1]])))
    elif points_of_freedom == 1:
        average_delta = np.median(dst_points - src_points, axis=0)
        transformation = np.float32(
            [[1, 0, average_delta[0]], [0, 1, average_delta[1]], [0, 0, 1]]
        )
        inliers = np.ones((len(src_points), 1))
    elif points_of_freedom == 0:
        transformation = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        inliers = np.ones((len(src_points), 1))
    return transformation, inliers


def transform_point(point, transformation):
    projected_hpoint = np.matmul(transformation, np.float32([*point, 1]))
    return np.float32(projected_hpoint[:-1]) / projected_hpoint[-1]


def associate_points(XA, XB, point_distance_threshold):
    if len(XA) == 0 or len(XB) == 0:
        return {}

    distance_matrix = scipy.spatial.distance.cdist(XA=XA, XB=XB)
    full_matrix = np.hstack(
        (distance_matrix, np.tile(point_distance_threshold, distance_matrix.shape))
    )
    return {
        a_point: (b_point if b_point < len(XB) else -1)
        for (a_point, b_point) in zip(
            *scipy.optimize.linear_sum_assignment(full_matrix)
        )
    }


def associate_points(XA, XB, point_distance_threshold):  # TODO
    if len(XA) == 0 or len(XB) == 0:
        return {}

    distance_matrix = scipy.spatial.distance.cdist(XA=XA, XB=XB)
    full_matrix = np.hstack(
        (distance_matrix, np.tile(point_distance_threshold, distance_matrix.shape))
    )
    return {
        a_point: (b_point if b_point < len(XB) else -1)
        for (a_point, b_point) in zip(
            *scipy.optimize.linear_sum_assignment(full_matrix)
        )
    }


def good_transformation(transformation, eps, translation_eps=np.inf):
    if transformation is None:
        return False
    sentinel = np.float32(
        [[eps, eps, translation_eps], [eps, eps, translation_eps], [eps, eps, eps]]
    )
    return np.all(np.abs(transformation - np.identity(3)) < sentinel)


class HoldDetector:
    def __init__(self):
        self.detect_fn = tf.saved_model.load(args.saved_model_dir)
        self.category_index = label_map_util.create_category_index_from_labelmap(
            "config/label_map.pbtxt", use_display_name=True
        )

    def raw_detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        return (
            detections["detection_boxes"],
            detections["detection_classes"],
            detections["detection_scores"],
        )

    def detect(self, frame, min_score):
        boxes, classes, scores = self.raw_detect(frame)
        height, width, layers = frame.shape
        return [
            # box order: top, left, bottom, right
            Detection(
                np.array(
                    [box[0] * height, box[1] * width, box[2] * height, box[3] * width]
                )
                .round()
                .astype(int)
            )
            for i, box in enumerate(boxes)
            if scores[i] >= min_score
        ]


@dataclass
class Detection:
    next_id = 0

    def __init__(self, box, fake=False):
        self.box = box
        self.point = np.array([((box[3] + box[1]) / 2), ((box[2] + box[0]) / 2)])
        self.fake = fake
        self.projected_point = self.point
        self.next_detection = None
        self.previous_detection = None
        self.color = (0, 0, 0)
        self.enabled = True
        self.root = None
        self.id = None
        self.label = None

    def track_length(self):
        if self.next_detection is None:
            return 1
        else:
            return self.next_detection.track_length() + 1

    def disable_track(self):
        self.enabled = False
        if self.next_detection:
            self.next_detection.disable_track()

    def draw(self, frame):
        # draw point
        point = (round(self.point[0]), round(self.point[1]))
        if self.fake:
            radius = 7
            thickness = 2
        else:
            radius = 0
            thickness = 15
        cv2.circle(
            img=frame,
            center=point,
            radius=radius,
            color=self.color,
            thickness=thickness,
        )

        # draw label
        cv2.putText(
            img=frame,
            text=str(self.id),
            org=(point[0] + 20, point[1]),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.color,
        )

        # draw projection
        if self.projected_point is not None:
            projection_center = (
                round(self.projected_point[0]),
                round(self.projected_point[1]),
            )
            cv2.arrowedLine(
                img=frame,
                pt1=point,
                pt2=projection_center,
                color=(255, 255, 255),
                thickness=2,
                tipLength=0.5,
            )

    def draw_box(self, frame):
        box = np.round(self.box)

        cv2.rectangle(
            img=frame,
            rec=(box[1], box[0], box[3] - box[1], box[2] - box[0]),
            color=(0, 0, 0),
            thickness=2,
        )


@dataclass
class DetectionFrame:
    def __init__(self, frame):
        self.frame = frame
        self.previous_detection_frame = None
        self.next_detection_frame = None
        self.detections = None
        self.relative_transformation = np.identity(3)
        self.accumulated_transformation = np.identity(3)
        self.transformation = np.identity(3)
        self.is_keyframe = True

    def detect_holds(self, hold_detector, min_score):
        self.detections = hold_detector.detect(self.frame, min_score)
        self.box_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        for d in self.detections:
            self.box_mask[d.box[0] : (d.box[2] + 1), d.box[1] : (d.box[3] + 1)] = 1

    def project_detections(self, max_distance=70):
        if not self.detections:
            return

        points = np.float32([d.point for d in self.detections])
        projected_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.frame,
            self.next_detection_frame.frame,
            points,
            None,
        )
        deltas = [
            delta
            for delta in (projected_points - points)[np.where(status == 1)[0]]
            if np.linalg.norm(delta) <= max_distance
        ]
        if len(deltas) == 0:
            estimated_velocity = np.float32([0, 0])
        else:
            estimated_velocity = np.median(deltas, axis=0).flatten()
        for detection in self.detections:
            detection.projected_point = detection.point + estimated_velocity

    def associate_detections(self, point_distance_threshold):
        if (
            self.next_detection_frame is None
            or not self.next_detection_frame.detections
        ):
            return
        reference_mapping = associate_points(
            XA=np.array([d.point for d in self.detections]),
            XB=np.array([d.point for d in self.next_detection_frame.detections]),
            point_distance_threshold=point_distance_threshold,
        )
        for detection_id, next_detection_id in reference_mapping.items():
            if next_detection_id != -1:
                detection = self.detections[detection_id]
                next_detection = self.next_detection_frame.detections[next_detection_id]
                detection.next_detection = next_detection
                next_detection.previous_detection = detection

    def calculate_relative_transformation(self, points_of_freedom):
        relevant_detections = [
            d
            for d in self.detections
            if d.next_detection is not None and not d.fake and not d.next_detection.fake
        ]
        src_points = np.float32([d.next_detection.point for d in relevant_detections])
        dst_points = np.float32([d.point for d in relevant_detections])
        self.relative_transformation, _ = compute_transform_matrix(
            src_points, dst_points, points_of_freedom=points_of_freedom
        )

    def calculate_relative_transformation_masked(
        self, points_of_freedom, point_distance_threshold=70
    ):
        if self.next_detection_frame is None:
            return

        src_points = cv2.goodFeaturesToTrack(
            image=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),
            maxCorners=0,
            qualityLevel=0.01,
            minDistance=0,
            mask=self.box_mask,
        )
        if src_points is None:
            return
        src_points = src_points.reshape(-1, 2)

        dst_points, flow_inliers, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=self.frame,
            nextImg=self.next_detection_frame.frame,
            prevPts=src_points,
            nextPts=None,
        )
        dst_points = dst_points.round().astype(int)

        flow_inlier_mask = np.array([x == 1 for x in flow_inliers[:, 0]], dtype=bool)
        for i, dst_point in enumerate(dst_points):
            if (
                np.any(dst_point) < 0
                or dst_point[1] >= self.next_detection_frame.box_mask.shape[0]
                or dst_point[0] >= self.next_detection_frame.box_mask.shape[1]
                or self.next_detection_frame.box_mask[dst_point[1], dst_point[0]] == 0
                or np.linalg.norm(dst_point - src_points[i]) > point_distance_threshold
            ):
                flow_inlier_mask[i] = False

        src_points = src_points[flow_inlier_mask]
        dst_points = dst_points[flow_inlier_mask]

        self.relative_transformation, transform_inliers = compute_transform_matrix(
            dst_points, src_points, points_of_freedom=points_of_freedom
        )

        transform_inlier_mask = np.array(
            [x == 1 for x in transform_inliers[:, 0]], dtype=bool
        )
        src_points = src_points[transform_inlier_mask]
        dst_points = dst_points[transform_inlier_mask]
        # for src_point, dst_point in zip(src_points, dst_points):
        #     cv2.arrowedLine(
        #         img=self.frame,
        #         pt1=tuple(src_point.round()),
        #         pt2=tuple(dst_point.round()),
        #         color=(255, 0, 0),
        #         thickness=2,
        #         tipLength=0.5,
        #     )

    def create_fake_detections(self):
        if self.next_detection_frame is None:
            return
        height, width, _ = self.frame.shape
        size = np.float32([width, height])
        zero = np.float32([0, 0])
        for detection in self.detections:
            if detection.next_detection is not None:
                continue
            fake_point = transform_point(detection.point, self.relative_transformation)
            if np.all(fake_point >= zero) and np.all(fake_point < size):
                fake_detection = Detection(fake_point, fake=True)
                self.next_detection_frame.detections.append(fake_detection)
                detection.next_detection = fake_detection
                fake_detection.previous_detection = detection
                fake_detection.projected_point = transform_point(
                    fake_detection.point,
                    self.next_detection_frame.relative_transformation,
                )
        self.next_detection_frame.associate_detections()

    def draw_detections(self):
        for detection in self.detections:
            detection.draw(self.frame)

    def draw_detection_boxes(self):
        for detection in self.detections:
            detection.draw_box(self.frame)

    def calculate_absolute_transformation(self, reference_points, points_of_freedom):
        relevant_detections = [d for d in self.detections if not d.fake and d.id != -1]

        if len(relevant_detections) < 3:
            self.transformation = np.tile(np.inf, (3, 3))
            return

        src_points = np.float32([d.point for d in relevant_detections])
        dst_points = reference_points[[d.id for d in relevant_detections]]
        self.transformation, _ = compute_transform_matrix(
            src_points, dst_points, points_of_freedom=points_of_freedom
        )

    def apply_absolute_transformation(self, reference_shape):
        self.frame = cv2.warpPerspective(
            src=self.frame, M=self.transformation, dsize=reference_shape
        )


class FrameManager:
    def __init__(self, reference_points, reference_shape):
        self.reference_points = reference_points
        self.reference_shape = reference_shape
        self.detection_frames = []
        self.palette = cv2.applyColorMap(
            np.arange(0, 255, dtype=np.uint8).reshape(1, 255, 1), cv2.COLORMAP_HSV
        ).squeeze(0)
        self.relative_view_transformation = np.identity(3)
        self.longest_gap = None

    def load_video(self, input_video):
        video_in = cv2.VideoCapture(input_video)
        self.fps = video_in.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = video_in.read()
            if not success:
                break
            self.detection_frames.append(DetectionFrame(frame))
        video_in.release()
        height, width, _ = self.detection_frames[0].frame.shape
        self.size = (width, height)
        for detection_frame, next_detection_frame in zip(
            self.detection_frames, self.detection_frames[1:]
        ):
            detection_frame.next_detection_frame = next_detection_frame
            next_detection_frame.previous_detection_frame = detection_frame

    def save_video(self, output_video):
        video_out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.size,
        )
        for detection_frame in self.detection_frames:
            video_out.write(detection_frame.frame)
        video_out.release()

    def save_preview(self, output_image):
        frames_dists = np.linspace(8, len(self.detection_frames) - 8, 3).astype(int)
        cv2.imwrite(
            output_image,
            np.amax([self.detection_frames[i].frame for i in frames_dists], axis=0),
        )

    def detect_holds(self, hold_detector, min_score):
        for detection_frame in self.detection_frames:
            detection_frame.detect_holds(hold_detector, min_score)
        self.num_detections = sum(
            len(frame.detections) for frame in self.detection_frames
        )

    def associate_detections(self, point_distance_threshold):
        for detection_frame in self.detection_frames[:-1]:
            detection_frame.project_detections()
            detection_frame.associate_detections(point_distance_threshold)

    def calculate_relative_transformations(self, points_of_freedom):
        for detection_frame in self.detection_frames:
            detection_frame.calculate_relative_transformation(points_of_freedom)

    def accumulate_relative_transformations(self):
        self.detection_frames[0].accumulated_transformation = np.float32(
            [
                [1, 0, 0],
                [0, 1, 4566 - self.detection_frames[0].frame.shape[1]],
                [0, 0, 1],
            ]
        )
        for detection_frame in self.detection_frames[1:]:
            detection_frame.accumulated_transformation = np.matmul(
                detection_frame.previous_detection_frame.relative_transformation,
                detection_frame.previous_detection_frame.accumulated_transformation,
            )

    def remove_short_tracks(self, min_length):
        for detection_frame in self.detection_frames:
            for detection in detection_frame.detections:
                if (
                    detection.previous_detection is None
                    and detection.track_length() < min_length
                ):
                    detection.disable_track()
        for detection_frame in self.detection_frames:
            detection_frame.detections = [
                d for d in detection_frame.detections if d.enabled
            ]

    def create_fake_detections(self):
        for detection_frame in self.detection_frames:
            detection_frame.create_fake_detections()

    def find_reference_mapping(self, point_distance_threshold):
        all_detection_points = np.array(
            [
                transform_point(
                    detection.point, detection_frame.accumulated_transformation
                )
                for detection_frame in self.detection_frames
                for detection in detection_frame.detections
            ]
        )

        _, (B, t) = pycpd.AffineRegistration(
            X=self.reference_points, Y=all_detection_points
        ).register()

        self.relative_view_transformation = np.array(
            [
                [B[0, 0], B[1, 0], t[0]],
                [B[0, 1], B[1, 1], t[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        for detection_frame in self.detection_frames:
            transformed_detection_points = np.array(
                [
                    transform_point(
                        detection.point,
                        np.matmul(
                            self.relative_view_transformation,
                            detection_frame.accumulated_transformation,
                        ),
                    )
                    for detection in detection_frame.detections
                ]
            )

            reference_mapping = associate_points(
                XA=transformed_detection_points,
                XB=self.reference_points,
                point_distance_threshold=point_distance_threshold,
            )

            for i, detection in enumerate(detection_frame.detections):
                detection.label = detection.id = reference_mapping.get(i, -1)

    def draw_detections(self):
        for detection_frame in self.detection_frames:
            detection_frame.draw_detections()

    def draw_detection_boxes(self):
        for detection_frame in self.detection_frames:
            detection_frame.draw_detection_boxes()

    def calculate_absolute_transformations(self, points_of_freedom):
        for detection_frame in self.detection_frames:
            detection_frame.calculate_absolute_transformation(
                self.reference_points, points_of_freedom
            )

    def clean_absolute_transformations(self, eps):
        n = len(self.detection_frames)
        for detection_frame in self.detection_frames:
            detection_frame.is_keyframe = good_transformation(
                detection_frame.transformation, eps=eps
            )

        prev_keyframe_distance = [None] * n
        next_keyframe_distance = [None] * n
        forward_transformations = [None] * n
        backward_transformations = [None] * n

        for i in range(n):
            if self.detection_frames[i].is_keyframe:
                prev_keyframe_distance[i] = 0
                forward_transformations[i] = self.detection_frames[i].transformation
            else:
                prev_keyframe_distance[i] = (
                    (prev_keyframe_distance[i - 1] + 1) if i > 0 else np.inf
                )
                forward_transformations[i] = np.matmul(
                    self.detection_frames[i - 1].relative_transformation,
                    forward_transformations[i - 1] if i > 0 else np.identity(3),
                )

        for i in reversed(range(n)):
            if self.detection_frames[i].is_keyframe:
                next_keyframe_distance[i] = 0
                backward_transformations[i] = self.detection_frames[i].transformation
            else:
                next_keyframe_distance[i] = (
                    (next_keyframe_distance[i + 1] + 1) if i < (n - 1) else np.inf
                )
                backward_transformations[i] = np.matmul(
                    np.linalg.inv(self.detection_frames[i].relative_transformation),
                    backward_transformations[i + 1] if i < (n - 1) else np.identity(3),
                )

        self.longest_gap = max(
            (dist for dist in next_keyframe_distance[1:] if dist != np.inf),
            default=np.inf,
        )

        for i in range(n):
            if self.detection_frames[i].is_keyframe:
                continue
            else:
                self.detection_frames[i].is_keyframe = False

            if prev_keyframe_distance[i] == np.inf:
                self.detection_frames[i].transformation = backward_transformations[i]
            elif next_keyframe_distance[i] == np.inf:
                self.detection_frames[i].transformation = forward_transformations[i]
            else:
                forward_ratio = next_keyframe_distance[i] / (
                    prev_keyframe_distance[i] + next_keyframe_distance[i]
                )
                self.detection_frames[i].transformation = forward_transformations[
                    i
                ] * forward_ratio + backward_transformations[i] * (1 - forward_ratio)

    def smoothen_absolute_transformations(
        self,
        smooth,
        initial_frames_count,
        initial_frames_weight,
        non_keyframes_weight,
    ):
        x = np.arange(len(self.detection_frames))
        y = np.array(
            [
                detection_frame.transformation
                for detection_frame in self.detection_frames
            ]
        )

        weights = np.ones(len(self.detection_frames))
        weights[
            [not d.is_keyframe for d in self.detection_frames]
        ] = non_keyframes_weight
        weights[:initial_frames_count] = initial_frames_weight

        yi = csaps.csaps(x, y, x, smooth=smooth, axis=0, weights=weights)
        # yi = scipy.signal.savgol_filter(x=y, window_length=49, polyorder=2, axis=0)

        for detection_frame, smoothed_matrix in zip(self.detection_frames, yi):
            detection_frame.transformation = smoothed_matrix

    def apply_absolute_transformations(self):
        for detection_frame in self.detection_frames:
            detection_frame.apply_absolute_transformation(self.reference_shape)
        self.size = self.reference_shape

    def draw_all_detections(self):
        for drawing_detection_frame in self.detection_frames:
            for detection_frame in self.detection_frames:
                for detection in detection_frame.detections:
                    transformed_point = transform_point(
                        detection.point,
                        np.matmul(
                            self.relative_view_transformation,
                            detection_frame.accumulated_transformation,
                        ),
                    )

                    color_map = [
                        (0, 0, 255),
                        (0, 255, 0),
                        (255, 0, 0),
                        (255, 0, 255),
                        (0, 255, 255),
                        (255, 255, 0),
                    ]

                    point = (round(transformed_point[0]), round(transformed_point[1]))
                    cv2.drawMarker(
                        drawing_detection_frame.frame,
                        point,
                        color=color_map[detection.id % 6]
                        if detection.id != -1
                        else (255, 255, 255),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        thickness=2,
                        markerSize=30,
                    )

    def draw_reference_points(self):
        for detection_frame in self.detection_frames:
            for i, point in enumerate(self.reference_points):
                circle_params = {
                    "img": detection_frame.frame,
                    "center": (round(point[0]), round(point[1])),
                    "radius": 0,
                }
                putText_params = {
                    "img": detection_frame.frame,
                    "text": str(i),
                    "org": (round(point[0]) + 20, round(point[1]) + 20),
                    "fontFace": cv2.FONT_HERSHEY_DUPLEX,
                    "fontScale": 2.0,
                }
                for (color, circle_thickness, text_thickness) in [
                    ((0, 0, 0), 30, 12),
                    ((255, 255, 255), 20, 2),
                ]:
                    cv2.circle(color=color, thickness=circle_thickness, **circle_params)
                    cv2.putText(color=color, thickness=text_thickness, **putText_params)

    def draw_keyframe_marks(self):
        for detection_frame in self.detection_frames:
            if not detection_frame.is_keyframe:
                continue
            cv2.circle(
                img=detection_frame.frame,
                center=(120, 120),
                color=(255, 255, 255),
                thickness=100,
                radius=0,
            )

    def get_debug_info(self):
        return {
            "total_detections": self.num_detections,
            # "valid_detections_ratio": sum(
            #     len(frame.detections) for frame in self.detection_frames
            # )
            # / self.num_detections,
            "used_detections_ratio": sum(
                detection.label != -1
                for frame in self.detection_frames
                for detection in frame.detections
            )
            / self.num_detections,
            "keyframe_ratio": sum(frame.is_keyframe for frame in self.detection_frames)
            / len(self.detection_frames),
            "first_keyframe_distance": next(
                (
                    i
                    for i, frame in enumerate(self.detection_frames)
                    if frame.is_keyframe
                ),
                np.inf,
            ),
            "last_keyframe_distance": next(
                (
                    i
                    for i, frame in enumerate(reversed(self.detection_frames))
                    if frame.is_keyframe
                ),
                np.inf,
            ),
            "longest_gap": self.longest_gap,
        }
    
    def save_absolute_transformations(self, filename):
        np.savetxt(filename, [
            d.transformation.reshape(9)
            for d in self.detection_frames
        ])


REFERENCE_POINTS = np.loadtxt("reference/wall_points.txt")
REFERENCE_SHAPE = (960, 4566)

def main():
    if not args.save_data and not args.save_video and not args.save_preview:
        parser.print_help()
        parser.exit("ERROR: please specify --save_data, --save_video or --save_preview")

    hold_detector = time_fn("Loading model", lambda: HoldDetector())

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "log.jsonl"), "w") as logfile:
        for input_video in args.input_video:
            output_video = os.path.join(
                args.out_dir, os.path.basename(input_video)
            )
            print(f"PROCESSING VIDEO: {input_video}", file=sys.stderr, flush=True)
            frame_manager = FrameManager(REFERENCE_POINTS, REFERENCE_SHAPE)
            time_fn("Loading video", lambda: frame_manager.load_video(input_video))
            time_fn(
                "Detecting holds",
                lambda: frame_manager.detect_holds(
                    hold_detector, min_score=args.detection_threshold
                ),
            )
            if args.save_video or args.save_preview:
                time_fn(
                    "Drawing detection boxes",
                    lambda: frame_manager.draw_detection_boxes(),
                )
            time_fn(
                "Associating detections",
                lambda: frame_manager.associate_detections(
                    point_distance_threshold=150
                ),
            )
            time_fn(
                "Removing short tracks",
                lambda: frame_manager.remove_short_tracks(min_length=args.min_track_length),
            )
            time_fn(
                "Calculating relative transformations",
                lambda: frame_manager.calculate_relative_transformations(
                    points_of_freedom=1
                ),
            )
            time_fn(
                "Accumulating relative transformations",
                lambda: frame_manager.accumulate_relative_transformations(),
            )
            
            time_fn(
                "Finding a mapping to reference points",
                lambda: frame_manager.find_reference_mapping(
                    point_distance_threshold=150
                ),
            )
            time_fn(
                "Calculating absolute transformations",
                lambda: frame_manager.calculate_absolute_transformations(
                    points_of_freedom=args.degrees_of_freedom//2
                ),
            )
            time_fn(
                "Cleaning absolute transformations",
                lambda: frame_manager.clean_absolute_transformations(eps=args.cleaning_eps),
            )
            time_fn(
                "Smoothing absolute transformations",
                lambda: frame_manager.smoothen_absolute_transformations(
                    smooth=args.csaps_smoothing,
                    initial_frames_count=12,
                    initial_frames_weight=100.0,
                    non_keyframes_weight=1.0,
                ),
            )
            if args.save_video or args.save_preview:
                time_fn(
                    "Transforming video",
                    lambda: frame_manager.apply_absolute_transformations(),
                )
                time_fn(
                    "Visualising detections", lambda: frame_manager.draw_all_detections()
                )
                time_fn(
                    "Visualising reference points",
                    lambda: frame_manager.draw_reference_points(),
                )

            if args.save_preview:
                time_fn(
                    "Saving preview",
                    lambda: frame_manager.save_preview(output_video + ".png"),
                )

            if args.save_video:
                time_fn("Marking keyframes", lambda: frame_manager.draw_keyframe_marks())
                time_fn(
                    "Saving video",
                    lambda: frame_manager.save_video(output_video),
                )

            if args.save_data:
                time_fn(
                    "Saving data",
                    lambda: frame_manager.save_absolute_transformations(output_video + ".txt.gz"),
                )

            if args.save_log:
                print(
                    {"file": input_video, **frame_manager.get_debug_info()},
                    flush=True,
                    file=logfile,
                )

            # Explicit resource release is necessary here
            del frame_manager
            time_fn("Reclaiming RAM", lambda: gc.collect())


main()
