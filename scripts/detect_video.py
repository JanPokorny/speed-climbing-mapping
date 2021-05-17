import tensorflow as tf
import os
import cv2
import numpy as np
import time
import argparse
from object_detection.utils import visualization_utils, label_map_util

parser = argparse.ArgumentParser(
    description="Run inference on video using the exported model."
)

parser.add_argument(
    "-o",
    "--video_out_dir",
    help="enable video export to directory",
    type=str,
    default=None,
)

parser.add_argument(
    "-O",
    "--data_out_dir",
    help="enable data export to directory",
    type=str,
    default=None,
)

parser.add_argument(
    "-p", "--preview", help="enable live preview", action="store_true", default=False
)

parser.add_argument(
    "-t",
    "--threshold",
    help="detection threshold, from interval <0.0, 1.0)",
    type=float,
    default=0.5,
)

parser.add_argument(
    "-m",
    "--saved_model_dir",
    help="saved model directory (containing saved_model.pb)",
    type=str,
    default="models/10k/export/saved_model",
)

parser.add_argument(
    "-l",
    "--label_map",
    help="label map file (label_map.pbtxt)",
    type=str,
    default="config/label_map.pbtxt",
)

parser.add_argument(
    "input_video", help="video to detect", type=str, default=None, nargs="+"
)

args = parser.parse_args()


def time_fn(label, fn):
    print(label + "... ", end="", flush=True)
    start_time = time.time()
    value = fn()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(" [{:.2f}s]".format(elapsed_time))
    return value


detect_fn = time_fn("Loading model", lambda: tf.saved_model.load(args.saved_model_dir))
category_index = label_map_util.create_category_index_from_labelmap(
    args.label_map, use_display_name=True
)

for input_video in args.input_video:
    print(f"Processing video {input_video}")
    video_in = cv2.VideoCapture(input_video)
    video_out = None

    frame_index = 0
    while True:
        success, frame = video_in.read()

        if not success:
            break

        if args.video_out_dir is not None and video_out is None:
            height, width, layers = frame.shape
            video_out = cv2.VideoWriter(
                os.path.join(args.video_out_dir, os.path.basename(input_video)),
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (width, height),
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = tf.convert_to_tensor(frame_rgb)[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )

        if args.preview or args.video_out_dir is not None:
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections["detection_boxes"],
                detections["detection_classes"],
                detections["detection_scores"],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=args.threshold,
                line_thickness=4,
                agnostic_mode=False,
            )

        if args.preview:
            cv2.imshow("Object detector", frame)

        if args.video_out_dir is not None:
            video_out.write(frame)
        
        if args.data_out_dir is not None:
            np.savetxt(
                fname=os.path.join(args.data_out_dir, os.path.basename(input_video) + f".{frame_index:05d}.txt"),
                X=np.hstack((detections["detection_boxes"], detections["detection_scores"][np.newaxis].T))
            )

        if cv2.waitKey(1) == ord("q"):
            break

        frame_index += 1

    video_in.release()
    if video_out is not None:
        video_out.release()
    cv2.destroyAllWindows()
