import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easy_ViTPose.easy_ViTPose.inference import VitInference
from utils.logger import logger

ViTPOSE_CPT = f"./easy_ViTPose/checkpoints/vitpose-h-ap10k.pth"
MODEL_TYPE = "h"
DATASET = "ap10k"
SINGLE_POSE = True
YOLO_SIZE = 96

class default_args():
    yolo_step = 1
    save_img = False
    save_video = True
    show_yolo = False
    show_raw_yolo = False
    conf_threshold = 0
    save_json = False
    is_video=True

def plot_keypoints(privileged_keypoints, output_dir, h):
    _, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(0, len(privileged_keypoints))
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    cmap = plt.get_cmap('viridis')

    for i, legs in enumerate(privileged_keypoints):
        for j, (y, x, _) in enumerate(legs):
            color = cmap(j / len(legs))
            ax.scatter(i, y, s=10, c=color)

    plt.savefig(f'{output_dir}/kp.png', bbox_inches='tight')

def detect_keypoints(video_path, output_dir):
    args = default_args()
    model = VitInference(ViTPOSE_CPT, MODEL_TYPE, DATASET,
                            YOLO_SIZE, is_video=args.is_video,
                            single_pose=SINGLE_POSE,
                            yolo_step=args.yolo_step)

    keypoints = []
    privileged_keypoints = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        logger.error("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream or file")
            break
        frame_shape = np.array(frame.shape[:2])
        
        frame_keypoints = model.inference(frame)
        keypoints.append(frame_keypoints)
        privileged_keypoint = []
        for i, kp in enumerate(frame_keypoints[0]):
            if i in [7, 10, 13, 16]:
                privileged_keypoint.append(kp)
        privileged_keypoints.append(privileged_keypoint)

    cap.release()

    plot_keypoints(privileged_keypoints, output_dir, frame_shape[0])

    np.save(f'{output_dir}/keypoints.npy', keypoints)
    np.save(f'{output_dir}/privileged_keypoints.npy', privileged_keypoints)
    np.save(f'{output_dir}/frame_shape.npy', frame_shape)

    return np.array(keypoints), np.array(privileged_keypoints), fps, frame_shape

if __name__ == '__main__':
    ...