import os
import sys
import argparse
import logging
from utils.logger import logger

from utils.keypoint_detection import detect_keypoints
from utils.gait_detection import detect_gait
from utils.visualization import plot_video_with_animation, plot_contact_sequence

def parse_args():
    parser = argparse.ArgumentParser(description='Contact sequence detection')
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--video_path', type=str, required=True, help='Video path')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='Output directory')
    parser.add_argument('--frame_rate', type=int, default=None, help='Frame rate')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

def main():
    args = parse_args()

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                filename=f'{args.output_dir}/log.log',
                filemode='a')

    logger.info(f"Task name: {args.task_name}, Video path: {args.video_path}")
    if not os.path.exists(args.video_path):
        logger.error(f"Video path {args.video_path} does not exist")
        sys.exit(1)

    keypoints, privileged_keypoints, fps, frame_shape = detect_keypoints(args.video_path, args.output_dir)
    if args.frame_rate is None:
        args.frame_rate = fps
    contact_arr, filtered_keypoints = detect_gait(privileged_keypoints, fps, args.output_dir)
    plot_contact_sequence(contact_arr, args.output_dir)
    plot_video_with_animation(contact_arr, privileged_keypoints, filtered_keypoints, args.video_path, args.output_dir, frame_rate=args.frame_rate)

    sys.exit(0)

if __name__ == '__main__':
    main()

