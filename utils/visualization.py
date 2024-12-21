import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import time

def plot_contact_sequence(data, output_dir):
    labels = ['LF', 'RF', 'LB', 'RB']

    fig, axs = plt.subplots(4, 1, figsize=(8, 2))
    axs = axs.flatten()

    for i in range(4):
        axs[i].bar(range(len(data)), data[:, i], width=1, align='edge', color='b', alpha=0.7)
        axs[i].set_ylabel(labels[i])
        axs[i].set_ylim(-0.1, 1.1)

        axs[i].tick_params(axis='y', which='both', length=0, labelbottom=False, labelleft=False)
        axs[i].tick_params(axis='x', which='both', length=0, labelbottom=False)

        plt.subplots_adjust(hspace=0)

        plt.savefig(f"{output_dir}/contact_sequence.png")

def plot_video_with_animation(data, keypoints, video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
    else:
        video_width0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 4)
    ax2.set_xticks([])
    ax2.set_yticks([])

    y_positions = np.array([0.5, 1.5, 2.5, 3.5])

    video_img = ax1.imshow(np.zeros((480, 640, 3)), animated=True, aspect='auto')

    ax1.set_xticks([])
    ax1.set_yticks([])

    red_dots = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_frame_index = 0
    video_playing = True

    def update(frame):
        nonlocal video_frame_index, video_playing

        if not video_playing:
            return

        for dot in red_dots:
            dot.remove()

        current_frame = data[frame]
        red_dots.clear()
        keypoints_to_mark = [False, False, False, False]

        for i in range(4):
            if current_frame[i] == 1:
                dot = ax2.plot(1, y_positions[i], 'ro', markersize=50)
                red_dots.append(dot[0])
                keypoints_to_mark[i] = True

        ret, frame_video = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)

            keypoint_0 = keypoints[video_frame_index][0]
            keypoint_1 = keypoints[video_frame_index][1]
            keypoint_2 = keypoints[video_frame_index][2]
            keypoint_3 = keypoints[video_frame_index][3]

            keypoint_0 = (keypoint_0[0], keypoint_0[1])
            keypoint_1 = (keypoint_1[0], keypoint_1[1])
            keypoint_2 = (keypoint_2[0], keypoint_2[1])
            keypoint_3 = (keypoint_3[0], keypoint_3[1])

            keypoint_0 = tuple(
                [int(keypoint_0[1] * video_width / video_width0), int(keypoint_0[0] * video_height / video_height0)]
            )
            keypoint_1 = tuple(
                [int(keypoint_1[1] * video_width / video_width0), int(keypoint_1[0] * video_height / video_height0)]
            )
            keypoint_2 = tuple(
                [int(keypoint_2[1] * video_width / video_width0), int(keypoint_2[0] * video_height / video_height0)]
            )
            keypoint_3 = tuple(
                [int(keypoint_3[1] * video_width / video_width0), int(keypoint_3[0] * video_height / video_height0)]
            )

            if keypoints_to_mark[0]:
                cv2.circle(frame_rgb, keypoint_0, 5, (0, 255, 0), -1)
            if keypoints_to_mark[1]:
                cv2.circle(frame_rgb, keypoint_1, 5, (255, 0, 0), -1)
            if keypoints_to_mark[2]:
                cv2.circle(frame_rgb, keypoint_2, 5, (0, 0, 255), -1)
            if keypoints_to_mark[3]:
                cv2.circle(frame_rgb, keypoint_3, 5, (0, 255, 255), -1)

            video_img.set_array(frame_rgb)

            video_frame_index += 1
        else:
            print("End of video stream or file")
            video_playing = False

        time.sleep(1 / frame_rate)
    
    ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / frame_rate, repeat=False)

    labels = ['LF', 'RF', 'LB', 'RB']
    for i in range(4):
        ax2.text(0.8, y_positions[i], labels[i], fontsize=12, verticalalignment='center', horizontalalignment='right')

    fig.tight_layout()

    plt.show()

    cap.release()
    return

if __name__ == '__main__':
    ...
