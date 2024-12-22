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

def plot_video_with_animation(data, keypoints, filtered_keypoints, video_path, output_dir, frame_rate=15):
    
    color_map = [[(0, 255, 0), (0, 1, 0)], [(255, 0, 0), (1, 0, 0)], [(0, 0, 255), (0, 0, 1)], [(0, 255, 255), (0, 1, 1)]]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout()
    ax2.set_xlim(0, len(filtered_keypoints))
    ax2.set_ylim(0, video_height)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.legend()

    lines = [ax2.plot([], [], label=f'Point {i + 1}', c=color_map[i][1])[0] for i in range(4)]

    # for i, ax in enumerate([ax2]):
    #     ax.set_title(f"Keypoints Animation")
    #     ax.set_xlabel("Frame")

    video_img = ax1.imshow(np.zeros((480, 640, 3)), animated=True, aspect='auto')

    ax1.set_xticks([])
    ax1.set_yticks([])

    video_frame_index = 0

    def update(frame):
        nonlocal video_frame_index

        current_frame = data[frame]
        keypoints_to_mark = [False, False, False, False]

        for i in range(4):
            if current_frame[i] == 1:
                keypoints_to_mark[i] = True

        ret, frame_video = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)    
            for i in range(4):
                keypoint = keypoints[video_frame_index][i]
                keypoint = tuple([int(keypoint[1]), int(keypoint[0])])
                if keypoints_to_mark[i]:
                    cv2.circle(frame_rgb, keypoint, 5, color_map[i][0], -1)

            video_img.set_array(frame_rgb)
            video_frame_index += 1
        else:
            print("End of video stream or file")
            return

        for i, line in enumerate(lines):
            line.set_data(range(frame + 1), filtered_keypoints[:frame + 1, i])

        time.sleep(1 / frame_rate)

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / frame_rate, repeat=False)

    ani.save(f"{output_dir}/video_animation.gif", writer='imagemagick', fps=frame_rate)

    cap.release()
    
if __name__ == '__main__':
    ...
