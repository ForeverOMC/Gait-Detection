import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

def lowpass_filter(arr, length=5):
    b = np.ones(length) / length
    a = 1
    filtered_arr = scipy.signal.filtfilt(b, a, arr)

    return filtered_arr


def pre_process(privileged_keypoint, length=5):
    filtered_keypoint = lowpass_filter(privileged_keypoint, length)

    # 异常值矫正
    n = len(privileged_keypoint)
    for i in range(1, n - 1):
        if abs(filtered_keypoint[i] - privileged_keypoint[i]) > 5:
            privileged_keypoint[i] = filtered_keypoint[i]

    filtered_keypoint = lowpass_filter(privileged_keypoint, length)

    return filtered_keypoint


def find_extrema(arr):
    max_extremas = []
    min_extremas = []
    n = len(arr)

    # 图像坐标系所以相反
    for i in range(n):
        if (i == 0 or arr[i] <= arr[i - 1]) and (i == n - 1 or arr[i] <= arr[i + 1]):
            max_extremas.append([i, arr[i]])
        if (i == 0 or arr[i] >= arr[i - 1]) and (i == n - 1 or arr[i] >= arr[i + 1]):
            min_extremas.append([i, arr[i]])

    return np.array(max_extremas), np.array(min_extremas)


def cal_period_length(max_extremas, min_extremas):
    start_idx = 1
    end_idx = len(min_extremas) - 1
    offset = 0

    if max_extremas[0][0] > min_extremas[0][0]:
        offset = -1

    period_lengths = []
    for i in range(start_idx, end_idx):
        period_lengths.append(abs(max_extremas[i+offset][1] - min_extremas[i][1]) +
                               abs(max_extremas[i+1+offset][1] - min_extremas[i][1]))

    period_lengths.sort()
    min_length, max_length = period_lengths[0], period_lengths[-1]
    period_lengths = [length for length in period_lengths 
                      if length > (max_length - min_length) / 2 + min_length]
    
    period_length = sum(period_lengths) / len(period_lengths)

    return period_length


def filter_extrema(max_extremas, min_extremas, period_length):
    filtered_max_extremas_idx = np.ones(len(max_extremas))
    filtered_min_extremas_idx = np.ones(len(min_extremas))

    start_idx = 0
    end_idx = len(min_extremas)
    offset = 0
    if max_extremas[0][0] > min_extremas[0][0]:
        filtered_min_extremas_idx[0] = 0
        start_idx = 1
        offset = -1
    if max_extremas[-1][0] < min_extremas[-1][0]:
        filtered_min_extremas_idx[-1] = 0
        end_idx = end_idx - 1

    for i in range(start_idx, end_idx):
        length = (abs(max_extremas[i+offset][1] - min_extremas[i][1]) +
                   abs(max_extremas[i+1+offset][1] - min_extremas[i][1]))
        if length < period_length * 0.5:
            if max_extremas[i+offset][1] < min_extremas[i][1]:
                filtered_max_extremas_idx[i+offset] = 0
            else:
                filtered_max_extremas_idx[i+1+offset] = 0
            filtered_min_extremas_idx[i] = 0

    filtered_max_extremas = max_extremas[filtered_max_extremas_idx.astype(bool)]
    filtered_min_extremas = min_extremas[filtered_min_extremas_idx.astype(bool)]

    return np.array(filtered_max_extremas), np.array(filtered_min_extremas)


def contact_detect(privileged_keypoint, min_extremas, preiod_length, threshold=0.001):
    n = len(privileged_keypoint)
    contact_idx = np.zeros(n)

    for i in range(len(min_extremas)):
        start_idx = end_idx = min_extremas[i][0].astype(int)
        while start_idx > 0 and abs(privileged_keypoint[start_idx] - min_extremas[i][1]) < preiod_length * threshold:
            start_idx -= 1
        while end_idx < n - 1 and abs(privileged_keypoint[end_idx] - min_extremas[i][1]) < preiod_length * threshold:
            end_idx += 1
        contact_idx[start_idx:end_idx+1] = 1

    return contact_idx


def post_process(privileged_keypoints, threshold=0.1):
    ...


def detect_gait(privileged_keypoints, fps, output_dir):
    contact_arr = np.zeros((len(privileged_keypoints), 4))
    length = 5
    foot_idx = 0
    again = False
    while foot_idx < 4:
        privileged_keypoint = privileged_keypoints[:, foot_idx, 0]

        privileged_keypoint = pre_process(privileged_keypoint, length)

        max_extremas, min_extremas = find_extrema(privileged_keypoint)
        period_length = cal_period_length(max_extremas, min_extremas)
        max_extremas, min_extremas = filter_extrema(max_extremas, min_extremas, period_length)

        n, m = len(min_extremas), len(privileged_keypoint) / fps
        f = m / n
        length_ = 3 + int((f - 1/2) / (1/4))
        if length_ != length and not again:
            length = length_
            again = True
            continue
        again = False

        contact_idx = contact_detect(privileged_keypoint, min_extremas, period_length)

        contact_arr[:, foot_idx] = contact_idx
        foot_idx += 1

    np.save(f'{output_dir}/contact_arr.npy', contact_arr)
    plot_contact_sequence(contact_arr, output_dir)
    return contact_arr


def plot_contact_sequence(contact_arr, output_dir):
    num_frames, num_feet = contact_arr.shape
    
    fig, axes = plt.subplots(num_feet, 1, figsize=(12, 6), sharex=True)

    for foot in range(num_feet):
        x = np.arange(num_frames)
        axes[foot].plot(x, contact_arr[:, foot], drawstyle='steps-post', label=f'Foot {foot + 1}')
        
        axes[foot].set_title(f'Contact Sequence for Foot {foot + 1}')
        axes[foot].set_ylim(-0.1, 1.1)
        axes[foot].set_yticks([0, 1])
        axes[foot].legend()
        axes[foot].grid()

    axes[-1].set_xlabel('Frames')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/contact_sequence_.png', dpi=300)


if __name__ == '__main__':
    pass
    # task = 'horse2'
    # contact_arr = detect_gait(task)
    # print(contact_arr)
    # plot_contact_sequence(contact_arr)