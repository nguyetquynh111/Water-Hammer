from vessel_segmentation_pytorch import DenseNet121
import matplotlib.pyplot as plt
import cv2
import os
import re
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
import math
from typing import List, Tuple
import torch
import gdown
import tempfile

# Global parameters and functions
WINDOW_SIZE = 20
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

MAX_VECTORS = 90
MAX_FRAMES = 75
SPLIT_FRAMES = 15
OVERLAY = 5
ITEMS = int((MAX_FRAMES - OVERLAY) / (SPLIT_FRAMES - OVERLAY)) + 1

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chmod(OUTPUT_DIR, 0o777)

def download_model_from_gdrive():
    """Download the model from Google Drive if it doesn't exist locally."""
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id=1yyRfJ0SpAEsn--ZLqMcDt9AtlwYqFDUP"
        gdown.download(url, model_path, quiet=False)
    return model_path

device = "cpu"
model = DenseNet121().to(device)
# Download and load model
model_path = download_model_from_gdrive()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def predict_image(model, cv_img, device, img_size=(512, 512)):
    """
    Same as above but skips PIL/transforms altogether.
    cv_img must be a numpy array H×W or H×W×3 (BGR).
    """
    # 1) gray + resize
    gray = cv2.cvtColor(
        cv_img, cv2.COLOR_BGR2GRAY) if cv_img.ndim == 3 else cv_img
    H1, W1 = img_size
    resized = cv2.resize(gray, (W1, H1), interpolation=cv2.INTER_AREA)

    # 2) to tensor [0,1]
    x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    x = x.to(device)

    # 3) predict & binarize
    model.eval()
    with torch.no_grad():
        out = model(x)                     # (1,1,H1,W1)
        mask = (out > 0.5).float().cpu().squeeze()  # (H1,W1), 0.0 or 1.0

    # 4) to 0/255 uint8
    mask_255 = (mask.numpy() * 255).astype(np.uint8)

    # 5) match: mask * resized image
    matched = cv2.bitwise_and(resized, resized, mask=mask_255)

    return mask_255, matched


def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour


def remove_catheter(image):
    original_image = image.copy()
    vessel_img, _ = predict_image(model, original_image, device)

    # remove catheter
    subtract_image = vessel_img
    _, binary = cv2.threshold(subtract_image, 50, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    if len(contours) > 1:
        mask = np.zeros_like(binary)
        try:
            cv2.drawContours(mask, [largest_contour], -
                             1, 255, thickness=cv2.FILLED)
            vessel_img = cv2.bitwise_and(
                subtract_image, subtract_image, mask=mask)
        except:
            print("fail here")

    resized_img = cv2.resize(original_image, (512, 512))
    pred_img = vessel_img/255.
    match_img = resized_img*pred_img
    return pred_img, match_img

# Sort windows


def find_catheter_last_point(catheter_img):
    width, height = np.where(catheter_img != 0)
    if len(width) == 0 or len(height) == 0:
        return 0, 0
    max_width = np.max(width)
    max_height = np.max(height)
    return max_height, max_width


def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def min_distance(x, y, vector, limit):
    if limit > 0:
        d = [distance(x, y, vector[i][0], vector[i][1]) for i in range(limit)]
        d.sort()
        return d[0]
    return 1000


def vectorize_one_image_using_center_line(img, previous_path=None):
    vector = np.zeros((MAX_VECTORS, 3), dtype=np.float32)
    STEP = 5

    pred_img, match_img = remove_catheter(img)

    centerline = skeletonize(pred_img.astype(int))
    centerline = centerline.astype(np.float32)

    if np.all(pred_img == 0):
        print("Fall here")
        return vector, None, None, match_img, pred_img

    try:
        img_with_rectangles = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
    except:
        img_with_rectangles = match_img.copy()
    centerline_with_rect = cv2.cvtColor(centerline * 255, cv2.COLOR_GRAY2BGR)
    index = 0
    WS12 = WINDOW_SIZE // 2
    IMAGE_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH)

    for y in range(0, IMAGE_DIM[0], STEP):
        for x in range(0, IMAGE_DIM[1], STEP):
            window = centerline[y:y + STEP, x:x + STEP]
            if np.count_nonzero(window) > 2:
                y_arr, x_arr = np.where(window == 1)
                x_w = int(x_arr.mean()) + x
                y_w = int(y_arr.mean()) + y

                if min_distance(x_w, y_w, vector, index) > WS12*0.1:
                    upper_left = (max(0, x_w - WS12), max(0, y_w - WS12))
                    lower_right = (min(IMAGE_WIDTH, x_w + WS12),
                                   min(IMAGE_HEIGHT, y_w + WS12))

                    if (lower_right[0] - upper_left[0]) <= 0 or (lower_right[1] - upper_left[1]) <= 0:
                        continue

                    window = match_img[upper_left[1]                                       :lower_right[1], upper_left[0]:lower_right[0]]
                    centerline[upper_left[1]:lower_right[1],
                               upper_left[0]:lower_right[0]] = 0

                    cv2.rectangle(img_with_rectangles, upper_left,
                                  lower_right, (0, 255, 0), 1)
                    cv2.rectangle(centerline_with_rect, upper_left,
                                  lower_right, (0, 255, 0), 1)

                    pixel_count = np.average(window)
                    vector[index] = [x_w, y_w, pixel_count]
                    index += 1
                    if index >= MAX_VECTORS:
                        vector = sort_by_distance(vector, pred_img)
                        return vector, img_with_rectangles, centerline_with_rect, match_img, pred_img

    vector = sort_by_distance(vector, pred_img, previous_path)
    return vector, img_with_rectangles, centerline_with_rect, match_img, pred_img

# Sort vectors


def most_left_upper_point(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    most_left_upper = points[0]

    for point in points[1:]:
        if point[0] == 0 and point[1] == 0:
            continue
        if point[0] < most_left_upper[0] or (point[0] == most_left_upper[0] and point[1] < most_left_upper[1]):
            most_left_upper = point

    return most_left_upper


def calculate_angle(point1, point2, point3):
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    cosine_theta = dot_product / (magnitude1 * magnitude2)
    cosine_theta = max(-1, min(1, cosine_theta))
    theta_rad = math.acos(cosine_theta)

    theta_deg = math.degrees(theta_rad)

    return theta_deg


def vector_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    center_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return max(0, center_distance - WINDOW_SIZE)


def calculate_ratio(previous_point: Tuple[float, float], current_point: Tuple[float, float], next_point: Tuple[float, float], image) -> float:
    if previous_point[0] != 0 and previous_point[1] != 0:
        angle = calculate_angle(previous_point, current_point, next_point)
    else:
        angle = 0

    line_overlapping = calculate_line_overlapping(
        image, next_point[:2], current_point[:2])
    return 0.7*angle + 0.3*line_overlapping


def get_line_points(image, point1, point2):
    blank_image = np.zeros_like(image)
    point1 = point1.astype(int)
    point2 = point2.astype(int)
    cv2.line(blank_image, (point1[0], point1[1]),
             (point2[0], point2[1]), 255, 1)
    line_points = np.column_stack(np.where(blank_image == 255))
    return line_points


def calculate_line_overlapping(binary_image, point1, point2):
    line_points = get_line_points(binary_image, point1, point2)
    overlap_count = 0
    total_line_points = len(line_points)
    for point in line_points:
        x, y = point
        if binary_image[x, y] != 0:
            overlap_count += 1
    overlap_percentage = (overlap_count / total_line_points) * 100
    return overlap_percentage


def find_overlap_points(x1, y1, x2, y2):
    x1_tl = x1 - WINDOW_SIZE // 2
    y1_tl = y1 - WINDOW_SIZE // 2
    x1_br = x1 + WINDOW_SIZE // 2
    y1_br = y1 + WINDOW_SIZE // 2

    x2_tl = x2 - WINDOW_SIZE // 2
    y2_tl = y2 - WINDOW_SIZE // 2
    x2_br = x2 + WINDOW_SIZE // 2
    y2_br = y2 + WINDOW_SIZE // 2

    # Calculate the overlapping region
    xmin = max(x1_tl, x2_tl)
    ymin = max(y1_tl, y2_tl)
    xmax = min(x1_br, x2_br)
    ymax = min(y1_br, y2_br)

    # Check if there's an overlap
    if xmin < xmax and ymin < ymax:
        return np.array([xmin, ymin, xmax, ymax])
    else:
        return []


def dfs(current_point: Tuple[float, float, float], remaining_points: List[Tuple[float, float, float]], path: List[Tuple[float, float, float]], queue: List[Tuple[float, float, float]], reset_branch: Tuple[float, float], previous_point: Tuple[float, float], image, previous_path=None) -> List[Tuple[float, float, float]]:
    if not (reset_branch[0] != 0 and reset_branch[1] != 0) or len(path) == 0:
        path.append(current_point)
    if not remaining_points:
        return path

    remaining_points.sort(key=lambda point: vector_distance(
        (point[0], point[1]), (current_point[0], current_point[1])))
    branches = []
    temp_branches = []
    for chosen_point in remaining_points[:5]:
        distance = vector_distance(
            (chosen_point[0], chosen_point[1]), (current_point[0], current_point[1]))
        if distance < 20.:
            temp_branches.append(chosen_point)
        else:
            break

    if len(temp_branches) > 1:
        for chosen_point in temp_branches:
            overlap_points = find_overlap_points(
                chosen_point[0], chosen_point[1], current_point[0], current_point[1])
            if len(overlap_points) > 0:
                xmin, ymin, xmax, ymax = overlap_points.astype(int)
                overlapping_part = image[ymin:ymax, xmin:xmax]
                lines = len(np.where(overlapping_part > 0)[0])
                if overlapping_part.shape[0]*overlapping_part.shape[1] == 0:
                    if calculate_line_overlapping(image, chosen_point[:2], current_point[:2]) > 90:
                        branches.append(chosen_point)
                elif (lines/(overlapping_part.shape[0]*overlapping_part.shape[1])) > 0.1:
                    branches.append(chosen_point)
                    continue
        else:
            if calculate_line_overlapping(image, chosen_point[:2], current_point[:2]) > 85.:
                branches.append(chosen_point)
    else:
        branches = temp_branches
    if len(branches) > 1:
        for time_appeared in branches[0:]:
            queue.append(current_point)
        branches.sort(key=lambda point: calculate_ratio(
            previous_point, current_point, point, image), reverse=True)

    if len(branches) > 0:
        previous_point = current_point
        if len(branches) > 1 and type(previous_path) != type(None):
            next_point = find_next_point(branches, len(
                path)-1, current_point[:2], previous_path)
        else:
            next_point = branches[0]
        reset_branch = [0, 0]
        remove_index = next(i for i, point in enumerate(
            remaining_points) if np.array_equal(point, next_point))
        remaining_points.pop(remove_index)
    else:
        if path[-1][0] != -1:
            path.append([-1, -1, -1])
        if len(queue) > 0:
            next_point = queue.pop(0)
            reset_branch = next_point[:2]
            previous_point = [0, 0]
        else:
            next_point = remaining_points.pop(0)
            reset_branch = [0, 0]
            previous_point = [0, 0]

    return dfs(next_point, remaining_points, path, queue, reset_branch, previous_point, image, previous_path)


def find_next_point(branches, current_index, current_point, previous_path):
    if (current_index+2) > len(previous_path):
        return branches[0]
    previous_point = previous_path[current_index][:2]
    next_point = previous_path[current_index+1][:2]

    if next_point[0] == -1:
        return branches[0]

    movement_vector = np.array(next_point) - np.array(previous_point)
    predict_point = np.array(current_point) + movement_vector

    branches.sort(key=lambda point: vector_distance(
        point[:2], predict_point[:2]))
    point = branches[0]
    return point


def sort_by_distance(points: List[Tuple[float, float, float]], image, previous_path=None) -> List[Tuple[float, float, float]]:
    if type(previous_path) == type(None):
        catheter_x, catheter_y = 0, 0
        if catheter_x != 0 and catheter_y != 0:
            remaining_points = [point for point in points]
            remaining_points.sort(key=lambda point: vector_distance(
                (point[0], point[1]), (catheter_x, catheter_y)))
            initial_point = remaining_points[0]
        else:
            initial_point = most_left_upper_point(points)
    else:
        initial_point = previous_path[0]

    points = [point for point in points if not np.array_equal(
        point, initial_point) and not np.array_equal(point, (0, 0, 0))]
    sorted_points = dfs(initial_point, points, [], [],
                        initial_point, [0, 0], image, previous_path)
    return sorted_points


def get_index_files(filename):
    return int(re.findall(r"_(\d+).png", filename)[0])


def process(patient_folder: str):
    frames = os.listdir(patient_folder)
    index_list = list(map(get_index_files, frames))
    sorted_frames = [filename for _, filename in sorted(zip(index_list, frames))]
    index_list = sorted(index_list)

    # Find max segmentation
    max_pixels_sum = 0
    min_index = 0
    max_index = 0
    for frame_index, frame in tqdm(enumerate(sorted_frames)):
        frame = os.path.join(patient_folder, frame)
        frame = cv2.imread(frame, 0)
        pred_img = remove_catheter(frame)
        pixels_sum = np.sum(pred_img)
        if pixels_sum == 0 and min_index != 0:
            max_index = frame_index
        if pixels_sum > max_pixels_sum:
            max_pixels_sum = pixels_sum
            min_index = frame_index
    if max_index < min_index and min_index != 0:
        max_index = len(sorted_frames)-1
    print(max_index, min_index)

    ITEMS = max_index - min_index + 1

    images = np.zeros((ITEMS, MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))
    vectors = np.zeros((ITEMS, MAX_VECTORS, 3))

    current_index = 0
    frames_count = 0
    previous_path = None
    for frame_index, frame in tqdm(enumerate(sorted_frames)):

        if frame_index<min_index:
            continue
        if frame_index>max_index:
            break

        frame = os.path.join(patient_folder, frame)
        frame = cv2.imread(frame, 0)
        if type(previous_path)==type(None):
            vector, img_with_rectangles, centerline_with_rect, match_img, pred_img = vectorize_one_image_using_center_line(frame)
            vector = np.array(vector)
            previous_path = vector
        else:
            vector, img_with_rectangles, centerline_with_rect, match_img, pred_img = vectorize_one_image_using_center_line(frame, previous_path)
            vector = np.array(vector)
        print(vector)
        image_from_vector = []
        filter_vector = []
        for v in vector:
            if len(image_from_vector)==MAX_VECTORS:
                break
            x, y, color = v
            if x==0 and y==0:
                continue
            if x==-1 and y==-1:
                if len(filter_vector)>0:
                    if filter_vector[-1][0]==-1:
                        continue
                small_image = np.zeros((WINDOW_SIZE, WINDOW_SIZE))
                filter_vector.append(v)
                image_from_vector.append(small_image)
                continue

            x = int(x)
            y = int(y)
            xmin = x-WINDOW_SIZE//2
            xmax = x+WINDOW_SIZE//2
            ymin = y-WINDOW_SIZE//2
            ymax = y+WINDOW_SIZE//2

            if xmax>IMAGE_WIDTH:
                xmin = xmin-(xmax-IMAGE_WIDTH)
                xmax = IMAGE_WIDTH
            if ymax>IMAGE_HEIGHT:
                ymin = ymin-(ymax-IMAGE_HEIGHT)
                ymax = IMAGE_HEIGHT
            if xmin<0:
                xmax = xmax + (0-xmin)
                xmin = 0
            if ymin<0:
                ymax = ymax + (0-ymin)
                ymin = 0

            small_image = match_img[ymin:ymax, xmin:xmax]
            small_pred_img = pred_img[y-1:y+1, x-1:x+1]
            if (np.sum(small_image)/(WINDOW_SIZE**2))>0.3 and len(np.where(small_pred_img==1)[0])>1:
                filter_vector.append(v)
                image_from_vector.append(small_image)
        print(len(image_from_vector))
        if len(image_from_vector)<1:
            continue
        images[frames_count][:len(image_from_vector)] = np.array(image_from_vector)
        vectors[frames_count][:len(image_from_vector)] = np.array(filter_vector)
        frames_count+=1
    # Generate plots
    plot_window_heatmap(vectors, save=True, show=False)
    plot_overall_window_trend(vectors, save=True, show=False)
    plot_selected_windows(vectors, save=True, show=False)


def plot_window_heatmap(vectors, save=True, show=True):
    data = np.array(vectors)
    data[data == -1] = 0
    window_means = data.mean(axis=2)
    img = window_means.T

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(img, aspect='auto', interpolation='nearest')
    ax.set_xlabel(f'Frame index (1…{data.shape[0]})')
    ax.set_ylabel('Window index (1…90)')
    ax.set_title('Mean Pixel Value Heatmap per Window')
    fig.colorbar(cax, label='Mean pixel value')

    if save:
        path = os.path.join(OUTPUT_DIR, "window_heatmap.png")
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved heatmap to {path}")
    if show:
        plt.show()

    plt.close(fig)


def plot_overall_window_trend(vectors, save=True, show=True):
    data = np.array(vectors)
    data[data == -1] = 0
    mean_vals = data.mean(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_vals, marker='o')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Average pixel value')
    ax.set_title('Overall Mean Pixel Value Over Frames')
    ax.grid(True)

    if save:
        path = os.path.join(OUTPUT_DIR, "overall_trend.png")
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved trend plot to {path}")
    if show:
        plt.show()

    plt.close(fig)


def plot_selected_windows(vectors, save=True, show=True):
    data = np.array(vectors)
    data[data == -1] = 0

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx in range(5):
        vals = data[:, idx, :].mean(axis=1)
        ax.plot(vals, label=f'Window {idx + 1}', marker='x')

    ax.set_xlabel('Frame index')
    ax.set_ylabel('Mean pixel value')
    ax.set_title('Pixel Value Trends for Selected Windows')
    ax.legend()
    ax.grid(True)

    if save:
        path = os.path.join(OUTPUT_DIR, "selected_windows.png")
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved selected windows plot to {path}")
    if show:
        plt.show()

    plt.close(fig)
