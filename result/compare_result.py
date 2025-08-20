import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_annotations(file_path):
    """读取标注文件并返回所有标注数据"""
    annotations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # 跳过不完整的行
                label = parts[0]
                confidence = float(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:])
                annotations.append((label, confidence, x1, y1, x2, y2))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return annotations


def create_composite_image(image, network1_image, network2_image, advantageous_annotations1):
    """创建一个合成图像，显示原图、网络2图、网络1图，并在各自右上角直接覆盖放大区域"""
    h, w = image.shape[:2]
    gap = 60

    # 合成最终图像
    composite_image = np.hstack([
        image,
        np.ones((h, gap, 3), dtype=np.uint8) * 255,
        network2_image,
        np.ones((h, gap, 3), dtype=np.uint8) * 255,
        network1_image
    ])

    return composite_image


def get_advantageous_annotations(annotations1, annotations2, compare_rate):
    """比较两个网络的标注：
    - 如果folder1的类别更多，只要求框数不比folder2少即可；
    - 如果类别数一样或更少，要求每类框都置信度更高，且至少一个高出0.1。
    """
    ann1_dict = {}
    ann2_dict = {}

    for ann in annotations1:
        label, conf, x1, y1, x2, y2 = ann
        ann1_dict.setdefault(label, []).append((label, conf, x1, y1, x2, y2))

    for ann in annotations2:
        label, conf, x1, y1, x2, y2 = ann
        ann2_dict.setdefault(label, []).append((label, conf, x1, y1, x2, y2))

    # 如果 folder1 类别数更多，只要求不缺框即可
    if len(ann1_dict) > len(ann2_dict):
        for label in ann2_dict:
            if label not in ann1_dict:
                return []
            if len(ann1_dict[label]) < len(ann2_dict[label]):
                return []
        # 不缺框，视为优势
        advantageous = []
        for label in ann1_dict:
            advantageous.extend(ann1_dict[label])
        return advantageous

    # 否则类别数一样或更少，必须置信度更高，且至少一个高出compare_rate
    diff_gt_01 = False
    for label in ann2_dict:
        if label not in ann1_dict:
            return []
        if len(ann1_dict[label]) < len(ann2_dict[label]):
            return []

        for i in range(len(ann2_dict[label])):
            try:
                conf1 = ann1_dict[label][i][1]
                conf2 = ann2_dict[label][i][1]
                if conf1 <= conf2:
                    return []
                if conf1 - conf2 > compare_rate:
                    diff_gt_01 = True
            except IndexError:
                return []

    if not diff_gt_01:
        return []

    # 置信度符合要求，记录这些框
    advantageous = []
    for label in ann1_dict:
        advantageous.extend(ann1_dict[label])
    return advantageous


def process_folders(original_folder, network1_folder, network2_folder, output_folder,compare_rate = 0.1, num_images=None):
    """主流程：读取、比较、生成合成图"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    img_files = list(Path(network1_folder).glob("*.jpg"))
    if num_images:
        img_files = img_files[:num_images]

    for img_path in tqdm(img_files, desc="Processing images"):
        name = img_path.stem
        original_img = cv2.imread(str(Path(original_folder) / f"{name}.jpg"))
        if original_img is None:
            continue

        net1_img = cv2.imread(str(Path(network1_folder) / f"{name}.jpg"))
        net2_img = cv2.imread(str(Path(network2_folder) / f"{name}.jpg"))
        if net1_img is None or net2_img is None:
            continue

        ann1 = read_annotations(Path(network1_folder) / f"{name}.txt")
        ann2 = read_annotations(Path(network2_folder) / f"{name}.txt")

        adv1 = get_advantageous_annotations(ann1, ann2, compare_rate)

        if adv1:
            comp = create_composite_image(original_img, net1_img, net2_img, adv1)
            out_path = Path(output_folder) / f"{name}_composite.jpg"
            cv2.imwrite(str(out_path), comp)


# 输入文件夹路径
original_folder = r"E:\python_project\yolo_change\ultralytics\voc\images\voc_test"
network1_folder = "VOC_compare_MDDF"
network2_folder = "VOC_compare_yolo"
output_folder = "compare_images/VOC"

# 运行处理函数
process_folders(original_folder, network1_folder, network2_folder, output_folder, 0.5)
