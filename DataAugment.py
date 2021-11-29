import random
from typing import Tuple
import albumentations as Augmenter
import cv2
from matplotlib import pyplot as plt
import gdown
import os
import numpy as np
from PIL import Image
from shutil import copy

# Pipeline for automated image augmentation
# Object Logic Workflow:
# dataset on gdrive -> Data riching loop:  Image files absolute path -> cv2 ImageReader      -> Numpy Array     |
#                                          Label files absolute path -> Utils.label_read()  |-> List of bboxes  |-> Albumentation Augmenter -> Transformed Image
#                                                                                           |-> List of labels  |

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


class Visualization:
    def visualize_single_bbox(img, shape, bbox, class_name, color=BOX_COLOR, thickness=2) -> np.array:
        """Visualizes a single bounding box on the image"""
        x_center_norm, y_center_norm, w_norm, h_norm = bbox

        w, h = w_norm*shape[1], h_norm*shape[0]

        x_min, y_min = (x_center_norm - w_norm/2)*shape[1], (y_center_norm - h_norm/2)*shape[0]

        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize_multi_bbox(self, image, shape, bboxes, category_ids, category_id_to_name) -> None:
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = self.visualize_single_bbox(img, shape, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)


class OsUtils:
    def get_gdrive(id: str) -> None:
        url = 'https://drive.google.com/u/1/uc?id={}&export=download'.format(id)
        output = './dataset.zip'
        gdown.download(url, output, quiet=False)

    def match_img_label(path: str, split: str) -> Tuple:
        '''
        path(str) is path to DATASET with structure:
        DATASET -> split -> images -> labels
        '''

        # /content/dataset/images/train/10.35.17.101_01_20210712140351896_MOTION_DETECTION.jpg
        # Image/Label Path = Dataset Path + images/labels + Split + filename

        images_path = '{}/images/{}'.format(path, split)
        labels_path = '{}/labels/{}'.format(path, split)

        # Read all files in images/labels dir
        images = []
        labels = []
        for f in os.listdir(images_path):
            images.append('{}/{}'.format(images_path, f))
        for f in os.listdir(labels_path):
            labels.append('{}/{}'.format(labels_path, f))

        return sorted(images), sorted(labels)

    def prepare_dir(path: str, split: str) -> None:
        os.makedirs('{}/Augmented/images/{}'.format(path, split), exist_ok=True)
        os.makedirs('{}/Augmented/images/{}'.format(path, split) + '/labels/', exist_ok=True)


class DatasetUtils:
    def label_read(path: str) -> Tuple:
        labels = []
        bboxes = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data = line.split()
                labels.append(int(data[0]))
                bboxes.append([float(i) for i in data[1:]])
        return labels, bboxes

    def report_dataset(path: str) -> str:
        # path = path to a folder containing /images/ and /labels/
        # train, val, public_test
        # no_mask, mask, incorrect_mask
        split = ['train', 'val', 'public_test']
        label = ['no_mask', 'mask', 'incorrect_mask']
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        file = 0
        for i in range(len(split)):
            for f in os.listdir('{}/labels/{}'.format(path, split[i])):
                with open('{}/labels/{}/{}'.format(path, split[i], f), 'r') as label_file:
                    for line in label_file.readlines():
                        try:
                            result[i][int(line[0])] += 1
                        except:
                            pass
                if(split[i] == 'train'):
                    file += 1
        result_str = 'Dataset include:\n'
        for i in range(len(split)):
            result_str += split[i] + ': '
            ratio = []
            for j in range(len(result[i])):
                result_str += str(result[i][j]) + ' ' + label[j]
                if j != len(result[i]) - 1:
                    result_str += ', '
                ratio.append(round(result[i][j] / sum(result[i]) * 100, 2))
            result_str += '\n(Ratio): {}% no_mask | {}% mask | {}% incorrect_mask\n'.format(ratio[0], ratio[1], ratio[2])
        return result_str + '\nTotal {} images'.format(file-1)


class AugmentWorker:
    def __call__(self, path: str, split: str, image: np.array, bboxes: list, labels: list, index: int, transformer: Augmenter.Compose, vers: int) -> None:
        for i in range(vers):
            transformed = transformer(image=image, bboxes=bboxes, category_ids=labels)
            data = transformed['image']
            # Save image
            img = Image.fromarray(data, 'RGB')
            img_path = '{}/Augmented/images/{}/img_{}_ver_{}.jpg'.format(path, split, index, i)
            img.save(img_path)
            # Save label
            label_path = '{}/Augmented/labels/{}/img_{}_ver_{}.txt'.format(path, split, index, i)
            f = open(label_path, "w+")
            for i in range(len(transformed['bboxes'])):
                f.write("%s %s %s %s %s\r\n" % (transformed['category_ids'][i], transformed['bboxes'][i][0],
                        transformed['bboxes'][i][1], transformed['bboxes'][i][2], transformed['bboxes'][i][3]))
            f.close()
