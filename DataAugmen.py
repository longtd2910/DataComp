import random
import albumentations as A
import cv2
from matplotlib import pyplot as plt

#Constant


class Visualization:
    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White
    #Draw Bounding Box and Label on Image (image, bbox -> def -> labeled image)
    def visualize_single_bbox(self, img, shape, bbox, class_name, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        x_center_norm, y_center_norm, w_norm, h_norm = bbox

        w , h = w_norm*shape[1], h_norm*shape[0]

        x_min, y_min = (x_center_norm - w_norm/2)*shape[1], (y_center_norm - h_norm/2)*shape[0]

        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), self.BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=self.TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize_multi_bbox(self, image, shape, bboxes, category_ids, category_id_to_name):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = self.visualize_single_bbox(img, shape, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)

