from wider import WIDER
import matplotlib.pyplot as plt
import cv2

import argparse


# press ctrl-C to stop the process
# for data in wider.next():

#     im = cv2.imread(data.image_name)

#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')

#     for bbox in data.bboxes:

#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1], fill=False,
#                           edgecolor='red', linewidth=3.5)
#             )

#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()
#     plt.show()
#     break

from yolo.yolo import YOLO
from PIL import ImageDraw, Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model-weights/YOLO_Face.h5',
                        help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt',
                        help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='cfg/face_classes.txt',
                        help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                        help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                        default=(416, 416), help='input image size')
    parser.add_argument('--image', default=False, action="store_true",
                        help='image detection mode')
    parser.add_argument('--video', type=str, default='samples/subway.mp4',
                        help='path to the video')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='image/video output path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get the arguments
    args = get_args()
    yolo=YOLO(args)

    # arg1: path to label
    # arg2: path to images
    # arg3: label file name
    wider = WIDER('./WIDER_val/wider_face_split',
                './WIDER_val/images',
                'wider_face_val.mat')


    imageList=list(wider.next())
    print(imageList)
    image = Image.open(imageList[0].image_name)
    res_image, _ = yolo.detect_image(image)
    res_image.show()
    yolo.close_session()