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
                        default=(768, 768), help='input image size')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='image/video output path')
    args = parser.parse_args()
    return args


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


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
    for i,item in enumerate(imageList):
        image = Image.open(item.image_name)
        res_image, _ = yolo.detect_image(image)
        draw = ImageDraw.Draw(res_image)
        for bbox in item.bboxes:
            #print(bbox)
            draw.rectangle([bbox[0], bbox[1], bbox[2],  bbox[3]],outline=(255, 0, 0))
        del draw
        #image.show()
        #res_image, bbox = yolo.detect_image(image)
        #res_image.show()
        try:
            res_image.save('./output/%d.jpg'%i, "JPEG")
        except IOError:
            print("cannot create pcture")
    yolo.close_session()