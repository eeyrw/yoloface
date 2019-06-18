from wider import WIDER
import matplotlib.pyplot as plt
import cv2
import argparse
from yolo.yolo import YOLO
from PIL import ImageDraw, Image
from tqdm import tqdm
import pickle
import datetime
import os
import shutil


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
                        default=(1024, 1024), help='input image size')
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

def createDir(dirPath):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)


def createDirOverwrite(dirPath):
	if os.path.exists(dirPath):
		shutil.rmtree(dirPath)
	os.makedirs(dirPath)

def saveExperimentResult(result, savePath):
    try:
        file = open(savePath, "wb")
        pickle.dump(result, file)
        file.close()
    except IOError as e:
        print(e)

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

    result=[]
    for i,item in tqdm(enumerate(imageList)):
        image = Image.open(item.image_name)
        res_image, predictBoxesAndScores ,timeConsumed= yolo.detect_image(image,isPrint=False)
        result.append({'imagePath':item.image_name,'groundTruthBoxes':item.bboxes,'scoresAndBoxes':predictBoxesAndScores,'timeConsumed':timeConsumed})
        # draw = ImageDraw.Draw(res_image)
        # for score,bbox in predictBoxesAndScores:
        #     print('Score:%2f'%score)
        #     print(bbox)
        #     draw.rectangle([bbox[0], bbox[1], bbox[2],  bbox[3]],outline=(0, 255, 0))
        # for bbox in item.bboxes:
        #     print(bbox)
        #     draw.rectangle([bbox[0], bbox[1], bbox[2],  bbox[3]],outline=(255, 0, 0))
        # del draw

        # try:
        #     res_image.save('./output/%d.jpg'%i, "JPEG")
        # except IOError:
        #     print("cannot create pcture")
    yolo.close_session()
    createDir('./result/')
    saveExperimentResult(result,'./result/Result-%s.data'%datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
