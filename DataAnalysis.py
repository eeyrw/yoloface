import pickle
import datetime
import os
import shutil
from pathlib import Path
from PIL import ImageDraw, Image

def createDir(dirPath):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)


def createDirOverwrite(dirPath):
	if os.path.exists(dirPath):
		shutil.rmtree(dirPath)
	os.makedirs(dirPath)


if __name__ == "__main__":
    with open('./result/Result-20190618-125235.data','rb') as f:
        result=pickle.load(f)
    if os.path.exists('./WIDER_RESULT/'):shutil.rmtree('./WIDER_RESULT/')
    for resultItem in result:
        #{'imagePath':item.image_name,'groundTruthBoxes':item.bboxes,'scoresAndBoxes':predictBoxesAndScores,'timeConsumed':timeConsumed})
        orginalImagePath=resultItem['imagePath']
        scoresAndBoxes=resultItem['scoresAndBoxes']
        groundTruthBoxes=resultItem['groundTruthBoxes']
        # Extract last two level
        imagePath=os.sep.join(os.path.normpath(orginalImagePath).split(os.sep)[-2:])
        txtPath=imagePath.replace('.jpg','.txt')
        destTxtPath=os.path.join('./WIDER_RESULT/',txtPath)
        destImagePath=os.path.join('./WIDER_RESULT/',imagePath)
        createDir(os.path.dirname(destTxtPath))
        # Please contact us to evaluate your detection results. An evaluation server will be available soon. 
        # The detection result for each image should be a text file, with the same name of the image. The detection results are organized by the event categories. For example, if the directory of a testing image is "./0--Parade/0_Parade_marchingband_1_5.jpg", the detection result should be writtern in the text file in "./0--Parade/0_Parade_marchingband_1_5.txt". The detection output is expected in the follwing format: 
        # ... 
        # < image name i > 
        # < number of faces in this image = im > 
        # < face i1 > 
        # < face i2 > 
        # ... 
        # < face im > 
        # ... 
        # Each text file should contain 1 row per detected bounding box, in the format "[left, top, width, height, score]". Please see the output example files and the README if the above descriptions are unclear.
        with open(destTxtPath,'w') as f:
            f.write(imagePath+'\n')
            f.write(str(len(scoresAndBoxes))+'\n')
            for score,box in scoresAndBoxes:
                f.write('%d %d %d %d %f'%(box[0],box[1],box[2]-box[0],box[3]-box[1],score)+'\n')
        
        image = Image.open(orginalImagePath)
        draw = ImageDraw.Draw(image)
        for score,bbox in scoresAndBoxes:
            draw.rectangle([bbox[0], bbox[1], bbox[2],  bbox[3]],outline=(0, 255, 0))
        for bbox in groundTruthBoxes:
            draw.rectangle([bbox[0], bbox[1], bbox[2],  bbox[3]],outline=(255, 0, 0))
        del draw

        try:
            image.save(destImagePath, "JPEG")
        except IOError:
            print("cannot create pcture")