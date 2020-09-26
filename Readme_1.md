# * 배달기사 일일 안전모 인증사진 정산 프로그램

#### 윈도우 폴더 분류를 해야하기 때문에 윈도우 환경인 쥬피터에서 실행합니다.

#### 예시 사진은 배달원 안전모 인증 사진으로 가정합니다.
```c
from time import sleep
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from glob import glob
import pandas as pd
#from PIL import image

frame_count = 0             # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           # used in post process loop, to get the no of specified class value.
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
```


# https://drive.google.com/file/d/1gh6knzmRZz-EH5qKPAtCDQDr4qamOhsL/view?usp=sharing
# weights 파일을 같은 폴더에 다운 받아 주세요. 여기서 같은 파일이라고 하면 jupyter notebook이 있는 폴더 입니다!!!!!!!

```c
# Load names of classes
classesFile = "obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
```
    

```c
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-obj.cfg";
modelWeights = "yolov3-obj_2400.weights";


net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
```
```c
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
```


```c
# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):

    global frame_count
# Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #print(label)            #testing
    #print(labelSize)        #testing
    #print(baseLine)         #testing

    label_name,label_conf = label.split(':')    #spliting into class & confidance. will compare it with person.
    if label_name == 'Helmet':
                                            #will try to print of label have people.. or can put a counter to find the no of people occurance.
                                        #will try if it satisfy the condition otherwise, we won't print the boxes or leave it.
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1


    #print(frame_count)
    if(frame_count> 0):
        return frame_count
```




```c
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs,cnt):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []               #have to fins which class have hieghest confidence........=====>>><<<<=======
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
               #this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can calculate it.
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
         #increase test counter till the loop end then print...

        #checking class, if it is a person or not

        my_class='Helmet'                   #======================================== mycode .....
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    #if(frame_count_out > 0):
#     print(frame_count_out)


    if count_person >= 1:
        path = 'use/'
        frame_name=os.path.basename(fn)             # trimm the path and give file name.
        cv.imwrite(str(path)+frame_name, frame)     # writing to folder.
        #print(type(frame))
        cnt['안전모 착용 횟수'][0]= cnt['안전모 착용 횟수'][0]+1
        cv.imshow('img',frame)
        cv.waitKey(800)
    else: 
        path = 'non/'
        frame_name=os.path.basename(fn)    
        cv.imwrite(str(path)+frame_name, frame)     # writing to folder.
        #print(type(frame))
        cnt['안전모 미착용 횟수'][0]=cnt['안전모 미착용 횟수'][0]+1
        cv.imshow('img',frame)
        cv.waitKey(800)
        
#     return(cnt)


    #cv.imwrite(frame_name, frame)
                                               #======================================mycode.........
```

```c
# 분류하여 저장될 폴더 2개를 명령어로 열어줍니다.

!start non #안전모 미착용
!start use # 안전모 착용
```

```c
# Process inputs

cnt = {'안전모 착용 횟수':[0],'안전모 미착용 횟수':[0]}


winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

for fn in glob('images/*.jpg'):

    frame = cv.imread(fn)
    frame_count =0

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, cnt)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    #print(t)
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #print(label)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #print(label)
    key = list(cnt.keys())
    val = list(cnt.values())
    print(key[0],':',val[0][0],',',key[1],':',val[1][0])
cv.destroyAllWindows()
        
    
cnt_df = pd.DataFrame(cnt)


cnt_df
# 0번째 배달 기사 주간 안전모 착용 정산 
# 안전모 착용 횟수 6회, 미착용 횟수 1회
```
    


# 마지막 입니다.

#### 참고 git : 5. 헬멧 이미지 분류 - https://github.com/BlcaKHat/yolov3-Helmet-Detection

