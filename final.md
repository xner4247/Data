
## 현재 위치 확인
```c
!pwd
```


## 0. darknet verJam git clone

데이터 청년 캠퍼스 프로젝트 정리해놓은 darknet git clone
```c
!git clone https://github.com/cyberjam/darknet.git '/content/darknet'
````
## 1.Data
## 1-1. Image Set Download(OIDv4 Toolkit)
 content의 Darknet OIDv4 Toolkit으로 이동
 ```c
 %cd /content/darknet/OIDv4_Toolkit_verJam/
 ````
 OIDv4 Toolkit을 실행하기 위한 library 설치
 
 ```c
 !pip install -r requirements.txt
 ```
 
 런타임 다시 실행하라고 하면 다시 실행 후 설치하면 됩니다.  
 모든 Requirement already satisfied 된다면 설치된겁니다.

## Human Hair 데이터 다운로드
data를 받기 위한 폴더 생성
```c
!mkdir /content/darknet/data_for_colab/data/
```
content의 darknet OIDv4 Toolkit으로 이동 후 main.py 실행 (다운로드 실행)
```c
%cd /content/darknet/OIDv4_Toolkit_verJam/  
!python main.py downloader -y --classes 'Human hair' --type_csv train --limit 400 # class id : 0
```
darknet > OIDv4_Toolkit_verJam에 OID 폴더가 만들어지며 OID 내 CSV 파일 이 위치해 있다.  
limit 400 이 부분은 다운 받을 데이터 개수를 설정하는 곳이다.  

## data 폴더 수 확인 400 hair image, 400 hair txt.
```c
!ls -l /content/darknet/data_for_colab/data | grep ^- | wc -l
```
## Human hair dataset 경로를 train test 분할
```c
%cd /content/darknet/Yolo_Training_GoogleColab/train_test_path_txt  
!python process.py
```

## hair와 helmet data를 각각 train test로 나누기 위해 폴더 분리 ( 기존 data 폴더 datahh로 이름 변경)
```c
!mv /content/darknet/data_for_colab/data /content/darknet/data_for_colab/datahh
```

## Helmet dataset을 다운로드위해 data 폴더 생성
```c
!mkdir /content/darknet/data_for_colab/data
```


## OIDv4_Tookit_vetJam>modules>downloader.py 에서 class id = 0 에서 1로 변경
```c
!sed -i 's/class_id=0/class_id=1/g' /content/darknet/OIDv4_Toolkit_verJam/modules/downloader.py
```

## Helmet 데이터 다운로드
content의 darknet OIDv4 Toolkit으로 이동
```c
%cd /content/darknet/OIDv4_Toolkit_verJam/

!python main.py downloader -y --classes 'Helmet' --type_csv train --limit 400 #1
```

## data 400장 helmet image, 400장 helmet image와 txt
```c
!ls -l /content/darknet/data_for_colab/data | grep ^- | wc -l  

%cd /content/darknet/Yolo_Training_GoogleColab/train_test_path_txt  

!python process.py
```
darknet > data_for_colab > data 에 보면 img와 txt download되어 있습니다  

## dathh에 있는 모든 폴더를 data로 옮기기
```c
!cp /content/darknet/data_for_colab/datahh/* /content/darknet/data_for_colab/data
```
## data 폴더 수 확인
```c
!ls -l /content/darknet/data_for_colab/data | grep ^- | wc -l
```
1600으로 나오는 이유는 각 폴더에 .jpg .txt 파일이 각각 400개 있고 두가지 카테고리의 항목을 가지고 있으므로 800*2 1600이 된다.  
darknet > data_for_colab 에 보면 train, test img path 저장된 text file 있습니다.

## 1-2. anchors 추출
### isnan 에러가 뜰경우도 있습니다만 anchors6를 삭제해주거나 anchor.py output_dir 마지막 /을 삭제하거나 붙이면 다시 됩니다.
```c
%cd /content/darknet/Yolo_Training_GoogleColab/anchors_calculation
!python anchors.py
```
## !!!!! 반드시 수정
### 출력 1번째줄 data_for_colab> .cfg 파일 anchors 수정 
### -> line 134, 176 anchors 수동으로 재설정 필요!!
```c
!cat /content/darknet/data_for_colab/anchors6.txt
```

## 압축풀기
```c
!zip -r /content/darknet.zip /content/darknet
```
1. data_for_colab> .cfg 파일 anchors 수정 -> 설정 필요!!  
2. class 2 -> 이미 설정 완료  
3. filter = (class +5)*3 = 21 -> 이미 설정 완료  
4. obj.data class 수정 -> 이미 설정 완료  
5. 이외 obj.names 수정 -> 이미 설정 완료  

## Train
# Train을 위한 필요한 Library 설치
```c
%cd /content/
!apt-get update
!apt-get upgrade
!apt-get install build-essential
!apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
!apt-get install libavcodec-dev libavformat-dev libswscale-d
!apt-get -y install cmake
!which cmake
!cmake --version
!apt-get install libopencv-dev
!apt-get install vim
```

make 전 content/darnet/Makefile의 GPU 및 OPENCV 설정 변경
```c
%cd /content/darknet

!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
print('Yeh-Ap!')
!sed -i 's/GPU=0/GPU=1/g' Makefile
print('Yeh-Ap!')
```

```c
%cd /content/
!apt install g++-5
!apt install gcc-5

!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
!update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
!update-alternatives --set cc /usr/bin/gcc
!update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
!update-alternatives --set c++ /usr/bin/g++
```

# GPU CUDA를 설치합니다
```c
!apt update -qq;
!wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!apt-get update -qq

!apt-get install cuda -y -qq 
!apt update
!apt upgrade
!apt install cuda-8.0 -y
```


## 2-2. GPU 설정
GPU 안먹힌다면 도구 > 런타임 유형변경 > 가속기에서 GPU 확인  
CUDA 설치 완료!

```c
import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)

print("'sup!'")

!/usr/local/cuda/bin/nvcc --version
```
결과
/device:GPU:0 # 이게 나와야 GPU 적용!
'Yeah-Ap!'
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61




## 2-3. Darknet 설치 및 훈련
Yolo 모델 버전들 중 가장 성능이 낮고 가벼운 버전을 사용하고, 데이터셋도 최소로 낮췄지만 train하여 weight를 추출하는데 상당한 시간이 듭니다.  

다운로드 받으신 데이터셋에 대한 train으로 weights 파일을 추출하여 진행하길 적극 권고하지만, 제가 추출한 weight 파일도 첨부 드립니다.  

다만, 제가 추출한 weight 파일은 다른 데이터셋에 대한 weight 파일이므로 cfg의 anchor 등 설정값이 맞지 않아 부정확한 결과가 나올수 있다는 점을 말씀드립니다.  

```c
# 제가 추출한 weight 파일로 돌려보시길 원하신다면 아래 주석을 풀고 진행하시면 됩니다.
# !cp /content/darknet/backup/yolov3-tiny-obj_1000_Human_hair_Helmet_2.weights /content/darknet/backup/yolov3-tiny-obj_1000.weights
```

```c
%cd /content/darknet
!make
```
## 결과물을 보여주기 위한 함수

```c
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
def download(path):
  from google.colab import files
  files.download(path)
  ```
    
train 시작  

1000 iter 마다 darknet> backup 폴더에 weight 파일을 자동 저장합니다.  

해당 YOLO 모델에서는 train 속도를 높이기 위해 성능이 비교적 낮은 버전을 사용하였으며  

train 설정 변수인 bach size등을 최대치로 올리지 않았습니다.  

train을 시작하는 코드입니다.  
```c
%cd /content/darknet
!./darknet detector train data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg data_for_colab/yolov3-tiny.conv.15 -dont_show
```

# Test 
명령어 맨끝 옵션 -thresh 0.1 에서 0과 1.0 사이 수를 넣어 임계치를 조정할수 있습니다.
```c
%cd /content/darknet
!./darknet detector test data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg /content/darknet/backup/yolov3-tiny-obj_1000.weights /content/darknet/data/helmetNonhelmet.jpg -i 0 -thresh 0.1
```

맨끝 이렇게 나오더라도 정상입니다.
Unable to init server: Could not connect: Connection refused  

(predictions:32093): Gtk-WARNING **: 11:17:59.609: cannot open display

## 사진을 디텍팅한 결과를 보기 위한 코드 입니다.
```C
imShow('/content/darknet/predictions.jpg')
```

# 3-2. Video test
Video-stream stopped! 에러가 뜬다면 해당 경로에 동영상이 없거나 경로 설정이 잘못된 경우 입니다.  

완료시 PPT에서 봤던 머리숱이 없는 외국인이 오토바이를 타는 detecting 하는 영상이 만들어집니다.  

```C
%cd /content/darknet
!./darknet detector demo data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg /content/darknet/backup/yolov3-tiny-obj_1000.weights  -dont_show '/content/darknet/data/2020-09-01_Untitled.mp4' -i 0 -out_filename ending.avi
```

# 해당 영상을 컴퓨터로 다운 받을수 있습니다.
```C
download('ending.avi')
```

## 3-3. Realtime
구글 colab 웹캠 연결  
https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb#scrollTo=buJCl90WhNfq  

```c

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
  ```
  
  ## 실시간 캡쳐 이미지 인식
실시간 웹캠 후 캡쳐
detect가 되지 않은 이미지가 나온경우
개개인 colab shell 설정에 따라서 os.sytem이 먹히지 않는 경우입니다.
```c
%cd /content/darknet
import os
from IPython.display import Image
try:
  filename = take_photo()  
  command = './darknet detector test data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg /content/darknet/backup/yolov3-tiny-obj_1000.weights /content/darknet/'+filename+' -i 0 -thresh 0.1'

  os.system(command)
  
  imShow('/content/darknet/predictions.jpg')
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
  ```
  
  ## 실시간 인식
  ```c
  # https://github.com/ultralytics/yolov3

import base64
import html
import io
import time

from IPython.display import display, Javascript
from google.colab.output import eval_js
import numpy as np
from PIL import Image
import cv2

def start_input():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 512, 512);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 512; //video.videoWidth;
      captureCanvas.height = 512; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function takePhoto(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)
  
def take_photo(label, img_data):
  data = eval_js('takePhoto("{}", "{}")'.format(label, img_data))
  return data
  ```
  

# 데이터 청년 캠퍼스 프로젝트 실시간 정리 git
```c
!git clone https://github.com/cyberjam/yolov3.git' /content/yolov3'
```
이미 트레이닝한 weight를 pt로 변환하여 파일 첨부

# clone
```c
%cd /content/yolov3
```

darknet은 c언어 기반이기 때문에 활용에 있어서 제약이 많습니다.  
그래서 python 기반으로 구현된 yolo를 사용하였습니다.  
그러기 위해서 c언어 기반 yolo에서 train한 weught 파일을   
pt(pytorch)파일로 변환해주는 작업을 합니다.  
```c
 %cd /content/yolov3
!python  -c "from models import *; convert('/content/darknet/data_for_colab/yolov3-tiny-obj.cfg', '/content/darknet/backup/yolov3-tiny-obj_1000.weights')"
# Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'
```

실시간 카운트를 하기 위한 과정입니다.
```c
import argparse
from sys import platform

from models import * 
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
# parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
parser.add_argument('--names', type=str, default='/content/yolov3/data/obj.names', help='*.names path')
parser.add_argument('--cfg', type=str, default='/content/yolov3/cfg/yolov3-tiny-obj.cfg', help='*.cfg path')
parser.add_argument('--weights', type=str, default='/content/darknet/backup/yolov3-tiny-obj_1000.pt', help='weights path')
# parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
opt = parser.parse_args(args = [])

# Initialize
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = Darknet(opt.cfg, opt.img_size)

# Load weights
attempt_download(opt.weights)
if opt.weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, opt.weights)

model.to(device).eval();

# Get names and colors
names = load_classes(opt.names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def js_reply_to_image(js_reply):
    """
    input: 
          js_reply: JavaScript object, contain image from webcam

    output: 
          image_array: image array RGB size 512 x 512 from webcam
    """
    jpeg_bytes = base64.b64decode(js_reply['img'].split(',')[1])
    image_PIL = Image.open(io.BytesIO(jpeg_bytes))
    image_array = np.array(image_PIL)

    return image_array

def get_drawing_array(image_array,cnt): 
    # cnt = {'NonHelmet':0,'Helmet':0}
    """
    input: 
          image_array: image array RGB size 512 x 512 from webcam

    output: 
          drawing_array: image RGBA size 512 x 512 only contain bounding box and text, 
                              channel A value = 255 if the pixel contains drawing properties (lines, text) 
                              else channel A value = 0
    """
    drawing_array = np.zeros([512,512,4], dtype=np.uint8)
    img = letterbox(image_array, new_shape=opt.img_size)[0]

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # (0 - 255) to (0.0 - 1.0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_array.shape).round()

        # Write results
        for *xyxy, conf, cls in det:
            if names[int(cls)] == 'DANGER! (human hair)':
              cnt['NonHelmet']+=1
            else:
              cnt['Helmet']+=1
            label = '%s %.2f' % (names[int(cls)], conf)
            plot_one_box(xyxy, drawing_array, label=label, color=colors[int(cls)])
            print(cnt)

    drawing_array[:,:,3] = (drawing_array.max(axis = 2) > 0 ).astype(int) * 255

    return drawing_array

def drawing_array_to_bytes(drawing_array):
    """
    input: 
          drawing_array: image RGBA size 512 x 512 
                              contain bounding box and text from yolo prediction, 
                              channel A value = 255 if the pixel contains drawing properties (lines, text) 
                              else channel A value = 0

    output: 
          drawing_bytes: string, encoded from drawing_array
    """

    drawing_PIL = Image.fromarray(drawing_array, 'RGBA')
    iobuf = io.BytesIO()
    drawing_PIL.save(iobuf, format='png')
    drawing_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(iobuf.getvalue()), 'utf-8')))
    # print(base64.b64encode(iobuf.getvalue()))
    return drawing_bytes
```

# 실시간 detect main 
```c
start_input()
label_html = 'Capturing...'
img_data = ''
count = 0 
cnt = {'NonHelmet':0,'Helmet':0}
while True:
    js_reply = take_photo(label_html, img_data)
    if not js_reply:
        break

    image = js_reply_to_image(js_reply)
    drawing_array = get_drawing_array(image,cnt) 
    drawing_bytes = drawing_array_to_bytes(drawing_array)
    img_data = drawing_bytes
```

출력시 아래는 괜찮은 경고문입니다.  
Consider using one of the following signatures instead:  
nonzero(*, bool as_tuple) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:766.)  
i, j = (x[:, 5:] > conf_thres).nonzero().t()  
종료시 출력되는 카메라를 클릭하면 정상적으로 종료됩니다.  
각 프레임당 객체 인식 수를 count 합니다.  
이후 객체를 tracking 하는 sort를 사용하여 중복 객체를 구별할수 있습니다.  

## Object Tracking

#### 프레임마다 같은 객체를 인지시키기 위해 Deep sort 알고리즘 적용
### 관련 git clone
```c
!git clone https://github.com/theAIGuysCode/yolov3_deepsort.git
```
# 필요한 library를 설치합니다.
```c
%cd yolov3_deepsort
!pip install -r requirements-gpu.txt
```
```c
# 우리가 darknet에서 만든 weights를 복사합니다.

#!cp '/content/darknet/yolov3_for_colab/backup/yolov3_custom_1000.weights' /content/yolov3_deepsort/weights 경로 확인 필요
```

# 경로 확인
```c
!pwd

%cd yolov3_deepsort/
```
```C
#yolov3_deepsort> yolov3_tf2 > models.py 에서
#yolov3 anchor를 변경해야합니다.

data = [0.21,0.29, 0.46,0.67, 0.83,1.19, 1.40,1.56, 1.65,2.38, 2.22,3.64, 3.82,5.77, 3.83,2.77, 7.45,8.66]
[(data[i],data[i+1]) for i in range(0,len(data),2)]
```

```C
#python 기반 tensorflow에서 실행하기 위해서 weughts파일을 tf 파일로 변환
#기준치 iou와 score 조정 필요

!python load_weights.py --weights '/content/drive/My Drive/YoloProject/backup/yolov3_semifinal_1000.weights' --output /content/yolov3_deepsort/weights/i1_s1.tf --num_classes 2
#yolov3_deepsort> object_tracker.py  ## i3_s3.weights iou = 0.3 score=0.3
```
### 현재 경로 확인
```c
!pwd
```

```c
# Object Tracking 동영상 실행
# 착용자 수 미착용자 수 반환
# object_tracker 수정

# !python object_tracker.py --video /content/yolov3_deepsort/data/video/KakaoTalk_20200915_15582966411111111111111111.mp4 --output ./data/video/h1_i7_s3_0915_pred.avi --weights ./weights/i7_i3.weights --num_classes 2 --classes ./data/labels/coco.names

# !python object_tracker.py --video /content/yolov3_deepsort/data/video/2020_0915_134812_038.MP4 --output ./data/video/ho1_i7_s3_0915_pred.avi --weights ./weights/i7_i3.weights --num_classes 2 --classes ./data/labels/coco.names
!python object_tracker.py --video '/content/drive/My Drive/YoloProject/moi/2020-09-16_누리네거리01.mp4' --output /content/drive/'My Drive'/YoloProject/result/2020-09-16_누리네거리01_tf_i1_s1_pred.avi --weights ./weights/i1_s1.tf --num_classes 2 --classes ./data/labels/coco.names

# ./data/labels/coco.names 수정
```

참고 git
1. 데이터셋 다운로더 - https://github.com/pythonlessons/OIDv4_ToolKit
2. Yolo 데이터셋 형식 참고 - https://github.com/RajashekarY/OIDv5_ToolKit-YOLOv3
3. Yolo 모델 - https://github.com/AlexeyAB/darknet
4. Custom 모델 훈련-https://github.com/rafiuddinkhan/Yolo-Training-GoogleColab
5. 헬멧 이미지 분류 - https://github.com/BlcaKHat/yolov3-Helmet-Detection
6. 실시간 객체 인식 - https://github.com/ultralytics/yolov3
7. 객체 추적 -https://github.com/theAIguysCode/yolov3_deepsort
문의 및 연락처

이메일 : jaminbread@kakao.com
핸드폰 : 010-2122-7772
