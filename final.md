## 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

## OIDv4 git 다운 받기
!git clone https://github.com/pythonlessons/OIDv4_ToolKit


## OID 이용 데이터가져오기
```c
%cd /content/drive/'My Drive'/OIDv4_ToolKit-master
````
### 필요한 패키지 설치
```c
!pip install -r /content/drive/'My Drive'/OIDv4_ToolKit-master/requirements.txt
```

### download.py 설정 변경
구글 드라이브 마운트를 하고 OID git을 다운받으면  /content/drive/My Drive/OIDv4_ToolKit-master/modules이 경로에 download.py 가 있습니다. 이 부분을 아래 코드를 복사해  
붙여 넣어 주세요 
```c
import os
import cv2
from tqdm import tqdm
from modules.utils import images_options
from modules.utils import bcolors as bc
from multiprocessing.dummy import Pool as ThreadPool

def download(args, df_val, folder, dataset_dir, class_name, class_code, class_list=None, threads = 20):
    '''
    Manage the download of the images and the label maker.
    :param args: argument parser.
    :param df_val: DataFrame Values
    :param folder: train, validation or test
    :param dataset_dir: self explanatory
    :param class_name: self explanatory
    :param class_code: self explanatory
    :param class_list: list of the class if multiclasses is activated
    :param threads: number of threads
    :return: None
    '''
    if os.name == 'posix':
        rows, columns = os.popen('stty size', 'r').read().split()
    elif os.name == 'nt':
        try:
            columns, rows = os.get_terminal_size(0)
        except OSError:
            columns, rows = os.get_terminal_size(1)
    else:
        columns = 50
    l = int((int(columns) - len(class_name))/2)

    print ('\n' + bc.HEADER + '-'*l + class_name + '-'*l + bc.ENDC)
    print(bc.INFO + 'Downloading {} images.'.format(args.type_csv) + bc.ENDC)
    df_val_images = images_options(df_val, args)

    images_list = df_val_images['ImageID'][df_val_images.LabelName == class_code].values
    images_list = set(images_list)
    print(bc.INFO + '[INFO] Found {} online images for {}.'.format(len(images_list), folder) + bc.ENDC)

    if args.limit is not None:
        import itertools
        print(bc.INFO + 'Limiting to {} images.'.format(args.limit) + bc.ENDC)
        images_list = set(itertools.islice(images_list, args.limit))

    if class_list is not None:
        class_name_list = '_'.join(class_list)
    else:
        class_name_list = class_name

    download_img(folder, dataset_dir, class_name_list, images_list, threads)
    if not args.sub:
        get_label(folder, dataset_dir, class_name, class_code, df_val, class_name_list, args)


def download_img(folder, dataset_dir, class_name, images_list, threads):
    '''
    Download the images.
    :param folder: train, validation or test
    :param dataset_dir: self explanatory
    :param class_name: self explanatory
    :param images_list: list of the images to download
    :param threads: number of threads
    :return: None
    '''
    image_dir = folder
    download_dir = os.path.join(dataset_dir, image_dir, class_name)
    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir)]
    images_list = list(set(images_list) - set(downloaded_images_list))

    pool = ThreadPool(threads)

    if len(images_list) > 0:
        print(bc.INFO + 'Download of {} images in {}.'.format(len(images_list), folder) + bc.ENDC)
        commands = []
        for image in images_list:
            path = image_dir + '/' + str(image) + '.jpg ' + '"' + download_dir + '"'
            command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + path                    
            commands.append(command)

        list(tqdm(pool.imap(os.system, commands), total = len(commands) ))

        print(bc.INFO + 'Done!' + bc.ENDC)
        pool.close()
        pool.join()
    else:
        print(bc.INFO + 'All images already downloaded.' +bc.ENDC)


def get_label(folder, dataset_dir, class_name, class_code, df_val, class_list, args):
    '''
    Make the label.txt files
    :param folder: trai, validation or test
    :param dataset_dir: self explanatory
    :param class_name: self explanatory
    :param class_code: self explanatory
    :param df_val: DataFrame values
    :param class_list: list of the class if multiclasses is activated
    :return: None
    '''
    if not args.noLabels:
        print(bc.INFO + 'Creating labels for {} of {}.'.format(class_name, folder) + bc.ENDC)

        image_dir = folder
        if class_list is not None:
            download_dir = os.path.join(dataset_dir, image_dir, class_list)
            label_dir = os.path.join(dataset_dir, folder, class_list, 'Label')
        else:
            download_dir = os.path.join(dataset_dir, image_dir, class_name)
            label_dir = os.path.join(dataset_dir, folder, class_name, 'Label')

        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
        images_label_list = list(set(downloaded_images_list))

        groups = df_val[(df_val.LabelName == class_code)].groupby(df_val.ImageID)
        for image in images_label_list:
            try:
                current_image_path = os.path.join(download_dir, image + '.jpg')
                dataset_image = cv2.imread(current_image_path)
                boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                file_name = str(image.split('.')[0]) + '.txt'
                # file_path = os.path.join(label_dir, file_name)download_dir
                
                ######################## JAM CODE #####################################
                file_path = os.path.join(download_dir, file_name) # jpg txt 함께 
                ######################## JAM CODE #####################################
                #file_path = os.path.join(label_dir, file_name) 
                
                if os.path.isfile(file_path):
                    f = open(file_path, 'a')
                else:
                    f = open(file_path, 'w')

                for box in boxes:
                    box[0] *= int(dataset_image.shape[1])
                    box[1] *= int(dataset_image.shape[1])
                    box[2] *= int(dataset_image.shape[0])
                    box[3] *= int(dataset_image.shape[0])

                    # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                    # print(class_name, box[0], box[2], box[1], box[3], file=f) 
                    # raw code

                    ###################### Jam code ############################
                    
                    # rafiuddinkhan의 git 참고 (Yolo-Training-GoogleColab/data_converstion/main/main.py) - 라벨 수동지정 코드
                    # I inspired rafiuddinkhan's git (Yolo-Training-GoogleColab/data_converstion/main/main.py) - making label code
                    # https://github.com/rafiuddinkhan/Yolo-Training-GoogleColab/blob/64e92f46e8050764126554439439f0136b456c10/data_converstion/main/main.py#L226
                    
                    width = dataset_image.shape[1] # 박스 가로 (box width)
                    height = dataset_image.shape[0] #


                    XMin, YMin, XMax, YMax = box[0], box[2], box[1], box[3]
                    x_center = (XMin + XMax) / float(2.0 * width) # 박스 너비 중간 ( box width's center)
                    y_center = (YMin + YMax) / float(2.0 * height) 
                    x_width = float(abs(XMax - XMin)) / width # 전체 너비 중 박스 너비의 비율 (0~1사이) ( ratio box width of image width )
                    y_height = float(abs(YMax - YMin)) / height 

                    # each row in a file is name of the class_name, XMin, YMin, XMax, YMax (left top right bottom)
                    print(1, x_center, y_center, x_width, y_height, file=f)
                    ###################### Jam code ############################

            except Exception as e:
                pass

        print(bc.INFO + 'Labels creation completed.' + bc.ENDC)
```

## Helmet Data 다운 받기
아래 코드를 실행 하기 전에 

!python main.py downloader -y --classes 'Human hair' --type_csv train --limit 1000  #### classes = 0

!ls -l /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet' | grep ^- | wc -l

!python main.py downloader -y --classes Helmet --type_csv train --limit 1000  #### classes = 1

# 이전 version 가져오기

%cd /content/

!unzip '/content/drive/My Drive/darknet_10.zip'

!cp -r /content/content/darknet /content/

!rm -r /content/content

#!cp -r '/content/drive/My Drive/yolov3_last.weights' /content/darknet/backup

#!cp -r '/content/darknet/backup/yolov3_1000.weights' '/content/drive/My Drive'

# already train

#!ls
#!rm -fr darknet
#!git clone https://github.com/AlexeyAB/darknet

#폴더 옴기기
#!rm -rf /content/darknet/data_for_colab # 이후 재민이가 다운로드된 txt와 jpg를 다크넷 안 data 폴더에 넣는다

#폴더 만들기
#!cp -r /content/drive/'My Drive'/data_for_colab /content/darknet

#!mkdir /content/drive/'My Drive'/darknet_2/data_for_colab/data
#!mkdir /content/drive/'My Drive'/darknet_2/data_for_colab/data/'Human_hair' 
#!mkdir /content/drive/'My Drive'/darknet_2/data_for_colab/data/Helmet 


#!cp -r /content/drive/'My Drive'/darknet_1/test.jpg /content/darknet
#!cp -r /content/drive/'My Drive'/darknet_1/bike1.mp4 /content/darknet


#OID_Toolkit에 있는 데이터 가져오기
#!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.txt /content/drive/'My Drive'/darknet_2/data_for_colab/data/'Human_hair'  
#!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.jpg /content/drive/'My Drive'/darknet_2/data_for_colab/data/'Human_hair'  # 여기까지 재민 이동
#!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.txt /content/drive/'My Drive'/darknet_2/data_for_colab/data/Helmet 
#!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.jpg /content/drive/'My Drive'/darknet_2/data_for_colab/data/Helmet

#!rm -rf /content/drive/'My Drive'/darknet_2/

#!cp -r /content/drive/'My Drive'/data_for_colab /content/darknet/data_for_colab 

!mkdir /content/darknet/data_for_colab/data
!mkdir /content/darknet/data_for_colab/data/'Human_hair' 
!mkdir /content/darknet/data_for_colab/data/Helmet
#!mkdir /content/darknet/data_for_colab/data/Motorcycle  

!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.txt /content/darknet/data_for_colab/data/'Human_hair'  
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.jpg /content/darknet/data_for_colab/data/'Human_hair'  # 여기까지 재민 이동
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.txt /content/darknet/data_for_colab/data/Helmet 
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.jpg /content/darknet/data_for_colab/data/Helmet
#!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Motorcycle'/*.txt /content/darknet/data_for_colab/data/Motorcycle 
!#cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Motorcycle'/*.jpg /content/darknet/data_for_colab/data/Motorcycle

!cp -r /content/darknet/cfg/yolov3.cfg /content/darknet/data_for_colab

#!ls -l /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/Motorcycle | grep ^- | wc -l
#!ls -l /content/darknet/data_for_colab/data/Helmet | grep ^- | wc -l
#!ls -l /content/darknet/data_for_colab/data/Human_hair | grep ^- | wc -l
#!ls -l /content/darknet/data_for_colab/data/Motorcycle | grep ^- | wc -l



!python /content/drive/'My Drive'/Yolo-Training-GoogleColab-master/train_test_conversion/process.py

!python /content/drive/'My Drive'/Yolo-Training-GoogleColab-master/anchors_calculation/anchors.py

!rm -r '/content/drive/My Drive/OIDv4_ToolKit-master/OID/Dataset/train/Helmet'
!rm -r '/content/drive/My Drive/OIDv4_ToolKit-master/OID/Dataset/train/Human_hair'

# train of install

%cd /content/darknet/

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
#Now let's get some YOLOv3 weights from the official site
#!wget https://pjreddie.com/media/files/yolov3.weights '/content/darknet'
#No here we're modifying the makefile to set OPENCV and GPU to 1

%cd /content/darknet
#No here we're modifying the makefile to set OPENCV and GPU to 1
!ls
!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
print('Yeh-Ap!')
!sed -i 's/GPU=0/GPU=1/g' Makefile
print('Yeh-Ap!')
!ls

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

#Now, here's a bunch of code that takes the longest to execute here but
#It's about installing CUDA and using the beautiful Tesla K80 GPU, so that
#Will worth it

!apt update -qq;
!wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!apt-get update -qq

#Here were are installing compilers and creating some links
!apt-get install cuda -y -qq #gcc-5 g++-5 
#!ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc
#!ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++
!apt update
!apt upgrade
!apt install cuda-8.0 -y

#Now let's see whether the GPU is here and CUDA was successfully installed!
import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)

print("'sup!'")

!/usr/local/cuda/bin/nvcc --version

%cd /content/darknet
!make

#Let's define some functions that will let us show images, and upload and 
#download files
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

#!ls #classes=1 2로 변환
#!./darknet detector train data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/data_for_colab/darknet53.conv.74 -dont_show 


!./darknet detector train data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/backup/yolov3_last.weights -dont_show 

!zip /content/darknet

!cp -r /content/drive/'My Drive'/test_2.jpg /content/darknet

%cd /content/darknet
!./darknet detector test /content/darknet/data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/backup/yolov3_4000.weights /content/darknet/test_3.jpg -dont-show -thresh 0.3

imShow('predictions.jpg')

!zip -r /content/drive/'My Drive'/darknet_10.zip /content/darknet

#!rm -r '/content/drive/My Drive/OIDv4_ToolKit-master/OID'

!cp -r /content/drive/'My Drive'/videoby.mp4 /content/darknet

%cd /content/darknet
!./darknet detector demo data_for_colab/obj.data data_for_colab/yolov3.cfg backup/yolov3_last.weights  -dont_show '/content/drive/My Drive/누리진짜2_1_0.75배속.mp4' -i 0 -out_filename riding.avi -thresh 0.3

download('riding.avi') 

!cp -r /content/darknet /content/drive/'My Drive'

#!./darknet detector calc_anchors Dataset/obj.data -num_of_clusters 5 -width 416 -height 416

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

%cd /content/darknet
import os
from IPython.display import Image
try:
  filename = take_photo()
  command = './darknet detector test data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg /content/darknet/backup/yolov3-tiny-obj_1000.weights /content/darknet/'+filename+' -i 0 -thresh 0.2'
  os.system(command)

  imShow('/content/darknet/predictions.jpg')

  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

#!zip -r /content/drive/'My Drive'/tracking.zip /content/TensorFlow-2.x-YOLOv3
#%cd /content
!unzip /content/drive/'My Drive'/tracking.zip

Looks like this, it will click the screen every 10 minutes so that you don't get kicked off for being idle! HACKS!

%cd /content/darknet

%cp -r /content/darknet/cfg/yolov4.cfg /content/darknet/data_for_colab

# train your custom detector! (uncomment %%capture below if you run into memory issues or your Colab is crashing)
# %%capture
!./darknet detector train data_for_colab/obj.data data_for_colab/yolov4.cfg data_for_colab/yolov4.conv.137 -dont_show

After training, you can observe a chart of how your model did throughout the training process by running the below command. It shows a chart of your average loss vs. iterations. For your model to be 'accurate' you should aim for a loss under 2.

!python3 -c "from utils import utils; utils.plot_results()"

# show chart.png of how custom object detector did with training
imShow('chart.png')

Here is what the chart.png should look like after an uninterrupted training! If you stop training or it crashes during training your chart will look like the above one but don't worry you can still check accuracy of your model in the next steps.


**TRICK**: If for some reason you get an error or your Colab goes idle during training, you have not lost your partially trained model and weights! Every 100 iterations a weights file called **yolov4-obj_last.weights** is saved to **mydrive/yolov4/backup/** folder (wherever your backup folder is). This is why we created this folder in our Google drive and not on the cloud VM. If your runtime crashes and your backup folder was in your cloud VM you would lose your weights and your training progress.

We can kick off training from our last saved weights file so that we don't have to restart! WOOHOO! Just run the following command but with your backup location.
```
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show
```

# kick off training from where it last saved
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show

# Step 6: Checking the Mean Average Precision (mAP) of Your Model
If you didn't run the training with the '-map- flag added then you can still find out the mAP of your model after training. Run the following command on any of the saved weights from the training to see the mAP value for that specific weight's file. I would suggest to run it on multiple of the saved weights to compare and find the weights with the highest mAP as that is the most accurate one!

**NOTE:** If you think your final weights file has overfitted then it is important to run these mAP commands to see if one of the previously saved weights is a more accurate model for your classes.

%cp -r /content/drive/'My Drive'/yolov3_semifinal_1000.weights /content/darknet/backup

%cp -r /content/drive/'My Drive'/yolov3.cfg /content/darknet/data_for_colab

%cd /content/darknet

!./darknet detector map data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/backup/yolov3_semifinal_1000.weights -show

!./darknet detector map /content/darknet/data_for_colab/obj.data /content/darknet/data_for_colab/yolov3.cfg /content/darknet/backup/yolov3_4000.weights -map

%cd /content

!git clone https://github.com/theAIGuysCode/yolov3_deepsort.git

%cd /content/yolov3_deepsort
!pip install -r requirements-gpu.txt

!cp '/content/darknet/backup/yolov3_final.weights' /content/yolov3_deepsort/weights

!pwd

%cd yolov3_deepsort/

data = [0.29,0.37, 0.67,0.81, 1.05,1.60, 1.84,2.00, 2.11,3.35, 3.00,5.71, 4.79,3.15, 5.46,7.49, 9.94,9.71]
[(data[i],data[i+1]) for i in range(0,len(data),2)]
#yolov3_deepsort> yolov3_tf2 > models.py


!python load_weights.py --weights '/content/yolov3_deepsort/weights/yolov3_final.weights' --output /content/yolov3_deepsort/weights/i3_s3.tf --num_classes 2
#yolov3_deepsort> object_tracker.py  ## i3_s3.weights iou = 0.3 score=0.3

# !python object_tracker.py --video /content/yolov3_deepsort/data/video/KakaoTalk_20200915_15582966411111111111111111.mp4 --output ./data/video/h1_i7_s3_0915_pred.avi --weights ./weights/i7_i3.weights --num_classes 2 --classes ./data/labels/coco.names

# !python object_tracker.py --video /content/yolov3_deepsort/data/video/2020_0915_134812_038.MP4 --output ./data/video/ho1_i7_s3_0915_pred.avi --weights ./weights/i7_i3.weights --num_classes 2 --classes ./data/labels/coco.names
!python object_tracker.py --video '/content/drive/My Drive/누리진짜2_1_0.75배속.mp4' --output /content/drive/'My Drive'/test_3_pred.avi --weights ./weights/i3_s3.tf --num_classes 2 --classes ./data/labels/coco.names

# ./data/labels/coco.names 수정

%cd /content/

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kCpwsWlTjRmeaeqgOoZcgHypvzhQbIpL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kCpwsWlTjRmeaeqgOoZcgHypvzhQbIpL" -O tesst2222222222.zip && rm -rf /tmp/cookies.txt

!git clone https://github.com/cyberjam/darknet.git '/content/darknet'

!rm -r /content/darknet

