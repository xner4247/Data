## 마운트를 하기전에 코랩 파일을 여시고 다른거 건들지 마시고 런타임에 런타임 유형 변경 GPU로 해주세요 이것을 안하고 돌리시다가 GPU 오류 뜨면 처음부터 다시하셔야 합니다 ^^

## 드라이브 마운트
```c
from google.colab import drive
drive.mount('/content/drive')
```
이 부분에서부터 알고 계셔야 할 부분은 '!' 를 사용하는 코드는 리눅스 기반의 명령어 입니다. 그렇기 때문에 경로 설정을 잘 해줘야 하기 때문에 /My Drive/ 가 들어 가는 부분이 있으면 My Drive 부분에 '' 따옴표를 꼭 붙여주셔야 합니다. 

## 내 드라이브로 경로 들어가기

```c
%cd /content/drive/My Drive
```

## OIDv4 git 다운 받기
```c
!git clone https://github.com/pythonlessons/OIDv4_ToolKit
```

### 필요한 패키지 설치
```c
!pip install -r /content/drive/'My Drive'/OIDv4_ToolKit-master/requirements.txt
```

### download.py 설정 변경
구글 드라이브 마운트를 하고 OID git을 다운받으면  /content/drive/My Drive/OIDv4_ToolKit-master/modules이 경로에 download.py 가 있습니다. 이 부분을 아래 코드를 복사해 붙여 넣어 주세요 
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

## Human hair Data 다운 받기
코드를 보시면 limit 뒤에 숫자가 있습니다. 이 부분은 다운 받을 데이터 수를 결정하는 것입니다. 
다 만드신 후에 폴더 이름을 Human_hair 로 만들어 줍니다. 다음에 실행 할 코드의 경로 설정을 위해서 입니다. 나중에 빠른 실행을 위해 사진은 모두 다운 받아주세요

```c
!python main.py downloader -y --classes 'Human hair' --type_csv train --limit 2500  #### classes = 0

#밑에 코드는 폴더 안에 있는 데이터 개수를 새는 코드 입니다.
!ls -l /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human hair' | grep ^- | wc -l
```
## Helmet Data 다운 받기 
위에 챕터를 실행 했으면 이 부분에서는 download.py에서의 print(0, x_center, y_center, x_width, y_height, file=f) 이 부분에서 0을 1로 바꿔주시고 실행해주세요 

```c
!python main.py downloader -y --classes Helmet --type_csv train --limit 2500  #### classes = 1

#밑에 코드는 폴더 안에 있는 데이터 개수를 새는 코드 입니다.
!ls -l /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet' | grep ^- | wc -l
```

## Darknet git 다운로드

```c
%cd /content
!git clone https://github.com/AlexeyAB/darknet
```


## 폴더 만들기
이 부분은 train을 위해 OID 에서 다운 받아온 파일을 Darknet 폴더로 가져오기 위한 코드 입니다.

```c
#먼저 mkdir을 이용해 darknet 안에 data for colab을 만들고 그 하위 폴더에 data 그 안에 human_hair와 helmet을 만들어 줍니다.

!mkdir /content/darknet/data_for_colab/  
!mkdir /content/darknet/data_for_colab/data
!mkdir /content/darknet/data_for_colab/data/'Human_hair' 
!mkdir /content/darknet/data_for_colab/data/Helmet

#이 부분은 oid를 이용해 이 코드를 실행 할 사람의 구글 드라이브 Oid에 있는 Huma_hair와 helmet .txt, .jpg를 darknet 안에있는 data_for_colab으로 가져오는 코드 입니다.

!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.txt /content/darknet/data_for_colab/data/'Human_hair'  
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Human_hair'/*.jpg /content/darknet/data_for_colab/data/'Human_hair'  
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.txt /content/darknet/data_for_colab/data/Helmet 
!cp /content/drive/'My Drive'/OIDv4_ToolKit-master/OID/Dataset/train/'Helmet'/*.jpg /content/darknet/data_for_colab/data/Helmet


#이 코드는 darknet에 있는 data_for_colab에 파일이 잘 들어가 있는지 개수를 확인하기 위한 코드 입니다. 
!ls -l /content/darknet/data_for_colab/data/Helmet | grep ^- | wc -l
!ls -l /content/darknet/data_for_colab/data/Human_hair | grep ^- | wc -l
```

## Anchor를 계산하고 train, test 나누기 위한 사전작업
```c
# 자신의 구글 드라이브에 git 다운 받기 

%cd /content/drive/My Drive
!git clonehttps://github.com/rafiuddinkhan/Yolo-Training-GoogleColab
```

### /content/drive/My Drive/Yolo-Training-GoogleColab-master/train_test_conversion 이 부분으로 들어가 process.py에 들어간다.

들어간 후에 이 코드를 복사해 procss.py에 붙여 넣는다.
```c
import glob, os

# Current directory
#current_dir = os.path.dirname(os.path.abspath(__file__))

#print(current_dir)

#current_dir = '/content/darknet/data_for_colab/data'
current_dir = '/content/darknet/data_for_colab/data/Helmet'
#current_dir = '/content/darknet/data_for_colab/data/Human_hair'

# Directory where the data will reside, relative to 'darknet.exe'
#path_data = './NFPAdataset/'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
#file_train = open('/content/darknet/data_for_colab/train.txt', 'w')
#file_test = open('/content/darknet/data_for_colab/test.txt', 'w')

file_train = open('/content/darknet/data_for_colab/train_1.txt', 'w')
file_test = open('/content/darknet/data_for_colab/test_1.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        counter = counter + 1
```
## train, text 나누기 이어서
이 코드를 먼저 실행한다.
```c
!python /content/drive/'My Drive'/Yolo-Training-GoogleColab-master/train_test_conversion/process.py
```
다음으로
1. current_dir = '/content/darknet/data_for_colab/data/Helmet' 이부분을 주석 처리해준다.  
2. #current_dir = '/content/darknet/data_for_colab/data/Human_hair' 이부분에 주석을 해제한다.  
3. 다음으로 아래 코드에서 위의 두줄의 코드의 주석을 해제하고 밑에 두 줄의 코드를 주석처리를 해준다.  

#file_train = open('/content/darknet/data_for_colab/train.txt', 'w')  
#file_test = open('/content/darknet/data_for_colab/test.txt', 'w')  

file_train = open('/content/darknet/data_for_colab/train_1.txt', 'w')  
file_test = open('/content/darknet/data_for_colab/test_1.txt', 'w')  
4. 그런다음에 아래 코드를 다시 실행한다.

```c
!python /content/drive/'My Drive'/Yolo-Training-GoogleColab-master/train_test_conversion/process.py
```
5. 그 다음 /content/darknet/data_for_colab에 생긴 test_1.txt 안에 있는 내용을 모두 복사해 test.txt에 붙여넣고 저장해준다.  
train_1.txt 안에 있는 내용도 train.txt 안에 복사해 넣어주고 저장한다.


### Anchor 계산

/content/drive/My Drive/Yolo-Training-GoogleColab-master/anchors_calculation 이부분에 들어가 아래 코드를 복사해 붙여넣어준다.
```c
from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import random
import math

width_in_cfg_file = 416.
height_in_cfg_file = 416.

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities)

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids))
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32.
        anchors[i][1]*=height_in_cfg_file/32.


    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))

    f.write('%f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X,centroids,eps,anchor_file):

    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = []
        iter+=1
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))

        #assign samples to centroids
        assignments = np.argmin(D,axis=1)

        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default = '/content/darknet/data_for_colab/train.txt',
                        help='path to filelist\n' )
    parser.add_argument('-output_dir', default = '/content/drive/My Drive/Yolo-Training-GoogleColab-master/anchors_calculation/anchors', type = str,
                        help='Output anchor directory\n' )
    parser.add_argument('-num_clusters', default = 9, type = int,
                        help='number of clusters\n' )


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    f = open(args.filelist)

    lines = [line.rstrip('\n') for line in f.readlines()]

    annotation_dims = []

    size = np.zeros((1,1,3))
    for line in lines:

        #line = line.replace('images','labels')
        #line = line.replace('img1','labels')
        line = line.replace('JPEGImages','labels')


        line = line.replace('.jpg','.txt')
        line = line.replace('.jpeg','.txt')
        line = line.replace('.jpg','.txt')
        print(line)
        f2 = open(line)
        for line in f2.readlines():
            line = line.rstrip('\n')
            w,h = line.split(' ')[3:]
            #print(w,h)
            annotation_dims.append(tuple(map(float,(w,h))))
    annotation_dims = np.array(annotation_dims)

    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters
            anchor_file = join( args.output_dir,'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)
```

위 단계를 하셨으면 밑에 코드를 실행 시키면 됩니다.

```c
!python /content/drive/'My Drive'/Yolo-Training-GoogleColab-master/anchors_calculation/anchors.py
```
실행시킨 후  /content/drive/My Drive/Yolo-Training-GoogleColab-master/anchors_calculation/anchors 이 경로에 들어가면 
anchor9.txt 파일이 있는데 맨 윗줄을 다 복사해서 가지고 있어야 합니다. 

### cfg 파일 옮기기 
darknet 안에 있는 cfg 파일을 darknet data_for_colab으로 옮기는 것입니다.
```c
!cp -r /content/darknet/cfg/yolov3.cfg /content/darknet/data_for_colab
```

### cfg 설정

옮겨진 cfg 파일에 들어가 아래 코드를 그대로 복사붙여넣기 한 후 위에 anchor 계산 값을 cfg 안에 anchor에 다 복사 붙여넣기 해줍니다.
총 3개가 있고 3 군데에 다 넣어주셔야 합니다.

```c
[net]
# Testing
batch=64
subdivisions=16
# Training
# batch=64
# subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
flip=0
learning_rate=0.001
burn_in=1000
max_batches = 4000
policy=steps
steps=3200,3600
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 6,7,8

anchors = 
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 3,4,5
anchors = 
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 0,1,2
anchors = 
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```



## 여기서 부터는 darknet에서 train을 하기 위한 코드 입니다. 그냥 실행 하시면 됩니다.
```c
%cd /content/darknet/
```
```c
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
```

```c
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

```c
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
```
```c
#Now let's see whether the GPU is here and CUDA was successfully installed!
import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)

print("'sup!'")

!/usr/local/cuda/bin/nvcc --version
```
```c
%cd /content/darknet
!make
```

```c
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
```

## 같이 첨부한 finaltest 압축 해제
1. yolov3_final.weights darknet53.conv.74 이 파일은 dakrnet/data_for_colab 에 넣어주시고
2. yolov3_final.weights은 darknet/backup에 넣어줍니다




## train 하는 코드 입니다. 그냥 실행 하시면 됩니다.
```c
#!./darknet detector train data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/data_for_colab/darknet53.conv.74 -dont_show 
```
```c
!cp -r /content/drive/'My Drive'/test_2.jpg /content/darknet
```

```c
%cd /content/darknet
!./darknet detector test /content/darknet/data_for_colab/obj.data data_for_colab/yolov3.cfg /content/darknet/backup/yolov3_4000.weights /content/darknet/test_3.jpg -dont-show -thresh 0.3
```
```
imShow('predictions.jpg')
```
```c
%cd /content/darknet
!./darknet detector demo data_for_colab/obj.data data_for_colab/yolov3.cfg backup/yolov3_last.weights  -dont_show '/content/drive/My Drive/누리진짜2_1_0.75배속.mp4' -i 0 -out_filename riding.avi -thresh 0.3
```
```c
download('riding.avi') 
```

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

```c
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
```

