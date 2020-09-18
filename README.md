
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
