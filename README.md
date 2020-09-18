
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
'''
