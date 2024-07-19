# Drug-Classification
알약; 알고먹는약   

카카오톡 챗봇으로 이용가능하며 사용방법과 사진 촬영 예시 제공으로 편리한 이용   
약의 이름과 복용법 제시, 추가 정보를 원할 시 약학정보원으로 이동 링크 제공   

## 실행영상   
<https://youtu.be/QjxSYqf6pGA>

## 구현
<img src="https://user-images.githubusercontent.com/49273782/168088223-b45c290d-ce68-4fca-bbd7-4434ebeee8d5.png" width="550px" height="200px"></img>
+ 전이학습을 통해 train : 728, val : 103 으로 학습 후 모델 추출

<img src="https://user-images.githubusercontent.com/49273782/168091042-f9d5157b-cd86-47de-82e6-53924f635eb7.png" width="550px" height="200px"></img>   
+ 서버로 전송된 사진 정보를 json파일로 변환
+ 학습된 모델로 분류 후 결과를 다시 json파일로 변환해서 카카오톡 서버로 전송
+ 카카오톡 서버에서 사용자에게해당 분류 결과에 알맞은 정보를 제공


Search Scopus Script
This guide explains how to activate the Conda environment and run the search_scopus.py script. Follow the steps below to get started.

#### API_KEY = '6b68c16912fcc075055369041cc6ef10'
```
conda activate scopus
python search_scopus.py
# Enter your API key when prompted
API_KEY = '6b68c16912fcc075055369041cc6ef10'
# Press Enter to execute
```

+ Input : Elsevier.xlsx
+ Output : Elsevier_ref_and_citation_check.csv
