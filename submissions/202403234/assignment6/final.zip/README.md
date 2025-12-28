# 일정 자동 분류 & 우선순위 기반 일정 관리 시스템  
6th Assignment — HUFS LAI ML 2025-2  
학번: 202403234  
이름: 양지현 (Yang Jihyun)

---

## 1. 프로젝트 개요

학기 중에는 시험, 과제, 발표, 학생회 활동 등 여러 일정이 동시에 몰려와  
어떤 일을 먼저 해야 하는지 판단하기 어렵다.  
일반 일정 관리 앱은 단순한 "할 일 기록" 수준에 머물러  
우선순위를 자동으로 분석하지 못한다.

본 프로젝트는 자연어로 입력된 일정 문장을 기반으로 다음 기능을 제공한다.

1. 일정 카테고리 자동 분류 (과제 / 시험 / 발표 / 기타)  
2. 일정 중요도 예측 (High / Medium / Low)  
3. 마감일 및 우선순위 기반 정렬  
4. 완료 체크 기능  
5. 일정 삭제 기능  
6. 일정 누적 저장

---

## 2. 폴더 구조

제출 폴더 구조는 다음과 같다.

assignment6/
│
├── final.zip 
│ ├── app.py 
│ ├── requirements.txt 
│ ├── category_model.pkl
│ ├── priority_model.pkl
│ ├── tfidf_vectorizer.pkl
│ ├── feature_scaler.pkl
│ └── README.md
│
└── report.pdf 



---

## 3. 실행 방법 (Execution Guide)

### (1) 가상환경 생성 및 활성화 (macOS 기준)

```bash
python3 -m venv venv
source venv/bin/activate
```

### (2) 패키지 설치

final.zip 압축을 풀고 해당 폴더로 이동한 후:
cd final
pip install -r requirements.txt

### (3) 서비스 실행
python app.py

### (4) 브라우저 접속
기본 접속 주소:
http://127.0.0.1:7860

---

## 4. 주요 기능
일정 자동 분류
입력된 문장을 기반으로 과제/시험/발표/기타로 자동 분류한다.
우선순위 예측
High / Medium / Low 3단계로 중요도를 예측한다.
우선순위 + 마감일 기반 정렬
예측된 우선순위 → 마감일 오름차순 기준으로 정렬해 보여준다.
일정 누적 저장
입력된 일정들이 테이블에 계속 저장되어 관리된다.
완료 체크 기능
완료된 일정은 드롭다운에서 선택 후 목록에서 제거할 수 있다.
일정 삭제 기능
잘못 입력한 일정은 삭제 버튼으로 제거할 수 있다.

---

## 5. 사용 모델 및 특징
TF-IDF 기반 텍스트 벡터화
StandardScaler 기반 numeric feature 정규화
Logistic Regression으로 카테고리/우선순위 예측
입력 feature
text
days_left
contains_keyword
## 6. 주의사항
app.py는 final 폴더 내부에서 실행해야 한다
모델 파일(.pkl)은 반드시 같은 폴더에 위치해야 한다
requirements 설치가 이루어지지 않으면 실행되지 않는다

---

## 7. 개발자 정보
Yang Jihyun
HUFS LAI ML 2025-2
