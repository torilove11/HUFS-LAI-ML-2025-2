## 1. 프로젝트 개요 

본 프로젝트는 Slack과 같은 메신저 서비스에서 사용되는 비정형 문장(메시지)을 분석하여 캘린더에 일정을 자동 등록하기 위한 핵심 개체명 추출 모듈을 개발하는 것을 목표로 합니다.

KoBERT-CRF 모델은 이 전체 시스템의 '개체명 추출 모듈(Extraction Module)' 역할을 담당하며, 일정 등록에 필요한 제목(SUB), 날짜/시간(DT), 장소(LOC) 정보를 문장에서 정확히 인식하고 추출하는 역할을 수행합니다.

-주요 파일 목록
training.ipynb: KoBERT-CRF 모델 학습 및 가중치 저장.

evaluation.ipynb: Test Set에 대한 최종 성능 평가 및 지표 계산.

inference.ipynb: 학습된 모델을 이용한 실제 메시지 추론 시연 및 추출된 개체명의 활용 방안 시연.

slack_calendar_data.csv: 학습에 사용된 합성 데이터셋.

best_kobert_crf_model.pt: 학습된 모델 가중치 파일.

## 2. 모델 아키텍처 및 학습 환경

모델 구조:	KoBERT + CRF (Conditional Random Field)
BERT Backbone:	skt/kobert-base-v1
개체명 태그:	B-SUB, I-SUB (제목), B-DT, I-DT (날짜/시간), B-LOC, I-LOC (장소), O (기타)
활용 라이브러리:	torch, transformers, torchcrf, seqeval
학습 장치:	Google Colab T4 GPU (CUDA) 환경

## 3. 데이터셋 및 전처리

### 3.1. 데이터 전처리 (BIO 태깅)

제공된 Slack 캘린더 데이터(`slack_calendar_data.csv`)를 활용하여 메시지 문장과 캘린더 정보를 **BIO(Beginning, Inside, Outside)** 형식의 개체명 태그로 변환했습니다.

* **토크나이저:** KoBERT 전용 `AutoTokenizer`를 사용하여 토큰화.
* **데이터 분할:** 전체 데이터를 **Train : Validation : Test = 8 : 1 : 1**의 비율로 분할했습니다.

### 3.2. 가중치 저장 위치

학습을 통해 생성된 최종 모델 가중치 파일은 아래 Google Drive 경로에 저장되어 있습니다.

> **파일명:** `best_kobert_crf_model.pt`
> **저장 경로:** `/content/drive/MyDrive/ML_assignment5/best_kobert_crf_model.pt`(링크: https://drive.google.com/file/d/1bRoFL5uRYGac-dNURlbNkkgTk5gim6D1/view?usp=sharing)

## 4. 평가 결과 (Evaluation Results)

`evaluation.ipynb`를 실행하여 Test Set (100개 데이터)에 대한 모델의 최종 성능을 측정했습니다.

### 4.1. 최종 Macro F1 Score

| Metric | Score |
| :--- | :--- |
| **Test Set Macro F1 Score** | **1.0000** |

### 4.2. 태그별 Classification Report

Macro F1 Score가 1.0000으로 측정됨에 따라, 모든 태그에 대한 Precision, Recall, F1 Score 또한 완벽한 결과를 보였습니다.

| Tag | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| B-DT | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| I-DT | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| B-LOC | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| I-LOC | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| B-SUB | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| I-SUB | 1.0000 | 1.0000 | 1.0000 | (토큰 수) |
| **Macro Avg** | **1.0000** | **1.0000** | **1.0000** | (총 토큰 수) |

**특이사항:** Macro F1 Score가 1.0000인 것은 **합성 데이터셋**의 특성상 Test Set에도 학습 데이터와 매우 유사하거나 동일한 패턴이 포함되어 있어 모델이 완벽하게 예측했기 때문으로 분석됩니다. (실제 환경에서는 극히 드문 결과입니다.)

## 5. 추론 결과 및 한계점 (Inference Results and Limitations)

### 5.1. 추론 시 Key Error 발생 및 분석

`inference.ipynb` 실행 시, 일부 예시 메시지에서 다음과 같은 오류와 함께 엔티티 추출이 불완전하게 종료되었습니다.

> `❌ 추론 중 KeyError 발생: 'LOC'` 또는 `'SUB'`

* **원인:** 추론 결과인 BIO 태그 시퀀스를 캘린더 항목으로 변환하는 `parse_bio_to_schedule` 함수에서, **모델이 예측한 시퀀스가 불완전**하여 발생했습니다. 예를 들어, `B-LOC` 다음에 `I-LOC`가 아닌 다른 태그(`O` 또는 `B-SUB`)를 예측할 경우, 현재 추출 중인 엔티티의 태그를 찾지 못하고 `KeyError`가 발생합니다.
* **결론:** 높은 F1 Score에도 불구하고, 실제 **새롭고 복잡한 문장 구조**에 대해서는 KoBERT-CRF 모델이 **태그 시퀀스의 일관성을 유지**하는 데 여전히 취약점을 보이며, 모델의 일반화 성능을 높이기 위한 추가적인 학습 데이터와 하이퍼파라미터 튜닝이 필요합니다.

### 5.2. 최종 목표 달성을 위한 추가 과제: 날짜/시간 정규화

현재 KoBERT-CRF 모델은 캘린더 자동 등록에 필요한 개체명을 성공적으로 추출했습니다. 그러나 추출된 날짜/시간 정보(DT 태그)는 여전히 **"다음 주 화요일 4시"**와 같은 비정형 문자열 형태입니다.

캘린더에 자동 등록을 완료하기 위해서는, 추출된 비정형 날짜/시간 정보를 컴퓨터가 인식할 수 있는 표준 형식(YYYY-MM-DD HH:MM:SS)으로 변환하는 '날짜/시간 정규화(Normalization)' 모듈이 후속적으로 반드시 필요합니다.

본 프로젝트의 KoBERT-CRF 모듈은 캘린더 자동 등록 시스템의 성공적인 첫 단계를 완성했습니다. 최종 목표 달성을 위해서는 이 NER 결과를 입력으로 받는 정규화 모듈 개발이 다음 과제로 남아 있습니다.
---

## 6. 결론

본 프로젝트는 KoBERT-CRF 구조를 활용하여 한국어 메신저 문장 내 개체명 인식 과제를 성공적으로 수행했습니다. NER 모듈은 캘린더 자동 등록 시스템의 핵심 엔진으로서 Test Set Macro F1 Score 1.0000을 달성하며 성능을 입증했습니다.

다만, 추론 과정에서 KeyError와 같은 시퀀스 불일치 오류를 확인했으며, 추출된 날짜/시간 정보의 정규화라는 후속 과제가 남아있습니다. 이 부분을 보완한다면 완전한 **'메시지 기반 일정 자동 등록 시스템'**을 구축할 수 있습니다.