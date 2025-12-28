import gradio as gr
import pandas as pd
import joblib
import os
from datetime import datetime
from scipy.sparse import hstack

# ========================================
# 1. 모델 파일 로드
# ========================================
MODEL_PATH = "."

category_model = joblib.load(os.path.join(MODEL_PATH, "category_model.pkl"))
priority_model = joblib.load(os.path.join(MODEL_PATH, "priority_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "feature_scaler.pkl"))

# ========================================
# 2. 일정 저장 리스트
# ========================================
schedule_entries = []

# ========================================
# 3. Numeric Feature 생성
# ========================================
KEYWORDS = ["과제", "시험", "발표", "제출", "마감", "레포트", "퀴즈"]

def make_numeric_features(text, due_date):
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
        today = datetime.today().date()
        days_left = max((due - today).days, 0)
    except:
        days_left = 0

    contains_keyword = int(any(k in text for k in KEYWORDS))
    return [[days_left, contains_keyword]], days_left


# ========================================
# 4. 예측 함수
# ========================================
def predict_schedule(text, due_date):
    X_num, days_left = make_numeric_features(text, due_date)
    X_num_scaled = scaler.transform(X_num)
    X_tfidf = tfidf.transform([text])
    X_input = hstack((X_tfidf, X_num_scaled))

    category = category_model.predict(X_input)[0]
    priority = priority_model.predict(X_input)[0]

    return category, priority, days_left


# ========================================
# 5. 일정 추가 기능
# ========================================
def add_schedule(text, due_date):
    global schedule_entries

    if not text or not due_date:
        df = pd.DataFrame(schedule_entries)
        sorted_df = df.sort_values(["priority", "days_left"])
        return df, sorted_df, list(range(len(df))), list(range(len(df)))

    category, priority, days_left = predict_schedule(text, due_date)

    schedule_entries.append({
        "done": False,
        "text": text,
        "due_date": due_date,
        "days_left": days_left,
        "category": category,
        "priority": priority,
    })

    df = pd.DataFrame(schedule_entries)
    sorted_df = df.sort_values(["priority", "days_left"])

    choices = list(range(len(df)))

    return df, sorted_df, choices, choices


# ========================================
# 6. 일정 삭제 기능
# ========================================
def delete_schedule(idx):
    global schedule_entries

    try:
        i = int(idx)
        if 0 <= i < len(schedule_entries):
            schedule_entries.pop(i)
    except:
        pass

    df = pd.DataFrame(schedule_entries)
    sorted_df = df.sort_values(["priority", "days_left"])
    choices = list(range(len(df)))

    return df, sorted_df, choices, choices


# ========================================
# 7. 완료 체크 기능
# ========================================
def mark_done(idx):
    global schedule_entries

    try:
        i = int(idx)
        schedule_entries.pop(i)
    except:
        pass

    df = pd.DataFrame(schedule_entries)
    sorted_df = df.sort_values(["priority", "days_left"])
    choices = list(range(len(df)))

    return df, sorted_df, choices, choices


# ========================================
# 8. UI 구성
# ========================================
with gr.Blocks(title="일정 자동 분류 & 우선순위 기반 일정 관리 시스템") as demo:

    gr.Markdown("## 📅 일정 자동 분류 & 우선순위 기반 일정 관리 시스템")

    with gr.Row():
        text_input = gr.Textbox(label="일정 내용 입력", placeholder="예: Python 과제 제출하기")
        due_input = gr.Textbox(label="마감일(YYYY-MM-DD)", placeholder="2025-12-10")

    add_button = gr.Button("➕ 일정 추가하기")

    # -------------------------
    # 🔥 우선순위 기반 순위표 (가장 위에!)
    # -------------------------
    gr.Markdown("## 🔥 지금 가장 먼저 해야 할 일 (우선순위 정렬)")
    sorted_table = gr.Dataframe(
        headers=["text", "due_date", "priority", "days_left", "category"],
        interactive=False
    )

    gr.Markdown("### ✔ 완료한 일정 표시")
    done_dropdown = gr.Dropdown(label="완료한 일정 선택", choices=[], interactive=True)
    done_button = gr.Button("✔ 완료 처리 (리스트에서 제거)")

    # -------------------------
    # 📌 전체 일정 목록
    # -------------------------
    gr.Markdown("## 📌 전체 일정 목록")
    schedule_table = gr.Dataframe(
        headers=["text", "due_date", "days_left", "category", "priority"],
        interactive=False
    )

    # -------------------------
    # ❌ 잘못 입력한 일정 삭제
    # -------------------------
    gr.Markdown("### ❌ 잘못 입력한 일정 삭제")
    delete_dropdown = gr.Dropdown(label="삭제할 일정 선택", choices=[], interactive=True)
    delete_button = gr.Button("❌ 삭제하기")

    # ---- 이벤트 연결 ----
    add_button.click(
        fn=add_schedule,
        inputs=[text_input, due_input],
        outputs=[schedule_table, sorted_table, delete_dropdown, done_dropdown]
    )

    delete_button.click(
        fn=delete_schedule,
        inputs=delete_dropdown,
        outputs=[schedule_table, sorted_table, delete_dropdown, done_dropdown]
    )

    done_button.click(
        fn=mark_done,
        inputs=done_dropdown,
        outputs=[schedule_table, sorted_table, delete_dropdown, done_dropdown]
    )


demo.launch()
