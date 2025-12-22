# ============================================
# 論文チャットボット（Streamlit + Embedding）
# ============================================

import streamlit as st
import openai
import numpy as np
import os
import streamlit as st

APP_PASSWORD = os.environ["litbot1222"]

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 LitBot ログイン")

    password = st.text_input(
        "パスワードを入力してください",
        type="password"
    )

    if st.button("ログイン"):
        if password == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.experimental_rerun()
        else:
            st.error("パスワードが違います")

    st.stop()  # ← ここ超重要

from PyPDF2 import PdfReader

# ===============================
# 設定
# ===============================
openai.api_key = os.environ["OPENAI_API_KEY"]

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.1"


# ===============================
# ユーティリティ
# ===============================
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def get_embedding(text):
    res = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="論文チャットボット", layout="wide")
st.title("📚 論文チャットボット")

st.markdown("""
PDF論文を登録して、ChatGPTのように質問してください。  
裏では **Embedding検索** が動いています。
""")

# ===============================
# PDFアップロード
# ===============================
uploaded_files = st.file_uploader(
    "論文PDFをアップロード（複数可）",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and "embeddings" not in st.session_state:
    with st.spinner("論文を読み込み・Embedding中..."):
        all_chunks = []
        all_embeddings = []

        for pdf in uploaded_files:
            reader = PdfReader(pdf)
            text = ""

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

            chunks = split_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "filename": pdf.name,
                    "text": chunk
                })
                all_embeddings.append(get_embedding(chunk))

        st.session_state["chunks"] = all_chunks
        st.session_state["embeddings"] = all_embeddings

    st.success("論文の準備が完了しました！")

# ===============================
# チャット
# ===============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("質問を入力してください")

if question and "embeddings" in st.session_state:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("回答生成中..."):
        q_emb = get_embedding(question)

        scores = [
            cosine_similarity(q_emb, emb)
            for emb in st.session_state["embeddings"]
        ]

        top_idx = np.argsort(scores)[-5:]

        context = ""
        for i in top_idx:
            chunk = st.session_state["chunks"][i]
            context += f"[論文: {chunk['filename']}]\n{chunk['text']}\n\n"

        prompt = f"""
以下は複数論文から抽出した関連部分です。
これを参考に質問に答えてください。

{context}

質問:
{question}
"""

        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer}
    )
    st.chat_message("assistant").write(answer)

elif question:
    st.warning("先にPDFをアップロードしてください。")
