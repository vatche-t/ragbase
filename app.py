import asyncio
import random

import streamlit as st
from dotenv import load_dotenv

from ragbase.chain import ask_question, create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.uploader import upload_files

load_dotenv()
LOADING_MESSAGES = [
    "در حال پردازش پرسش شما، لطفاً شکیبا باشید...",
    "اطلاعات در حال تحلیل دقیق و عمیق هستند...",
    "پاسخ شما به زودی آماده خواهد شد. لطفاً منتظر بمانید...",
]

@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message("assistant", avatar="🤖")
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if isinstance(event, str):
                full_response += event
                message_placeholder.markdown(full_response)
            elif isinstance(event, list):
                documents.extend(event)
        for i, doc in enumerate(documents):
            with st.expander(f"منبع #{i+1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    st.sidebar.title("آپلود فایل")
    st.sidebar.markdown("لطفاً فایل‌های PDF خود را بارگذاری کنید.")
    uploaded_files = st.sidebar.file_uploader(
        "آپلود فایل‌های PDF", type=["pdf"], accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.sidebar.warning("لطفاً یک فایل PDF بارگذاری کنید.")
        st.stop()

    # Use `st.spinner` in the main area, not in the sidebar
    with st.spinner("در حال تحلیل فایل‌ها..."):
        return build_qa_chain(uploaded_files)

def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        emoji = "🤖" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=emoji):
            st.markdown(message["content"])


def show_chat_input(chain):
    prompt = st.chat_input("سوال خود را اینجا وارد کنید:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))


st.set_page_config(page_title="پایگاه پاسخ‌رسانی", page_icon="🤖")

st.markdown(
    """
<style>
    body {
        direction: rtl;
        text-align: right;
        font-family: "Tahoma", sans-serif;
    }
    .st-chat {
        direction: rtl;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام! چه چیزی می‌خواهید درباره اسناد خود بدانید؟",
        }
    ]

if Config.CONVERSATION_MESSAGES_LIMIT > 0 and Config.CONVERSATION_MESSAGES_LIMIT <= len(
    st.session_state.messages
):
    st.warning("به حد مجاز گفتگو رسیده‌اید. صفحه را بازخوانی کنید تا گفتگو جدیدی شروع شود.")
    st.stop()

chain = show_upload_documents()
show_message_history()
show_chat_input(chain)