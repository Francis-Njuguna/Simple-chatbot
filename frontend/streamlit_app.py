"""Amref International University Help Desk RAG Chatbot."""

import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Amref Help Desk Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .user-bubble {
        background: #1a5276;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 85%;
    }
    .assistant-bubble {
        background: #f0f4f8;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 90%;
        border: 1px solid #e0e0e0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        background: #e8f5e9;
        color: #2e7d32;
    }
    .source-link {
        font-size: 0.85rem;
        color: #1a5276;
    }
    .footer-text {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        padding: 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    .typing-indicator {
        color: #888;
        font-style: italic;
    }
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
"""

DARK_CSS = """
<style>
    .main-header { color: #64b5f6 !important; }
    .assistant-bubble {
        background: #2d2d2d !important;
        color: #e0e0e0 !important;
        border-color: #444 !important;
    }
</style>
"""


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "session_id": None,
        "dark_mode": False,
        "selected_category": None,
        "history_sessions": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)


def fetch_categories() -> list[str]:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/categories")
            if response.status_code == 200:
                return response.json()
    except httpx.HTTPError:
        pass
    return []


def fetch_history_sessions() -> list[dict]:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/history")
            if response.status_code == 200:
                return response.json()
    except httpx.HTTPError:
        pass
    return []


def send_chat_message(message: str, category: str | None) -> dict | None:
    payload = {"message": message}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id
    if category:
        payload["category"] = category

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{API_BASE}/chat", json=payload)
            if response.status_code == 200:
                return response.json()
            st.error(f"API error: {response.status_code} - {response.text}")
    except httpx.HTTPError as exc:
        st.error(f"Connection error: {exc}")
    return None


def submit_feedback(message_id: str, rating: int) -> None:
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{API_BASE}/feedback",
                json={"message_id": message_id, "rating": rating},
            )
    except httpx.HTTPError:
        pass


def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://www.amref.ac.ke/wp-content/uploads/2021/03/amref-logo.png",
            width=180,
        )
        st.markdown("### Help Desk Assistant")
        st.caption("Powered by RAG + ChromaDB")

        st.divider()

        st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)

        categories = fetch_categories()
        if categories:
            selected = st.selectbox(
                "Filter by Category",
                options=["All Categories"] + categories,
                index=0,
            )
            st.session_state.selected_category = (
                None if selected == "All Categories" else selected
            )
        else:
            st.session_state.selected_category = None

        st.divider()
        st.markdown("**Search History**")

        sessions = fetch_history_sessions()
        st.session_state.history_sessions = sessions
        for sess in sessions[:10]:
            title = sess.get("title", "Chat session")
            sid = sess.get("session_id", "")
            if st.button(f"📄 {title[:30]}", key=f"hist_{sid}", use_container_width=True):
                st.session_state.session_id = sid
                load_session_history(sid)

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

        st.divider()
        st.caption("Knowledge Base")
        st.link_button(
            "Open Help Desk",
            "https://helpdesk.amref.ac.ke/knowledgebase.php",
            use_container_width=True,
        )


def load_session_history(session_id: str) -> None:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/history/{session_id}")
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "metadata": m.get("metadata"),
                        "message_id": m["id"],
                    }
                    for m in data["messages"]
                ]
    except httpx.HTTPError:
        pass


def render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)

        metadata = msg.get("metadata") or {}
        confidence = metadata.get("confidence", 0)
        if confidence:
            st.markdown(
                f'<span class="confidence-badge">Confidence: {confidence:.0%}</span>',
                unsafe_allow_html=True,
            )

        images = metadata.get("images", [])
        if images:
            st.markdown("**Relevant Images**")
            cols = st.columns(min(len(images), 3))
            for i, img in enumerate(images[:3]):
                with cols[i]:
                    img_path = img.get("filepath", "")
                    if img_path.startswith("/static"):
                        img_url = f"{BACKEND_URL}{img_path}"
                    elif img_path.startswith("http"):
                        img_url = img_path
                    else:
                        img_url = f"{BACKEND_URL}/static/images/{img.get('filename', '')}"
                    try:
                        st.image(img_url, caption=img.get("caption", ""), use_container_width=True)
                    except Exception:
                        st.caption(img.get("caption", "Image unavailable"))

        sources = metadata.get("sources", [])
        if sources:
            with st.expander("📚 Sources & References"):
                for src in sources:
                    title = src.get("title", "Article")
                    url = src.get("url", "")
                    score = src.get("score", 0)
                    st.markdown(
                        f"- [{title}]({url}) "
                        f"<span class='source-link'>(relevance: {score:.0%})</span>",
                        unsafe_allow_html=True,
                    )

        message_id = msg.get("message_id")
        if message_id:
            rating = st.feedback("stars", key=f"fb_{message_id}")
            if rating is not None:
                submit_feedback(message_id, int(rating) + 1)


def main() -> None:
    init_session_state()
    apply_theme()
    render_sidebar()

    st.markdown('<div class="main-header">🎓 Amref Help Desk Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about LMS, Student Portal, '
        'Microsoft Authenticator, VAS Exams, SMOWL, Email, and more.</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask a question about the Help Desk knowledge base...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching knowledge base..."):
            placeholder = st.empty()
            placeholder.markdown(
                '<div class="typing-indicator">Assistant is typing...</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.3)

            response = send_chat_message(
                prompt,
                st.session_state.selected_category,
            )
            placeholder.empty()

        if response:
            st.session_state.session_id = response["session_id"]
            assistant_msg = {
                "role": "assistant",
                "content": response["answer"],
                "metadata": {
                    "confidence": response.get("confidence", 0),
                    "sources": response.get("sources", []),
                    "images": response.get("images", []),
                },
                "message_id": response.get("message_id"),
            }
            st.session_state.messages.append(assistant_msg)
            st.rerun()

    st.markdown(
        '<div class="footer-text">Powered by FastAPI + LangChain + ChromaDB | '
        'Amref International University</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
