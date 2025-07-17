import streamlit as st
import pickle
import json
import os
import random
import time
from typing import List, Dict, Any
from streamlit_chat import message
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

load_dotenv()

# Haystack imports
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

# Google AI integration - Custom Component
import google.generativeai as genai
from haystack import component, default_from_dict, default_to_dict

@component
class CustomGoogleAIGenerator:
    """
    Một component Haystack tùy chỉnh để gọi trực tiếp API Gemini của Google.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def to_dict(self):
        return default_to_dict(self, api_key=self.api_key, model_name=self.model_name)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return {"replies": [response.text]}
        except Exception as e:
            return {"replies": [f"Xin lỗi, đã có lỗi xảy ra: {e}"]}

# Thiết lập page config với theme hiện đại
st.set_page_config(
    page_title="AI Math Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Gia sư Toán AI thông minh cho học sinh lớp 9"
    }
)

# Custom CSS cho giao diện hiện đại
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app styling */
    .main {
        padding: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        margin-left: 20%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Bot message */
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        margin-right: 20%;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: rgba(255,255,255,0.1);
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 20%;
        animation: pulse 2s infinite;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: white;
        animation: typingDots 1.5s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.3s; }
    .typing-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes typingDots {
        0%, 60%, 100% { opacity: 0.3; }
        30% { opacity: 1; }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f093fb;
        box-shadow: 0 0 20px rgba(240, 147, 251, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Status indicators */
    .status-online {
        color: #4ade80;
        font-weight: 600;
    }
    
    .status-thinking {
        color: #fbbf24;
        font-weight: 600;
    }
    
    /* Math expression styling */
    .math-expression {
        background: rgba(255,255,255,0.1);
        padding: 8px 12px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# Kiểm tra API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("⚠️ Không tìm thấy API key. Vui lòng cấu hình biến môi trường.")
    st.stop()

query_params = st.query_params
if "healthcheck" in query_params:
    st.write("ok ✅")
    st.stop()

@st.cache_resource
def load_resources():
    """Load và khởi tạo tất cả tài nguyên của hệ thống"""
    
    # Load documents
    try:
        with open("embedded_documents.pkl", "rb") as f:
            documents = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy dữ liệu học liệu")
        st.stop()
    
    # Load videos
    try:
        with open("videos.json", "r", encoding="utf-8") as f:
            videos_data = json.load(f)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy dữ liệu video")
        st.stop()
    
    # Initialize document store
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    # Initialize components
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    text_embedder = SentenceTransformersTextEmbedder(
        model="bkai-foundation-models/vietnamese-bi-encoder"
    )
    
    # Templates
    informer_template = """Bạn là một gia sư toán AI. Dựa vào lịch sử trò chuyện gần đây và thông tin từ sách giáo khoa, hãy trả lời câu hỏi của học sinh.

--- LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY ---
{{ conversation_history }}
---

--- THÔNG TIN SÁCH GIÁO KHOA (TỪ RAG) ---
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
---

Dựa vào cả hai nguồn thông tin trên, hãy trả lời câu hỏi cuối cùng của người dùng một cách chính xác và đúng ngữ cảnh.

Câu hỏi cuối cùng: {{ query }}

Hãy trả lời bằng tiếng Việt, giải thích rõ ràng từng bước:"""

    practice_template = """Bạn là gia sư toán sáng tạo. Học sinh cần luyện tập: '{{ student_weakness }}'.

Hãy tạo 2 bài tập mới và đề xuất 1 video phù hợp từ danh sách:
{{ video_cheatsheet_json }}

Trả lời theo format:
### 🎯 BÀI TẬP LUYỆN TẬP
1. [Bài tập 1]
2. [Bài tập 2]

### 📹 VIDEO ĐỀ XUẤT
**[Tên video]**
🎬 Link: https://www.youtube.com/playlist?list=PL5q2T2FxzK7XY4s9FqDi6KCFEpGr2LX2D"""

    insight_template = """Phân tích hội thoại và trả về JSON:

{{ conversation_history }}

Output: {"misunderstood_concepts": ["concept1", "concept2"], "sentiment": "emotion"}"""

    verifier_template = """Kiểm tra tính chính xác của lời giải:

Câu hỏi: {{ query }}
Lời giải: {{ informer_answer }}

Output: {"is_correct": true/false, "correction_suggestion": "gợi ý nếu sai"}"""

    intent_template = """Phân loại ý định từ hội thoại:

{{ conversation_history }}

Chọn một trong: greeting_social, math_question, request_for_practice, expression_of_stress, off_topic"""

    # Create prompt builders
    informer_prompt_builder = PromptBuilder(template=informer_template, required_variables=["documents", "query", "conversation_history"])
    practice_prompt_builder = PromptBuilder(template=practice_template, required_variables=["student_weakness", "video_cheatsheet_json"])
    insight_prompt_builder = PromptBuilder(template=insight_template, required_variables=["conversation_history"])
    verifier_prompt_builder = PromptBuilder(template=verifier_template, required_variables=["query", "informer_answer"])
    intent_prompt_builder = PromptBuilder(template=intent_template, required_variables=["conversation_history"])
    
    # Create generator
    generator = CustomGoogleAIGenerator(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create RAG pipeline
    informer_pipeline = Pipeline()
    informer_pipeline.add_component("text_embedder", text_embedder)
    informer_pipeline.add_component("retriever", retriever)
    informer_pipeline.add_component("prompt_builder", informer_prompt_builder)
    informer_pipeline.add_component("generator", generator)
    
    # Connect components
    informer_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    informer_pipeline.connect("retriever.documents", "prompt_builder.documents")
    informer_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    
    return {
        "informer_pipeline": informer_pipeline,
        "generator": generator,
        "practice_prompt_builder": practice_prompt_builder,
        "insight_prompt_builder": insight_prompt_builder,
        "verifier_prompt_builder": verifier_prompt_builder,
        "intent_prompt_builder": intent_prompt_builder,
        "videos_data": videos_data,
        "document_store": document_store
    }

def classify_intent(conversation_history: str, resources: Dict) -> str:
    """Phân loại ý định người dùng"""
    try:
        # Cải thiện prompt để phân loại chính xác hơn
        improved_intent_template = """Phân loại ý định từ hội thoại sau:

{{ conversation_history }}

Phân loại thành một trong các loại sau:
- 'greeting_social': Chào hỏi, xã giao, cảm ơn, tạm biệt
- 'math_question': Câu hỏi về toán học, yêu cầu giải bài tập, tính toán
- 'request_for_practice': Yêu cầu bài tập luyện tập, muốn thực hành
- 'expression_of_stress': Biểu hiện căng thẳng, mệt mỏi, nản lòng
- 'off_topic': Chủ đề hoàn toàn không liên quan đến học tập

Chỉ trả về MỘT từ duy nhất từ danh sách trên:"""

        # Tạo prompt builder mới với template cải thiện
        intent_prompt_builder = PromptBuilder(
            template=improved_intent_template, 
            required_variables=["conversation_history"]
        )
        
        prompt = intent_prompt_builder.run(conversation_history=conversation_history)
        result = resources["generator"].run(prompt=prompt["prompt"])
        intent = result["replies"][0].strip().lower()
        
        # Debug: In ra intent để kiểm tra
        print(f"DEBUG - User input: {conversation_history.split('User: ')[-1] if 'User: ' in conversation_history else 'N/A'}")
        print(f"DEBUG - Classified intent: {intent}")
        
        valid_intents = ['greeting_social', 'math_question', 'request_for_practice', 'expression_of_stress', 'off_topic']
        
        # Nếu intent không hợp lệ, thử phân loại thủ công
        if intent not in valid_intents:
            # Kiểm tra từ khóa toán học
            math_keywords = ['giải', 'tính', 'phương trình', 'bài tập', 'toán', 'xác suất', 'thống kê', 'hình học', 'đại số']
            user_input = conversation_history.split('User: ')[-1] if 'User: ' in conversation_history else ''
            
            if any(keyword in user_input.lower() for keyword in math_keywords):
                intent = 'math_question'
            else:
                intent = 'greeting_social'
        
        return intent
    except Exception as e:
        print(f"DEBUG - Intent classification error: {e}")
        return 'greeting_social'

def informer_agent(query: str, conversation_history: str, resources: Dict) -> str:
    """Agent giải toán dựa trên RAG"""
    try:
        result = resources["informer_pipeline"].run({
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query, "conversation_history": conversation_history}
        })
        return result["generator"]["replies"][0]
    except:
        return "Xin lỗi, tôi không thể giải bài này lúc này."

def verifier_agent(query: str, informer_answer: str, resources: Dict) -> Dict:
    """Agent kiểm tra tính đúng đắn"""
    try:
        prompt = resources["verifier_prompt_builder"].run(query=query, informer_answer=informer_answer)
        result = resources["generator"].run(prompt=prompt["prompt"])
        return json.loads(result["replies"][0])
    except:
        return {"is_correct": True, "correction_suggestion": ""}

def insight_agent(conversation_history: str, resources: Dict) -> Dict:
    """Agent phân tích điểm yếu"""
    try:
        prompt = resources["insight_prompt_builder"].run(conversation_history=conversation_history)
        result = resources["generator"].run(prompt=prompt["prompt"])
        return json.loads(result["replies"][0])
    except:
        return {"misunderstood_concepts": [], "sentiment": "neutral"}

def practice_agent(student_weakness: str, resources: Dict) -> str:
    """Agent tạo bài tập"""
    try:
        video_cheatsheet = []
        for video in resources["videos_data"]:
            video_cheatsheet.append({
                "title": video["title"],
                "keywords": video["keywords"],
                "summary": video["summary_for_llm"]
            })
        
        video_json = json.dumps(video_cheatsheet, ensure_ascii=False)
        prompt = resources["practice_prompt_builder"].run(
            student_weakness=student_weakness,
            video_cheatsheet_json=video_json
        )
        result = resources["generator"].run(prompt=prompt["prompt"])
        return result["replies"][0]
    except:
        return "Xin lỗi, tôi không thể tạo bài tập lúc này."

def problem_solving_engine(query: str, conversation_history: str, resources: Dict) -> str:
    """
    Đây là "cỗ máy" con, kết hợp Informer và Verifier.
    Nó nhận một câu hỏi và trả về một câu trả lời cuối cùng đã được xác thực.
    """
    print("DEBUG: Problem-Solving Engine activated.")
    
    # 1. Informer Agent tạo ra bản nháp
    informer_answer = informer_agent(query, conversation_history, resources) 
    
    # 2. Verifier Agent kiểm tra bản nháp đó
    verification = verifier_agent(query, informer_answer, resources)
    
    # 3. Trả về kết quả cuối cùng dựa trên sự xác thực
    if verification.get("is_correct", True): # Mặc định là True nếu có lỗi
        return informer_answer
    else:
        correction = verification.get("correction_suggestion", "")
        # Tạo một câu trả lời an toàn, thừa nhận sự không chắc chắn
        return f"🔍 Tôi đã xem xét lại và thấy có một chút chưa chính xác. {correction} Tôi sẽ cần tìm hiểu thêm về vấn đề này để có câu trả lời tốt hơn."


def tutor_agent_response(user_input: str, conversation_history: List, resources: Dict, supabase: Client, user_id: str, display_name: str) -> str:
    """Agent chính điều phối các agent khác"""
    
    # Tạo lịch sử để phân tích
    history_str = "\n".join([f"{'User' if i%2==0 else 'Bot'}: {msg}" 
                            for i, msg in enumerate(conversation_history[-10:])])
    history_str += f"\nUser: {user_input}"
    
    # Phân loại ý định
    intent = classify_intent(history_str, resources)
    
    # Xử lý theo ý định
    if intent == "greeting_social":
        responses = [
            "Xin chào! Tôi là gia sư AI của bạn 😊 Hôm nay chúng ta học gì nhé?",
            "Chào bạn! Tôi sẵn sàng giúp bạn giải toán 📚 Có câu hỏi gì không?",
            "Hi! Cảm ơn bạn đã tin tưởng tôi 💪 Bắt đầu thôi!",
            "Chào bạn thân mến! Toán học thú vị lắm đó ✨ Hãy hỏi tôi nhé!"
        ]
        return random.choice(responses)
    
    elif intent == "math_question":
        print("DEBUG: Tutor Agent is calling the Problem-Solving Engine.")
        return problem_solving_engine(user_input, conversation_history, resources)
    
    elif intent == "request_for_practice":
        # Tạo bài tập
        insights = insight_agent(history_str, resources)
        
        if insights["misunderstood_concepts"]:
            weakness = insights["misunderstood_concepts"][0]
            practice_response = practice_agent(weakness, resources)
            return f"🎯 Tôi thấy bạn cần luyện tập **{weakness}**:\n\n{practice_response}"
        else:
            practice_response = practice_agent("phương trình bậc nhất", resources)
            return f"📝 **Bài tập luyện tập:**\n\n{practice_response}"
    
    elif intent == "expression_of_stress":
        stress_responses = [
            "Tôi hiểu cảm giác của bạn 😊 Hãy nghỉ ngơi 5 phút rồi quay lại nhé!",
            "Đừng lo lắng! Toán học cần thời gian 💪 Chúng ta từ từ thôi!",
            "Thở sâu nhé! Mọi vấn đề đều có lời giải 🌟 Tôi sẽ giúp bạn!"
        ]
        return random.choice(stress_responses)
    
    else:  # off_topic
        return """🤖 **Tôi chuyên về Toán học:**

📐 Giải bài tập lớp 9
📝 Tạo bài luyện tập  
🎥 Đề xuất video học
💪 Hỗ trợ tinh thần

Bạn có câu hỏi Toán nào không? 😊"""

def init_supabase_client():
    """Khởi tạo Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Thiếu thông tin Supabase. Vui lòng cấu hình biến môi trường.")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

def render_chat_message(message: str, is_user: bool, key: str):
    """Render tin nhắn chat với animation"""
    css_class = "user-message" if is_user else "bot-message"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def should_trigger_proactive_practice(conversation_history: List) -> bool:
    """
    Kiểm tra xem có nên kích hoạt luồng luyện tập chủ động không
    (sau mỗi 3-4 lượt chat về toán)
    """
    if len(conversation_history) < 6:  # Ít nhất 3 lượt hỏi đáp
        return False
    
    # Đếm số lượt chat về toán trong 8 tin nhắn gần nhất
    recent_messages = conversation_history[-8:]
    math_count = 0
    
    for i in range(0, len(recent_messages), 2):  # Chỉ đếm tin nhắn của user
        if i < len(recent_messages):
            # Đơn giản hóa: giả sử tin nhắn chứa số hoặc dấu = là câu hỏi toán
            user_msg = recent_messages[i]
            if any(char in user_msg for char in "0123456789=+-*/()"):
                math_count += 1
    
    return math_count >= 3


def show_typing_indicator():
    """Hiển thị indicator khi bot đang suy nghĩ"""
    return st.markdown('''
        <div class="typing-indicator">
            <span style="margin-right: 10px;">🤖 Đang suy nghĩ ...</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

def handle_modern_auth(supabase: Client):
    """Xử lý authentication với UI hiện đại"""
    
    # Kiểm tra session
    try:
        session = supabase.auth.get_session()
        if session and session.user and session.user.email_confirmed_at:
            if "user" not in st.session_state:
                st.session_state.user = session.user
    except:
        if "user" in st.session_state:
            del st.session_state.user
    
    # Nếu chưa đăng nhập
    if "user" not in st.session_state or st.session_state.user is None:
        
        # Welcome message
        st.markdown('''
            <div class="welcome-message">
                <h1>🤖 Chào mừng đến với Gia sư AI</h1>
                <p style="font-size: 1.2em; margin: 1rem 0;">
                    Hệ thống gia sư Toán thông minh với 5 AI Agent chuyên nghiệp
                </p>
                <p style="opacity: 0.9;">
                    Đăng nhập để bắt đầu hành trình học tập cá nhân hóa
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        # Auth tabs
        tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        
        with tab1:
            with st.form("login_form"):
                st.subheader("Đăng nhập tài khoản")
                email = st.text_input("📧 Email", placeholder="example@email.com")
                password = st.text_input("🔒 Mật khẩu", type="password")
                login_btn = st.form_submit_button("Đăng nhập", use_container_width=True)
                
                if login_btn:
                    if email and password:
                        try:
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            if response.user and response.user.email_confirmed_at:
                                st.session_state.user = response.user
                                st.success("✅ Đăng nhập thành công!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("⚠️ Vui lòng xác thực email trước khi đăng nhập!")
                        except Exception as e:
                            if "invalid login credentials" in str(e).lower():
                                st.error("❌ Email hoặc mật khẩu không đúng")
                            else:
                                st.error(f"❌ Lỗi đăng nhập: {str(e)}")
                    else:
                        st.warning("⚠️ Vui lòng nhập đầy đủ thông tin")
        
        with tab2:
            with st.form("register_form"):
                st.subheader("Tạo tài khoản mới")
                display_name = st.text_input("👤 Tên của bạn", placeholder="Nguyễn Văn A")
                new_email = st.text_input("📧 Email", placeholder="example@email.com")
                new_password = st.text_input("🔒 Mật khẩu", type="password")
                register_btn = st.form_submit_button("Đăng ký", use_container_width=True)
                
                if register_btn:
                    if display_name and new_email and new_password:
                        try:
                            # --- THAY ĐỔI 3: GỬI KÈM TÊN TRONG OPTIONS ---
                            response = supabase.auth.sign_up({
                                "email": new_email, 
                                "password": new_password,
                                "options": {
                                    "data": {
                                        "display_name": display_name
                                    }
                                }
                            })
                            if response.user:
                                st.success("🎉 Đăng ký thành công!")
                                st.info("📧 Vui lòng kiểm tra email để xác thực tài khoản")
                        except Exception as e:
                            if "already registered" in str(e).lower():
                                st.error("❌ Email đã được đăng ký")
                            else:
                                st.error(f"❌ Lỗi đăng ký: {str(e)}")
                    else:
                        st.warning("⚠️ Vui lòng nhập đầy đủ Tên, Email và Mật khẩu")
        
        # Feature showcase
        st.subheader("🚀 Tính năng nổi bật")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
                <div class="feature-card">
                    <h3>🧠 5 AI Agent thông minh</h3>
                    <p>Hệ thống đa tác nhân chuyên nghiệp cho trải nghiệm học tập tối ưu</p>
                </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
                <div class="feature-card">
                    <h3>📚 Dựa trên SGK chính thức</h3>
                    <p>Nội dung chuẩn theo chương trình Toán lớp 9</p>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
                <div class="feature-card">
                    <h3>🎯 Học tập cá nhân hóa</h3>
                    <p>Phân tích điểm yếu và đề xuất bài tập phù hợp</p>
                </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
                <div class="feature-card">
                    <h3>🎥 Video bài giảng</h3>
                    <p>Kho video phong phú với lời giải chi tiết</p>
                </div>
            ''', unsafe_allow_html=True)
        
        return False
    
    return True

def main():
    """Hàm chính của ứng dụng"""
    
    # Khởi tạo Supabase
    supabase = init_supabase_client()
    
    # Kiểm tra và xử lý authentication
    # Hàm này sẽ hiển thị form đăng nhập/đăng ký và dừng app nếu chưa đăng nhập
    if not handle_modern_auth(supabase):
        return
    
    # Nếu đã đăng nhập, lấy thông tin user
    user = st.session_state.user
    user_id = user.id

    display_name = user.user_metadata.get("display_name", user.email)
    
    # Load resources (chỉ chạy khi đã đăng nhập thành công)
    with st.spinner("🚀 Đang khởi tạo hệ thống AI..."):
        resources = load_resources()
    
    # --- Giao diện chính sau khi đăng nhập ---
    
    # Header
    st.markdown(f'''
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <h1>🤖 Gia sư Toán AI</h1>
            <p class="status-online">● Online - Sẵn sàng hỗ trợ {display_name}</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Khởi tạo session state cho cuộc trò chuyện
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Thêm tin nhắn chào mừng đầu tiên
        welcome_msg = "Xin chào! Tôi là gia sư AI của bạn 😊. Hôm nay chúng ta cùng học Toán nhé!"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg, "intent": "greeting_social"})

    # Container để chứa các tin nhắn chat
    chat_placeholder = st.container()
    with chat_placeholder:
        for i, msg_data in enumerate(st.session_state.messages):
            is_user = msg_data["role"] == "user"
            # Sử dụng hàm render tùy chỉnh
            render_chat_message(msg_data["content"], is_user, key=f"msg_{i}")

    # Input của người dùng được đặt ở dưới cùng
    if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
        # Thêm và hiển thị tin nhắn của người dùng
        st.session_state.messages.append({"role": "user", "content": user_input, "intent": "unknown"})
        with chat_placeholder:
             render_chat_message(user_input, is_user=True, key=f"user_{len(st.session_state.messages)}")
        
        # Hiển thị indicator "đang suy nghĩ"
        with chat_placeholder:
            typing_indicator_placeholder = show_typing_indicator()
        
        # Xử lý bằng Tutor Agent
        bot_response = tutor_agent_response(
            user_input, 
            [msg["content"] for msg in st.session_state.messages], 
            resources, 
            supabase, 
            user_id,
            display_name=display_name
        )
        
        # Xóa indicator và thêm phản hồi của bot
        typing_indicator_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with chat_placeholder:
            render_chat_message(bot_response, is_user=False, key=f"bot_{len(st.session_state.messages)}")

        # Kiểm tra luồng luyện tập chủ động
        if should_trigger_proactive_practice([msg["content"] for msg in st.session_state.messages]):
            with chat_placeholder:
                proactive_typing_placeholder = show_typing_indicator()
            
            # Lấy hồ sơ người dùng để đề xuất bài tập
            try:
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
                insights = insight_agent(history_str, resources)
                
                if insights and insights.get("misunderstood_concepts"):
                    weakness = insights["misunderstood_concepts"][0]
                    practice_response = practice_agent(weakness, resources)
                    proactive_msg = f"💡 **Tôi nhận thấy bạn có thể cần luyện tập thêm về *{weakness}*. Đây là một số gợi ý:**\n\n{practice_response}"
                    
                    proactive_typing_placeholder.empty()
                    st.session_state.messages.append({"role": "assistant", "content": proactive_msg})
                    with chat_placeholder:
                        render_chat_message(proactive_msg, is_user=False, key=f"proactive_{len(st.session_state.messages)}")
                else:
                    proactive_typing_placeholder.empty() # Xóa indicator nếu không có gì để đề xuất
            except Exception as e:
                proactive_typing_placeholder.empty()
                st.warning(f"Không thể tạo đề xuất chủ động: {str(e)}")
        
        # Rerun để cuộn xuống tin nhắn mới nhất
        st.rerun()

    # Sidebar với thông tin khi đã đăng nhập
    with st.sidebar:
        st.header(f"👤 Chào, {display_name}")
        st.caption(f"Email: {user.email}")
        
        if st.button("Đăng xuất", use_container_width=True):
            supabase.auth.sign_out()
            # Xóa các session state liên quan đến user
            keys_to_delete = ["user", "messages"]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Đã đăng xuất!")
            time.sleep(1)
            st.rerun()
        
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()