from altair import Bin
from numpy import tri
import streamlit as st
import pickle
import json
import os
import random
import time
import re
from typing import List, Dict, Any
from streamlit_chat import message
from dotenv import load_dotenv
from datetime import datetime
from supabase_utils import init_supabase_client, update_user_profile
from supabase import Client
from sympy import Rem
from PIL import Image
import io

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
        
        # --- THAY ĐỔI 1: ĐỊNH NGHĨA CẤU HÌNH AN TOÀN ---
        # Tạm thời vô hiệu hóa các bộ lọc an toàn để gỡ lỗi.
        # Điều này giúp xác định xem có phải lỗi do bị chặn hay không.
        # Lưu ý: Chỉ nên dùng mức độ này để debug, không nên dùng trong sản phẩm thực tế.
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # --- THAY ĐỔI 2: ĐỊNH NGHĨA CẤU HÌNH SINH VĂN BẢN ---
        self.generation_config = genai.types.GenerationConfig(
            # Đặt nhiệt độ rất thấp để mô hình trả lời một cách máy móc, bám sát prompt
            temperature=0.1,
            # Đảm bảo mô hình có đủ không gian để tạo ra JSON
            max_output_tokens=1024 
        )

        # Khởi tạo model với các cấu hình mới
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings # <-- Áp dụng cấu hình an toàn
        )


    def to_dict(self):
        return default_to_dict(self, api_key=self.api_key, model_name=self.model_name)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(self, prompt_parts: List[Any]): # <-- THAY ĐỔI 1: Thay đổi đầu vào
        """
        Gửi một prompt đa phương thức (văn bản và hình ảnh) đến API Gemini.
        """
        try:
            # --- THAY ĐỔI 2: Xử lý hình ảnh ---
            # Chuyển đổi dữ liệu bytes của ảnh thành đối tượng PIL Image
            processed_parts = []
            for part in prompt_parts:
                if isinstance(part, bytes): # Kiểm tra xem có phải là dữ liệu ảnh không
                    try:
                        img = Image.open(io.BytesIO(part))
                        processed_parts.append(img)
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh: {e}")
                else:
                    processed_parts.append(part) # Giữ nguyên nếu là text
            # --- KẾT THÚC THAY ĐỔI 2 ---

            response = self.model.generate_content(processed_parts)
            print(f"DEBUG: [Generator] Raw response from Gemini: {response.text}")
            return {"replies": [response.text]}
        except Exception as e:
            print(f"ERROR: [Generator] Lỗi trong lúc gọi API: {e}")
            return {"replies": [f"Xin lỗi, đã có lỗi xảy ra khi kết nối với mô hình AI."]}

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
    informer_template = """
        Bạn là một Gia sư Toán AI chuyên nghiệp. Vai trò của bạn là cung cấp một lời giải hoặc một lời giải thích chi tiết, chính xác và dễ hiểu cho học sinh lớp 9.

        **QUY TRÌNH CỦA BẠN:**
        1.  **Đọc Lịch sử Trò chuyện:** Hiểu rõ bối cảnh và câu hỏi trước đó của học sinh.
        2.  **Nghiên cứu Tài liệu:** Tham khảo kỹ các thông tin từ sách giáo khoa được cung cấp.
        3.  **Trả lời câu hỏi cuối cùng:** Dựa vào cả lịch sử và tài liệu, hãy trả lời câu hỏi cuối cùng của học sinh.

        **YÊU CẦU TRÌNH BÀY:**
        -   Sử dụng ngôn ngữ sư phạm, rõ ràng, từng bước một.
        -   Sử dụng Markdown để định dạng các công thức toán học, các đề mục và nhấn mạnh các điểm quan trọng.
        -   Luôn trả lời bằng tiếng Việt.

        ---
        **LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:**
        {{ conversation_history }}
        ---
        **THÔNG TIN SÁCH GIÁO KHOA (TỪ RAG):**
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        ---

        **Câu hỏi cuối cùng của học sinh:** {{ query }}

        **Lời giải chi tiết của bạn:**
        """

    practice_template = """
        Bạn là một chuyên gia ra đề thi và tư vấn học liệu môn Toán.

        **NHIỆM VỤ:**
        Dựa trên **chủ đề yếu** của học sinh và **danh sách video** được cung cấp, hãy thực hiện 2 việc:

        1.  **Tạo 2 Bài tập Mới:**
            -   Các bài tập phải liên quan trực tiếp đến chủ đề yếu.
            -   Độ khó tương đương chương trình lớp 9.
            -   Bài tập phải hoàn toàn mới, không được trùng lặp với các ví dụ phổ biến.
        2.  **Đề xuất 1 Video Phù hợp nhất:**
            -   Chọn ra MỘT video từ danh sách có nội dung liên quan chặt chẽ nhất đến chủ đề yếu.

        **THÔNG TIN ĐẦU VÀO:**
        -   **Chủ đề yếu của học sinh:** '{{ student_weakness }}'
        -   **Danh sách video có sẵn (JSON):** {{ video_cheatsheet_json }}

        **YÊU CẦU OUTPUT:**
        Chỉ trả lời theo định dạng Markdown dưới đây, không thêm bất kỳ lời dẫn hay giải thích nào khác.

        ### 🎯 BÀI TẬP CỦNG CỐ
        1.  **Bài 1:** [Nội dung câu hỏi bài tập 1]
        2.  **Bài 2:** [Nội dung câu hỏi bài tập 2]


        ### 📹 VIDEO ĐỀ XUẤT
        **[Tên video]**
        🎬 Link: https://www.youtube.com/playlist?list=PL5q2T2FxzK7XY4s9FqDi6KCFEpGr2LX2D"""

    insight_template = """
        Bạn là một chuyên gia phân tích giáo dục. Nhiệm vụ của bạn là đọc kỹ đoạn hội thoại và xác định chính xác những khái niệm toán học mà học sinh đang hiểu sai.

        **HƯỚNG DẪN:**
        - Đọc kỹ toàn bộ hội thoại.
        - Tập trung vào những câu hỏi hoặc nhận định của 'User' thể hiện sự nhầm lẫn hoặc thiếu kiến thức.
        - Dựa trên sự nhầm lẫn đó, xác định khái niệm toán học cốt lõi bị hiểu sai.
        - Chỉ trả lời bằng một đối tượng JSON duy nhất theo định dạng sau. Không thêm bất kỳ giải thích hay văn bản nào khác.

        **VÍ DỤ:**
        ---
        Hội thoại:
        User: hệ thức Vi-ét dùng để làm gì?
        Assistant: ...
        User: vậy nếu phương trình vô nghiệm thì vẫn tính tổng và tích các nghiệm được đúng không?
        ---
        JSON Output:
        {"misunderstood_concepts": ["điều kiện áp dụng hệ thức Vi-ét"], "sentiment": "confused"}
        ---

        **BÂY GIỜ, HÃY PHÂN TÍCH HỘI THOẠI SAU:**

        **Hội thoại:**
        {{ conversation_history }}

        **JSON Output:**
        """

    verifier_template = """Bin là một người kiểm định chất lượng toán học cực kỳ khó tính và chính xác.
        Nhiệm vụ của bạn là kiểm tra xem lời giải được đề xuất có hoàn toàn đúng về mặt toán học và logic hay không.

        **Câu hỏi của học sinh:** {{ query }}

        **Lời giải được đề xuất:** {{ informer_answer }}

        **YÊU CẦU:**
        Hãy kiểm tra từng bước, từng công thức và kết quả cuối cùng. Sau đó, chỉ trả lời bằng một đối tượng JSON duy nhất theo định dạng sau.

        **JSON Output:**
        {"is_correct": [true hoặc false], "correction_suggestion": "[Nếu sai, hãy giải thích ngắn gọn và chính xác lỗi sai nằm ở đâu. Nếu đúng, để trống chuỗi này.]"}
        """

    intent_template = """
        Bạn là một hệ thống phân loại ý định cực kỳ chính xác. Dựa vào câu hỏi cuối cùng của người dùng, hãy phân loại nó vào MỘT trong các loại sau.

        **ĐỊNH NGHĨA CÁC LOẠI:**
        - 'greeting_social': Chào hỏi, xã giao, cảm ơn, tạm biệt.
        - 'math_question': Bất kỳ câu hỏi nào liên quan trực tiếp đến kiến thức toán học, bao gồm giải bài tập, tính toán, hỏi định nghĩa, hỏi công thức, hỏi tính chất.
        - 'request_for_practice': Yêu cầu bài tập luyện tập, muốn thực hành.
        - 'expression_of_stress': Biểu hiện căng thẳng, mệt mỏi, nản lòng.
        - 'study_support': Hỏi về phương pháp học chung, cách để tiến bộ, tìm kiếm động lực.
        - 'off_topic': Chủ đề hoàn toàn không liên quan đến học tập.

        **VÍ DỤ:**
        ---
        User: Chào bạn
        Phân loại: greeting_social
        ---
        User: Giải giúp mình phương trình x^2 + 5x - 6 = 0
        Phân loại: math_question
        ---
        User: hệ thức Vi-ét dùng để làm gì?  <-- VÍ DỤ MỚI QUAN TRỌNG
        Phân loại: math_question
        ---
        User: Bài này khó quá, mình nản thật
        Phân loại: expression_of_stress
        ---
        User: Có bài nào tương tự để mình luyện tập thêm không?
        Phân loại: request_for_practice
        ---
        User: Làm sao để học tốt môn hình học không gian?
        Phân loại: study_support
        ---
        User: Giá vàng hôm nay bao nhiêu?
        Phân loại: off_topic
        ---

        **Bây giờ, hãy phân loại lịch sử chat sau. Chỉ trả về MỘT từ duy nhất.**

        **Lịch sử chat:**
        {{ conversation_history }}

        **Phân loại:**
        """

    # --- TEMPLATES CHO TUTOR AGENT ---

    # Prompt tổng quát định hình vai trò và tính cách
    tutor_master_prompt = """
    Bạn là một Gia sư Toán AI, một người bạn đồng hành học tập thông minh, thấu cảm và chuyên nghiệp.
    Vai trò của bạn là phản hồi lại học sinh một cách phù hợp nhất dựa trên ý định của họ.
    Luôn sử dụng ngôn ngữ tích cực, khuyến khích và thân thiện. Luôn trả lời bằng tiếng Việt.
    """

    # Prompt cho intent 'greeting_social'
    greeting_template = """
    {{ master_prompt }}

    **Bối cảnh:** Học sinh đang bắt đầu cuộc trò chuyện hoặc nói những câu xã giao (chào hỏi, cảm ơn).
    **Nhiệm vụ:** Hãy phản hồi lại một cách thân thiện, tự nhiên và mời gọi họ bắt đầu buổi học.

    **Lịch sử chat gần đây:**
    {{ conversation_history }}

    **Lời chào thân thiện của bạn:**
    """

    # Prompt cho intent 'expression_of_stress'
    stress_template = """
    {{ master_prompt }}

    **Bối cảnh:** Học sinh đang thể hiện sự căng thẳng, mệt mỏi hoặc nản lòng về việc học.
    **NHIỆM VỤ CỰC KỲ QUAN TRỌNG:**
    1.  **Đồng cảm:** Thể hiện rằng bạn hiểu cảm giác của họ.
    2.  **Bình thường hóa:** Cho họ biết rằng cảm giác này là bình thường.
    3.  **Gợi ý giải pháp AN TOÀN:** Đề xuất những hành động đơn giản như nghỉ ngơi, hít thở sâu.
    4.  **TUYỆT ĐỐI KHÔNG:** Đóng vai chuyên gia tâm lý, không đưa ra lời khuyên phức tạp.

    **Lịch sử chat gần đây:**
    {{ conversation_history }}

    **Lời động viên an toàn và thấu cảm của bạn:**
    """

    # Prompt cho intent 'study_support'
    support_template = """
    {{ master_prompt }}

    **Bối cảnh:** Học sinh đang hỏi về phương pháp học tập, cách để tiến bộ hoặc tìm kiếm động lực.
    **Nhiệm vụ:** Hãy đưa ra những lời khuyên chung, hữu ích và mang tính động viên về việc học Toán. Bạn có thể gợi ý về các chức năng của mình (giải bài tập, tạo luyện tập,...).

    **Lịch sử chat gần đây:**
    {{ conversation_history }}

    **Lời khuyên và hỗ trợ của bạn:**
    """

    # Prompt cho intent 'off_topic'
    off_topic_template = """
    {{ master_prompt }}

    **Bối cảnh:** Học sinh đang hỏi một câu hoàn toàn không liên quan đến toán học hoặc học tập.
    **Nhiệm vụ:** Hãy lịch sự từ chối trả lời và nhẹ nhàng hướng cuộc trò chuyện quay trở lại chủ đề chính là học Toán.

    **Lịch sử chat gần đây:**
    {{ conversation_history }}

    **Lời từ chối khéo léo của bạn:**
    """

    # Create prompt builders
    informer_prompt_builder = PromptBuilder(template=informer_template, required_variables=["documents", "query", "conversation_history"])
    practice_prompt_builder = PromptBuilder(template=practice_template, required_variables=["student_weakness", "video_cheatsheet_json"])
    insight_prompt_builder = PromptBuilder(template=insight_template, required_variables=["conversation_history"])
    verifier_prompt_builder = PromptBuilder(template=verifier_template, required_variables=["query", "informer_answer"])
    intent_prompt_builder = PromptBuilder(template=intent_template, required_variables=["conversation_history"])
    
    greeting_prompt_builder = PromptBuilder(template=greeting_template, required_variables=["master_prompt", "conversation_history"])
    stress_prompt_builder = PromptBuilder(template=stress_template, required_variables=["master_prompt", "conversation_history"])
    support_prompt_builder = PromptBuilder(template=support_template, required_variables=["master_prompt", "conversation_history"])
    off_topic_prompt_builder = PromptBuilder(template=off_topic_template, required_variables=["master_prompt", "conversation_history"])
    
    # Create generator
    generator = CustomGoogleAIGenerator(api_key=os.getenv("GOOGLE_API_KEY"))
    
    return {
        "informer_prompt_builder": informer_prompt_builder,
        "generator": generator,
        "practice_prompt_builder": practice_prompt_builder,
        "insight_prompt_builder": insight_prompt_builder,
        "verifier_prompt_builder": verifier_prompt_builder,
        "intent_prompt_builder": intent_prompt_builder,
        "videos_data": videos_data,
        "document_store": document_store,
        "tutor_master_prompt": tutor_master_prompt,
        "greeting_prompt_builder": greeting_prompt_builder,
        "stress_prompt_builder": stress_prompt_builder,
        "support_prompt_builder": support_prompt_builder,
        "off_topic_prompt_builder": off_topic_prompt_builder,
        "retriever": retriever,
        "text_embedder": text_embedder
    }

def classify_intent(conversation_history: str, resources: Dict) -> str:
    """Phân loại ý định người dùng"""
    valid_intents = ['greeting_social', 'math_question', 'request_for_practice', 'expression_of_stress', 'study_support', 'off_topic']
    
    try:
        # 1. Lấy công cụ đã được chuẩn bị sẵn
        prompt_builder = resources["intent_prompt_builder"]
        
        # 2. Sử dụng công cụ đó để tạo prompt cuối cùng
        prompt_text = prompt_builder.run(conversation_history=conversation_history)["prompt"]
        
        # --- SỬA LỖI: GÓI PROMPT VĂN BẢN VÀO MỘT DANH SÁCH ---
        result = resources["generator"].run(prompt_parts=[prompt_text])
        intent = result["replies"][0].strip().lower()
        
        # Debug: In ra intent để kiểm tra
        # Sửa logic trích xuất user input từ conversation history
        user_input_debug = "N/A"
        if 'User: ' in conversation_history:  # Sửa thành chữ hoa 'User: '
            # Tìm tin nhắn user cuối cùng
            lines = conversation_history.split('\n')
            for line in reversed(lines):
                if line.strip().startswith('User: '):  # Sửa thành chữ hoa 'User: '
                    user_input_debug = line.replace('User: ', '').strip()
                    break
        
        print(f"DEBUG - User input: {user_input_debug}")
        print(f"DEBUG - Classified intent: {intent}")
        print(f"DEBUG - Conversation history format: {conversation_history[:200]}...")  # In ra 200 ký tự đầu để debug
        
        # Nếu intent không hợp lệ, thử phân loại thủ công
        if intent not in valid_intents:
            # Kiểm tra từ khóa toán học
            math_keywords = ['giải', 'tính', 'phương trình', 'bài tập', 'toán', 'xác suất', 'thống kê', 'hình học', 'đại số']
            
            # Sử dụng logic trích xuất user input đã sửa
            user_input_for_fallback = "N/A"
            if 'User: ' in conversation_history:
                lines = conversation_history.split('\n')
                for line in reversed(lines):
                    if line.strip().startswith('User: '):
                        user_input_for_fallback = line.replace('User: ', '').strip()
                        break
            
            if any(keyword in user_input_for_fallback.lower() for keyword in math_keywords):
                intent = 'math_question'
            else:
                intent = 'greeting_social'
        
        return intent
    except Exception as e:
        print(f"DEBUG - Intent classification error: {e}")
        return 'greeting_social'

def informer_agent(query: str, conversation_history_str: str, resources: Dict) -> str:
    """Agent giải toán dựa trên RAG"""
    try:
        result = resources["informer_pipeline"].run({
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query, "conversation_history": conversation_history_str}
        })
        return result["generator"]["replies"][0]
    except:
        return "Xin lỗi, tôi không thể giải bài này lúc này."

def verifier_agent(query: str, informer_answer: str, resources: Dict) -> Dict:
    """Agent kiểm tra tính đúng đắn"""
    try:
        prompt_text = resources["verifier_prompt_builder"].run(query=query, informer_answer=informer_answer)["prompt"]
        result = resources["generator"].run(prompt_parts=[prompt_text])
        # Logic trích xuất JSON từ Markdown (quan trọng)
        llm_reply_string = result["replies"][0]

        json_match = re.search(r"\{.*\}", llm_reply_string, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {"is_correct": True, "correction_suggestion": "Lỗi parse verifier"}
    except Exception as e:
        print(f"ERROR: [Verifier Agent] Lỗi: {e}")
        return {"is_correct": True, "correction_suggestion": ""}

def insight_agent(conversation_history: str, resources: Dict) -> Dict:
    """Agent phân tích điểm yếu, với logic trích xuất JSON thông minh."""
    try:
        prompt_builder = resources["insight_prompt_builder"]
        prompt_text = prompt_builder.run(conversation_history=conversation_history)["prompt"]
        
        print("\n" + "="*50)
        print("DEBUG: [Insight Agent] PROMPT CUỐI CÙNG GỬI ĐẾN GEMINI:")
        print(prompt_text)
        print("="*50 + "\n")

        result = resources["generator"].run(prompt_parts=[prompt_text])
        llm_reply = result["replies"][0]

        # --- LOGIC TRÍCH XUẤT JSON TỪ MARKDOWN ---
        # Sử dụng regex để tìm chuỗi bắt đầu bằng { và kết thúc bằng }
        # re.DOTALL cho phép . khớp với cả ký tự xuống dòng
        json_match = re.search(r"\{.*\}", llm_reply, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(0)
            print(f"DEBUG: [Insight Agent] Đã trích xuất chuỗi JSON: {json_string}")
            # Bây giờ mới parse chuỗi JSON đã được làm sạch
            return json.loads(json_string)
        else:
            # Nếu không tìm thấy JSON nào, ghi nhận lỗi và trả về mặc định
            print(f"ERROR: [Insight Agent] Không tìm thấy chuỗi JSON hợp lệ trong phản hồi của LLM: {llm_reply}")
            return {"misunderstood_concepts": [], "sentiment": "neutral"}

    except json.JSONDecodeError as e:
        print(f"ERROR: [Insight Agent] Lỗi khi parse JSON đã trích xuất: {e}")
        return {"misunderstood_concepts": [], "sentiment": "neutral"}
    except Exception as e:
        print(f"ERROR: [Insight Agent] Đã xảy ra lỗi không xác định: {e}")
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
        prompt_text = resources["practice_prompt_builder"].run(
            student_weakness=student_weakness,
            video_cheatsheet_json=video_json
        )["prompt"]
        
        result = resources["generator"].run(prompt_parts=[prompt_text])
        return result["replies"][0]
    except:
        return "Xin lỗi, tôi không thể tạo bài tập lúc này."

def problem_solving_engine(
    query_text: str, 
    query_image: bytes, 
    conversation_history_str: str, 
    resources: Dict
) -> str:
    """
    Cỗ máy giải quyết vấn đề đa năng, TÁI SỬ DỤNG informer_prompt_builder.
    """
    print("DEBUG: Multimodal Problem-Solving Engine activated.")
    
    try:
        # --- GIAI ĐOẠN 1: TRÍCH XUẤT NGỮ CẢNH TỪ ẢNH (NẾU CÓ) ---
        extracted_text_from_image = ""
        if query_image:
            print("DEBUG: [Stage 1] Image detected. Calling Gemini for OCR...")
            try:
                ocr_prompt_parts = [
                    "Bạn là một hệ thống OCR toán học siêu chính xác. Hãy đọc và trích xuất toàn bộ văn bản từ hình ảnh sau đây. Chỉ trả về phần văn bản được trích xuất.", 
                    query_image
                ]
                ocr_result = resources["generator"].run(prompt_parts=ocr_prompt_parts)
                extracted_text_from_image = ocr_result["replies"][0]
                print(f"DEBUG: [Stage 1] Text extracted from image: '{extracted_text_from_image}'")
            except Exception as e:
                print(f"ERROR: [Stage 1] OCR failed: {e}")
                extracted_text_from_image = "Không thể đọc được nội dung từ hình ảnh."

        full_query_text = (query_text + " " + extracted_text_from_image).strip()
        print(f"DEBUG: [Stage 1.5] Full query text: '{full_query_text}'")

        # --- GIAI ĐOẠN 2: RAG DỰA TRÊN NGỮ CẢNH ĐẦY ĐỦ ---
        context_docs = []
        if full_query_text:
            try:
                print("DEBUG: [Stage 2] Starting RAG retrieval...")
                embedding = resources["text_embedder"].run(text=full_query_text)["embedding"]
                print("DEBUG: [Stage 2] Embedding created successfully")
                context_docs = resources["retriever"].run(query_embedding=embedding)["documents"]
                print(f"DEBUG: [Stage 2] Retrieved {len(context_docs)} documents")
            except Exception as e:
                print(f"ERROR: [Stage 2] RAG retrieval failed: {e}")
                context_docs = []

        # --- GIAI ĐOẠN 3: XÂY DỰNG PROMPT BẰNG CÁCH KẾT HỢP ---
        print("DEBUG: [Stage 3] Building final prompt...")
        
        try:
            # 3a. Sử dụng PromptBuilder để tạo phần văn bản của prompt
            informer_prompt_builder = resources["informer_prompt_builder"]
            print("DEBUG: [Stage 3a] Got informer_prompt_builder")
            
            text_prompt_result = informer_prompt_builder.run(
                query=query_text if query_text else "Giải bài toán trong hình.",
                conversation_history=conversation_history_str,
                documents=context_docs
            )
            print("DEBUG: [Stage 3a] Prompt builder ran successfully")
            
            text_part = text_prompt_result["prompt"]
            print(f"DEBUG: [Stage 3a] Generated text prompt length: {len(text_part)} chars")
            
        except Exception as e:
            print(f"ERROR: [Stage 3a] Prompt building failed: {e}")
            # Fallback to simple prompt
            text_part = f"""Bạn là gia sư toán AI. Hãy giải bài toán sau:

Câu hỏi: {query_text if query_text else "Giải bài toán trong hình"}
Nội dung từ hình: {extracted_text_from_image}

Lịch sử: {conversation_history_str}

Hãy trả lời chi tiết bằng tiếng Việt:"""

        # 3b. Xây dựng prompt_parts cuối cùng
        final_prompt_parts = [text_part] # Phần đầu tiên là toàn bộ prompt văn bản
        
        if query_image:
            # Thêm hình ảnh vào cuối
            final_prompt_parts.append("\n**Hình ảnh đính kèm:**")
            final_prompt_parts.append(query_image)
            
        print(f"DEBUG: [Stage 3b] Final prompt parts count: {len(final_prompt_parts)}")
            
        # --- GIAI ĐOẠN 4: GỌI GENERATOR VÀ VERIFIER ---
        print("DEBUG: [Stage 4] Calling Gemini for final answer...")
        try:
            final_result = resources["generator"].run(prompt_parts=final_prompt_parts)
            informer_answer = final_result["replies"][0]
            print(f"DEBUG: [Stage 4] Got answer, length: {len(informer_answer)} chars")
        except Exception as e:
            print(f"ERROR: [Stage 4] Gemini call failed: {e}")
            return f"Xin lỗi, tôi không thể xử lý câu hỏi này lúc này. Lỗi: {str(e)}"

        # --- BƯỚC XÁC THỰC (VERIFIER) ---
        try:
            print("DEBUG: [Stage 5] Starting verification...")
            verification_query = full_query_text if full_query_text else "Phân tích bài toán trong hình ảnh"
            verification = verifier_agent(verification_query, informer_answer, resources)
            print(f"DEBUG: [Stage 5] Verification result: {verification}")
            
            if verification.get("is_correct", True):
                return informer_answer
            else:
                correction = verification.get("correction_suggestion", "")
                return f"🔍 Tôi đã xem xét lại và thấy có một chút chưa chính xác. {correction}"
        except Exception as e:
            print(f"ERROR: [Stage 5] Verification failed: {e}")
            # Trả về kết quả không qua verification
            return informer_answer

    except Exception as e:
        print(f"ERROR: [Problem-Solving Engine] Critical error: {str(e)}")
        import traceback
        print(f"ERROR: [Problem-Solving Engine] Traceback: {traceback.format_exc()}")
        return f"Xin lỗi, đã có lỗi nghiêm trọng khi xử lý yêu cầu: {str(e)}"


def tutor_agent_response(user_input: str, intent: str, conversation_history_str: str, resources: Dict, supabase: Client, user_id: str, display_name: str) -> str:
    """
    Agent chính, bây giờ CHỈ xử lý các intent giao tiếp.
    Các câu hỏi toán học đã được xử lý bởi problem_solving_engine.
    """
    print(f"DEBUG: Tutor Agent is handling a communication intent: '{intent}'")
    
    # Chọn đúng prompt builder dựa trên intent
    if intent == "greeting_social":
        prompt_builder = resources["greeting_prompt_builder"]
    elif intent == "expression_of_stress":
        prompt_builder = resources["stress_prompt_builder"]
    elif intent == "study_support":
        prompt_builder = resources["support_prompt_builder"]
    elif intent == "request_for_practice":
        # request_for_practice cũng là một dạng giao tiếp
        print("DEBUG: Tutor Agent is triggering the Practice Flow.")
        insights = insight_agent(conversation_history_str, resources)
        if insights and insights.get("misunderstood_concepts"):
            weakness = insights["misunderstood_concepts"][0]
            return practice_agent(weakness, resources)
        else:
            return practice_agent("các chủ đề toán lớp 9 tổng quát", resources)
    else: # Mặc định là off_topic
        prompt_builder = resources["off_topic_prompt_builder"]
        
    try:
        # Xây dựng và chạy prompt
        prompt_text = prompt_builder.run(
            master_prompt=resources["tutor_master_prompt"],
            conversation_history=conversation_history_str
        )["prompt"]
        
        result = resources["generator"].run(prompt_parts=[prompt_text])
        return result["replies"][0]
    except Exception as e:
        print(f"ERROR: Could not generate response for intent '{intent}': {e}")
        return "Rất xin lỗi, tôi đang gặp một chút sự cố."

def render_chat_message(content: str, is_user: bool, key: str, image: bytes = None):
    """Render tin nhắn chat, có thể kèm ảnh."""
    css_class = "user-message" if is_user else "bot-message"
    
    # Nếu có ảnh, hiển thị nó trước
    if image:
        st.image(image, width=250)
        
    # Hiển thị nội dung text
    if content:
        st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

def should_trigger_proactive_practice(conversation_history: List[Dict[str, str]]) -> bool:
    """
    Kiểm tra xem có nên kích hoạt luồng luyện tập chủ động không
    bằng cách đếm số lượng intent 'math_question' đã được lưu.
    """
    print("\n--- DEBUG: [should_trigger_proactive_practice] Bắt đầu kiểm tra điều kiện ---")

    # Cần ít nhất 3 cặp hỏi-đáp (6 tin nhắn)
    if len(conversation_history) < 6:
        print("DEBUG: Kích hoạt = False. Lý do: Lịch sử chat quá ngắn.")
        return False
    
    # Lấy ra intent của 3 tin nhắn gần nhất của người dùng
    user_intents = [msg['intent'] for msg in conversation_history if msg['role'] == 'user'][-3:]
    
    if len(user_intents) < 3:
        print("DEBUG: Kích hoạt = False. Lý do: Không có đủ 3 lượt tương tác từ người dùng.")
        return False

    print(f"DEBUG: Phân tích 3 intent gần nhất của người dùng: {user_intents}")
    
    # Đếm xem có bao nhiêu trong số đó là 'math_question'
    math_question_count = user_intents.count('math_question')
    
    # Kích hoạt nếu có ít nhất 2 câu hỏi toán
    should_trigger = math_question_count >= 2

    print(f"DEBUG: Tổng số intent 'math_question': {math_question_count}/3.")
    print(f"DEBUG: Kích hoạt = {should_trigger}.")
    print("--- KẾT THÚC KIỂM TRA ---")
    
    return should_trigger


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
    # XÓA TOÀN BỘ KHỐI if user_input: CŨ VÀ THAY BẰNG KHỐI NÀY

    with st.form(key="chat_form", clear_on_submit=True):
        # Chia layout thành 2 cột: 1 cho ảnh, 1 cho text và nút gửi
        col1, col2 = st.columns([1, 4])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Đính kèm ảnh", 
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed"
            )

        with col2:
            user_text = st.text_input(
                "Nhập câu hỏi của bạn...",
                placeholder="Nhập câu hỏi hoặc mô tả cho ảnh...",
                label_visibility="collapsed"
            )
        
        # Nút gửi chung cho cả form, đặt ở ngoài layout cột để nó chiếm toàn bộ chiều rộng
        submit_button = st.form_submit_button(label="Gửi")

    # --- XỬ LÝ SAU KHI NGƯỜI DÙNG NHẤN NÚT GỬI ---
    if submit_button and (user_text or uploaded_image):
        
        # Đọc dữ liệu ảnh nếu có
        image_data = uploaded_image.getvalue() if uploaded_image else None
        
        # BƯỚC 1: LƯU VÀ HIỂN THỊ TIN NHẮN CỦA USER
        st.session_state.messages.append({
            "role": "user", 
            "content": user_text, 
            "image": image_data, # Lưu dữ liệu bytes của ảnh
            "intent": "unknown"
        })
        with chat_placeholder:
             # Nhớ đảm bảo hàm render_chat_message của bạn đã được cập nhật để nhận tham số 'image'
             render_chat_message(user_text, is_user=True, image=image_data, key=f"user_{len(st.session_state.messages)}")

        # BƯỚC 2: PHÂN LOẠI INTENT (DỰA TRÊN PHẦN TEXT)
        history_str_for_llm = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-10:] if msg['content']])
    
        # Phân loại intent (vẫn dựa trên text)
        detected_intent = classify_intent(history_str_for_llm, resources)
        st.session_state.messages[-1]["intent"] = detected_intent
        
        # Hiển thị indicator "đang suy nghĩ"
        with chat_placeholder:
            typing_indicator_placeholder = show_typing_indicator()

        # --- LOGIC ĐIỀU PHỐI MỚI ---
        # Nếu có ảnh, hoặc nếu intent là câu hỏi toán, hãy gọi cỗ máy đa năng
        if uploaded_image or detected_intent == "math_question":
            bot_response = problem_solving_engine(
                query_text=user_text,
                query_image=image_data,
                conversation_history_str=history_str_for_llm,
                resources=resources
            )
        else:
            # Các intent giao tiếp khác chỉ xử lý text
            bot_response = tutor_agent_response(
                user_input=user_text, 
                intent=detected_intent,
                conversation_history_str=history_str_for_llm,
                resources=resources,
                supabase=supabase,
                user_id=user_id,
                display_name=display_name
            )
        
        typing_indicator_placeholder.empty()

        # BƯỚC 4: LƯU VÀ HIỂN THỊ PHẢN HỒI CỦA BOT
        st.session_state.messages.append({"role": "assistant", "content": bot_response, "intent": detected_intent, "image": None})
        with chat_placeholder:
            render_chat_message(bot_response, is_user=False, key=f"bot_{len(st.session_state.messages)}")

        # --- BƯỚC 5: KIỂM TRA PROACTIVE VỚI HÀM MỚI THÔNG MINH ---
        # Hàm mới này sẽ đọc intent trực tiếp từ st.session_state.messages
        if should_trigger_proactive_practice(st.session_state.messages):
    
            # Hiển thị indicator trên giao diện để người dùng biết hệ thống đang làm thêm việc
            with chat_placeholder:
                proactive_typing_placeholder = show_typing_indicator()
            
            try:
                # --- DEBUG: Thông báo bắt đầu luồng proactive ---
                print("\n--- DEBUG: [Proactive Flow] Bắt đầu luồng phân tích và đề xuất ---")
                
                # Tạo history_str để gửi cho Insight Agent
                history_str_for_insight = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-10:]])
                
                # Gọi Insight Agent
                print("DEBUG: [Proactive Flow] Gọi Insight Agent...")
                insights = insight_agent(history_str_for_insight, resources)
                print(f"DEBUG: [Proactive Flow] Insight Agent trả về: {insights}")
                
                # --- LOGIC XỬ LÝ KẾT QUẢ CỦA INSIGHT ---
                # Kiểm tra xem insight có hợp lệ và có chứa khái niệm bị hiểu sai không
                if insights and isinstance(insights, dict) and insights.get("misunderstood_concepts"):
                    
                    concepts = insights["misunderstood_concepts"]
                    last_weakness = concepts[0]
                    user_email = user.email
                    
                    profile_data_to_save = {
                        "email": user_email, 
                        "misunderstood_concepts": concepts,
                        "last_weakness": last_weakness,
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    print(f"DEBUG: [Proactive Flow] Chuẩn bị cập nhật profile cho user_id: {user_id}")
                    
                    # GỌI HÀM UPDATE ĐÃ ĐƯỢC IMPORT
                    update_user_profile(supabase, user_id, profile_data_to_save)
                    
                    st.toast("✅ Đã phân tích và cập nhật hồ sơ học tập!", icon="🧠")
                    print(f"DEBUG: [Proactive Flow] Phát hiện điểm yếu: '{last_weakness}'. Gọi Practice Agent...")
                    
                    # Gọi Practice Agent để tạo bài tập
                    practice_response = practice_agent(last_weakness, resources)
                    
                    # Tạo tin nhắn đề xuất
                    proactive_msg = f"💡 **Phân tích nhanh:** Dựa trên các câu hỏi vừa rồi, tôi nhận thấy bạn có thể cần luyện tập thêm về chủ đề **'{last_weakness}'**. Đây là một số gợi ý cho bạn:\n\n{practice_response}"
                    
                    # Xóa indicator và thêm tin nhắn mới vào state
                    proactive_typing_placeholder.empty()
                    st.session_state.messages.append({"role": "assistant", "content": proactive_msg, "intent": "proactive_suggestion"})
                    
                    # Hiển thị tin nhắn proactive
                    with chat_placeholder:
                        render_chat_message(proactive_msg, is_user=False, key=f"proactive_{len(st.session_state.messages)}")
                
                else:
                    # --- DEBUG: Trường hợp không có gì để đề xuất ---
                    print("DEBUG: [Proactive Flow] Insight Agent không tìm thấy điểm yếu nào cụ thể. Bỏ qua đề xuất.")
                    # Xóa indicator một cách thầm lặng, không làm gì thêm
                    proactive_typing_placeholder.empty()

            except Exception as e:
                # --- DEBUG: Bắt lỗi trong luồng proactive ---
                print(f"ERROR: [Proactive Flow] Đã xảy ra lỗi: {str(e)}")
                proactive_typing_placeholder.empty()
                # Có thể hiển thị một cảnh báo nhỏ trên UI nếu cần
                # st.warning("Không thể tạo đề xuất chủ động lúc này.")

        # Rerun để cập nhật giao diện
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