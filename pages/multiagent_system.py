import streamlit as st
import os
from supabase import create_client, Client
from datetime import datetime
import json

# Thiết lập page config
st.set_page_config(
    page_title="Multi-Agent System - Gia sư Toán AI",
    page_icon="🤖",
    layout="wide"
)

def init_supabase_client() -> Client:
    """
    Khởi tạo Supabase client
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.error("Không tìm thấy SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
        st.stop()
    
    return create_client(url, key)

def display_system_architecture():
    """
    Hiển thị kiến trúc hệ thống
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                text-align: center;'>
        <h1 style='color: white; margin-bottom: 1rem; font-size: 2.5rem;'>
            🤖 An Adaptive Multi-Agent Tutoring System
        </h1>
        <p style='color: #f0f0f0; font-size: 1.2rem; margin-bottom: 0;'>
            Hệ thống gia sư thông minh với kiến trúc đa tác nhân thích ứng
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder cho hình ảnh system architecture
    st.markdown("### 📊 Kiến trúc Hệ thống")
    
    # Container cho hình ảnh với styling đẹp
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("images/system.jpg", use_container_width=True)

def display_agents():
    """
    Hiển thị chi tiết từng Agent với giao diện được tối ưu hóa
    """
    # Header section
    st.markdown("""
    <div style="margin: 3rem 0 2rem 0;">
        <h2 style="text-align: center; color: #333; margin-bottom: 0.5rem; font-size: 2.5rem;">
            🤖 Chi tiết các AI Agents
        </h2>
        <p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
            Khám phá hệ sinh thái AI agents thông minh và chuyên biệt
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CSS được tối ưu hóa - loại bỏ các effect phức tạp
    st.markdown("""
    <style>
    .agent-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .agent-header {
        padding: 2rem;
        color: white;
    }
    
    .agent-title {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .agent-title-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
    }
    
    .agent-role {
        background: rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .agent-body {
        padding: 2rem;
        background: #fafafa;
    }
    
    .agent-features {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .features-title {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
    }
    
    .features-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-item {
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        display: flex;
        align-items: flex-start;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-bullet {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 1rem;
        margin-top: 0.5rem;
        flex-shrink: 0;
    }
    
    .feature-text {
        flex: 1;
        line-height: 1.6;
    }
    
    .feature-name {
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .image-placeholder {
        background: rgba(255,255,255,0.9);
        border: 2px dashed #ddd;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    .placeholder-text {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    .placeholder-path {
        font-size: 0.9rem;
        color: #666;
        font-family: monospace;
        background: rgba(0,0,0,0.05);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Định nghĩa thông tin các agents
    agents = [
        {
            "number": "1",
            "name": "Informer Agent",
            "icon": "🧠",
            "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "color": "#667eea",
            "role": "Chuyên gia giải toán với khả năng truy xuất thông tin thông minh",
            "image": "images/informer_agent.png",
            "features": [
                "RAG Pipeline: Sử dụng Retrieval-Augmented Generation để tìm kiếm thông tin từ sách giáo khoa",
                "Embedding Search: Tìm kiếm ngữ nghĩa thông minh với Vietnamese bi-encoder",
                "Detailed Solutions: Sinh lời giải chi tiết, từng bước dễ hiểu",
                "Context Awareness: Hiểu ngữ cảnh câu hỏi để đưa ra câu trả lời phù hợp"
            ]
        },
        {
            "number": "2",
            "name": "Practice Agent",
            "icon": "📝",
            "gradient": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "color": "#f093fb",
            "role": "Chuyên gia tạo bài tập và đề xuất tài liệu học tập",
            "image": "images/practice_agent.png",
            "features": [
                "Exercise Generation: Tạo bài tập mới phù hợp với điểm yếu của học sinh",
                "Video Recommendation: Đề xuất video học tập từ knowledge base",
                "Adaptive Difficulty: Điều chỉnh độ khó bài tập theo trình độ",
                "Creative Problems: Soạn câu hỏi sáng tạo, không có trong sách"
            ]
        },
        {
            "number": "3",
            "name": "Insight Agent",
            "icon": "🔍",
            "gradient": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "color": "#4facfe",
            "role": "Chuyên gia phân tích tâm lý và học tập của học sinh",
            "image": "images/insight_agent.png",
            "features": [
                "Weakness Analysis: Phân tích điểm yếu từ lịch sử hội thoại",
                "Sentiment Detection: Nhận diện cảm xúc và tâm trạng học sinh",
                "Learning Pattern: Phát hiện patterns trong cách học của học sinh",
                "Personalized Insights: Đưa ra những insight cá nhân hóa"
            ]
        },
        {
            "number": "4",
            "name": "Verifier Agent",
            "icon": "✅",
            "gradient": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
            "color": "#a8edea",
            "role": "Chuyên gia kiểm tra chất lượng và tính chính xác",
            "image": "images/verifier_agent.png",
            "features": [
                "Solution Verification: Kiểm tra tính đúng đắn của lời giải",
                "Error Detection: Phát hiện và chỉ ra lỗi sai",
                "Quality Assurance: Đảm bảo chất lượng câu trả lời",
                "Correction Suggestions: Đưa ra gợi ý sửa lỗi"
            ]
        },
        {
            "number": "5",
            "name": "Tutor Agent (Orchestrator)",
            "icon": "🎯",
            "gradient": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
            "color": "#fcb69f",
            "role": "Trung tâm điều phối và quản lý toàn bộ hệ thống",
            "image": "images/tutor_agent.png",
            "features": [
                "Intent Classification: Phân loại ý định người dùng (greeting, math_question, practice, stress, etc.)",
                "Agent Orchestration: Điều phối các agent khác theo luồng phù hợp",
                "Conversation Management: Quản lý bối cảnh và luồng hội thoại",
                "Proactive Learning: Chủ động đề xuất bài tập khi phát hiện cơ hội",
                "Emotional Support: Hỗ trợ tâm lý khi học sinh gặp khó khăn"
            ]
        }
    ]
    
    # Hiển thị từng agent
    for i, agent in enumerate(agents):
        # Container chính
        st.markdown(f"""
        <div class="agent-container">
            <div class="agent-header" style="background: {agent['gradient']};">
                <div class="agent-title">
                    <span class="agent-title-icon">{agent['icon']}</span>
                    <span>Agent {agent['number']}: {agent['name']}</span>
                </div>
                <div class="agent-role">
                    <strong>Vai trò:</strong> {agent['role']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Body với layout 2 cột
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="agent-body">
                <div class="agent-features">
                    <div class="features-title">
                        <span style="margin-right: 0.5rem;">🔧</span>
                        <span>Chức năng chính</span>
                    </div>
                    <ul class="features-list">
            """, unsafe_allow_html=True)
            
            for feature in agent['features']:
                parts = feature.split(': ', 1)
                if len(parts) == 2:
                    st.markdown(f"""
                    <li class="feature-item">
                        <div class="feature-bullet" style="background: {agent['color']};"></div>
                        <div class="feature-text">
                            <span class="feature-name" style="color: {agent['color']};">{parts[0]}:</span>
                            <span>{parts[1]}</span>
                        </div>
                    </li>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <li class="feature-item">
                        <div class="feature-bullet" style="background: {agent['color']};"></div>
                        <div class="feature-text">{feature}</div>
                    </li>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                st.image(agent['image'], use_container_width=True, caption=f"Agent {agent['number']}: {agent['name']}")
            except:
                st.markdown(f"""
                <div class="agent-body">
                    <div class="image-placeholder">
                        <div class="placeholder-icon" style="color: {agent['color']};">{agent['icon']}</div>
                        <div class="placeholder-text">Agent {agent['number']} Image</div>
                        <div class="placeholder-path">{agent['image']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Đóng container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Spacing giữa các agents
        if i < len(agents) - 1:
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

def display_system_components():
    """
    Hiển thị các thành phần của hệ thống
    """
    st.markdown("## 🏗️ Các Thành phần Hệ thống")
    
    # Orchestration Hub
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h3 style='color: white; margin-bottom: 0.5rem;'>
                🎯 Orchestration Hub
            </h3>
            <p style='color: #f0f0f0; margin-bottom: 0;'>
                Trung tâm điều phối và quản lý tất cả các tác nhân trong hệ thống
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Hai flows chính
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h3 style='color: #8B0000; margin-bottom: 1rem;'>
                📝 Flow A: Problem-Solving
            </h3>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;'>
                <h4 style='color: #8B0000; margin-bottom: 0.5rem;'>RAG & Verification</h4>
                <ul style='color: #2F4F4F; margin-bottom: 0;'>
                    <li>Xử lý truy vấn RAG</li>
                    <li>Giải quyết yêu cầu</li>
                    <li>Xác thực và soạn thảo</li>
                    <li>Cung cấp câu trả lời cuối cùng</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h3 style='color: #2F4F4F; margin-bottom: 1rem;'>
                🎯 Flow B: Personalization
            </h3>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;'>
                <h4 style='color: #2F4F4F; margin-bottom: 0.5rem;'>Learning Engine</h4>
                <ul style='color: #2F4F4F; margin-bottom: 0;'>
                    <li>Phân tích thông tin người dùng</li>
                    <li>Gửi chủ đề phù hợp</li>
                    <li>Truy vấn dữ liệu</li>
                    <li>Tạo bài tập cá nhân hóa</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_workflow_details():
    """
    Hiển thị chi tiết quy trình làm việc - được tối ưu hóa
    """
    st.markdown("## 🔄 Quy trình Hoạt động")
    
    # Workflow steps
    workflow_steps = [
        {
            "step": "1",
            "title": "Nhận câu hỏi từ học sinh",
            "description": "Hệ thống nhận và phân tích câu hỏi từ người dùng",
            "icon": "❓",
            "color": "#FF6B6B"
        },
        {
            "step": "2", 
            "title": "Orchestration Hub phân tích",
            "description": "Trung tâm điều phối quyết định luồng xử lý phù hợp",
            "icon": "🎯",
            "color": "#4ECDC4"
        },
        {
            "step": "3",
            "title": "Xử lý song song",
            "description": "Flow A giải quyết vấn đề, Flow B cá nhân hóa trải nghiệm",
            "icon": "⚡",
            "color": "#45B7D1"
        },
        {
            "step": "4",
            "title": "Cập nhật hồ sơ học sinh",
            "description": "Hệ thống cập nhật thông tin học tập và điểm yếu",
            "icon": "📊",
            "color": "#96CEB4"
        },
        {
            "step": "5",
            "title": "Trả lời cuối cùng",
            "description": "Cung cấp câu trả lời được xác thực và cá nhân hóa",
            "icon": "✅",
            "color": "#FECA57"
        }
    ]
    
    for i, step in enumerate(workflow_steps):
        # Tạo layout cho từng bước
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"""
            <div style='width: 70px; 
                        height: 70px; 
                        background: {step['color']}; 
                        border-radius: 50%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        margin: 0 auto;'>
                <span style='font-size: 1.8rem;'>{step['icon']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: {step['color']}20; 
                        padding: 1rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem;
                        border-left: 4px solid {step['color']};'>
                <h4 style='color: #333; margin-bottom: 0.5rem;'>
                    Bước {step['step']}: {step['title']}
                </h4>
                <p style='color: #666; margin-bottom: 0;'>
                    {step['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Thêm mũi tên giữa các bước (trừ bước cuối)
        if i < len(workflow_steps) - 1:
            st.markdown("""
            <div style='text-align: center; margin: 1rem 0;'>
                <span style='font-size: 1.5rem; color: #ccc;'>⬇️</span>
            </div>
            """, unsafe_allow_html=True)

def display_technical_features():
    """
    Hiển thị các tính năng kỹ thuật
    """
    st.markdown("## 🛠️ Tính năng Kỹ thuật")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h4 style='color: white; margin-bottom: 1rem;'>
                🔍 RAG (Retrieval-Augmented Generation)
            </h4>
            <ul style='color: #f0f0f0; margin-bottom: 0;'>
                <li>Truy xuất thông tin từ cơ sở dữ liệu</li>
                <li>Tăng cường khả năng sinh text</li>
                <li>Đảm bảo tính chính xác cao</li>
                <li>Xác thực câu trả lời</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h4 style='color: white; margin-bottom: 1rem;'>
                🎯 Personalization Engine
            </h4>
            <ul style='color: #f0f0f0; margin-bottom: 0;'>
                <li>Phân tích điểm yếu học sinh</li>
                <li>Tạo bài tập cá nhân hóa</li>
                <li>Theo dõi tiến độ học tập</li>
                <li>Đề xuất nội dung phù hợp</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h4 style='color: white; margin-bottom: 1rem;'>
                🧠 Adaptive Learning
            </h4>
            <ul style='color: #f0f0f0; margin-bottom: 0;'>
                <li>Học thích ứng theo năng lực</li>
                <li>Điều chỉnh độ khó tự động</li>
                <li>Phản hồi thời gian thực</li>
                <li>Tối ưu hóa trải nghiệm</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h4 style='color: white; margin-bottom: 1rem;'>
                📊 User Profile Management
            </h4>
            <ul style='color: #f0f0f0; margin-bottom: 0;'>
                <li>Lưu trữ lịch sử học tập</li>
                <li>Phân tích pattern học tập</li>
                <li>Cập nhật thông tin real-time</li>
                <li>Báo cáo tiến độ chi tiết</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_benefits():
    """
    Hiển thị lợi ích của hệ thống - được tối ưu hóa
    """
    st.markdown("## 🌟 Lợi ích của Hệ thống")
    
    benefits = [
        {
            "icon": "🎯",
            "title": "Học tập cá nhân hóa",
            "description": "Mỗi học sinh nhận được trải nghiệm học tập được tùy chỉnh theo nhu cầu riêng",
            "color": "#FF6B6B"
        },
        {
            "icon": "⚡",
            "title": "Phản hồi tức thì",
            "description": "Hệ thống đa tác nhân cho phép xử lý song song, đưa ra phản hồi nhanh chóng",
            "color": "#4ECDC4"
        },
        {
            "icon": "🧠",
            "title": "Học thích ứng thông minh",
            "description": "AI phân tích và điều chỉnh phương pháp dạy theo tiến bộ của từng học sinh",
            "color": "#45B7D1"
        },
        {
            "icon": "📊",
            "title": "Theo dõi tiến độ chi tiết",
            "description": "Báo cáo và phân tích tiến độ học tập giúp tối ưu hóa quá trình học",
            "color": "#96CEB4"
        },
        {
            "icon": "🔍",
            "title": "Độ chính xác cao",
            "description": "Hệ thống RAG và xác thực đảm bảo thông tin chính xác và đáng tin cậy",
            "color": "#FECA57"
        },
        {
            "icon": "🌐",
            "title": "Mở rộng dễ dàng",
            "description": "Kiến trúc đa tác nhân cho phép mở rộng và thêm tính năng mới linh hoạt",
            "color": "#FF9FF3"
        }
    ]
    
    # Hiển thị benefits trong lưới 2x3
    for i in range(0, len(benefits), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(benefits):
                benefit = benefits[i]
                st.markdown(f"""
                <div style='background: {benefit['color']}20; 
                            padding: 1.5rem; 
                            border-radius: 10px; 
                            margin-bottom: 1rem;
                            border-left: 4px solid {benefit['color']};
                            height: 130px;'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.8rem; margin-right: 0.5rem;'>{benefit['icon']}</span>
                        <h4 style='color: #333; margin: 0;'>{benefit['title']}</h4>
                    </div>
                    <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                        {benefit['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if i + 1 < len(benefits):
                benefit = benefits[i + 1]
                st.markdown(f"""
                <div style='background: {benefit['color']}20; 
                            padding: 1.5rem; 
                            border-radius: 10px; 
                            margin-bottom: 1rem;
                            border-left: 4px solid {benefit['color']};
                            height: 130px;'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.8rem; margin-right: 0.5rem;'>{benefit['icon']}</span>
                        <h4 style='color: #333; margin: 0;'>{benefit['title']}</h4>
                    </div>
                    <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                        {benefit['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """
    Hàm chính của trang Multi-Agent System
    """
    # Hiển thị kiến trúc hệ thống
    display_system_architecture()

    # Hiển thị các tác nhân
    display_agents()
    
    # Hiển thị các thành phần hệ thống
    display_system_components()
    
    # Hiển thị quy trình hoạt động
    display_workflow_details()
    
    # Hiển thị tính năng kỹ thuật
    display_technical_features()
    
    # Hiển thị lợi ích
    display_benefits()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                margin-top: 2rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>
            🚀 Trải nghiệm Hệ thống Multi-Agent ngay hôm nay!
        </h3>
        <p style='color: #f0f0f0; font-size: 1.1rem; margin-bottom: 0;'>
            Hệ thống gia sư AI thông minh với kiến trúc đa tác nhân tiên tiến, 
            mang đến trải nghiệm học tập cá nhân hóa và hiệu quả nhất.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()