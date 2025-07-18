import streamlit as st
from datetime import datetime
from supabase_utils import init_supabase_client, get_user_profile

# Thiết lập page config
st.set_page_config(
    page_title="Hồ sơ Học tập - Gia sư Toán AI",
    page_icon="👤",
    layout="wide"
)

def display_header(user_email: str, user_id: str, profile: dict):
    """
    Hiển thị header với thông tin người dùng
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;'>
        <h1 style='color: white; margin-bottom: 1rem; font-size: 2.5rem; text-align: center;'>
            👤 Hồ sơ Học tập của bạn
        </h1>
        <p style='color: #f0f0f0; font-size: 1.2rem; text-align: center; margin-bottom: 0;'>
            Theo dõi tiến trình và điểm yếu cần cải thiện
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Thông tin người dùng trong cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h3 style='color: white; margin-bottom: 0.5rem; display: flex; align-items: center;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>📧</span>
                Email
            </h3>
            <p style='color: #f0f0f0; margin-bottom: 0; font-size: 1.1rem;'>
                {user_email}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        updated_time = "Chưa có dữ liệu"
        if profile.get("updated_at"):
            try:
                updated_dt = datetime.fromisoformat(profile["updated_at"].replace("Z", "+00:00"))
                updated_time = updated_dt.strftime('%d/%m/%Y %H:%M:%S')
            except:
                updated_time = "Không xác định"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;'>
            <h3 style='color: white; margin-bottom: 0.5rem; display: flex; align-items: center;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🕒</span>
                Cập nhật lần cuối
            </h3>
            <p style='color: #f0f0f0; margin-bottom: 0; font-size: 1.1rem;'>
                {updated_time}
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_weakness_and_stats(profile: dict):
    """
    Hiển thị điểm yếu gần nhất và thống kê
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎯 Điểm yếu gần nhất")
        
        if profile.get("last_weakness"):
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                        padding: 1.5rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem;'>
                <h3 style='color: #8B0000; margin-bottom: 0.5rem; text-align: center;'>
                    {profile['last_weakness']}
                </h3>
                <p style='color: #2F4F4F; margin-bottom: 0; text-align: center; font-style: italic;'>
                    💡 Hệ thống sẽ ưu tiên tạo bài tập về chủ đề này
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1.5rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem;
                        border: 2px dashed #ccc;'>
                <h4 style='color: #666; text-align: center; margin-bottom: 0.5rem;'>
                    Chưa có dữ liệu phân tích
                </h4>
                <p style='color: #888; text-align: center; margin-bottom: 0;'>
                    Hãy chat với gia sư AI để hệ thống có thể phân tích điểm yếu của bạn!
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📊 Thống kê tổng quan")
        
        misunderstood_count = len(profile.get("misunderstood_concepts", []))
        
        # Tạo màu động dựa trên số lượng
        if misunderstood_count == 0:
            color_gradient = "linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)"
            text_color = "#2F4F4F"
            icon = "🎉"
        elif misunderstood_count <= 3:
            color_gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            text_color = "#8B0000"
            icon = "⚠️"
        else:
            color_gradient = "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)"
            text_color = "#ffffff"
            icon = "🔥"
        
        st.markdown(f"""
        <div style='background: {color_gradient}; 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;
                    text-align: center;'>
            <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
                <span style='font-size: 2rem; margin-right: 0.5rem;'>{icon}</span>
                <h3 style='color: {text_color}; margin: 0;'>
                    Khái niệm cần cải thiện
                </h3>
            </div>
            <div style='font-size: 3rem; font-weight: bold; color: {text_color}; margin-bottom: 0.5rem;'>
                {misunderstood_count}
            </div>
            <p style='color: {text_color}; margin: 0; opacity: 0.9;'>
                {'Tuyệt vời!' if misunderstood_count == 0 else 'Cần cải thiện' if misunderstood_count <= 3 else 'Cần tập trung cao'}
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_concepts_list(profile: dict):
    """
    Hiển thị danh sách khái niệm cần cải thiện
    """
    st.markdown("## 📝 Danh sách khái niệm cần cải thiện")
    
    misunderstood_concepts = profile.get("misunderstood_concepts", [])
    
    if misunderstood_concepts:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin-bottom: 1.5rem;'>
            <p style='color: #2d3436; margin: 0; text-align: center; font-weight: 500;'>
                📚 Dưới đây là những khái niệm mà hệ thống phát hiện bạn còn gặp khó khăn
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị concepts trong grid
        cols = st.columns(2)
        for i, concept in enumerate(misunderstood_concepts):
            col_idx = i % 2
            with cols[col_idx]:
                # Màu sắc xoay vòng cho từng concept
                colors = [
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
                    "#FECA57", "#FF9FF3", "#54a0ff", "#5f27cd"
                ]
                color = colors[i % len(colors)]
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}20, {color}10); 
                            padding: 1rem; 
                            border-radius: 10px; 
                            margin-bottom: 1rem;
                            border-left: 4px solid {color};'>
                    <div style='display: flex; align-items: center;'>
                        <span style='background: {color}; 
                                     color: white; 
                                     border-radius: 50%; 
                                     width: 30px; 
                                     height: 30px; 
                                     display: flex; 
                                     align-items: center; 
                                     justify-content: center; 
                                     margin-right: 0.75rem;
                                     font-weight: bold;'>
                            {i+1}
                        </span>
                        <h4 style='color: #333; margin: 0; flex: 1;'>
                            {concept}
                        </h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    text-align: center;
                    margin-bottom: 1.5rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🎉</div>
            <h3 style='color: white; margin-bottom: 1rem;'>
                Tuyệt vời! Chưa có khái niệm nào được đánh dấu là yếu.
            </h3>
            <p style='color: #dff9fb; margin-bottom: 1rem;'>
                Điều này có thể có nghĩa là:
            </p>
            <div style='background: rgba(255,255,255,0.1); 
                        padding: 1rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem;'>
                <p style='color: #dff9fb; margin: 0;'>
                    • Bạn chưa chat đủ với gia sư AI để hệ thống phân tích<br>
                    • Bạn đang làm rất tốt với các bài toán đã thảo luận
                </p>
            </div>
            <p style='color: #dff9fb; margin: 0;'>
                💡 <strong>Gợi ý:</strong> Hãy tiếp tục chat và hỏi nhiều câu hỏi toán học để gia sư AI có thể đưa ra các phân tích và đề xuất phù hợp nhé!
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_learning_suggestions(misunderstood_concepts: list):
    """
    Hiển thị gợi ý học tập
    """
    if misunderstood_concepts:
        st.markdown("## 🎯 Gợi ý học tập")
        
        suggestions = [
            {
                "icon": "✅",
                "title": "Tạo bài tập luyện tập",
                "description": "Yêu cầu gia sư AI tạo bài tập luyện tập cụ thể về các khái niệm yếu",
                "color": "#00b894"
            },
            {
                "icon": "🧠",
                "title": "Giải thích lại khái niệm",
                "description": "Hỏi gia sư AI giải thích lại các khái niệm khó hiểu bằng cách khác",
                "color": "#6c5ce7"
            },
            {
                "icon": "📹",
                "title": "Xem video bài giảng",
                "description": "Tham khảo các video bài giảng được đề xuất theo level phù hợp",
                "color": "#fd79a8"
            },
            {
                "icon": "📅",
                "title": "Luyện tập đều đặn",
                "description": "Dành 15-30 phút mỗi ngày để thực hành những khái niệm yếu",
                "color": "#fdcb6e"
            }
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {suggestion['color']}20, {suggestion['color']}10); 
                            padding: 1.5rem; 
                            border-radius: 10px; 
                            margin-bottom: 1rem;
                            border-left: 4px solid {suggestion['color']};
                            height: 140px;'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{suggestion['icon']}</span>
                        <h4 style='color: #333; margin: 0;'>{suggestion['title']}</h4>
                    </div>
                    <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                        {suggestion['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Cách sử dụng
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-top: 1rem;'>
            <h4 style='color: white; margin-bottom: 1rem; text-align: center;'>
                💬 Cách sử dụng
            </h4>
            <div style='background: rgba(255,255,255,0.1); 
                        padding: 1rem; 
                        border-radius: 8px;'>
                <p style='color: #ddd; margin: 0; text-align: center;'>
                    Quay lại trang chính và nói với gia sư AI:<br>
                    <strong style='color: white;'>"Tôi muốn luyện tập"</strong> hoặc 
                    <strong style='color: white;'>"Cho tôi bài tập về [tên chủ đề]"</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_footer():
    """
    Hiển thị footer
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Làm mới dữ liệu", type="primary", use_container_width=True):
            st.rerun()
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                text-align: center;
                margin-top: 2rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>
            💡 Về hồ sơ học tập
        </h3>
        <p style='color: #f0f0f0; margin-bottom: 0;'>
            Dữ liệu được cập nhật tự động khi bạn chat với gia sư AI.<br>
            Hồ sơ này giúp cá nhân hóa trải nghiệm học tập của bạn để đạt hiệu quả tốt nhất.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_login_required():
    """
    Hiển thị thông báo cần đăng nhập
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff7675 0%, #fd79a8 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                text-align: center;
                margin: 2rem 0;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>⚠️</div>
        <h2 style='color: white; margin-bottom: 1rem;'>
            Bạn cần đăng nhập để xem hồ sơ học tập
        </h2>
        <p style='color: #ffeaa7; font-size: 1.2rem; margin-bottom: 0;'>
            👈 Vui lòng đăng nhập ở sidebar của trang chính
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """
    Hàm chính của trang Hồ sơ học tập
    """
    # Kiểm tra xem người dùng đã đăng nhập chưa
    if "user" not in st.session_state or st.session_state.user is None:
        display_login_required()
        st.stop()
    
    # Khởi tạo Supabase client
    supabase = init_supabase_client()
    
    # Lấy thông tin người dùng hiện tại
    user = st.session_state.user
    user_email = user.email
    user_id = user.id
    
    # Lấy hồ sơ học tập
    with st.spinner("Đang tải hồ sơ học tập..."):
        profile = get_user_profile(supabase, user_id)

    # Kiểm tra xem profile có dữ liệu không
    if not profile:
        st.error("Không thể tải hoặc tạo hồ sơ của bạn. Vui lòng thử lại.")
        st.stop()
    
    # Hiển thị header
    display_header(user_email, user_id, profile)
    
    # Hiển thị điểm yếu và thống kê
    display_weakness_and_stats(profile)
    
    st.markdown("---")
    
    # Hiển thị danh sách khái niệm
    display_concepts_list(profile)
    
    # Hiển thị gợi ý học tập
    misunderstood_concepts = profile.get("misunderstood_concepts", [])
    display_learning_suggestions(misunderstood_concepts)
    
    # Hiển thị footer
    display_footer()

if __name__ == "__main__":
    main()