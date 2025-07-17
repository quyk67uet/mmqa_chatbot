# 🤖 Chatbot Gia sư Toán AI Đa Agent

Một ứng dụng chatbot thông minh sử dụng kiến trúc đa agent để hỗ trợ học sinh lớp 9 học Toán theo chương trình Việt Nam.

## 🎯 Tính năng chính

### 5 Agent thông minh:
1. **Informer Agent**: Giải bài toán dựa trên sách giáo khoa (RAG)
2. **Practice Agent**: Tạo bài tập và đề xuất video phù hợp
3. **Insight Agent**: Phân tích điểm yếu và cảm xúc học sinh
4. **Verifier Agent**: Kiểm tra tính đúng đắn của câu trả lời
5. **Tutor Agent**: Điều phối các agent và quản lý hội thoại

### Các luồng tương tác:
- ✅ **Giải toán chi tiết**: Trả lời từng bước với kiểm tra chéo
- 🎯 **Luyện tập chủ động**: Tự động đề xuất bài tập dựa trên điểm yếu
- 💚 **Hỗ trợ cảm xúc**: Nhận biết stress và phản hồi đồng cảm
- 🚫 **Từ chối an toàn**: Chỉ tập trung vào toán học

## 🛠️ Thiết lập

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Thiết lập Google AI API Key
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Hoặc trên Windows:
```cmd
set GOOGLE_API_KEY=your_api_key_here
```

### 3. Chuẩn bị dữ liệu
Đảm bảo có 2 file trong thư mục gốc:
- `embedded_documents.pkl`: Documents sách giáo khoa đã được embedding
- `videos.json`: Danh sách thông tin video bài giảng

### 4. Chạy ứng dụng
```bash
streamlit run app.py
```

## 📚 Cấu trúc hệ thống

### Kiến trúc Multi-Agent
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Tutor     │────│  Informer   │────│  Verifier   │
│   Agent     │    │   Agent     │    │   Agent     │
│ (Điều phối) │    │   (RAG)     │    │ (Kiểm tra)  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌─────────────┐
│  Practice   │    │   Insight   │
│   Agent     │    │   Agent     │
│(Tạo bài tập)│    │(Phân tích)  │
└─────────────┘    └─────────────┘
```

### Luồng xử lý
1. **Phân loại ý định** → Xác định loại câu hỏi
2. **Điều phối agent** → Gọi agent phù hợp
3. **Xử lý và kiểm tra** → Tạo câu trả lời + verification
4. **Phân tích chủ động** → Đề xuất bài tập theo điểm yếu

## 🎮 Cách sử dụng

### Các loại câu hỏi được hỗ trợ:

1. **Câu hỏi toán học**:
   - "Giải phương trình x + 5 = 10"
   - "Tính diện tích hình tròn bán kính 3cm"
   - "Chứng minh định lý Pythagore"

2. **Yêu cầu luyện tập**:
   - "Cho tôi bài tập về phương trình"
   - "Tôi muốn luyện tập thêm"

3. **Biểu đạt stress**:
   - "Tôi mệt quá"
   - "Khó hiểu quá"
   - "Không làm được"

4. **Câu hỏi ngoài chuyên môn**:
   - Hệ thống sẽ từ chối và hướng về toán học

## 🔧 Công nghệ sử dụng

- **Framework**: Haystack-AI 2.x
- **LLM**: Google Gemini 1.5 Pro
- **Embedding**: Vietnamese BI-Encoder
- **UI**: Streamlit + Streamlit-Chat
- **Vector Store**: InMemoryDocumentStore

## 📊 Tối ưu hóa

### Caching thông minh:
- `@st.cache_resource` cho việc tải models
- Session state cho lịch sử chat
- Embedding được cache tự động

### Memory Management:
- Giới hạn lịch sử chat (10 tin nhắn gần nhất)
- Lazy loading cho các component

## 🤝 Đóng góp

Dự án này là case study cho AIQAM'25 Workshop. Nếu bạn muốn đóng góp:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - Chi tiết xem file LICENSE

## 🎓 Tác giả

Dự án được phát triển cho Workshop AIQAM'25 về Interactive QA Systems.

---
**Lưu ý**: Cần thiết lập Google AI API key để sử dụng. Tham khảo [Google AI Studio](https://makersuite.google.com/app/apikey) để lấy key. 