import os
from supabase import create_client, Client
from datetime import datetime
import streamlit as st 

@st.cache_resource
def init_supabase_client() -> Client:
    """Khởi tạo và trả về Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("Không tìm thấy SUPABASE_URL hoặc SUPABASE_KEY.")
        st.stop()
    return create_client(url, key)

def get_user_profile(supabase: Client, user_id: str) -> dict:
    """Lấy hồ sơ người dùng từ database. Nếu chưa có, tạo một hồ sơ mặc định."""
    try:
        response = supabase.table("user_profiles").select("*").eq("id", user_id).single().execute()
        print(f"DEBUG: [Supabase] Retrieved profile: {response.data}")
        print(f"DEBUG: [Supabase] updated_at value: {response.data.get('updated_at')} (type: {type(response.data.get('updated_at'))})")
        return response.data
    except Exception:
        try:
            default_profile = {
                "id": user_id,
                "misunderstood_concepts": [],
                "last_weakness": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            print(f"DEBUG: [Supabase] Creating default profile with updated_at: {default_profile['updated_at']}")
            response = supabase.table("user_profiles").insert(default_profile).execute()
            print(f"DEBUG: [Supabase] Created profile: {response.data[0]}")
            return response.data[0]
        except Exception as e:
            st.error(f"Không thể tạo hồ sơ mặc định: {e}")
            return {}

def update_user_profile(supabase: Client, user_id: str, profile_data: dict):
    """
    Cập nhật (hoặc chèn nếu chưa có) hồ sơ người dùng trong database.
    """
    if not user_id:
        st.warning("Lỗi hệ thống: Đang cố gắng cập nhật hồ sơ mà không có User ID.")
        return

    try:
        response = supabase.table("user_profiles").upsert({
            "id": user_id,
            **profile_data
        }).execute()
        print(f"DEBUG: [Supabase] Cập nhật profile thành công: {response.data}")
    except Exception as e:
        st.error(f"Lỗi khi cập nhật hồ sơ trên Supabase: {e}")
        print(f"ERROR: [Supabase] Lỗi khi cập nhật profile: {e}")