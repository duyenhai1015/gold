# dashboard_DIAGNOSTIC.py (Chỉ dùng để kiểm tra lỗi)
import streamlit as st
from pymongo import MongoClient
import os

st.set_page_config(page_title="Kiểm tra Kết nối", layout="wide")
st.title("🔬 Bộ Chẩn Đoán Kết Nối MongoDB Atlas")

MONGO_URI = os.environ.get("MONGODB_ATLAS_URI")

# 1. Kiểm tra xem Secret có tồn tại không
st.subheader("Bước 1: Kiểm tra Biến Môi trường (Secret)")
if not MONGO_URI:
    st.error("❌ LỖI: Không tìm thấy Secret 'MONGODB_ATLAS_URI'.")
    st.info("Giải pháp: Vào 'Settings' -> 'Secrets' của app này và đảm bảo bạn đã đặt Key (Khóa) là 'MONGODB_ATLAS_URI'.")
    st.stop()
else:
    st.success("✅ Đã tìm thấy Secret 'MONGODB_ATLAS_URI'.")
    st.code(f"Giá trị (bị ẩn): {MONGO_URI[:15]}...{MONGO_URI[-20:]}", language="text")

# 2. Kiểm tra xem chuỗi có đúng không
st.subheader("Bước 2: Kiểm tra Cú pháp Chuỗi kết nối")
if "gold_pipeline" not in MONGO_URI:
    st.error("❌ LỖI: Chuỗi kết nối của bạn bị THIẾU tên Database (phòng).")
    st.info("Nó phải kết thúc bằng '/gold_pipeline?appName=Cluster0'.")
    st.code(f"Chuỗi của bạn: {MONGO_URI}", language="text")
    st.stop()
else:
    st.success("✅ Chuỗi kết nối có chứa 'gold_pipeline'.")

# 3. Thử kết nối và đếm
st.subheader("Bước 3: Thử Kết nối và Đếm Dữ liệu")
with st.spinner("Đang thử kết nối tới Atlas (Timeout sau 5 giây)..."):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["gold_pipeline"]
        collection = db["gold_prices"]

        # Kiểm tra kết nối
        client.server_info() # Lệnh này buộc phải kết nối

        st.success("✅ KẾT NỐI THÀNH CÔNG!")

        # Thử đếm
        count = collection.count_documents({})

        if count > 0:
            st.balloons()
            st.success(f"🎉 TUYỆT VỜI! Đã tìm thấy {count} bản ghi!")
            st.info("Bây giờ bạn có thể khôi phục lại file dashboard.py (V5.1) cũ.")
        else:
            st.warning(f"⚠️ Đã kết nối, nhưng tìm thấy 0 bản ghi.")
            st.info("Lý do: Bạn đã kết nối đúng, nhưng bạn đã chạy backfill_data.py vào một Cluster khác.")

    except Exception as e:
        st.error(f"❌ LỖI KẾT NỐI NGHIÊM TRỌNG:")
        st.code(e, language="text")

        if "Authentication failed" in str(e):
            st.warning("Gợi ý: Lỗi 'Authentication failed'. Mật khẩu 'nhom5vuive' trong Secret của bạn bị SAI.")
        elif "connect: connection refused" in str(e) or "Timeout" in str(e):
            st.warning("Gợi ý: Lỗi 'Timeout/Refused'. Firewall (Tường lửa) của bạn chưa mở (chưa set '0.0.0.0/0'), hoặc tên Cluster (a5bcwew) bị sai.")
        else:
            st.warning("Gợi ý: Một lỗi mạng không xác định. Hãy kiểm tra lại toàn bộ chuỗi kết nối.")
