import re
import streamlit as st
import pandas as pd
from underthesea import word_tokenize, sentiment

# ============================================================================
# HÀM PHÁT HIỆN SPAM
# ============================================================================
def detect_spam(text: str) -> bool:

    if not text or pd.isna(text):
        return False

    t = str(text).lower().strip()

    # ------------------------------------------------------------------------
    # KIỂM TRA LINK
    # ------------------------------------------------------------------------
    if re.search(
        r"https?://|www\.|\.com|\.vn|\.net|\.org|bit\.ly|tinyurl",
        t
    ):
        return True

    # ------------------------------------------------------------------------
    # KIỂM TRA SỐ ĐIỆN THOẠI
    # ------------------------------------------------------------------------
    if re.search(r"(\d[\d\.\-\s]{8,}\d)", t):
        return True

    # ------------------------------------------------------------------------
    # TỪ KHÓA SPAM
    # ------------------------------------------------------------------------
    spam_keywords = [
        "inbox",
        "ib",
        "lien he",
        "liên hệ",
        "zalo",
        "giá rẻ",
        "miễn phí",
        "khuyến mãi",
        "sale",
        "giảm giá",
        "đặt hàng",
        "ship cod",
        "hotline",
        "contact",
        "dm",
        "direct",
        "order",
        "đơn hàng",
        "quảng cáo",
        "ads"
    ]

    if any(keyword in t for keyword in spam_keywords):
        return True

    # ------------------------------------------------------------------------
    # LẶP KÝ TỰ BẤT THƯỜNG
    # ------------------------------------------------------------------------
    if re.search(r"(.)\1{5,}", t):
        return True

    return False


# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="Auto Label NLP - Bình luận Facebook",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CSS GIAO DIỆN
# ============================================================================
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

/* Dataframe */
div[data-baseweb="table"] {
    font-family: 'Segoe UI', Arial, sans-serif !important;
    font-size: 15px !important;
}

div[data-baseweb="table"] th {
    background-color: #4CAF50 !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 8px !important;
}

div[data-baseweb="table"] td {
    padding: 10px 8px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Progress */
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}

</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================
st.title("📝 Lab NLP: Tự động gán nhãn & phân tách từ")

st.markdown("""
### Ứng dụng NLP tiếng Việt sử dụng Underthesea

Chức năng:
- ✂️ Phân tách từ tiếng Việt
- 😊 Phân tích cảm xúc
- 🚨 Phát hiện spam Facebook comments
- 📊 Thống kê dữ liệu
- ⬇️ Xuất file CSV kết quả
""")

st.markdown("---")


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("📁 Upload dữ liệu")

uploaded_file = st.sidebar.file_uploader(
    "Chọn file CSV chứa bình luận",
    type=["csv"],
    help="File CSV phải có cột: id và text"
)

st.sidebar.markdown("---")
st.sidebar.caption("NLP Vietnamese - Underthesea")


# ============================================================================
# KIỂM TRA FILE
# ============================================================================
if uploaded_file is None:

    st.info("👆 Vui lòng upload file CSV để bắt đầu.")

    st.stop()


# ============================================================================
# ĐỌC FILE CSV
# ============================================================================
try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"❌ Không thể đọc file CSV: {e}")

    st.stop()


# ============================================================================
# KIỂM TRA CỘT
# ============================================================================
required_cols = {"id", "text"}

if not required_cols.issubset(df.columns):

    st.error("""
    ❌ File CSV phải chứa:
    - cột id
    - cột text
    """)

    st.stop()


# ============================================================================
# THÔNG TIN FILE
# ============================================================================
st.success(f"✅ Đã tải thành công {len(df)} bình luận")


# ============================================================================
# XEM TRƯỚC DỮ LIỆU
# ============================================================================
st.subheader("👀 Xem trước dữ liệu")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ============================================================================
# XỬ LÝ NLP
# ============================================================================
st.markdown("---")

st.subheader("⚙️ Đang xử lý NLP...")

progress_bar = st.progress(0)

status_text = st.empty()

tokenized_list = []

sentiment_list = []


for i, row in df.iterrows():

    text = str(row["text"]).strip()

    # ------------------------------------------------------------------------
    # TOKENIZE
    # ------------------------------------------------------------------------
    try:

        tokens = word_tokenize(
            text,
            format="text"
        )

    except:

        tokens = text

    tokenized_list.append(tokens)

    # ------------------------------------------------------------------------
    # SENTIMENT
    # ------------------------------------------------------------------------
    try:

        label = sentiment(text)

    except:

        label = "neutral"

    sentiment_list.append(label)

    # ------------------------------------------------------------------------
    # UPDATE PROGRESS
    # ------------------------------------------------------------------------
    progress = (i + 1) / len(df)

    progress_bar.progress(progress)

    status_text.text(
        f"Đang xử lý bình luận {i+1}/{len(df)}..."
    )


# ============================================================================
# GÁN KẾT QUẢ
# ============================================================================
df["tokenized"] = tokenized_list

df["sentiment_label"] = sentiment_list

df["spam"] = df["text"].apply(
    lambda x: detect_spam(str(x))
)

# Spam label
df["spam_label"] = df["spam"].map({
    True: "spam",
    False: "không spam"
})

df["spam_label_vn"] = df["spam"].map({
    True: "Spam",
    False: "Không spam"
})

# Sentiment VN
sentiment_vn_map = {
    "positive": "Tích cực",
    "negative": "Tiêu cực",
    "neutral": "Trung lập"
}

df["sentiment_label_vn"] = (
    df["sentiment_label"]
    .map(sentiment_vn_map)
    .fillna(df["sentiment_label"])
)

progress_bar.empty()

status_text.empty()

st.success("🎉 Hoàn tất xử lý tất cả bình luận!")


# ============================================================================
# THỐNG KÊ
# ============================================================================
st.markdown("---")

st.subheader("📊 Thống kê kết quả")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Tổng bình luận",
        len(df)
    )

with col2:

    positive_count = (
        df["sentiment_label"] == "positive"
    ).sum()

    st.metric(
        "Tích cực",
        positive_count
    )

with col3:

    spam_count = df["spam"].sum()

    st.metric(
        "Spam",
        spam_count
    )


# ============================================================================
# BIỂU ĐỒ
# ============================================================================
chart_col1, chart_col2 = st.columns(2)

# ---------------------------------------------------------------------------
# CẢM XÚC
# ---------------------------------------------------------------------------
with chart_col1:

    st.subheader("📈 Phân bố cảm xúc")

    sentiment_counts = (
        df["sentiment_label_vn"]
        .value_counts()
    )

    st.bar_chart(
        sentiment_counts,
        use_container_width=True
    )


# ---------------------------------------------------------------------------
# SPAM
# ---------------------------------------------------------------------------
with chart_col2:

    st.subheader("🚫 Phân bố Spam")

    spam_counts = (
        df["spam_label_vn"]
        .value_counts()
    )

    st.bar_chart(
        spam_counts,
        use_container_width=True
    )


# ============================================================================
# BẢNG KẾT QUẢ
# ============================================================================
st.markdown("---")

st.subheader("📋 Kết quả chi tiết")

display_cols = [
    "id",
    "text",
    "tokenized",
    "spam_label_vn",
    "sentiment_label_vn"
]

st.dataframe(
    df[display_cols],
    use_container_width=True,
    hide_index=True,
    height=600,
    column_config={
        "id": st.column_config.TextColumn(
            "ID",
            width="small"
        ),

        "text": st.column_config.TextColumn(
            "Nội dung bình luận",
            width="large"
        ),

        "tokenized": st.column_config.TextColumn(
            "Phân tách từ",
            width="large"
        ),

        "spam_label_vn": st.column_config.TextColumn(
            "Phát hiện Spam",
            width="medium"
        ),

        "sentiment_label_vn": st.column_config.TextColumn(
            "Cảm xúc",
            width="medium"
        ),
    }
)


# ============================================================================
# EXPORT CSV
# ============================================================================
st.markdown("---")

csv_data = (
    df[display_cols]
    .to_csv(index=False)
    .encode("utf-8-sig")
)

st.download_button(
    label="⬇️ Tải file kết quả CSV",
    data=csv_data,
    file_name="auto_label_binh_luan_facebook.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================================
# FOOTER
# ============================================================================
st.caption("""
✅ File CSV export dùng UTF-8 BOM để hiển thị đúng tiếng Việt trên Excel.
""")