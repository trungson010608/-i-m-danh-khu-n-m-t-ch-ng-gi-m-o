import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import face_recognition
import os
import sqlite3
import pandas as pd
from datetime import datetime
import math
import unicodedata
# ==========================================
# KHỞI TẠO CẤU TRÚC THƯ MỤC VÀ DATABASE
# ==========================================
DB_NAME = "attendance_system.db"
IMAGE_DIR = "known_faces"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Bảng lớp học (Thêm cột max_students)
    cursor.execute('''CREATE TABLE IF NOT EXISTS classes (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        max_students INTEGER DEFAULT 50)''')
    
    # Bảng sinh viên (Thêm cột dob - Ngày sinh)
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        dob TEXT,
                        class_id TEXT,
                        image_path TEXT)''')
    
    # Bảng lịch sử điểm danh (Thêm cột type để phân biệt: Vào lớp, Giữa giờ, Ra về)
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        timestamp TEXT,
                        type TEXT)''')
    
    # Thêm dữ liệu mẫu nếu bảng lớp học trống
    cursor.execute("SELECT COUNT(*) FROM classes")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO classes VALUES (?, ?, ?)", [
            ("CNTT1", "Công nghệ thông tin 1", 45),
            ("CNTT2", "Công nghệ thông tin 2", 50),
            ("ANM", "An toàn bảo mật", 30)
        ])
    conn.commit()
    conn.close()

init_db()

# ==========================================
# CÁC HÀM PHỤ TRỢ (TOÁN HỌC & XỬ LÝ CHUỖI)
# ==========================================
def calculate_ear(eye):
    """Tính toán tỉ lệ khung mắt (EAR) để phát hiện chớp mắt"""
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def remove_accents(input_str):
    """Loại bỏ dấu tiếng Việt để OpenCV hiển thị không bị lỗi ???"""
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

# ==========================================
# CACHE AI MODEL ĐỂ TRÁNH LOAD LẠI NHIỀU LẦN
# ==========================================
@st.cache_resource
def load_anti_spoof_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    model_path = '/Users/user/Downloads/Face_Login_System/anti_spoof_cnn.pth'
    
    if not os.path.exists(model_path):
        st.error(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file model tại '{model_path}'!")
    else:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ ĐÃ LOAD TRỌNG SỐ MODEL THÀNH CÔNG!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file model: {e}")
            
    model.to(device)
    model.eval()
    return model, device

anti_spoof_model, device = load_anti_spoof_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_registered_faces():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM students")
    rows = cursor.fetchall()
    conn.close()
    
    known_encodings = []
    known_metadata = []
    
    for row in rows:
        s_id, s_name = row
        img_path = os.path.join(IMAGE_DIR, f"{s_id}.jpg")
        if os.path.exists(img_path):
            try:
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_metadata.append({"id": s_id, "name": s_name})
            except Exception as e:
                pass
    return known_encodings, known_metadata

# ==========================================
# GIAO DIỆN STREAMLIT - SIDEBAR NAVIGATION
# ==========================================
st.set_page_config(page_title="Hệ thống Điểm danh Khuôn mặt", layout="wide")
st.sidebar.title("ĐỒ ÁN TỐT NGHIỆP")
st.sidebar.write("Ứng dụng Điểm danh kết hợp Chống điểm danh giả mạo")
menu = st.sidebar.radio(
    "DANH MỤC CHỨC NĂNG",
    ["Trang chủ", "Quản lý lớp học", "Quản lý sinh viên", "Điểm danh", "Danh sách sinh viên", "Lịch sử điểm danh", "Thống kê báo cáo"]
)

def get_db_connection():
    return sqlite3.connect(DB_NAME)

# ==========================================
# CHỨC NĂNG 1: TRANG CHỦ
# ==========================================
if menu == "Trang chủ":
    st.title("🏫 Trang chủ Hệ thống")
    st.subheader("Tổng quan số lượng lớp học và sinh viên")
    
    conn = get_db_connection()
    df_classes = pd.read_sql_query("SELECT * FROM classes", conn)
    df_students = pd.read_sql_query("SELECT class_id, COUNT(*) as total_students FROM students GROUP BY class_id", conn)
    conn.close()
    
    df_dashboard = pd.merge(df_classes, df_students, left_on='id', right_on='class_id', how='left').fillna(0)
    df_dashboard['total_students'] = df_dashboard['total_students'].astype(int)
    
    col1, col2 = st.columns(2)
    col1.metric("Tổng số lớp học", len(df_classes))
    col2.metric("Tổng số sinh viên hệ thống", df_dashboard['total_students'].sum())
    
    st.markdown("---")
    st.write("### Danh sách chi tiết các lớp học")
    st.table(df_dashboard[['id', 'name', 'max_students', 'total_students']].rename(
        columns={'id': 'Mã Lớp', 'name': 'Tên Lớp học', 'max_students': 'Sĩ số tối đa', 'total_students': 'Sĩ số hiện tại'}))

# ==========================================
# CHỨC NĂNG THÊM: QUẢN LÝ LỚP HỌC
# ==========================================
elif menu == "Quản lý lớp học":
    st.title("🏫 Quản lý thông tin Lớp học")
    tab1, tab2, tab3 = st.tabs(["➕ Thêm lớp học", "📝 Sửa lớp học", "❌ Xoá lớp học"])
    
    with tab1:
        st.write("### Tạo lớp học mới")
        with st.form("add_class_form"):
            c_id = st.text_input("Mã lớp học (Ví dụ: CNTT3)*")
            c_name = st.text_input("Tên lớp học*")
            c_max = st.number_input("Số lượng sinh viên tối đa", min_value=1, max_value=200, value=50)
            submit_c = st.form_submit_button("Lưu lớp học")
            
            if submit_c:
                if not c_id or not c_name:
                    st.error("Vui lòng nhập đầy đủ thông tin mã lớp và tên lớp!")
                else:
                    conn = get_db_connection()
                    exist = conn.execute("SELECT id FROM classes WHERE id = ?", (c_id,)).fetchone()
                    if exist:
                        st.warning(f"Mã lớp '{c_id}' đã tồn tại!")
                    else:
                        conn.execute("INSERT INTO classes VALUES (?, ?, ?)", (c_id, c_name, c_max))
                        conn.commit()
                        st.success(f"✅ Đã thêm lớp: {c_name}")
                    conn.close()
                    st.rerun()

    with tab2:
        st.write("### Cập nhật thông tin lớp học")
        conn = get_db_connection()
        classes_list = conn.execute("SELECT * FROM classes").fetchall()
        conn.close()
        
        if classes_list:
            c_dict = {row[1]: row[0] for row in classes_list}
            selected_c_edit = st.selectbox("Chọn lớp cần sửa", list(c_dict.keys()), key="edit_c_select")
            c_id_edit = c_dict[selected_c_edit]
            
            conn = get_db_connection()
            c_data = conn.execute("SELECT * FROM classes WHERE id = ?", (c_id_edit,)).fetchone()
            conn.close()
            
            with st.form("edit_class_form"):
                new_c_name = st.text_input("Tên lớp mới", value=c_data[1])
                new_c_max = st.number_input("Sĩ số tối đa mới", min_value=1, value=int(c_data[2]))
                update_c_btn = st.form_submit_button("Cập nhật")
                
                if update_c_btn:
                    conn = get_db_connection()
                    conn.execute("UPDATE classes SET name = ?, max_students = ? WHERE id = ?", (new_c_name, new_c_max, c_id_edit))
                    conn.commit()
                    conn.close()
                    st.success("Cập nhật thông tin lớp thành công!")
                    st.rerun()

    with tab3:
        st.write("### Xóa lớp học khỏi hệ thống")
        conn = get_db_connection()
        classes_list = conn.execute("SELECT * FROM classes").fetchall()
        conn.close()
        
        if classes_list:
            c_dict_del = {row[1]: row[0] for row in classes_list}
            selected_c_del = st.selectbox("Chọn lớp cần xóa", list(c_dict_del.keys()), key="del_c_select")
            
            if st.button("Tiến hành Xóa Lớp", type="primary"):
                conn = get_db_connection()
                # Kiểm tra xem lớp có sinh viên không
                has_students = conn.execute("SELECT COUNT(*) FROM students WHERE class_id = ?", (c_dict_del[selected_c_del],)).fetchone()[0]
                if has_students > 0:
                    st.error(f"Không thể xóa! Lớp học đang có {has_students} sinh viên học tập.")
                else:
                    conn.execute("DELETE FROM classes WHERE id = ?", (c_dict_del[selected_c_del],))
                    conn.commit()
                    st.success("Đã xóa lớp thành công!")
                conn.close()
                st.rerun()

# ==========================================
# CHỨC NĂNG 2: QUẢN LÝ SINH VIÊN (NÂNG CẤP NGÀY SINH & SĨ SỐ)
# ==========================================
elif menu == "Quản lý sinh viên":
    st.title("🗂️ Quản lý thông tin Sinh viên")
    
    conn = get_db_connection()
    classes_dict = {row[1]: row[0] for row in conn.execute("SELECT * FROM classes").fetchall()}
    conn.close()
    
    tab1, tab2, tab3 = st.tabs(["➕ Thêm sinh viên", "📝 Sửa sinh viên", "❌ Xoá sinh viên"])
    
    with tab1:
        st.write("### Đăng ký sinh viên mới")
        with st.form("add_student_form"):
            s_id = st.text_input("Mã số sinh viên (MSSV)*")
            s_name = st.text_input("Họ và tên sinh viên*")
            s_dob = st.date_input("Ngày tháng năm sinh", min_value=datetime(1980, 1, 1), max_value=datetime.now())
            s_class_name = st.selectbox("Chọn lớp học", list(classes_dict.keys()))
            uploaded_file = st.file_uploader("Tải lên ảnh chân dung mẫu*", type=['jpg', 'jpeg', 'png'])
            submit_btn = st.form_submit_button("Lưu đăng ký")
            
            if submit_btn:
                if not s_id or not s_name or not uploaded_file:
                    st.error("Vui lòng điền đầy đủ thông tin và tải ảnh lên!")
                else:
                    img_path = os.path.join(IMAGE_DIR, f"{s_id}.jpg")
                    try:
                        conn = get_db_connection()
                        # KIỂM TRA SĨ SỐ TỐI ĐA CỦA LỚP HỌC
                        selected_class_id = classes_dict[s_class_name]
                        class_info = conn.execute("SELECT name, max_students FROM classes WHERE id = ?", (selected_class_id,)).fetchone()
                        current_students = conn.execute("SELECT COUNT(*) FROM students WHERE class_id = ?", (selected_class_id,)).fetchone()[0]
                        
                        check_exist = conn.execute("SELECT id FROM students WHERE id = ?", (s_id,)).fetchone()
                        
                        if check_exist:
                            st.warning(f"⚠️ Thêm thất bại: Mã số sinh viên '{s_id}' đã tồn tại!")
                            conn.close()
                        elif current_students >= class_info[1]:
                            st.error(f"❌ Không thể thêm sinh viên! Lớp '{class_info[0]}' đã đầy sĩ số tối đa ({class_info[1]} bạn).")
                            conn.close()
                        else:
                            uploaded_file.seek(0)
                            image = Image.open(uploaded_file).convert('RGB')
                            rgb_img = np.array(image, dtype=np.uint8)
                            encodings = face_recognition.face_encodings(rgb_img)
                            
                            if len(encodings) == 0:
                                st.error("❌ Không tìm thấy khuôn mặt trong ảnh mẫu! Hãy chụp ảnh rõ mặt hơn.")
                                conn.close()
                            else:
                                # Lưu ảnh chuẩn bằng OpenCV liên tục bộ nhớ
                                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(img_path, bgr_img)
                                
                                dob_str = s_dob.strftime('%Y-%m-%d')
                                conn.execute("INSERT INTO students VALUES (?, ?, ?, ?, ?)", (s_id, s_name, dob_str, selected_class_id, img_path))
                                conn.commit()
                                conn.close()
                                st.success(f"✅ Đã thêm sinh viên thành công: {s_name}")
                    except Exception as e:
                        st.error(f"Lỗi hệ thống khi xử lý: {str(e)}")

    with tab2:
        st.write("### Thay đổi thông tin sinh viên")
        s_id_to_edit = st.text_input("Nhập MSSV cần sửa thông tin")
        if s_id_to_edit:
            conn = get_db_connection()
            student = conn.execute("SELECT * FROM students WHERE id = ?", (s_id_to_edit,)).fetchone()
            conn.close()
            
            if student:
                with st.form("edit_student_form"):
                    new_name = st.text_input("Họ tên mới", value=student[1])
                    current_dob = datetime.strptime(student[2], '%Y-%m-%d') if student[2] else datetime.now()
                    new_dob = st.date_input("Ngày sinh mới", value=current_dob)
                    current_class_id = student[3]
                    current_class_name = [k for k, v in classes_dict.items() if v == current_class_id][0]
                    new_class_name = st.selectbox("Lớp học mới", list(classes_dict.keys()), index=list(classes_dict.keys()).index(current_class_name))
                    
                    update_btn = st.form_submit_button("Cập nhật thông tin")
                    if update_btn:
                        conn = get_db_connection()
                        dob_str = new_dob.strftime('%Y-%m-%d')
                        conn.execute("UPDATE students SET name = ?, dob = ?, class_id = ? WHERE id = ?", (new_name, dob_str, classes_dict[new_class_name], s_id_to_edit))
                        conn.commit()
                        conn.close()
                        st.success("Cập nhật thông tin sinh viên thành công!")
            else:
                st.warning("Không tìm thấy sinh viên có MSSV này.")

    with tab3:
        st.write("### Xoá sinh viên khỏi hệ thống")
        s_id_to_del = st.text_input("Nhập MSSV cần xoá")
        if st.button("Tiến hành Xoá", type="primary"):
            conn = get_db_connection()
            student = conn.execute("SELECT * FROM students WHERE id = ?", (s_id_to_del,)).fetchone()
            if student:
                img_path_del = os.path.join(IMAGE_DIR, f"{s_id_to_del}.jpg")
                if os.path.exists(img_path_del):
                    os.remove(img_path_del)
                conn.execute("DELETE FROM students WHERE id = ?", (s_id_to_del,))
                conn.commit()
                st.success(f"Đã xoá hoàn toàn sinh viên có MSSV: {s_id_to_del}")
            else:
                st.error("Không tìm thấy sinh viên cần xoá.")
            conn.close()

# ==========================================
# CHỨC NĂNG 3: ĐIỂM DANH ĐA ĐIỂM (MỖI 50 PHÚT)
# ==========================================
elif menu == "Điểm danh":
    st.title("📸 Camera Điểm danh tích hợp Kiểm tra Đa điểm thời gian")
    
    # 1. BẢO VỆ CHỌN CHẾ ĐỘ ĐIỂM DANH ĐÚNG THỰC TẾ LỚP HỌC
    att_mode = st.selectbox("CHỌN CHẾ ĐỘ ĐIỂM DANH", [
        "Vào lớp (Đầu giờ)",
        "Xác nhận giữa giờ (Mỗi 50 phút)",
        "Điểm danh về (Ra về)"
    ])
    
    known_encodings, known_metadata = load_registered_faces()
    
    if len(known_encodings) == 0:
        st.warning("⚠️ Hệ thống chưa có dữ liệu mẫu sinh viên. Vui lòng qua mục Quản lý sinh viên thêm trước!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_attendance = st.button("🎬 BẮT ĐẦU ĐIỂM DANH", key="start_cam", type="primary", use_container_width=True)
        with col2:
            stop_attendance = st.button("🛑 DỪNG LẠI", key="stop_cam", use_container_width=True)
        
        notification_space = st.empty()
        FRAME_WINDOW = st.empty()
        
        if start_attendance:
            cap = cv2.VideoCapture(1)
            st.session_state['run_camera'] = True
            
            consecutive_live_frames = 0
            REQUIRED_FRAMES = 3
            has_blinked = False
            eyes_closed_frames = 0
            
            MIN_FACE_SIZE = 110
            MAX_FACE_SIZE = 400
            
            # --- 2 BIẾN MỚI THÊM VÀO ĐỂ CHỐNG LAG ---
            process_this_frame = True # Dùng để luân phiên bật/tắt AI
            cached_results = []       # Dùng để nhớ khung hình chữ nhật và chữ
            
            while st.session_state.get('run_camera', True):
                ret, frame = cap.read()
                if not ret: break
                
                # CHỈ CHẠY AI (NHẬN DIỆN MẶT & CHỐNG GIẢ MẠO) 1 NỬA SỐ LẦN
                if process_this_frame:
                    cached_results = [] # Xóa bộ nhớ cũ
                    
                    # 1. THU NHỎ ẢNH 50% ĐỂ TÌM VỊ TRÍ MẶT CHO NHANH
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    raw_face_locations = face_recognition.face_locations(rgb_small_frame)
                    # Nhân tọa độ lên gấp đôi để trả về kích thước gốc
                    face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in raw_face_locations]
                    
                    if face_locations:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Trích xuất đặc trưng trên ảnh to để đảm bảo chính xác
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            face_width = right - left
                            face_height = bottom - top
                            
                            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                                cached_results.append({"box": (left, top, right, bottom), "color": (0, 255, 255), "text": "Tien lai gan hon!"})
                                continue
                            if face_width > MAX_FACE_SIZE or face_height > MAX_FACE_SIZE:
                                cached_results.append({"box": (left, top, right, bottom), "color": (0, 165, 255), "text": "Lui ra xa mot chut!"})
                                continue
                                
                            # LỌC CHỐNG GIẢ MẠO CNN
                            h, w, _ = frame.shape
                            as_top, as_bottom = max(0, top), min(h, bottom)
                            as_left, as_right = max(0, left), min(w, right)
                            
                            prob_live = 0.0
                            try:
                                face_img_anti_spoof = rgb_frame[as_top:as_bottom, as_left:as_right]
                                if face_img_anti_spoof.size != 0:
                                    pil_image = Image.fromarray(face_img_anti_spoof)
                                    input_tensor = transform(pil_image).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        outputs = anti_spoof_model(input_tensor)
                                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                                        prob_live = probabilities[0].item()
                            except: pass
                            
                            # ĐO LƯỜNG CHỚP MẮT
                            ear_avg = 1.0
                            landmarks = face_recognition.face_landmarks(rgb_frame, [(top, right, bottom, left)])
                            if len(landmarks) > 0:
                                left_eye = landmarks[0]['left_eye']
                                right_eye = landmarks[0]['right_eye']
                                ear_avg = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                                
                                if ear_avg < 0.22:
                                    eyes_closed_frames += 1
                                else:
                                    if eyes_closed_frames >= 1: has_blinked = True
                                    eyes_closed_frames = 0
                                    
                            # KIỂM TRA TÍNH CHẤT GIẢ MẠO BẢO VỆ KÉP
                            if prob_live < 0.80 and ear_avg >= 0.22:
                                has_blinked = False
                                consecutive_live_frames = 0
                                cached_results.append({"box": (left, top, right, bottom), "color": (0, 0, 255), "text": "GIA MAO!"})
                                continue
                                
                            if not has_blinked:
                                cached_results.append({"box": (left, top, right, bottom), "color": (255, 255, 0), "text": "Vui long CHOP MAT!"})
                                continue
                                
                            consecutive_live_frames += 1
                            if consecutive_live_frames < REQUIRED_FRAMES:
                                cached_results.append({"box": (left, top, right, bottom), "color": (0, 255, 255), "text": f"Dang quet... ({consecutive_live_frames}/{REQUIRED_FRAMES})"})
                                continue
                                
                            # ĐỐI SÁNH DANH TÍNH
                            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                            name, s_id = "Unknown", None
                            
                            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = known_metadata[best_match_index]["name"]
                                    s_id = known_metadata[best_match_index]["id"]
                                    
                            if s_id:
                                conn = get_db_connection()
                                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                today_str = datetime.now().strftime('%Y-%m-%d')
                                
                                # LOGIC THỜI GIAN NGHIÊM NGẶT
                                if att_mode == "Vào lớp (Đầu giờ)":
                                    check = conn.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND type = 'Vào lớp' AND timestamp LIKE ?", (s_id, f"{today_str}%")).fetchone()[0]
                                    if check == 0:
                                        conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, 'Vào lớp')", (s_id, now_str))
                                        conn.commit()
                                        notification_space.success(f"✅ [{att_mode}] Thành công: {name}")
                                    else:
                                        notification_space.warning(f"🔔 Bạn đã quét 'Vào lớp' ngày hôm nay rồi!")
                                        
                                elif att_mode == "Xác nhận giữa giờ (Mỗi 60 phút)":
                                    last_log = conn.execute("SELECT timestamp, type FROM attendance WHERE student_id = ? AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1", (s_id, f"{today_str}%")).fetchone()
                                    if not last_log:
                                        notification_space.error(f"❌ Bạn chưa quét điểm danh 'Vào lớp' đầu giờ! Không được xác nhận giữa giờ.")
                                    else:
                                        last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S')
                                        diff_mins = (datetime.now() - last_time).total_seconds() / 60.0
                                        if diff_mins > 60.0:
                                            notification_space.error(f"❌ QUÁ 60 PHÚT! Khoảng cách lượt quét trước là {int(diff_mins)} phút. Bạn đã bị đánh vắng hôm nay!")
                                        else:
                                            conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, 'Giữa giờ')", (s_id, now_str))
                                            conn.commit()
                                            notification_space.success(f"✅ [Xác nhận giữa giờ] Thành công: {name} (Khoảng cách: {int(diff_mins)} phút)")
                                            
                                elif att_mode == "Điểm danh về (Ra về)":
                                    has_in = conn.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND type = 'Vào lớp' AND timestamp LIKE ?", (s_id, f"{today_str}%")).fetchone()[0]
                                    if has_in == 0:
                                        notification_space.error(f"❌ Dữ liệu không hợp lệ! Bạn không có log 'Vào lớp' đầu giờ.")
                                    else:
                                        last_log = conn.execute("SELECT timestamp FROM attendance WHERE student_id = ? AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1", (s_id, f"{today_str}%")).fetchone()
                                        last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S')
                                        diff_mins = (datetime.now() - last_time).total_seconds() / 60.0
                                        if diff_mins > 50.0:
                                            notification_space.error(f"❌ QUÁ 60 PHÚT kể từ lần tương tác cuối! Không thể điểm danh về hợp lệ.")
                                        else:
                                            conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, 'Ra về')", (s_id, now_str))
                                            conn.commit()
                                            notification_space.success(f"🎉 Tạm biệt {name}! Bạn đã hoàn thành điểm danh ra về hợp lệ.")
                                conn.close()
                                
                                display_name = remove_accents(name)
                                cached_results.append({"box": (left, top, right, bottom), "color": (0, 255, 0), "text": "OK"})
                                
                                has_blinked = False
                                consecutive_live_frames = 0
                            else:
                                cached_results.append({"box": (left, top, right, bottom), "color": (255, 165, 0), "text": "Unknown"})

                # LẬT CÔNG TẮC: Lần lặp tiếp theo sẽ không chạy AI, chỉ vẽ hình
                process_this_frame = not process_this_frame
                
                # LUÔN LUÔN VẼ LÊN HÌNH TỪ BỘ ĐỆM CACHED
                for draw in cached_results:
                    left, top, right, bottom = draw["box"]
                    color = draw["color"]
                    text = draw["text"]
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                    cv2.putText(frame, text, (left, top - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

                FRAME_WINDOW.image(frame, channels="BGR")
                
                if stop_attendance:
                    st.session_state['run_camera'] = False
                    cap.release()
                    st.rerun()
                    break

# ==========================================
# CHỨC NĂNG 4: DANH SÁCH SINH VIÊN THEO LỚP
# ==========================================
elif menu == "Danh sách sinh viên":
    st.title("📋 Danh sách Sinh viên theo lớp")
    
    conn = get_db_connection()
    classes_df = pd.read_sql_query("SELECT * FROM classes", conn)
    
    selected_class_name = st.selectbox("Chọn lớp để xem danh sách", classes_df['name'].tolist())
    selected_class_id = classes_df[classes_df['name'] == selected_class_name]['id'].values[0]
    
    query = """
        SELECT students.id as [MSSV], students.name as [Họ và Tên], students.dob as [Ngày Sinh], classes.name as [Tên Lớp]
        FROM students
        JOIN classes ON students.class_id = classes.id
        WHERE students.class_id = ?
    """
    df_st_list = pd.read_sql_query(query, conn, params=(selected_class_id,))
    conn.close()
    
    st.write(f"### Danh sách lớp {selected_class_name} (Sĩ số hiện tại: {len(df_st_list)})")
    st.dataframe(df_st_list, use_container_width=True)

# ==========================================
# CHỨC NĂNG 5: LỊCH SỬ ĐIỂM DANH
# ==========================================
elif menu == "Lịch sử điểm danh":
    st.title("🕒 Nhật ký Điểm danh chi tiết")
    
    conn = get_db_connection()
    classes_df = pd.read_sql_query("SELECT * FROM classes", conn)
    selected_class_name = st.selectbox("Chọn lớp kiểm tra lịch sử", classes_df['name'].tolist())
    selected_class_id = classes_df[classes_df['name'] == selected_class_name]['id'].values[0]
    
    query = """
        SELECT students.id as [MSSV], students.name as [Họ Tên], classes.name as [Lớp],
               attendance.type as [Trạng thái Quét], attendance.timestamp as [Thời gian quét]
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        JOIN classes ON students.class_id = classes.id
        WHERE students.class_id = ?
        ORDER BY attendance.timestamp DESC
    """
    df_history = pd.read_sql_query(query, conn, params=(selected_class_id,))
    conn.close()
    
    st.dataframe(df_history, use_container_width=True)

# ==========================================
# CHỨC NĂNG 6: THỐNG KÊ BÁO CÁO (KIỂM TRA CHẶT CHẼ ĐA ĐIỂM)
# ==========================================
elif menu == "Thống kê báo cáo":
    st.title("📊 Thống kê Báo cáo Chuyên sâu (Kiểm tra quy định 50 phút)")
    
    select_date = st.date_input("Chọn ngày xem thống kê", datetime.now())
    date_str = select_date.strftime('%Y-%m-%d')
    
    conn = get_db_connection()
    classes_df = pd.read_sql_query("SELECT * FROM classes", conn)
    selected_class_name = st.selectbox("Chọn lớp xem báo cáo", classes_df['name'].tolist())
    selected_class_id = classes_df[classes_df['name'] == selected_class_name]['id'].values[0]
    
    # Lấy toàn bộ danh sách sinh viên của lớp đã chọn
    students_of_class = conn.execute("SELECT id, name FROM students WHERE class_id = ?", (selected_class_id,)).fetchall()
    
    report_data = []
    total_present = 0
    total_absent = 0
    
    for s_id, s_name in students_of_class:
        # Lấy tất cả lượt quét của sinh viên này trong ngày được chọn xếp theo thứ tự thời gian tăng dần
        logs = conn.execute("""
            SELECT timestamp, type FROM attendance
            WHERE student_id = ? AND timestamp LIKE ?
            ORDER BY timestamp ASC
        """, (s_id, f"{date_str}%")).fetchall()
        
        status = "Vắng mặt"
        reason = "Không đi học"
        
        if logs:
            has_in = any(r[1] == "Vào lớp" for r in logs)
            has_out = any(r[1] == "Ra về" for r in logs)
            
            if not has_in:
                reason = "Thiếu quét đầu giờ 'Vào lớp'"
            elif not has_out:
                reason = "Thiếu quét cuối giờ 'Ra về'"
            else:
                # Kiểm tra khoảng cách giữa tất cả các lượt quét liên tiếp xem có cái nào cách nhau quá 50 phút không
                violated = False
                max_gap = 0
                for i in range(len(logs) - 1):
                    t1 = datetime.strptime(logs[i][0], '%Y-%m-%d %H:%M:%S')
                    t2 = datetime.strptime(logs[i+1][0], '%Y-%m-%d %H:%M:%S')
                    gap = (t2 - t1).total_seconds() / 60.0
                    if gap > max_gap:
                        max_gap = gap
                    if gap > 50.0:
                        violated = True
                
                if violated:
                    reason = f"Vi phạm: Có khoảng giãn quét quá 50 phút ({int(max_gap)} phút)"
                else:
                    status = "Có mặt hợp lệ"
                    reason = f"Đủ quy trình ({len(logs)} lượt tương tác)"
                    total_present += 1
                    
        if status == "Vắng mặt":
            total_absent += 1
            
        report_data.append({
            "MSSV": s_id,
            "Họ và Tên": s_name,
            "Kết quả ngày": status,
            "Chi tiết/Lý do": reason
        })
        
    conn.close()
    
    # Hiển thị Widget
    m1, m2, m3 = st.columns(3)
    m1.metric("Sĩ số lớp", len(students_of_class))
    m2.metric("Số bạn CÓ MẶT HỢP LỆ", total_present, delta=f"{total_present} bạn")
    m3.metric("Số bạn BỊ TÍNH VẮNG", total_absent, delta=f"-{total_absent} bạn", delta_color="inverse")
    
    st.markdown("---")
    st.write(f"### Bảng đánh giá chi tiết ngày học: {date_str}")
    df_report = pd.DataFrame(report_data)
    if not df_report.empty:
        st.dataframe(df_report, use_container_width=True)
