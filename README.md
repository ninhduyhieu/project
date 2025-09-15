# 5G Cellular Network Simulation

## 1. Cài đặt thư viện cần thiết
Chạy các lệnh sau để cài đặt môi trường Python:

```bash
pip install numpy matplotlib pandas pillow
```

---

## 2. Cách chạy chương trình
File chính: **`gpt_map_hex_CI_UMa_clean.py`**

Khi chạy, chương trình sẽ yêu cầu nhập các thông số sau:

1. **Đường dẫn ảnh** (local path hoặc URL)  
   - Ví dụ:  
     ```
     C:\Users\HP\Downloads\khu_vuc.png
     ```
   - Nếu bỏ trống: chương trình sẽ tự sinh ra lưới tòa nhà mẫu.

2. **Chiều dài HCN (m)**  
   - Ví dụ: `7000`

3. **Chiều rộng HCN (m)**  
   - Ví dụ: `5000`

4. **Số lượng UE**  
   - Ví dụ: `10`

5. **Tỷ lệ Manhattan (0..1, mặc định 0.7)**  
   - Ví dụ: `0.8`  
   - Giá trị này quyết định tỷ lệ UE di chuyển theo mô hình Manhattan so với ngẫu nhiên.

6. **Buildings trong ảnh là MÀU TỐI? (y/n, Enter=auto)**  
   - `y`: coi vùng tối trong ảnh là tòa nhà  
   - `n`: coi vùng sáng trong ảnh là tòa nhà  
   - Enter (để trống): tự động xác định bằng thuật toán Otsu

---

## 3. Kết quả
- Chương trình sẽ hiển thị hoạt cảnh di chuyển UE, vị trí BS, LOS/NLOS.  
- Sau khi mô phỏng kết thúc, dữ liệu log (tọa độ UE, công suất thu Prx, SINR, Packet Loss, handover…) sẽ được lưu ra file CSV với tên dạng:

```
du_lieu_cong_suat_nhan_ues{SỐ_UE}_vantoc{VẬN_TỐC_TB}_thoigianbuoc{Δt}_ptx{Ptx}_buoc{STEPS}_hom{HOM}.csv
```

---

## 4. Ví dụ chạy
```bash
python gpt_map_hex_CI_UMa_clean.py
```

Ví dụ nhập dữ liệu:
```
Dán ĐƯỜNG DẪN ẢNH (local path hoặc URL): C:\Users\HP\Downloads\area.png
Nhập chiều DÀI HCN (m): 7000
Nhập chiều RỘNG HCN (m): 5000
Nhập số UE: 10
Tỷ lệ Manhattan (0..1, mặc định 0.7): 0.8
Buildings trong ảnh là MÀU TỐI? (y/n, Enter=auto): y
```

Kết quả: hiển thị bản đồ mô phỏng, UE di chuyển, và file CSV log được xuất ra thư mục chạy.
