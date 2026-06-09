# collection-va-ocr

Hướng dẫn chạy OCR hàng loạt theo danh sách trong file va-info-list.json.

## 1. Clone mã nguồn

```bash
git clone https://github.com/msang77335/collection-va-ocr.git
cd collection-va-ocr
```

## 2. Chuẩn bị dữ liệu đầu vào

Cần 2 thành phần:

- File danh sách: va-info-list.json
- Thư mục ảnh: va-frontid-images/

### Định dạng va-info-list.json

Mỗi phần tử trong mảng cần có tối thiểu các trường sau:

- recordId
- bankAccountNumber
- filename

Ví dụ:

```json
[
  {
    "recordId": "0a0f201a-4e0b-41f2-a884-8b60944c4663",
    "bankAccountNumber": "96988900324453",
    "filename": "0a0f201a-4e0b-41f2-a884-8b60944c4663_96988900324453_frontIdCard.jpg"
  }
]
```

### Yêu cầu quan trọng về filename

Giá trị filename trong va-info-list.json phải trùng khớp chính xác với tên file thực tế trong thư mục va-frontid-images/.

Ví dụ:

- Trong JSON: filename = abc.jpg
- Bắt buộc tồn tại file: va-frontid-images/abc.jpg

Nếu không tồn tại, script sẽ báo lỗi File không tồn tại cho phần tử đó.

## 3. Cài đặt môi trường Node.js

Yêu cầu Node.js 18+ (khuyến nghị 20+).

```bash
npm install
```

## 4. Cài đặt môi trường Python

Script Node gọi trực tiếp Python tại đường dẫn .venv/bin/python3.
Vì vậy bắt buộc tạo virtualenv tên .venv ở thư mục gốc của project.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 5. Chạy file process-ocr-images

```bash
node process-ocr-images.js
```

Script sẽ:

- Đọc va-info-list.json
- Lấy từng filename để tìm ảnh trong va-frontid-images/
- Gọi ocr_card.py để OCR từng ảnh
- Ghi kết quả theo thời gian thực ra file ocr-results_YYYY-MM-DDTHH-mm-ss.json

## 6. Kết quả đầu ra

Sau khi chạy xong, file output có dạng:

- ocr-results_2026-06-09T05-04-47.json (tên file thay đổi theo thời điểm chạy)

Mỗi phần tử kết quả gồm:

- recordId
- bankAccountNumber
- filename
- success
- ocrData (nếu thành công) hoặc error (nếu thất bại)
- processedAt

## 7. Lỗi thường gặp

- Lỗi: Python venv không tồn tại
  Nguyên nhân: Chưa tạo .venv đúng tên hoặc tạo sai vị trí.
  Cách xử lý: Tạo lại theo bước 4.

- Lỗi: File không tồn tại: <filename>
  Nguyên nhân: filename trong va-info-list.json không khớp với file thực tế.
  Cách xử lý: Kiểm tra lại tên file trong va-frontid-images/.

- OCR engine chưa được cài
  Nguyên nhân: Chưa cài đủ thư viện Python trong requirements.txt.
  Cách xử lý: Kích hoạt .venv và chạy lại pip install -r requirements.txt.
