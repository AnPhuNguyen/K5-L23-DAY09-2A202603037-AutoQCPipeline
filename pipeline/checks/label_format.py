"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    # Đọc ngưỡng từ cấu hình để đổi quy tắc ở một nơi, không ghi số cố định trong code.
    lc = cfg["label_format"]
    issues = []

    # Xem từng ảnh và từng file nhãn YOLO tương ứng.
    for image in ds.image_files():
        stem = image.stem
        p = ds.label_path(stem)
        if not p.exists():
            continue

        # previous chỉ giữ các box hợp lệ đã gặp trước đó để tìm box trùng ở dòng sau.
        previous = []
        for line_no, text in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            # Dòng trống không phải lỗi YOLO; enumerate vẫn giữ đúng số thứ tự dòng gốc.
            if not text.strip():
                continue
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                # Một dòng hỏng không làm cả script dừng; ghi BAD_FORMAT rồi bỏ qua dòng này.
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem,
                                    "Dòng không đúng 5 cột số YOLO", line_no))
                continue

            # ID lớp phải thuộc danh sách 0, 1, 2, 3 thay vì nhận ID tùy ý.
            if not 0 <= cls < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                                    f"ID lớp {cls} ngoài phạm vi", line_no))
                continue
            # Box không được có chiều rộng hoặc chiều cao bằng/chưa lớn hơn 0.
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem,
                                    "Box có chiều rộng hoặc chiều cao <= 0", line_no))
                continue

            # YOLO dùng tâm (cx, cy) và kích thước (w, h).
            # Tính hai mép để biết box còn nằm trong khung ảnh [0, 1] hay không.
            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2
            # overflow là mức tràn lớn nhất ở một trong bốn mép; giới hạn nhỏ được phép.
            overflow = max(-left, right - 1, -top, bottom - 1, 0)
            if overflow > lc["overflow_tol"]:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    "Box tràn ra ngoài ảnh", line_no,
                                    {"overflow": overflow}))

            # Chỉ so box cùng lớp; IoU càng gần 1 thì hai box càng giống nhau.
            if any(box_cls == cls and iou_xywh((cls, cx, cy, w, h)[1:], box[1:]) > lc["dup_box_iou"]
                   for box_cls, box in previous):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem,
                                    "Box trùng lớp với box trước đó", line_no))

            # Chuyển kích thước chuẩn hoá sang pixel để đánh giá box nhỏ.
            size = ds.image_size(stem)
            if size:
                iw, ih = size
                short_side = min(w * iw, h * ih)
                # Cạnh ngắn quá nhỏ là cảnh báo, không chặn pipeline tự động.
                if short_side < lc["min_box_px"]:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem,
                                        "Box nhỏ hơn ngưỡng dự án", line_no))
                # Diện tích gần phủ hết ảnh có thể là box quá lớn.
                if w * h > lc["max_box_frac"]:
                    issues.append(Issue("label_format", WARN, "BOX_HUGE", stem,
                                        "Box chiếm diện tích quá lớn", line_no))

            # Đưa box hiện tại vào danh sách các dòng trước đó cho kiểm tra ở dòng sau.
            previous.append((cls, (cls, cx, cy, w, h)))

    return issues
