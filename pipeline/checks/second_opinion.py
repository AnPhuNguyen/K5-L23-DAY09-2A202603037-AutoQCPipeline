"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    # Tất cả ngưỡng lấy từ cấu hình để bước calibration có thể thay đổi kết quả.
    sc = cfg["second_opinion"]
    # Nếu người gọi không truyền dự đoán thì đọc cache yolo26l, không cần GPU.
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []
    # fp_conf có thể thấp hơn conf vì bước kiểm box thừa dùng dự đoán rộng hơn.
    fp_conf = sc["fp_conf"]

    # Chỉ xem ảnh có cả file nhãn và có dự đoán trong cache.
    for image in ds.image_files():
        stem = image.stem
        p = ds.label_path(stem)
        if not p.exists() or stem not in preds:
            continue

        size = ds.image_size(stem)
        if not size:
            continue
        iw, ih = size

        # P gồm dự đoán đạt độ tin cậy và diện tích tối thiểu của model checker.
        predictions = preds.get(stem, [])
        main_predictions = [
            pred for pred in predictions
            if pred[1] >= sc["conf"]
            and pred[5] * ih * pred[4] * iw >= sc["min_area_px"]
        ]

        # Lưu nhãn hợp lệ và số dòng gốc; dòng hỏng được module khác báo, không làm đây chết.
        valid_labels = []
        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= cls < len(ds.classes):
                valid_labels.append((line_no, cls, (cls, cx, cy, w, h)))

        # Xử lý từng dự đoán theo đúng thứ tự trong cache để kết quả ổn định.
        for pred in main_predictions:
            cls, conf, cx, cy, w, h = pred
            box = (cls, cx, cy, w, h)
            # cands là mọi nhãn có vị trí chồng đủ ngưỡng, bất kể tên lớp.
            cands = [
                label for label in valid_labels
                if iou_xywh(box[1:], label[2][1:]) >= sc["iou_missing"]
            ]
            # Không có nhãn nào gần vật thì model nghi rằng đang thiếu nhãn.
            if not cands:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING",
                                    stem, "Model thấy vật thể không có nhãn khớp",
                                    None, {"conf": conf}))
                continue

            # Nếu có nhãn cùng lớp, chỉ cờ box lỏng khi IoU tốt nhất vẫn thấp.
            same = [label for label in cands if label[1] == cls]
            if same:
                best_iou = max(iou_xywh(box[1:], label[2][1:]) for label in same)
                if best_iou < sc["iou_loose"]:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX",
                                        stem, "Dự đoán đúng lớp nhưng box không khớp",
                                        None, {"conf": conf, "iou": best_iou}))
                continue

            # Không cùng lớp: sai lớp chỉ khi vị trí khớp khá cao.
            other = [label for label in cands if label[1] != cls]
            if other:
                best_iou = max(iou_xywh(box[1:], label[2][1:]) for label in other)
                if best_iou >= sc["iou_match"]:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS",
                                        stem, "Vị trí khớp nhưng khác lớp",
                                        None, {"conf": conf, "iou": best_iou}))

        # FALSE_POS xem dự đoán ở ngưỡng fp_conf, bao gồm cả dự đoán diện tích nhỏ.
        fp_predictions = [
            pred for pred in predictions
            if pred[1] >= fp_conf
        ]
        # Nếu nhãn không khớp dự đoán nào thì báo đúng số dòng gốc của nhãn đó.
        for line_no, cls, label in valid_labels:
            if all(iou_xywh(label[1:], pred[2:]) < sc["iou_missing"]
                   for pred in fp_predictions):
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS",
                                    stem, "Box nhãn không khớp dự đoán nào",
                                    line_no, {}))

    return issues
