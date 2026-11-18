"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    # fixes là bằng chứng mọi thay đổi; pipeline ghi danh sách này vào 03_fixes.csv.
    fixes = []
    # Những mã này liên quan cả ảnh chứ không chỉ một dòng nhãn nên loại cả hai.
    removal_codes = {
        "DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST",
        "CORRUPT_IMAGE", "BAD_SIZE",
    }

    # Chỉ sửa lỗi bắt buộc; cảnh báo để người duyệt xem, không thay đổi tự động.
    error_issues = [i for i in issues if i.severity == ERROR]
    # Gom tên ảnh cần loại để xử lý từng ảnh đúng một lần.
    removed = {i.image for i in error_issues if i.code in removal_codes}

    # Xóa khỏi bản làm việc ds, không xóa file gốc trong data/raw.
    for stem in sorted(removed):
        ip = ds.image_path(stem)
        lp = ds.label_path(stem)
        if ip and ip.exists():
            ip.unlink()
        if lp.exists():
            lp.unlink()
        fixes.append({
            "image": stem,
            "line": "",
            "action": "remove_image",
            "before": "ảnh + nhãn",
            "after": "đã loại",
        })

    # Gom các issue theo file nhãn để ghi lại file chỉ một lần sau khi sửa.
    group = defaultdict(list)
    for issue in error_issues:
        if issue.image in removed or issue.image in {"<dataset>", "data.yaml"}:
            continue
        group[issue.image].append(issue)

    # Những mã này làm dòng nhãn không còn hợp lệ nên action là drop_line.
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    for stem, stem_issues in group.items():
        lp = ds.label_path(stem)
        if not lp.exists():
            continue

        # splitlines() giữ thứ tự dòng gốc để các issue.line không bị dịch sau khi xóa.
        lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        drop_lines = {i.line for i in stem_issues if i.code in drop_codes and i.line is not None}
        clip_lines = {i.line for i in stem_issues if i.code == "BOX_OUT_OF_IMAGE" and i.line is not None}

        # Sắp xếp theo số dòng gốc để xử lý không lệch vị trí.
        for line_no in sorted(drop_lines | clip_lines):
            if line_no < 1 or line_no > len(lines):
                continue
            before = lines[line_no - 1]
            # Nếu cùng dòng vừa tràn box vừa trùng box thì drop thắng, ghi một fix thôi.
            if line_no in drop_lines:
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": before, "after": "",
                })
                lines[line_no - 1] = None
                continue

            # Chỉ cắt box còn parse được thành đúng 5 số YOLO.
            try:
                cls, cx, cy, w, h = parse_yolo_line(before)
            except ValueError:
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": before, "after": "",
                })
                lines[line_no - 1] = None
                continue

            # Tính phần giao của box với khung ảnh chuẩn hoá [0, 1].
            left = max(0.0, cx - w / 2)
            right = min(1.0, cx + w / 2)
            top = max(0.0, cy - h / 2)
            bottom = min(1.0, cy + h / 2)
            nw, nh = right - left, bottom - top
            if nw <= 0 or nh <= 0:
                # Box nằm hoàn toàn ngoài ảnh thì không còn box hợp lệ; xóa dòng.
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": before, "after": "",
                })
                lines[line_no - 1] = None
            else:
                # Từ mép trái/phải/trên/dưới tính lại tâm và kích thước box đã cắt.
                ncx, ncy = (left + right) / 2, (top + bottom) / 2
                after = f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                fixes.append({
                    "image": stem, "line": line_no, "action": "clip_box",
                    "before": before, "after": after,
                })
                lines[line_no - 1] = after

        # Lọc None để dòng bị xóa biến mất, không để lại dòng trống trong file.
        remaining = [line for line in lines if line is not None]
        lp.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")

    # Thiếu file nhãn là việc người phải gán; máy không được tự tạo file rỗng.
    for issue in error_issues:
        if issue.code == "MISSING_LABEL_FILE" and issue.image not in removed:
            fixes.append({
                "image": issue.image,
                "line": "",
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

    # Chỉ tên lớp sai mới được ghi lại data.yaml; không sửa nhãn ngữ nghĩa.
    if any(i.code == "BAD_CLASS_NAME" for i in error_issues):
        if ds.yaml_path.exists():
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            d["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                                    encoding="utf-8")
        fixes.append({
            "image": "data.yaml",
            "line": "",
            "action": "rename_classes",
            "before": "tên lớp không hợp lệ",
            "after": ", ".join(cfg["classes"]),
        })

    return fixes
