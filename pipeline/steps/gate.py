"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    # Lỗi ERROR là lỗi bắt buộc: Rework chặn phát hành dù người đã duyệt ảnh hay chưa.
    n_errors = sum(i.severity == ERROR for i in issues)
    if n_errors:
        return {
            "decision": "Rework",
            "n_errors": n_errors,
            "n_warn_images": 0,
            "n_images": len(image_stems),
            "warn_ratio": 0.0,
            "reasons": [f"Còn {n_errors} lỗi bắt buộc; phải sửa hoặc Rework"],
        }

    # So tên ảnh chỉ với danh sách ảnh thực tế để bỏ issue dataset hoặc ảnh đã bị loại.
    stems = set(image_stems)
    # set() gom nhiều cảnh báo của cùng một ảnh thành một ảnh, mẫu số vẫn là số ảnh.
    warned = {
        i.image for i in issues
        if i.severity == WARN
        and i.image in stems
        and i.image not in reviewed
    }
    n_images = len(stems)
    # Mẫu số là tổng số ảnh; không nhân số cảnh báo với số issue trong ảnh.
    warn_ratio = round(len(warned) / n_images, 4) if n_images else 0.0
    threshold = cfg["gate"]["warn_ratio_hold"]

    # Nếu đạt ngưỡng warning, cần người xem thêm nên quyết định Hold thay vì Release.
    if warn_ratio >= threshold:
        decision = "Hold"
        reason = f"Tỉ lệ ảnh còn cảnh báo {len(warned)}/{n_images} = {warn_ratio:.4f} đạt ngưỡng Hold {threshold:.4f}"
    else:
        decision = "Release"
        reason = f"Tỉ lệ ảnh còn cảnh báo {len(warned)}/{n_images} = {warn_ratio:.4f} dưới ngưỡng Hold {threshold:.4f}"

    return {
        "decision": decision,
        "n_errors": 0,
        "n_warn_images": len(warned),
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": [reason],
    }
