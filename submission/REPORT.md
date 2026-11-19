# Báo cáo Day 09 — Nguyến An Phú, 2A202603037

## 1. Kết luận trước
pipeline kết luận **Hold**? Bao nhiêu ảnh vào = 224 ảnh, bao nhiêu ảnh ra dataset 205? Lý do chính (1–3 câu): kích thước một số ảnh nhỏ so với quy định (cạnh ảnh không ngắn hơn 250 px). Ngoài ra, nhiều ảnh gần như là lặp lại của ảnh khác có hình tương tự

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `BAD_CLASS_ID` | 6 | máy | ID lớp nằm ngoài phạm vi cho phép (thường gặp nhất: ID 4). |
| `BAD_CLASS_NAME` | 2 | máy | Tên lớp trong `data.yaml` không khớp tên lớp quy định. |
| `BAD_FORMAT` | 3 | máy | Dòng nhãn không đúng 5 cột số theo định dạng YOLO. |
| `BAD_SIZE` | 3 | máy | Cạnh ngắn của ảnh nhỏ hơn 250 px (phổ biến nhất: 180 px). |
| `BOX_HUGE` | 3 | người | Box chiếm diện tích ảnh quá lớn. |
| `BOX_OUT_OF_IMAGE` | 8 | máy | Box vượt ra ngoài biên ảnh. |
| `BOX_TINY` | 6 | người | Box nhỏ hơn ngưỡng của dự án. |
| `BOX_ZERO` | 4 | máy | Box có chiều rộng hoặc chiều cao bằng/nhỏ hơn 0. |
| `CLASS_RARE` | 4 | người | Một số lớp có quá ít box (phổ biến nhất: lớp 7 có 1 box). |
| `CORRUPT_IMAGE` | 2 | máy | Không mở/đọc được ảnh. |
| `DUP_BOX` | 7 | máy | Box bị lặp cùng lớp với box đứng trước. |
| `DUP_IMAGE_EXACT` | 8 | máy | Ảnh trùng từng byte với ảnh khác. |
| `DUP_IMAGE_NEAR` | 2 | máy | Ảnh trùng gần đúng theo pHash. |
| `EMPTY_LABEL` | 12 | người | File nhãn rỗng; cần xác nhận ảnh nền hay bị thiếu nhãn. |
| `LEAK_TEST` | 4 | máy | Ảnh trùng với ảnh trong kho test/val. |
| `MISSING_LABEL_FILE` | 6 | người | Ảnh chưa có file nhãn, cần người gán nhãn. |
| `POSSIBLE_FALSE_POS` | 29 | người | Box nhãn không khớp với dự đoán nào của model. |
| `POSSIBLE_LOOSE_BOX` | 45 | người | Model dự đoán đúng lớp nhưng box không khớp box nhãn. |
| `POSSIBLE_MISSING` | 152 | người | Model thấy vật thể nhưng không có box nhãn khớp. |
| `POSSIBLE_WRONG_CLASS` | 22 | người | Vị trí box khớp nhưng nhãn lớp khác dự đoán. |

Ghi chú: bảng gồm cả dòng `error` và `warn`; số lượng là số dòng trong CSV. Các lỗi có autofix được giao máy; `MISSING_LABEL_FILE` và các cảnh báo cần người xem.

vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy? Với ngân sách 80 ảnh, người xem được bao nhiêu lỗi ngữ nghĩa?
```
05_review_queue.csv xếp ảnh dựa theo thứ tự giảm dần của cột 'priority'
Với ngân sách 80 ảnh, người xem được 80 lỗi ngữ nghĩa
```

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 4 | 1.00/0.50 -> 1.00/0.67 | cải thiện thêm recall - tỉ lệ bắt lỗi |
| `second_opinion.conf` | 0.25 | 0.5 | 0.56/0.83 -> 0.56/0.83 | cải thiện thêm recall - tỉ lệ bắt lỗi |

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) chứng minh gate chặn đúng: chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt nên ảnh thiếu file nhãn còn nguyên), chỉ ra kết luận **Rework**, mã thoát 20 và việc **không xuất** dataset
```
" Bước 5/5  KHÔNG xuất: còn lỗi bắt buộc (Rework) — trả về bước 3"
Rework = mã thoát 20
```

. (Đừng sửa tay `work/run/ds`: pipeline dựng lại thư mục này từ raw mỗi lần chạy.) 

(b) ≥ 2 điểm mù có bằng chứng (box lỏng nhẹ, nhầm lớp mà model lớn cũng nhầm, ảnh trùng biến đổi mạnh…).

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa train: ghi "chưa chạy" và lý do, rồi bỏ qua phần còn lại của mục này. Nếu đã train: dán bảng từ `submission/results/comparison.md` (model, epoch, seed, số ảnh train mỗi biến thể).

```md
| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 205 | 2 | 0.711 ± 0.021 | 0.500 ± 0.009 | 0.753 ± 0.054 | 0.620 ± 0.016 |

Chênh mAP50 (clean − raw) = **-0.015**.
```


## 6. Quyết định và ghi chú AI
(a) quyết định cuối; nếu Hold/Rework cần làm gì để thành Release? (b) Dùng AI viết code? Ghi 3 dòng: prompt chính, cách bạn kiểm code AI, 1 lần AI sai. Không dùng: ghi "không dùng".


(a) quyết định: Release
(b) dùng AI
- prompt chính:
```
theo như CHECKPOINTS.md, hiện đã setup môi trường, cài thư viện yêu cầu, chạy pytest đạt kết quả lúc đầu 55 failed, 24 passed (đúng như dự kiến: 5 check đã viết sẵn). bài toán được mô tả trong README.md

yêu cầu: thực hiện Bước 2 ghi trong GUIDE.md, để khi chạy pytest có thể pass hết toàn bộ kiểm tra, đồng thời thỏa mãn yêu cầu bài toán như trên. chú ý RULES.md và docs\PROMPTING.md

task: đọc toàn bộ các file liên quan / được đề cập trong các file md, rồi cho tôi một kế hoạch để tôi kiểm tra. sau khi chấp thuận thì mới triển khai.
```

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
