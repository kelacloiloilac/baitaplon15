# BTL15 - Xấp xỉ hàm Gaussian hai chiều bằng MLP

Bài làm dùng PyTorch để xấp xỉ hàm mật độ Gaussian hai chiều

$$
f(x) = \frac{1}{2\pi\sqrt{\det(C)}} \exp\left(-\frac{1}{2}(x-\mu)^T C^{-1}(x-\mu)\right)
$$

$$
\mu = [0, 0]^T, \quad C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}
$$

$$
-2 \le x_1 \le 2, \quad -4 \le x_2 \le 4
$$

và thực hiện ba nhóm thực nghiệm có kiểm soát:

| Nhóm | Biến độc lập      | Giá trị           | Cố định       |
| ----- | --------------------- | ------------------- | ---------------- |
| A     | Số mẫu train        | 200, 400, ..., 2000 | B = 10, C = 1    |
| B     | Số nơ-ron tầng ẩn | 1, 2, ..., 10       | A = 1200, C = 1  |
| C     | Số tầng ẩn         | 1, 2, 3, 4          | A = 1200, B = 10 |

Mỗi cấu hình chạy với ba seed `42, 43, 44`; kết quả báo cáo trung bình ± độ lệch chuẩn.
Validation dùng để chọn checkpoint/cấu hình; test 100 mẫu chỉ dùng một lần cho mô hình cuối.

## Cấu trúc thư mục

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
```

## Chạy lại từ đầu

chay notebook trong `src/`

## Cấu hình huấn luyện

| Siêu tham số          | Giá trị |
| ----------------------- | --------- |
| Batch size              | 64        |
| Số epoch tối đa      | 300       |
| Optimizer               | Adam      |
| Learning rate           | 1e-3      |
| Activation              | tanh      |
| Early-stopping patience | 40        |
| Thiết bị              | CPU       |

Dữ liệu: train 2000 mẫu (seed 42), validation 400 mẫu (seed 43), test 100 mẫu (seed 44).
Ba tập được sinh độc lập, không chia lại từ một tập lớn. Seed huấn luyện `42, 43, 44` chỉ
điều khiển khởi tạo trọng số và thứ tự xáo trộn; dữ liệu cố định cho mọi cấu hình.

## Kết quả chính

- Mô hình được chọn theo validation: `C_depth_4_seed_42` (A = 1200, B = 10, C = 4).
- Validation MSE: `2.5995e-06`.
- Test MSE / MAE / RMSE: `2.3988e-06` / `0.0013` / `0.0015`.
