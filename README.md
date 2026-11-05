# BioCLIP LoRA cho phân loại bệnh lá cà chua

Kho mã nguồn và kết quả thực nghiệm của dự án trên máy của tôi.

- [`experiments/`](experiments/): mã huấn luyện, đánh giá, few-shot, ablation và kiểm tra PlantDoc.
- [`src/`](src/): mã mô hình và tiện ích.
- [`public_results/`](public_results/): cấu hình, lịch sử huấn luyện, chỉ số đánh giá và dự đoán từng ảnh cho các thí nghiệm v3/v3.1.
- [`public_results/protocols/`](public_results/protocols/): giao thức thí nghiệm đã dùng.
- [`reference_conservative_d8_v2.json`](reference_conservative_d8_v2.json): manifest chia dữ liệu nguồn, cùng tệp kiểm toán đi kèm.

Repo không chứa báo cáo, ảnh dữ liệu gốc hoặc checkpoint mô hình. Các đường dẫn ảnh trong kết quả là định danh của dữ liệu cục bộ; xem nguồn dữ liệu và giấy phép trước khi tải hoặc sử dụng lại. Tập test PlantVillage là tập lịch sử, không phải test mù mới.
