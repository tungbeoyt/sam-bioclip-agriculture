# Huấn luyện bổ sung sau phản biện DG35

Bộ mã này tạo thí nghiệm mới, không thay kết quả cũ và không bảo đảm trước ưu thế của BioCLIP. Chưa chạy GPU trong môi trường soạn mã. Không đưa số liệu dự kiến vào bài báo.

## 1. Cài vào máy Ubuntu có GPU

Giải nén gói mã vào `experiments/revision_v2` trong dự án sam-bioclip-agriculture đang có dữ liệu và môi trường GPU. Giữ nguyên bộ mã v1, kết quả cũ và môi trường cũ để đối chiếu. Các lệnh dưới chạy từ thư mục gốc dự án.

```bash
source .venv/bin/activate  # đổi thành venv GPU hiện dùng nếu khác
python -m pip install ImageHash
python -c "import torch, torchvision, open_clip, peft, sklearn, scipy, PIL, matplotlib; print(torch.__version__, torch.cuda.is_available())"
python -m pip freeze > environment_revision_v2.txt
export PV=data/cls/plantvillage/color
export OLD=experiments/results/tomato_7000_20260925_113448/split_manifest.json
export CODE=experiments/revision_v2
```

Không cần cài lại torch/torchvision nếu môi trường cũ chạy được. Base cần numpy, matplotlib, Pillow, open_clip_torch, peft, scikit-learn, tqdm và CUDA. Nếu GPU không hỗ trợ BF16, trainer tự dùng FP32; ghi rõ khi so thời gian.

## 2. Rà soát dữ liệu trước khi train chính

Tạo manifest toàn bộ ảnh duy nhất bằng prepare-only (không train):

```bash
python "$CODE/train_revision.py" --method linear --data-root "$PV" --reference-manifest "$OLD" --output-dir experiments/results/v2_prepare --prepare-only
python "$CODE/audit_groups.py" candidates --manifest experiments/results/v2_prepare/source_full_manifest.json --root "$PV" --review near_duplicate_review.csv
```

Mở CSV và xem từng cặp ảnh: điền `yes` nếu cùng lá/đối tượng hoặc cùng ảnh qua crop/góc/biến đổi; `no` nếu khác. pHash chỉ sinh ứng viên, có thể bỏ sót và nhận nhầm. Bổ sung thủ công cặp cùng đối tượng nếu có metadata; không coi rà soát này là chứng minh độc lập theo cây. Tất cả dòng phải có quyết định. Nếu nhóm có nhãn mâu thuẫn, cần kiểm tra nhãn trước.

```bash
python "$CODE/audit_groups.py" split --manifest experiments/results/v2_prepare/source_full_manifest.json --root "$PV" --review near_duplicate_review.csv --output reference_grouped_v2.json --per-class 100
export REF=reference_grouped_v2.json
export RESULTS=experiments/results/revision_v2_grouped
```

Chia lại giữ toàn bộ nhóm cùng split; val/test ít nhất 100 ảnh/lớp (có thể dư do nhóm). Đây là tái chia dữ liệu lịch sử: nhiều ảnh đã từng được train, KHÔNG gọi test mù mới. Tất cả run mới khởi tạo pretrained gốc. Muốn test mù thực sự: thu thập nguồn ảnh mới theo lá/cây, khóa danh sách và chỉ đánh giá sau khi khóa mô hình. Không gộp kết quả split mới với số liệu cũ.

## 3. Kiểm tra GPU trước

```bash
python "$CODE/test_v2.py"
python "$CODE/experiment_v2.py" smoke --data-root "$PV" --reference-manifest "$REF" --results "$RESULTS"
```

Smoke chạy 7 cấu hình, gồm attention và both; chỉ dùng vài batch validation, không dùng test. Phải chạy qua bước calibration và lưu checkpoint được. Nếu lỗi, gửi nguyên traceback và environment_revision_v2.txt; chưa chạy train chính. QKV adapter dùng parametrization trên in_proj_weight của 12 MultiheadAttention; MLP dùng PEFT. V2 đặt dropout adapter bằng 0 cho cả MLP và QKV để so vị trí cùng điều kiện; khác v1 (MLP dropout 0.1), nên phải train lại anchor r8 và không trộn kết quả v1 vào ablation. Attention QKV dùng cập nhật hạng thấp trên ma trận fused QKV, không phải ba adapter hạng r độc lập.

## 4. Chạy theo từng giai đoạn

```bash
python "$CODE/experiment_v2.py" train --phase core --data-root "$PV" --reference-manifest "$REF" --results "$RESULTS"
python "$CODE/experiment_v2.py" select --phase core --data-root "$PV" --reference-manifest "$REF" --results "$RESULTS"
python "$CODE/experiment_v2.py" evaluate --phase core --data-root "$PV" --reference-manifest "$REF" --results "$RESULTS"
```

Core: Linear Probe, LoRA r8 MLP, MobileNetV2, ResNet50 và BioCLIP full fine-tuning. Mỗi cấu hình thử 3 LR head (0.001, 0.003, 0.01), LR backbone/adapter = head/10, 3 seed, tối đa 150 epoch, patience 20; mỗi phương pháp 9 run tuning validation-only. Chọn LR bằng trung bình best validation CE của ba seed. Bộ tìm kiếm và ngân sách epoch giống nhau, FLOPs/thời gian không giống nhau; đây là ngân sách số thử bằng nhau, không phải ngân sách tính toán bằng nhau.

Nếu select báo best tại giới hạn epoch: dùng results root mới, tăng --epochs 300 cho toàn bộ tuning cùng phase; chưa nhìn test để quyết định. Best trước giới hạn chỉ là kiểm tra biên, không chứng minh tối ưu toàn cục. Nếu không đạt kết quả tốt với ba LR, mở rộng tìm kiếm chỉ dựa validation và công bố đủ ngân sách. Test chỉ chạy khi selection.json đã được khóa.

Lặp lại ba lệnh train/select/evaluate với `--phase fewshot`, rồi `--phase ablation`. Fewshot gồm linear, LoRA, MobileNetV2 × K=1/2/4/8/16 × 3 LR × 3 seed (135 run). Core 45 run; ablation 108 run. Có thể chạy theo từng phase và gửi kết quả từng đợt. Không cần chạy một lần toàn bộ. Chạy lại cùng lệnh train để tiếp tục last.pt; không sửa mã giữa run. Nếu OOM, chọn root mới và giảm batch-size đồng nhất cho phase. Full fine-tuning cần nhiều bộ nhớ hơn LoRA.

Few-shot dùng support lồng nhau theo seed; vẫn dùng toàn bộ validation có nhãn. Báo cáo ngân sách là 10K nhãn train CỘNG số nhãn validation; không gọi là chỉ có 10K nhãn. Ablation có r4/r8/r16 × MLP/attention/both và factorial weighting sqrt/none × smoothing 0/0.05 tại r8 MLP. Không phải factorial toàn bộ vị trí × loss.

## 5. PlantDoc và zero-shot

Tải đúng phiên bản Cropped PlantDoc từ nguồn chính thức của bộ dữ liệu, lưu URL/commit/license và support. Tệp plantdoc_mapping_review.json là ánh xạ dự kiến 7 lớp; phải kiểm tra nghĩa nhãn và ảnh trước khi dùng. Không tự thêm yellow virus hoặc tomato leaf thành TYLCV/healthy. Rà soát ảnh trùng và gần trùng giữa nguồn và đích; script chỉ tự loại trùng điểm ảnh chính xác. Ghi lại quyết định rà soát, không dùng nhãn đích để tuning.

```bash
export PD=/duong/dan/Cropped-PlantDoc
python "$CODE/experiment_v2.py" evaluate --phase core --data-root "$PV" --reference-manifest "$REF" --results "$RESULTS" --plantdoc-root "$PD" --mapping "$CODE/plantdoc_mapping_review.json" --mapping-reviewed
```

Lệnh giữ nguyên checkpoint và T từ validation PlantVillage; báo Macro-F1 trên lớp có support, đầu ra vẫn 10 lớp nên dự đoán ngoài tập giao tính sai. So chuyển miền phải đánh giá lại source trên cùng phạm vi nhãn đích: chạy evaluate_revision.py --run-dir RUN --output-dir OUT_MOI --root "$PV" --mapping "$CODE/plantdoc_mapping_review.json" (dataset mặc định source). Chưa lấy Macro-F1 nguồn 10 lớp trừ Macro-F1 đích 7 lớp. Khi dùng mapping ở source, chỉ lọc ground truth, vẫn giữ đầu ra 10 lớp; script tự tính macro trên các lớp có support như miền đích. ECE/NLL/Brier trước/sau T nằm trong metrics.json, không fit lại T trên PlantDoc. Công bố support từng lớp.

Zero-shot dùng một run nguồn đã chọn trong selection.json chỉ để lấy danh sách lớp/manifest; không học head hoặc dùng trọng số fine-tune:

```bash
python "$CODE/evaluate_revision.py" --run-dir /duong/dan/run_da_chon --output-dir "$RESULTS/zero_shot_source" --root "$PV" --zero-shot
python "$CODE/evaluate_revision.py" --run-dir /duong/dan/run_da_chon --output-dir "$RESULTS/zero_shot_plantdoc" --dataset plantdoc --root "$PD" --mapping "$CODE/plantdoc_mapping_review.json" --mapping-reviewed --zero-shot
```

Zero-shot hiện dùng T=1; không lặp 3 seed giống hệt để tạo std giả. Prompt được cố định trong class_prompts, chưa tối ưu theo test.

## 6. Gửi minh chứng về để viết lại bài

```bash
python "$CODE/summarize_results.py" --results-root "$RESULTS" --zip revision_v2_results.zip
```

Gửi zip và environment_revision_v2.txt, reference_grouped_v2.json, near_duplicate_review.csv, thông tin nguồn PlantDoc. Giữ best.pt/last.pt tại máy GPU, không cần gửi checkpoint nặng. Zip chứa lịch sử tuning, selection, predictions, calibration và manifest; kiểm tra không có dữ liệu riêng tư trước khi công khai.

Sau khi có số liệu mới: vẽ learning curve K theo mean±std 3 seed; báo cả trước/sau T và CI từng seed; McNemar dùng cùng ảnh/nhãn, không gộp 3 seed thành 3N. Kiểm định hiện tại chỉ so LoRA chuẩn với baseline nguồn; nếu mở rộng target/ablation cần định nghĩa family so sánh rồi hiệu chỉnh Holm tương ứng. Với ảnh theo nhóm, CI độc lập theo ảnh có hạn chế; dùng bootstrap theo nhóm nếu muốn suy luận trên đối tượng.

Chỉ khi nhận kết quả thật mới viết lại paper 8–10 trang và phụ lục. Các kết luận về tính mới, ưu thế BioCLIP, hiệu quả tham số so full fine-tuning và hiệu chuẩn ngoài miền phải bám kết quả. Việc công khai artifact GitHub/Zenodo và gộp thư mục bản nộp cần thực hiện trên tài khoản/thư mục thực tế; chưa được hoàn tất bởi gói mã này.
