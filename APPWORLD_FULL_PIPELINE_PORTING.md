# Port các thay đổi full pipeline của ACE sang AppWorld

Tài liệu này ghi lại các thay đổi đã triển khai trong commit `ba1f036` trên
nhánh `test-full-pipeline`, đồng thời mô tả cách áp dụng cùng hành vi cho bộ
ACE chạy trên AppWorld.

> Trạng thái hiện tại: thư mục `ace-appworld/` trong workspace đang rỗng, vì
> vậy tài liệu dùng tên hàm và điểm tích hợp của ACE hiện tại. Khi đưa source
> AppWorld vào workspace, hãy ánh xạ chúng sang orchestrator/training loop
> tương ứng thay vì sao chép cứng đường dẫn.

## Mục tiêu và invariant

Pipeline AppWorld sau khi sửa phải giữ bốn invariant sau:

1. Sample đúng ngay từ lần generate đầu tiên chỉ được cập nhật counter
   `helpful/harmful`; nó không được ADD, UPDATE, DELETE hoặc MERGE nội dung
   playbook.
2. Failure thật và failure tổng hợp được lưu trong hai memory bank độc lập:
   `M_real` và `M_adv`.
3. Cả hai bank giữ nguyên giá trị `failure_memory_top_k` từ cấu hình. Không tự
   giảm `top_k` khi tách bank.
4. Một adversarial failure chỉ được học qua đúng một đường: Curator cập nhật
   playbook trực tiếp, hoặc lưu vào `M_adv` nếu Curator không áp dụng operation
   nào; tuyệt đối không làm cả hai.

## 1. Tắt content update từ correct samples

### Hành vi cũ

Sau khi generator trả lời đúng, Reflector vẫn tag các bullet, rồi Curator vẫn
được gọi theo `curator_frequency`. Điều này cho phép một nhận xét mang tính thủ
tục trên câu vốn đã đúng tạo hoặc sửa rule và gây regression.

### Hành vi mới

Lưu trạng thái correctness ngay sau lần generate đầu tiên:

```python
final_answer = extract_answer(gen_response)
is_correct = data_processor.answer_is_correct(final_answer, target)
pre_train_answer = final_answer
pre_train_was_correct = is_correct
```

Nhánh correct vẫn chạy Reflector và cập nhật counter:

```python
if bullet_tags:
    self.playbook = update_bullet_counts(self.playbook, bullet_tags)
```

Nhưng Curator chỉ được chạy nếu sample sai ngay từ đầu:

```python
if not pre_train_was_correct and step % curator_frequency == 0:
    self.playbook, self.next_global_id, operations, _ = self.curator.curate(...)
```

Phải dùng `pre_train_was_correct`, không dùng giá trị `is_correct` ở cuối các
reflection round. Sample sai ban đầu nhưng được reflection sửa đúng vẫn là một
failure signal hợp lệ và vẫn được phép đề xuất content update.

### Áp dụng trong AppWorld

Trong hàm xử lý một trajectory/sample AppWorld:

- Chụp `pre_train_was_correct` ngay sau execution/evaluation đầu tiên.
- Nếu đúng: chạy tagging/counter update nếu cần, nhưng bỏ qua Curator và mọi
  hygiene pass có thể thay đổi nội dung playbook.
- Nếu sai: giữ nguyên reflection, regeneration và Curator pipeline.
- Điều kiện này phải dựa trên success của trajectory đầu tiên, không dựa trên
  success sau retry hoặc reflection.

## 2. Tách `M_real` và `M_adv`, giữ nguyên `top_k`

### Khởi tạo

Thay một `FailureMemoryBank` dùng chung bằng hai bank:

```python
self.real_failure_memory = FailureMemoryBank(
    encoder=shared_encoder,
    top_k=failure_memory_top_k,
    mode=failure_memory_mode,
)

adversarial_encoder = shared_encoder or self.real_failure_memory._encode
self.adversarial_failure_memory = FailureMemoryBank(
    encoder=adversarial_encoder,
    top_k=failure_memory_top_k,
    mode=failure_memory_mode,
)
```

Hai bank đều nhận đúng `failure_memory_top_k`. Khi RAE không bật, `M_adv` dùng
encoder của `M_real` để không nạp hai bản BGE-M3 vào bộ nhớ; dữ liệu và index
vẫn tách biệt.

Để giữ tương thích với code cũ:

```python
self.failure_memory = self.real_failure_memory
```

Nên bổ sung hai accessor:

```python
def _get_real_failure_memory(self):
    return getattr(self, "real_failure_memory", None) or getattr(
        self, "failure_memory", None
    )

def _get_adversarial_failure_memory(self):
    return getattr(self, "adversarial_failure_memory", None)
```

Không fallback từ `M_adv` sang `self.failure_memory`, vì fallback đó sẽ làm
adversarial reflection đọc failure thật và phá source isolation.

### Routing bắt buộc

| Nguồn sample | Memory dùng khi Reflector retrieve | Memory dùng khi lưu |
|---|---|---|
| AppWorld trajectory thật | `M_real` | `M_real` |
| Adversarial/synthetic trajectory | `M_adv` | `M_adv`, chỉ khi chưa được Curator xử lý |
| Sample đúng | Không retrieve memory | Không lưu failure |

Với AppWorld, “real” là các task/trajectory lấy từ train split hoặc execution
environment thật. “Adversarial” chỉ là task do adversarial agent tổng hợp.

### Log và snapshot riêng

Không để hai bank dùng chung snapshot path. Cấu hình hiện tại của ACE là:

```text
<run>/failure_memory_v2.jsonl
<run>/detailed_llm_logs/adversarial_failure_memory/failure_memory_v2.jsonl
```

Trong AppWorld có thể đổi tên rõ hơn thành:

```text
<run>/memory/real/failure_memory_v2.jsonl
<run>/memory/adversarial/failure_memory_v2.jsonl
```

Điều quan trọng là `set_log_dir()` của hai bank phải nhận hai thư mục khác
nhau. Không resume `M_real` từ snapshot cũ từng chứa cả hai source; hãy migrate
theo trường `source` hoặc bắt đầu một run mới.

## 3. Loại double-update của adversarial failure

### Hành vi cũ

Một adversarial failure được lưu vào FMB trước, sau đó Curator lập tức cập nhật
playbook. Failure đã được xử lý vẫn tiếp tục được retrieve ở các episode sau,
tạo vòng khuếch đại cùng một signal.

### Hành vi mới

Thứ tự xử lý phải là:

```text
adversarial execution sai
  -> adversarial reflection dùng M_adv
  -> chạy Curator
  -> nếu có operation thực sự được áp dụng: không lưu M_adv
  -> nếu không có operation được áp dụng: lưu unresolved failure vào M_adv
```

Không chỉ kiểm tra list operation có rỗng hay không. Một operation có thể được
đề xuất nhưng bị executor bỏ qua. Hãy xác định operation thực sự được áp dụng:

```python
curator_applied = any(
    operation.get("_execution_status", "applied") == "applied"
    for operation in curator_operations
)
```

Chỉ lưu failure khi `curator_applied` là false:

```python
if (
    adversarial_failure_memory is not None
    and not curator_applied
    and reflection_content not in ("(empty)", "")
):
    adversarial_failure_memory.add_verified(
        ...,
        source="adversarial",
    )
```

Không gọi `record_curator_result()` cho một adversarial memory entry vừa được
Curator xử lý, vì entry đó không còn được tạo. Đây chính là cơ chế loại
double-update.

## 4. Mapping từ ACE hiện tại sang AppWorld

Các điểm tham chiếu trong implementation đã sửa:

| Chức năng | ACE hiện tại | Điểm cần tìm trong AppWorld |
|---|---|---|
| Khởi tạo hai bank | `ACE.__init__` trong `ace/ace.py` | Constructor của trainer/orchestrator |
| Tạo snapshot riêng | `ACE.run` | Hàm tạo run directory/log directory |
| Route standard failure | `_train_single_sample` | Hàm train/evaluate một AppWorld trajectory |
| Gate correct sample | `_train_single_sample` | Ngay trước lệnh gọi Curator |
| Route adversarial reflection | `_run_adversarial_episode` | Adversarial execution/reflection loop |
| Gate double-update | `_run_adversarial_episode` | Ngay sau Curator result |

Khi source AppWorld có tên hàm khác, tìm theo các pattern:

```bash
rg -n "FailureMemoryBank|failure_memory|curator\.curate|reflect_on_correct|adversarial" ace-appworld
```

## 5. Test cần port sang AppWorld

ACE đã thêm `tests/test_pipeline_interactions.py`. Khi port, giữ tối thiểu ba
test tương đương.

### Test A — source isolation và giữ `top_k`

- Khởi tạo pipeline với `failure_memory_top_k=10`.
- Xác nhận có hai instance memory khác nhau.
- Xác nhận cả hai instance đều nhận `top_k=10`.
- Xác nhận alias cũ, nếu còn dùng, trỏ tới `M_real`.

### Test B — correct sample không gọi Curator

- Mock generator trả lời đúng ngay lần đầu.
- Mock Reflector tag một bullet là helpful.
- Xác nhận helpful counter tăng.
- Xác nhận `curator.curate()` không được gọi.

### Test C — adversarial failure chỉ dùng một update path

Chạy hai subcase:

1. Curator áp dụng một operation: `M_adv.add_verified()` phải không được gọi.
2. Curator không áp dụng operation: `M_adv.add_verified()` phải được gọi đúng
   một lần.

Trong cả hai case:

- Adversarial Reflector phải nhận `M_adv`.
- `M_real.add_verified()` không được gọi.

## 6. Tiêu chí nghiệm thu trên AppWorld

Port được xem là hoàn tất khi:

- Correct trajectory không tạo thay đổi nội dung playbook.
- Counter evidence từ correct trajectory vẫn được cập nhật.
- Standard reflection không bao giờ retrieve entry từ `M_adv`.
- Adversarial reflection không bao giờ retrieve entry từ `M_real`.
- `top_k` trong cả hai bank đúng bằng config ban đầu.
- Mỗi adversarial failure tạo tối đa một trong hai side effect: playbook
  mutation hoặc memory insertion.
- Resume run đọc đúng hai snapshot riêng.
- Toàn bộ test AppWorld cũ và ba test tương tác mới đều pass.

## 7. File và commit tham chiếu

- Implementation: `ace/ace.py`
- Regression tests: `tests/test_pipeline_interactions.py`
- Commit: `ba1f036` — `Separate failure memories and gate playbook updates`
- Branch: `test-full-pipeline`

