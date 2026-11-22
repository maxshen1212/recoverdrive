# Plan：加入 PRAM-R baseline，然後收斂

## Context

報告現有的 baseline 是 B0、B1–B3（DriveAgent 式）和 checks-only。文獻調查顯示，最接近我們、也最新的 VLM 方法是
**PRAM-R**（arXiv 2603.04222，TUM，2026-03）：用同一個 Qwen3-VL-8B，先算每個 sensor 的品質指標，再讓 VLM 給
可信度，低於門檻就停用。審稿人或老師一定會問「跟 PRAM-R 比如何」，所以把它加成最後一個 baseline，之後不再擴充。

已和你確認：
- **判斷單位：每個 sensor 各一個分數**（6 台鏡頭 + LiDAR）。原文是每個模態一個分數；改寫成逐 sensor 才能算
  「指對 sensor」和 recovery 成功。報告註明這是改寫。
- **test split：全部 10 種故障 × seeds 0–2，只跑一次**（你已同意這一次 test run）。所有參數只在 dev 上決定。

PRAM-R 原文沒有的部分（重問、投票、驗證、VLM 出錯處理）**不幫它加**，保持忠實。原文沒有公開程式碼，照論文重寫。

## PRAM-R 怎麼對應到我們的系統（輸入 → 輸出）

| 步驟 | 輸入 | 輸出 | 跟原文的差別 |
|---|---|---|---|
| 1. 品質指標（不用 VLM） | 每台鏡頭的影像；LiDAR 點雲 | 鏡頭：亮度、對比、邊緣密度。LiDAR：點密度（2–50 m 點數 ÷ 固定格數）、雜訊比例（0.5 m 內沒有鄰點的點的比例）、反射率比（平均 intensity ÷ dev 中位數） | 不用 radar、天氣、地圖（我們的系統沒有這些輸入） |
| 2. 標成高低 | 每個指標值 + dev 乾淨畫面的正常範圍（p5–p95） | 「低／正常／高」標籤，用原文 Table I 的說法（例如亮度低 → 照明不足） | — |
| 3. VLM router（每幀 1 次呼叫） | 指標與標籤、上一幀的指標和可信度（memory）；設定 B 另加鏡頭拼圖 | 嚴格 JSON：每個 sensor 的可信度 r∈[0,1]、場景複雜度、每個 sensor 的使用旗標 | 原文 router 是 1 Hz（雙迴路），我們每幀 2 Hz（=原文 ablation「w/o dual-loop」，反應更快，對它有利）。memory 不放門檻後的開關狀態，才能離線精確調門檻 |
| 4. 門檻 + 遲滯 | r、門檻 θ（鏡頭和 LiDAR 各一個）、δ | 每個 sensor 開或關：關著時 r ≥ θ+δ 才打開；開著時 r < θ−δ 才關掉 | 原文關閉條件是 ≤；改成 <，δ=0 時才和原文的 reliable set（r ≥ θ）一致 |
| 5. 接到我們的評估 | 被關掉的 sensor | 關掉 = 隔離；有隔離 → degraded；動作交給同一個規則 shield | 使用旗標（省運算用）不拿來隔離，因為它不是故障判斷 |

E1 的定義照 checks-only：有隔離就算警報，怪「可信度最低的被隔離 sensor」，不會怪 "agent"。VLM 輸出壞掉或逾時：
**沿用上一幀狀態，不發警報**（原文沒有出錯處理）。

## 要改的檔案

新檔：
- `src/recoverdrive/pramr.py`（純函式，不 import methods.py）：`camera_indicators`、`lidar_indicators`（`scipy.spatial.cKDTree` 算孤立點）、
  `load_reference`、`label`、`pramr_schema`（扁平 key：`<SENSOR>_reliability`、`<SENSOR>_use`、`scene_complexity`，沿用現有
  `vlm.validate_json`）、`pramr_messages`、`latch_step`（用整數千分位比較，避免 0.6−0.05 的浮點誤差）、`outcome`，以及凍結常數
  `SCENE_IMAGE`、`THETA_CAMERA`、`THETA_LIDAR`、`DELTA`、`MAX_TOKENS = 384`。所有 PRAM-R prompt 都放這裡，`vlm.py` 不動，
  既有 cache key 不受影響。
- `scripts/pramr_reference.py`：只在 dev 乾淨 keyframe 上算 p5／median／p95，拒絕 `--split test`，輸出
  `configs/pramr_reference.yaml`（要 commit；附 git 狀態）。也印出「至少一個指標超出範圍」的乾淨畫面比例，只記錄、不拿來調。
- `scripts/pramr_tune.py`：把 log 裡的 r 重播進 `latch_step`。`--check` 驗證重播和線上輸出一致；格點搜尋拒絕 test tag。
- `tests/test_pramr.py`。

修改：
- `src/recoverdrive/methods.py`：`class PramR`，名稱 `pramr_text_full`、`pramr_image_full`（dev 比較用）和 `pramr_full`（凍結版）。
  `_full` 結尾自動啟用 recovery 指標。加進 `build_methods` 的 `known`（main 上約第 448 行）並註冊。`extra` 記錄 attempts、
  r 和乾淨答案的 r、指標、前後狀態、參數。參考 `ChecksOnly`（main 上 331–348）和 `rules_outcome`。
- `src/recoverdrive/agent.py`：A2 現在遇到沒有 `verdict`／`sensor` 的 schema 會 `KeyError`，讓所有 A2 幀變成錯誤。
  在建 sensor 清單前加分支：有 `_reliability` key 就用新的 `flip_reliability`；其他 schema 走原本的路，行為不變。
  `flip_reliability` 比照 `flip_verdict` 的「一次答錯」：全部 r ≥ 0.5 時，隨機一個 sensor 改成 1−r；否則把 < 0.5 的都改成 1−r。
- `src/recoverdrive/faults.py`：A1 `missing_field` 只刪 verdict／status／sensor／confidence，對 PRAM-R 的答案沒作用（對它有利）。
  加 fallback：一個都沒刪到時，刪第一個 key。既有 schema 都含這些 key，不受影響。
- `scripts/make_figures.py`：加 `pramr_full` 顯示名稱和獨立的 `have_pramr` 區塊（比照 `test_checks_only`，約 275–309 行），
  放進 S1、S2、S5；S5 的比較過濾改成 `startswith("ours_full - ")`。
- `scripts/run_eval.py`：`--split test` 且方法是 PRAM-R 時，要求 `configs/pramr_reference.yaml` 已被 git 追蹤。

不動：`configs/eval.yaml`（凍結）、Ours 和其他方法。

## 步驟

所有指令都在 `/data/maxshen/recoverdrive` 執行，先 `source scripts/env.sh`。cache 和 outputs 是相對路徑，所以要在主 checkout 跑。
開 vLLM 前先 `nvidia-smi` 選空閒 GPU；停止時用 `pkill -u maxshen -INT -f "[v]llm serve"`。

1. **開分支**：`git switch main && git switch -c pramr-baseline`。`e2e-vlm` 分支（end-to-end 探索）不動、不合併。
2. **先寫 golden 測試**（改程式前先在原程式上跑過）：固定 `ours_schema`、`textchain_schema`、`diagnosis_schema` 的 A2 輸出字串，
   以及 A1 `missing_field` 的輸出。
3. **實作**上面的檔案和 `tests/test_pramr.py`，涵蓋：
   - 指標：黑圖、雜訊圖、空點雲、扇區缺口、孤立點。
   - 標籤。
   - 遲滯的邊界：r=0.55、θ=0.60、δ=0.05 要關掉。
   - 不同 θ／δ 下送出的 prompt 完全相同。
   - 輸出無效時沿用狀態；第二幀的 prompt 出現 memory。
   - A1 對新 schema 有效；A2 和 `agent_recovered`。
   - 怪 sensor 時同分的處理；`build_methods`／`needs_vlm`。
4. **驗證沒動到既有行為**：`uv run pytest -q`、`uv run ruff check src tests scripts`、
   `uv run python scripts/replay_ablation.py --tag test_final --check`（要 4830/4830），再離線比對 `test_final` log 裡每個 A2 和
   A1 `missing_field` 的損壞字串，確認和原本一樣。commit。
5. **正常範圍**：`uv run python scripts/pramr_reference.py --split dev --out configs/pramr_reference.yaml`，commit。
6. **smoke test**（只修格式）：dev、`scene-0061`、C3 和 L1、兩種設定，tag `smoke_pramr`，要 0 個無效輸出。
7. **dev 篩選**（3 個 vLLM server，約 1.5–2 小時）：
   `run_eval.py --split dev --faults C1 C2 C3 C4 L1 L2 L4 --seeds 0 --methods pramr_text_full pramr_image_full --workers 20 --vlm-url ... --tag dev_pramr_s0`
   預期 42 runs、1,701 frames、0 錯誤，和 `vlm_screen_s0` 同一批畫面。
8. **離線調參**（先寫好規則，再看結果）：`pramr_tune.py --tag dev_pramr_s0 --check`（每個方法 1701/1701 一致），然後格點搜尋：
   2 種設定 × θ_cam 0.30–0.80 × θ_lidar 0.30–0.80（間隔 0.05）× δ {0, 0.05, 0.10}。
   選擇規則：**誤報率 ≤ 0.10 中 F1 最高**；同分選誤報較低，再同分選 δ 較小。要設上限，因為一直報警的監控在 dev 上 F1 也有約 0.66。
9. **凍結**：把選中的常數寫進 `pramr.py`，更新測試，commit。再用 `pramr_full` 跑一次 dev（tag `dev_pramr_s0_frozen`），
   應該全部命中 cache，結果和選中的那一列相同。dev 比較：
   `bootstrap_ci.py --tags vlm_screen_s0 dev_pramr_s0_frozen --methods ours_full checks_only pramr_full --reference ours_full`。
10. **test 只跑一次**（約 1.5–2.5 小時）：確認 `git status --porcelain --untracked-files=no` 為空，再跑
    `run_eval.py --split test --faults all --seeds 0 1 2 --methods pramr_full --workers 20 --vlm-url ... --tag test_pramr`。
    預期 120 runs、4,830 frames、0 錯誤。之後跑 `compare_flags.py --a test_final --b test_pramr` 和
    `pramr_tune.py --tag test_pramr --check`。關掉 vLLM。
11. **報告**：
    - `bootstrap_ci.py --tags test_final test_checks_only test_pramr --methods B1 B2 B3 ours_detect ours_full checks_only pramr_full --reference ours_full`；確認既有列沒變。
    - 改好的 `make_figures.py` 重新產生表格。
    - **主比較用「只看 sensor 故障」的表**；agent 故障另外報告。PRAM-R 沒有重問，每個 agent 故障幀都會受影響，要在文中說明。
    - 更新 `docs/experiments.md`、`docs/superpowers/runs/results-provenance.md`（註明 vLLM build 與 `test_final` 不同）、`FutureWork.md`（Step 2 的 PRAM-R 項目完成）、本機 `REPORT.md`。
    - 更新 Google 報告 Doc（E1／E2 表加一列、一段結果、related work 一句）和 provenance Doc。
12. **收尾**：寫 run log，merge 到 `main`（fast-forward），push（HTTPS），確認 GPU 已釋放。

## 公平性注意事項（會寫進報告）

- dev 數字對 PRAM-R 偏樂觀，因為它在同一批 1,701 幀上調參；只有 test 數字是乾淨的。
- prompt 只用原文 Table I 的說法，不加入我們的故障知識（例如「數字完全相同代表凍結」）。
- 設定 B 只看當前鏡頭拼圖，不看前一幀拼圖和 LiDAR 俯視圖；時間資訊只來自 memory。
- 不放進 CARLA（E3），因為正常範圍是 nuScenes dev 的數值。

## 驗證（完成的定義）

- `uv run pytest -q` 全過、`ruff` 無錯誤、`replay_ablation --check` 4830/4830、golden 測試和 agent-fault log 比對都一致。
- `pramr_tune --check`：dev 1701/1701，test 4830/4830。
- `dev_pramr_s0_frozen` 的指標等於調參時選中的那一列。
- `test_pramr`：120 runs、4,830 frames、0 錯誤；`compare_flags` 和 `test_final` 一致。
- 報告表格出現 PRAM-R 一列，既有數字不變；vLLM 已關閉。

## 時程（目標 12/1 簡報前完成）

| 時間 | 工作 |
|---|---|
| 第 1 天 | 步驟 1–6：實作、測試、正常範圍、smoke |
| 第 2 天 | 步驟 7–10：dev 篩選、調參、凍結、test run |
| 第 3 天 | 步驟 11–12：報告、Doc、merge、push |
