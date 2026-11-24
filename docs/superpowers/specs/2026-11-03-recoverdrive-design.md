# RecoverDrive 設計文件

> 狀態：設計已在對話中逐段確認，並已實作完成（2026-11-24）。以下正文保留原始設計；實作與原設計的差異列在「實作差異」。
> 範圍：整體方法、實驗設計、各階段計畫（不含日期與分工）

> **實作差異（2026-11-24 補記，正文未改）**
> - §1.2：在 sensor 故障上，checks-only（不用 VLM）與 Ours-full 的 F1 相同（都是 0.714）；VLM 的增益只出現在 agent 故障。
> - §1.6、§8.1：後來另外使用 gvl2（8 張 GPU，driver 565）；兩台伺服器都改用 CUDA 12 build。
> - §2.2、§4.2：VLM 輸出欄位是 `verdict`（`ok` / `uncertain` / `fault:<sensor>`），不是 `status` + `sensor`。「只在 flag 時呼叫 VLM」的 gate 只做為 ablation（F1 降到 0.601）。
> - §3.2：L3（ghost objects）沒有實作。
> - §3.4：CARLA 的 ego 車用自寫的 lane-keeping 與速度控制（D11），不用 autopilot；只監控 2 台相機（Town01_Opt）。
> - §4.3：沒有加入「重新觀察相鄰相機」。
> - §5.1：另外加了 checks-only、PRAM-R-style 兩個 baseline（事後加入，各自只跑一次 test split）與 9 個 VLM-centric 變體（只在 dev split）。checks-only 不用 Qwen3-VL-8B。
> - §5.3 E5：沒有跑 32B 和 LoRA；Thinking 只跑 seed 0。
> - §5.4：「test split 只跑一次」適用於凍結的主實驗；之後的 baseline 與 ablation 是另外核准的 test run。
> - §5.3、§6：課程要求的部分（R1–R6 的 STL 需求、shield 與 controller 的證明、falsification、assurance case）不在本 spec，見 `docs/requirements-and-verification.md`。

## 1. 目標與背景

### 1.1 要做什麼
以 DriveAgent（arXiv 2505.02123）為出發點，建立一個能**偵測感測器與 agent 故障、嘗試恢復、並在必要時安全降級**的自駕感知監控系統：

detect → diagnose → recover → verify → fallback

### 1.2 主軸
- **第一層（基礎）**：用 Qwen3-VL + SAM 3 + 確定性幾何檢查，改良 DriveAgent 的故障偵測。
- **第二層（主要貢獻）**：在偵測之上加入 recovery 與 safety fallback。

### 1.3 成功的定義
- 有一條可以 demo 的完整流程（nuScenes 離線 + CARLA 閉環）。
- 報告中有和 baseline（DriveAgent 式文字鏈、CoT、Self-Refine）在相同條件下的比較數字，以及 ablation。

### 1.4 已確定的決策
| 項目 | 決定 |
|---|---|
| 主要資料 | nuScenes-mini，離線注入故障 |
| 閉環實驗 | CARLA（0.9.x，`-RenderOffScreen`），量測碰撞率 |
| 系統架構 | 混合式：數學交給程式、判斷交給 VLM（做法 B） |
| Baseline | 用同一個 Qwen3-VL 重新實作 DriveAgent 式文字鏈（做法 A） |
| 相機端偵測 | SAM 3（文字提示）；Qwen3-VL grounding 保留作 ablation |
| LiDAR 端 | 標註框 + 框內點數（暫不使用深度偵測器） |
| VLM | Qwen3-VL-8B-Instruct，以 vLLM 提供 OpenAI 相容 API |

### 1.5 不在範圍內
- 重現 DriveAgent 論文的數字（官方程式碼無法執行，R2/R3 資料未釋出）。
- 實車部署或嚴格的即時性要求。
- MVP 階段偵測 LiDAR 幽靈物件（L3）。

### 1.6 運算環境
實驗室 Linux 伺服器，4 × RTX 6000 Ada（48 GB），NVIDIA driver 580 / CUDA 13.0，無螢幕。與他人共用，執行時以 `CUDA_VISIBLE_DEVICES` 指定 GPU。

## 2. 系統架構

### 2.1 資料流
```
資料來源轉接器（nuScenes / CARLA）
  → 故障注入器（僅評估時啟用）
  → LiDAR 物件（標註框 + 點數）   ‖   SAM 3 相機偵測（2D 框）
  → 一致性檢查（Python，確定性）
  → 診斷 agent（Qwen3-VL，JSON 輸出）
  → Recovery controller（重問 / 重新觀察 / 隔離 → verify）
  → Safety shield + fallback（規則式）
  → 動作 + 記錄（JSONL + Rerun）
```

### 2.2 模組與介面
每個模組都是「資料類別進、資料類別出」的函式，可以各自用假資料測試。

| 資料類別 | 主要欄位 |
|---|---|
| `Frame` | `source`（nuscenes / carla）、`scene_id`、`frame_idx`、`timestamp`、`cameras`（名稱 → 影像、內參、外參）、`lidar`（N×4 點雲、外參）、`ego_pose`、`gt_boxes`（3D 框 + 類別） |
| `FaultLabel` | `fault_id`（C1…A3 或 none）、`sensor`（camera:<名稱> / lidar / agent / none）、`severity`（low / medium / high）、`episode_id` |
| `CheckReport` | 每個相機的 `match_rate`、`lidar_visible_ratio`、`point_count`、與上一幀的變化量、`agent_output_valid`、觸發的旗標 |
| `Diagnosis` | `status`（ok / fault / uncertain）、`sensor`、`fault_type`、`confidence`（0–1）、`evidence`（文字） |
| `RecoveryResult` | `steps_taken`、`final_state`（healthy / degraded / unresolved）、`isolated_sensors`、`final_diagnosis` |
| `Action` | `action`（continue / slow_down / stop）、`reason` |

### 2.3 模型服務
- **Qwen3-VL**：vLLM 在一張 GPU 上提供 OpenAI 相容 API。所有呼叫走同一個 client，換模型（4B / 8B / 32B、Instruct / Thinking，或 GPT-4o）只改設定檔。JSON 輸出用 vLLM 的 schema 約束解碼。
- **SAM 3**：與 Qwen 共用 GPU 或另一張卡，以函式或小型服務呼叫。權重需在 Hugging Face 同意授權後下載。

### 2.4 兩個資料來源共用介面
nuScenes 與 CARLA 各寫一個轉接器，都輸出 `Frame`。轉接器之後的所有模組完全相同。

## 3. 資料與故障注入

### 3.1 nuScenes-mini
- 10 個場景，約 400 個關鍵幀（2 Hz）。
- **以場景切分**：6 個開發場景（調 prompt、校正門檻、選擇性微調），4 個測試場景（只用於最終報告）。同一場景的幀不會同時出現在兩邊。
- 測試集太小時，從完整的 nuScenes 驗證集再加入場景。

### 3.2 故障清單
| 代號 | 類別 | 故障 | 注入方式 |
|---|---|---|---|
| C1 | 相機 | 視角錯置 | 用 CAM_BACK 影像取代 CAM_FRONT |
| C2 | 相機 | 角度偏移 | 影像旋轉或平移 |
| C3 | 相機 | 畫質劣化 | 模糊、雜訊、過曝、全黑 |
| C4 | 相機 | 畫面凍結 | 重複上一幀影像 |
| L1 | LiDAR | 掉點 | 隨機掉點、少 beam、扇區遺失 |
| L2 | LiDAR | 校正偏移 | 點雲整體平移或旋轉 |
| L3 | LiDAR | 幽靈物件 | 插入不存在的點群（MVP 不偵測） |
| L4 | LiDAR | 完全失效 | 空點雲 |
| A1 | Agent | 輸出損壞 | JSON 格式錯誤或缺欄位 |
| A2 | Agent | 答案錯誤 | 反轉某一次呼叫的診斷結果 |
| A3 | Agent | 逾時 | 模擬模型無回應 |

### 3.3 注入規則
- **MVP 只做 C1、L1、A1**，其餘在階段 3 加入。
- 約 50% 的幀不注入故障，用來量誤報與不必要的 fallback。
- 故障以「片段」形式出現：從某幀開始持續數幀（至少 2 幀），讓時間檢查與重新觀察有意義。
- 每種故障有 low / medium / high 三種嚴重程度。
- 注入位置與參數由 random seed 決定。

### 3.4 CARLA
- 1 張地圖、3–5 條短路線，包含會讓故障變危險的情境（前車急煞、行人穿越）。
- 車輛由 CARLA 內建 autopilot 駕駛；系統只能輸出 continue / slow_down / stop。
- 同步模式：模擬器等待模型回應。故障在感測器回呼時用同一個故障注入器注入。

## 4. 偵測 → 診斷 → Recovery → Verify → Fallback

### 4.1 偵測：一致性檢查（Python）
- **跨感測器**：50 m 內、框內點數 ≥ k 的 3D 標註框投影到各相機，與 SAM 3 的 2D 框以 IoU ≥ 0.3 配對，得到每個相機的配對率。
- **LiDAR 健康度**：50 m 內標註框被 LiDAR 看到的比例、總點數、各方位覆蓋率。
- **時間一致性**：與上一幀相比，配對率與點數的驟降；影像完全相同時判為凍結。
- **Agent 輸出**：JSON 格式驗證、逾時偵測。
- 初始值 k = 5、IoU = 0.3、範圍 50 m；所有門檻在開發集上校正後固定。

### 4.2 診斷：Qwen3-VL
- 輸入：當前與前一幀影像、LiDAR 鳥瞰圖、`CheckReport` 的文字摘要。
- 輸出：符合 schema 的 `Diagnosis` JSON。
- MVP 每一幀都呼叫；之後實作「只在檢查可疑時呼叫」，作為延遲優化並量測效果。

### 4.3 Recovery controller
依序嘗試，每幀最多一輪：
1. **重新詢問**：輸出格式錯誤時重問，最多 2 次。懷疑故障時以 temperature 0.7 取樣 3 次多數決，一致比例作為信心。
2. **重新觀察**：檢查下一個關鍵幀（MVP），或視野重疊的相鄰相機（之後加入），判斷故障是暫時還是持續。
3. **隔離感測器**：確認故障的感測器標記為不可用，用剩餘感測器重跑檢查，進入降級模式。

### 4.4 Verify
在剩餘感測器上重跑一致性檢查。一致且信心 ≥ 門檻 → `degraded`（成功恢復）；否則 → `unresolved`。沒有故障 → `healthy`。

### 4.5 Safety shield（規則式，不使用 LLM）
| 狀態 | 動作 |
|---|---|
| healthy | continue（接受原本的駕駛動作） |
| degraded，前方感知仍可用 | slow_down（限速） |
| degraded 且失去前方感知，或 unresolved | stop（有控制地停車） |
| 任何狀態下碰撞時間低於門檻（CARLA，初始 2 秒） | 強制煞車 |

## 5. 評估

### 5.1 比較的方法
所有方法使用相同的 Qwen3-VL-8B、資料與故障。

| 方法 | 說明 |
|---|---|
| B0 | 不監控，永遠 continue |
| B1 | DriveAgent 式文字鏈，zero-shot |
| B2 | DriveAgent 式文字鏈 + CoT |
| B3 | DriveAgent 式文字鏈 + Self-Refine |
| Ours-detect | 一致性檢查 + SAM 3 + Qwen 診斷，不做 recovery |
| Ours-full | Ours-detect + recovery + verify + shield |

B1–B3 的診斷結果接上同一個 safety shield，使 fallback 的比較只反映偵測與 recovery 的差異。

### 5.2 理想動作與指標定義
每種情況有一組「可接受動作」：

| 情況 | 可接受動作 |
|---|---|
| 沒有故障 | continue |
| Agent 故障（A1–A3） | continue 或 slow_down |
| 感測器故障，嚴重程度 low，或只影響側邊 / 後方相機 | continue 或 slow_down |
| 感測器故障，嚴重程度 medium 以上，且影響前鏡頭或 LiDAR | slow_down 或 stop |

C1（視角錯置）與 L4（完全失效）沒有程度之分，固定視為 high。

指標定義（依保守程度排序：continue < slow_down < stop）：
- **不必要的 fallback**：動作比可接受集合中最保守的動作還保守。
- **漏掉的 fallback**：動作比可接受集合中最不保守的動作還不保守。
- **Recovery 成功**：感測器故障時，`final_state = degraded` 且被隔離的正是實際故障的感測器；Agent 故障時，重新詢問後得到格式正確且與無故障版本一致的診斷。

### 5.3 實驗
| 實驗 | 問題 | 設定 | 指標 / 產出 |
|---|---|---|---|
| E0 前導實驗 | 流程跑得起來嗎？訊號存在嗎？ | 1 場景、CAM_FRONT、C1 | 配對率分布圖、混淆矩陣、延遲 |
| E1 偵測與診斷 | 能否抓到故障、判斷正確感測器？ | 所有方法 × 故障 × 嚴重程度 | P / R / F1、診斷正確率、誤報率 |
| E2 Recovery 與 fallback | 能否恢復？fallback 是否過度保守？ | shield 輸出 vs 理想動作 | recovery 成功率、恢復所需幀數、不必要 / 漏掉的 fallback 比例 |
| E3 閉環安全 | 能否避免碰撞？ | CARLA：B0、B1、Ours-full | 碰撞率、路線完成率、不必要的停車 |
| E4 延遲 | 夠不夠快？時間花在哪？ | 各元件計時 | 延遲分解 |
| E5 Ablation | 哪個零件最重要？ | 一次換掉或拿掉一個零件 | ablation 表格 |

E5 的項目：SAM 3 vs Qwen grounding；拿掉一致性檢查；拿掉多次取樣 / 重新觀察 / 感測器隔離；Qwen3-VL 4B vs 8B（時間夠加 32B）；Instruct vs Thinking。加分項目：LoRA 微調。

### 5.4 規則｀
- E1、E2 先在開發集上跑，用來調整；定案後在測試集上只跑一次，報告只用測試集數字。
- 有隨機性的設定跑 3 個 seed，報告平均值 ± 標準差。

## 6. 階段計畫

| 階段 | 內容 | 完成標準 | 解鎖的實驗 |
|---|---|---|---|
| 0 環境與介面 | Python 環境、vLLM + Qwen3-VL、SAM 3、nuScenes-mini + devkit、Rerun；定義第 2.2 節的資料類別 | 一支腳本讀一幀資料、各呼叫一次 Qwen 和 SAM 3、記錄到 Rerun | — |
| 1 MVP pipeline | nuScenes 轉接器；C1 / L1 / A1 注入；LiDAR 點數；投影與配對；簡單診斷；評分腳本 | 一個指令跑完開發場景 3 種故障並輸出指標表 | E0 |
| 2 Recovery 與 baseline | recovery controller、verify、shield、理想動作標籤；B0–B3 | 開發集上有 Ours-full 與 B0–B3 的比較數字 | E1、E2（開發集） |
| 3 完整故障集 | 其餘故障、嚴重程度、門檻校正後固定 | 測試集上的主要表格完成 | E1、E2（測試集） |
| 4 CARLA 閉環 | CARLA 轉接器、autopilot、控制介面、場景腳本（階段 2 完成後可與階段 3 同時進行） | 能完整跑完路線並套用 shield 動作 | E3 |
| 5 Ablation | 依第 5.3 節替換或移除零件；延遲優化 | 所有表格與圖完成後凍結結果 | E4、E5 |
| 6 簡報與報告 | 投影片、demo 影片（Rerun 錄影或 CARLA）、報告 | — | — |

## 7. 前導實驗（E0）細節

**目的**：確認每個模組接得起來，以及核心訊號「LiDAR 與相機對不上」在故障時確實出現。不追求效果。

**範圍**：nuScenes-mini 1 個場景（約 40 幀）、只用 CAM_FRONT、只做 C1（一半的幀換成 CAM_BACK 影像）。不做 recovery、shield、baseline、CARLA。

**步驟**：
1. 用 nuScenes devkit 讀出每幀的 CAM_FRONT 影像、LiDAR 點雲、校正參數、3D 標註框。
2. 50 m 內的標註框計算框內點數，保留 ≥ 5 點者。
3. 投影到 CAM_FRONT 得到 2D 框，只保留畫面內的。
4. SAM 3 以提示 car、truck、bus、pedestrian、bicycle、motorcycle 偵測同一張影像。
5. 以 IoU ≥ 0.3 配對，計算配對率。
6. **檢查點**：畫出正常幀與錯置幀的配對率分布。兩者明顯分開才繼續；否則先檢查投影與配對。
7. 加入 Qwen3-VL：影像 + 檢查摘要 → `{status, sensor, confidence}` JSON（schema 約束）。
8. 與故障標籤比對，輸出混淆矩陣並記錄每幀延遲。
9. 抽幾幀記錄到 Rerun，將投影框與 SAM 3 框疊在影像上目視確認。

**完成標準**：40 幀全部跑完無錯誤；配對率分布明顯分開；Qwen JSON 100% 可解析；每幀延遲都有記錄。

E0 的程式即為階段 1 的基礎，之後各階段在這條流程上擴充。實作計畫依階段撰寫，第一份涵蓋階段 0–1（含 E0）。

## 8. 風險與已知限制

### 8.1 風險與退路
| 風險 | 退路 |
|---|---|
| 套件與 CUDA 13 驅動不相容 | 使用 Docker 或固定版本；必要時退回 Qwen2.5-VL |
| 共用 GPU 被占用 | 與實驗室協調固定一張卡；必要時改用 4B 模型 |
| CARLA 閉環進度落後 | 停損：階段 3 結束前仍無法閉環，E3 改為僅 demo、不量碰撞率；nuScenes 結果獨立成立 |
| 測試集太小、數字不穩 | 從 nuScenes 驗證集加入場景 |
| E0 配對率分不開 | 先排查投影、座標系與類別對應，再調整範圍與點數門檻 |

### 8.2 已知限制（寫入報告）
- LiDAR 端以標註框 + 點數代替偵測器，相當於假設一個只受點數影響的理想偵測器；因此無法偵測幽靈物件（L3）。
- nuScenes 為離線資料，碰撞相關結論僅來自 CARLA。
- 理想動作規則是人為定義，fallback 指標的意義受此規則影響。
