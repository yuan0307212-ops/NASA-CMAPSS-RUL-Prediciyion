# AERO LAB — NASA C-MAPSS RUL Prediction / HW2

完整可執行的 Flask 多元線性回歸網站，使用你提供的 NASA C-MAPSS FD001–FD004 資料。

## 本機執行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

瀏覽器開啟 http://127.0.0.1:5000 。

## 網站功能

- FD001–FD004 資料切換，預設 FD002，另有 FD004 進階情境。
- 資料品質、缺失值、固定欄位、RUL 分布、原始資料預覽。
- 單台引擎感測器時間序列與 RUL 走勢。
- SelectKBest / RFE / Lasso / All feature selection。
- Multiple Linear Regression / Ridge / Lasso 互動訓練。
- 依引擎 ID 的 GroupShuffleSplit，避免同一引擎跨訓練與驗證。
- MAE、MSE、RMSE、R²；驗證／測試集基準模型比較。
- 測試引擎最後觀測週期的 RUL 預測與經驗區間。
- CSV 自訂資料匯入：數值欄位回歸與 SelectKBest。
- CRISP-DM 六階段與作業繳交檢核。

## 數據與方法

每筆原始資料 26 欄：引擎 ID 1 + cycle 1 + operation settings 3 + sensors 21。排除 ID 後最多 25 個輸入特徵。RUL 訓練標籤 = 同台引擎最後故障 cycle − 當前 cycle。

`engine.py` 的 `fit`：
1. 依引擎分組 80/20 分割訓練／驗證。
2. 在訓練部分進行特徵選擇（SelectKBest、RFE 或 Lasso）。
3. 標準化與回歸建模；驗證指標與平均值基準。
4. 用全部訓練資料及選定欄位重新擬合模型；測試資料只取每台引擎最後一個觀測週期，和 RUL_FDxxx.txt 一一對應。
5. 顯示測試指標與絕對殘差分位數的經驗區間。**經驗區間不是經過校準的正式信賴區間或預測區間。**

> 方法限制：驗證集以所有週期作為觀測單位，測試集以每台引擎最後週期作為觀測單位；因此兩者評估情境不同。C-MAPSS 是模擬資料，並非真實飛機安全預測服務。線性模型可能產生負 RUL，介面保留原始模型評估結果；僅在區間下界做 0 截斷。

## 部署到 Render

1. 建立 GitHub repository，將本資料夾全部上傳。
2. Render → New → Web Service → Connect GitHub repository。
3. Environment: Python；Build Command: `pip install -r requirements.txt`。
4. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`。
5. 部署完成後將 Render 網址與 GitHub 網址交給老師。

預設輸出 JSON 已產生，網站第一次打開不用等訓練。重新訓練需要運算資源，Render 免費方案可能有冷啟動、RAM 或逾時限制。

## LMS 作業要求與待辦

- [x] Kaggle 公開資料集；25 個候選輸入特徵（不同 FD 子集的常數欄位另排除）。
- [x] Multiple Linear Regression、Feature Selection、模型評估、CRISP-DM。
- [x] 兩種以上問題切換（FD002/FD004，另可選 FD001/FD003）。
- [x] 自訂 CSV 回歸上傳。
- [ ] 你需要實際建立 GitHub repo 與部署公開網站。
- [ ] 你需要使用 NotebookLM 研究相關資料，撰寫 **至少 100 字**真實摘要，標示「NotebookLM 摘要」。
- [ ] 你需要把完整 ChatGPT 對話匯出 PDF，標示「GPT 輔助內容」。
- [ ] 如需正式報告書，請另行整理並繳交。

## 資料來源

- Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- NASA PCoE: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.

註：NASA 官網另一筆名稱類似的 **C-MAPSS Aircraft Engine Simulator Data** 目前標示不可下載，與此處 NASA PCoE 公開的 **Turbofan Engine Degradation Simulation** 資料集不同。


## Light Pro 2026 更新

- 全站淺色航空研究視覺系統、藍白卡片與行動裝置響應式排版。
- 首頁新增 6 步研究導覽、研究問題、RUL 視覺解說、FD001–FD004 互動情境卡片。
- 新增「研究知識庫」頁：Linear / Ridge / Lasso、MAE / RMSE / R²、FAQ。
- Plotly 圖表調整為適合白色背景的配色。
- 保留原版 Flask API、真實資料、模型訓練與 CSV 匯入。
- 離線 HTML 只展示內嵌的預設分析結果，重新訓練、上傳 CSV 與任意引擎查詢須啟動 Flask。


## Caramel & Cream 米橘色主題
此版本採白底、咖啡色文字、米色與暖橘色點綴；圖表同步改為暖色系。功能與資料皆保留。


## 首頁主題標示更新
首頁以「飛機引擎還能運轉多久？」為大標題，副標題「利用多元線性回歸預測引擎剩餘使用壽命」，並以白底米橘色系呈現核心研究問題。

## NotebookLM 研究摘要

「作業與報告」頁面已加入基於 NASA 原始論文的研究摘要草稿、來源連結及 NotebookLM 提示詞。**請注意此草稿不是 NotebookLM 的輸出**；請依 `研究摘要與NotebookLM操作指南.md` 操作後替換成真正生成的摘要。

## NotebookLM 摘要更新
已將使用者提供的 NotebookLM 研究摘要整合至「作業繳交」頁面，並保存於 `NotebookLM研究摘要_正式版.md`。原始輸出未包含逐句來源引用。
