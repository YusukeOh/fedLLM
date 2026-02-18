# プロジェクト立ち上げとフェーズ0の完了

<!-- status: decided -->
<!-- created: 2026-02-18 -->
<!-- updated: 2026-02-18 -->
<!-- related: proposal/implementation_plan.md, proposal/research_proposal.md -->
<!-- chat_refs: 479526e2-7fc1-4bd9-aaa9-e39c878f2a72 -->

## ステータス

- **状態**: decided
- **作成日**: 2026-02-18
- **最終更新**: 2026-02-18

## 背景

プロポーザル「Macroeconomic Forecasting with LLM speaking Fed's dialect」に基づき、プロジェクトの実装計画を策定し、フェーズ0（環境構築・先行研究の再現）を実行した。

## 議論の経緯

### 2026-02-18: 実装計画の策定

- プロポーザルを精読し、6フェーズの実装計画を策定（`proposal/implementation_plan.md`）
- **重要な制約**: 数値データに関する限り、FRED-MDが唯一のデータソースである。Time-LLMオリジナルが使用していた電力・天気等の汎用データセットは使用しない
- 推奨ディレクトリ構成、技術スタック、主要リスクと対策を文書化

### 2026-02-18: フェーズ0の実施

以下の5タスクを完了した。

#### 0-1. 環境構築

- Python 3.10 + PyTorch 2.10 + Transformers 5.2
- GPU: NVIDIA RTX A6000 × 2（各49GB VRAM）
- `requirements.txt` を整備

#### 0-2. Time-LLM公式リポジトリの調査

- [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM) を `references/Time-LLM/` にクローン
- アーキテクチャの要点:
  - **ReprogrammingLayer**: Cross-attentionによりパッチ化された時系列をLLMの語彙埋め込み空間に射影。`mapping_layer` で全語彙 → `num_tokens=1000` トークンに圧縮後、パッチ埋め込みがquery、語彙埋め込みがkey/valueとなるattentionで線形結合を算出
  - **Prompt-as-Prefix**: データ定義 + 統計量（min/max/median/trend/top-5 lags）をテキスト化し、LLMの入力埋め込みとしてreprogrammed embeddingの前に結合
  - LLMパラメータは完全凍結。学習対象はReprogramming層・mapping_layer・PatchEmbedding・output projectionのみ
  - サポートLLM: LLaMA-7B（dim=4096）、GPT-2（dim=768）、BERT（dim=768）

#### 0-3. FRED-MDデータパイプライン

- `src/data/fred_md.py` を作成
- St. Louis Fed から 2026-01 ヴィンテージ（最新）を自動ダウンロード
- McCracken & Ng (2016) の tcode 変換（7種: level, Δ, Δ², log, Δlog, Δ²log, %change）を適用
- 結果: **802ヶ月 × 122変数**（1959-03 〜 2025-12）、欠損値ゼロ
- Time-LLM互換の PyTorch Dataset クラス（train/val/test = 70/10/20 分割）
- 月次データ向けにデフォルトパラメータを調整: `seq_len=36`, `pred_len=12`, `patch_len=6`, `stride=3`

#### 0-4. VARベースライン

- `src/baselines/var_model.py` を作成
- Expanding-window方式（AICによるラグ自動選択、max_lags=13）
- 8主要ターゲット変数 × 4ホライズン（h=1,3,6,12）の結果:

| Target   | h=1 MSE    | h=3 MSE    | h=6 MSE    | h=12 MSE   |
|----------|------------|------------|------------|------------|
| INDPRO   | 0.000044   | 0.000051   | 0.000058   | 0.000060   |
| CPIAUCSL | 0.000009   | 0.000009   | 0.000009   | 0.000010   |
| UNRATE   | 0.023502   | 0.025102   | 0.026713   | 0.029447   |
| PAYEMS   | 0.000001   | 0.000001   | 0.000002   | 0.000002   |
| FEDFUNDS | 0.074687   | 0.102663   | 0.087473   | 0.064225   |
| GS10     | 0.054692   | 0.060179   | 0.058079   | 0.052277   |
| M2SL     | 0.000012   | 0.000015   | 0.000016   | 0.000017   |
| HOUST    | 0.007070   | 0.010254   | 0.017418   | 0.039965   |

完全な結果は `results/var_baseline_summary.csv` に保存済み。

#### 0-5. 評価パイプライン

- `src/evaluation/metrics.py`: MSE, RMSE, MAE, MAPE, sMAPE, MASE（季節性=12）
- `scripts/run_all_baselines.py`: 複数ターゲット×ホライズンの一括評価

### 2026-02-18: FedTimeLLMモデルの実装

- `src/models/time_llm/model.py`: Time-LLMをFRED-MD向けに適応した `FedTimeLLM` クラス
- `src/models/time_llm/layers.py`: Normalize, PatchEmbedding, ReprogrammingLayer, FlattenHead
- プロンプトテンプレートをマクロ経済予測用に特化（"forecast the next N months given ..."）
- `configs/default.yaml`: GPT-2を軽量実験用デフォルトに設定

## 結論

フェーズ0の全タスクが完了し、以下が確立された:

1. FRED-MD専用のデータパイプラインが安定動作
2. Time-LLMの再実装がFRED-MDデータと正しくインターフェース
3. VARベースラインの定量結果が得られ、比較基準が確立

## 残課題・今後の検討事項

- [ ] フェーズ1: FOMC声明文の収集・Fed語彙辞書の構築
- [ ] `src/models/fed_prompt/` の実装（Step 1: Prompt-as-Prefix としてのFOMC声明文）
- [ ] GPT-2での軽量実験を先行し、LLaMA-7Bでの本番実験に移行

## 参考資料

- Jin, M. et al. (2024) "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", ICLR 2024
- McCracken, M. W. & Ng, S. (2016) "FRED-MD: A Monthly Database for Macroeconomic Research", JBES 34(4)
- Carriero, A. et al. (2025) "Macroeconomic Forecasting with Large Language Models", arXiv
