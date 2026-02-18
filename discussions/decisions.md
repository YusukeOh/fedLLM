# 確定した意思決定レジストリ

<!-- このファイルは、プロジェクトの主要な設計判断を一覧管理する。 -->
<!-- 各エントリは決定確定時に追記し、詳細は個別の議論ファイルを参照する。 -->
<!-- AIセッションはこのファイルを読むことで、プロジェクトの現在の方針を把握できる。 -->

## 凡例

| ステータス | 意味 |
|-----------|------|
| **confirmed** | 確定済み。実装に反映してよい |
| **tentative** | 暫定方針。変更の可能性あり |
| **superseded** | 別の決定により置き換え済み |

---

## DEC-001: プロジェクト基盤構成

- **日付**: 2025-02-18
- **ステータス**: confirmed
- **決定内容**: フェーズ0〜5の段階的実装計画に基づきプロジェクトを推進する
- **詳細**: [proposal/implementation_plan.md](../proposal/implementation_plan.md)
- **チャット参照**: [実装計画の起案](13616bec-5950-4666-b5a8-787b702e7c53)

## DEC-002: バックボーンLLMの選定

- **日付**: 2025-02-18
- **ステータス**: tentative
- **決定内容**: メインのバックボーンLLMとしてLLaMA-3.1-8Bを採用。再現実験（フェーズ0）はLLaMA-7B、開発・デバッグはGPT-2で実施
- **根拠**: 128k語彙（LLaMA-1/2の4倍）により、経済用語のReprogramming空間が大幅に拡大。A6000 1枚（48GB）で十分動作可能
- **詳細**: [001_backbone_llm_selection.md](001_backbone_llm_selection.md)
- **チャット参照**: [バックボーンLLMレビュー](65860807-18d1-44a9-96b4-f65961779c7a)

## DEC-003: FRED-MD変数選択の方針

- **日付**: 2026-02-18
- **ステータス**: confirmed
- **決定内容**: FRED-MD 128系列からコア29変数を選定。4つの視点（Fed対外コミュニケーション全期間、マクロ経済予測文献、ナウキャスティング文献）で精査し確定
- **根拠**:
  - Boivin & Ng (2006): 変数スクリーニングが因子推定を改善
  - McCracken & Ng (2016): 8因子すべてをカバー
  - NBER一致指標4項目を完備（W875RX1, PAYEMS, INDPRO, CMRMTSPLx）
  - FOMC声明文・議事録・SEPで言及される主要変数を網羅（1959–2026年の全期間にわたり検証）
  - Conference Board LEI: FRED-MD対応可能な8項目中8項目をカバー
- **29変数の構成**: 実体経済5, 労働市場6, 物価6, 住宅2, 設備投資・在庫2, 金融環境4, 為替1, マネー・信用2, センチメント1
- **2026-02-18 修正**: CES0600000007（週平均労働時間、AWHMANと重複 r=0.965）→ CES0600000008（時間当たり賃金）に差替え。賃金チャネル復活
- **詳細**: [002_fred_md_variable_selection.md](002_fred_md_variable_selection.md)
- **チャット参照**: [FRED-MD変数選択（多角的レビュー）](THIS_SESSION)

## DEC-004: フェーズ0の完了

- **日付**: 2025-02-18
- **ステータス**: confirmed
- **決定内容**: フェーズ0（環境構築・Time-LLM再現・VARベースライン）を完了とする
- **成果物**: FRED-MDデータパイプライン（802ヶ月×122変数）、Time-LLMモデル再実装、VARベースライン（8変数×4ホライズン）、評価パイプライン
- **チャット参照**: [実装計画の起案](13616bec-5950-4666-b5a8-787b702e7c53)

## DEC-005: 前処理パイプラインの設計

- **日付**: 2026-02-18
- **ステータス**: confirmed
- **決定内容**: FRED-MD前処理パイプラインを以下の通り確定
  1. **始期**: 1978-02（全29変数が構造的欠損なく揃う最早時点）
  2. **変換前補間**: 生データの内部NaN穴を線形補間（tcode変換によるNaN伝播を防止）。末尾NaNは補完しない
  3. **tcode変換**: McCracken & Ng公式コード。HOUST/PERMIT: tcode=4（アブレーション候補: tcode=5）
  4. **外れ値処理**: Winsorization（median ± 10×IQR で打ち切り）。McCracken & NgのNaN→補間方式ではなく、ボルカー期等の極端な金融政策シグナルの方向・時期を保持
  5. **正規化**: StandardScaler（訓練セットfit）→ RevIN（モデル内部）
  6. **データ分割**: 70/10/20（train/val/test）
- **根拠**: FEDFUNDSボルカー期に15ヶ月分が10×IQR超。NaN置換は金融政策シグナルを消去するが、Winsorization は方向を保持しつつ StandardScaler の歪みを防止
- **詳細**: [003_preprocessing_pipeline.md](003_preprocessing_pipeline.md)
- **チャット参照**: [前処理パイプライン設計](fec25999-527e-411c-878c-c73f665ac98b)

## DEC-006: [UNPUB]トークンによる未公表データの明示的処理

- **日付**: 2026-02-18
- **ステータス**: tentative
- **決定内容**: 末尾 jagged edges（変数ごとの公表ラグによる未公表データ）を、学習可能な [UNPUB] 埋込みにより明示的にモデルに認知させる
  - PatchEmbedding層に `unpub_embedding` パラメータを追加。未公表パッチを学習可能な埋込みで置換
  - 訓練時: 既知の公表ラグ構造に基づくシミュレーション（データ拡張、確率30%）により [UNPUB] の意味を学習
  - PaPでの公表ステータス伝達は不要と判断（埋込みで十分。統計量は公表済みデータのみから計算）
- **根拠**: 切詰め（速い変数の先行シグナル破棄）、ffill（偽情報注入）、per-variable end（実装複雑・CI前提）のいずれの欠点も回避。NLPの [MASK] トークンに着想。ナウキャスティング文献（Bok et al., 2018）における情報到着の概念と整合
- **詳細**: [003_preprocessing_pipeline.md](003_preprocessing_pipeline.md)
- **チャット参照**: [前処理パイプライン設計](fec25999-527e-411c-878c-c73f665ac98b)

## DEC-007: Prompt-as-Prefixの再設計（Domain-Anchored PaP）

- **日付**: 2026-02-18
- **ステータス**: confirmed
- **決定内容**: Time-LLMの統計ベースPaPを、変数名・経済的役割・変換意味を含むDomain-Anchored PaPに置換
  - プロンプトにはパッチから推論不可能な情報のみを載せる（変数名、経済的役割、変換の経済学的意味）
  - 記述統計（min, max, median, trend, lags）は原則除外（アブレーション候補）
  - **時期情報は除外**（凍結LLMの事前学習コーパスを経由するlook-aheadバイアスの経路となるため）
  - [UNPUB]ステータスはプロンプトでは伝達しない（埋込みで十分）
- **根拠**: 記述統計はパッチ埋込みと冗長。変数名・変換意味は凍結LLMの時代不変のドメイン知識を活性化する。トークン数も60–80→25–35に削減
- **詳細**: [004_pap_design_and_lookahead.md](004_pap_design_and_lookahead.md)
- **チャット参照**: [PaP再設計とlook-ahead対策](fec25999-527e-411c-878c-c73f665ac98b)

## DEC-008: LLM内在的look-aheadバイアスへの対処

- **日付**: 2026-02-18
- **ステータス**: confirmed
- **決定内容**: 凍結LLMの事前学習コーパスを経由するlook-aheadバイアスを認識し、以下の統制実験で定量化する
  1. 時間的シャッフル検定（FOMC声明文をランダムな予測時点に割当て）
  2. LLM訓練打ち切り日テスト（打ち切り前後でFOMC効果を分離評価）
  3. GPT-2バックボーン検証（打ち切り~2019年末のGPT-2で2020年以降を評価）
  4. Fedマクロ経済モデル仮想データ検証
- **根拠**: Step 1b（FOMC声明文PaP）はlook-aheadリスクが高い。Step 2（Fed語彙Reprogramming）はリスクが低い（語彙埋込みは時代不変の概念）。look-aheadの定量的分離自体を方法論的貢献として位置づけ
- **詳細**: [004_pap_design_and_lookahead.md](004_pap_design_and_lookahead.md)
- **チャット参照**: [PaP再設計とlook-ahead対策](fec25999-527e-411c-878c-c73f665ac98b)

## DEC-009: フェーズ1の完了とフェーズ2移行

- **日付**: 2026-02-18
- **ステータス**: confirmed
- **決定内容**: フェーズ1（データ収集・前処理）の全タスクを完了とし、フェーズ2（Step 1: Prompt-as-Prefix実験）に移行する
- **成果物**:
  - **FRED-MD前処理パイプライン**: `src/data/fred_md.py` — 変換前線形補間→tcode変換→始期1978-02→Winsorization(10×IQR)→StandardScaler→公表マスク生成。575ヶ月×29変数で動作確認済み
  - **変数差替え**: CES0600000007→CES0600000008（configs/default.yaml, src/data/fred_md.py）
  - **FOMC文書収集**: 244声明文＋308議事録（1993–2025）→ `data/raw/fomc/`
  - **Fed語彙辞書**: GPT-2トークナイザーで5,000トークン → `data/vocabulary/fed_vocab_gpt2.json`
  - **Domain-Anchored PaP**: 29変数プロンプト辞書 `src/data/prompt_dictionary.py`、`model.py` の `forecast()` をper-variable prompt生成に改修
  - **時間軸アライメント**: `src/data/temporal_alignment.py`
  - **前処理設定**: `configs/default.yaml` にpreprocessingセクション追加（始期、Winsorization、公表ラグtier等）
- **フェーズ2の実装予定**:
  - Step 1a: Domain-Anchored PaPでのベースライン学習・評価
  - Step 1b: FOMC声明文をPaPに追加した実験
  - [UNPUB]トークンの `layers.py` 実装（DEC-006）
  - look-ahead統制実験の設計（DEC-008）
- **チャット参照**: [フェーズ1完了・コード実装](THIS_SESSION)
