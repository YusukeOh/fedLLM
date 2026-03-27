# 概念空間アーキテクチャ：モジュール設計・FOMC概念学習・LLMの役割再定義

<!-- status: active -->
<!-- created: 2026-03-27 -->
<!-- updated: 2026-03-27 -->
<!-- related: decisions.md#DEC-011, 005_interpretability_and_cross_variable.md, 006_llm_value_proposition_vs_var.md -->
<!-- chat_refs: THIS_SESSION -->

## ステータス

- **状態**: active
- **作成日**: 2026-03-27
- **最終更新**: 2026-03-27

## 背景

005（解釈可能性とCI）および006（LLMの有用性 vs VAR）の議論を通じて、以下が確立された：

- 変数間インタラクションの捕捉にLLMは不要
- 凍結LLMのtransformer層の予測精度への寄与は不確実
- VARに全般的に勝つことは目標にすべきではない
- VARにない固有の価値は：(a) FOMC声明文の直接統合、(b) 動的語彙ベースのナラティブ
- 各コンポーネントの寄与を分離検証可能な設計が必要

本議論では、これらの知見を踏まえたアーキテクチャの具体的設計を扱う。

## 議論の経緯

### 2026-03-27: モジュール分離設計の提案

各コンポーネントの寄与を個別に検証可能にするため、3モジュール構成を提案：

**Module A（多変量時系列エンコーダ、LLM不要）**
```
29変数 × T月 → PatchEmbedding(per-variable)
  → Cross-Variable Attention (投射層 + 残差接続)
  → 時系列表現 (B, N, n_patches, d_model)
```
変数間インタラクションの捕捉を担当。純粋にデータ駆動。

**Module B（語彙アンカー解釈層、LLM埋込み行列のみ）**
```
時系列表現 → Reprogramming(Fed語彙埋込みをkey/value)
  → 語彙空間での表現 (B, N, n_patches, d_llm)
```
RQ2の解釈メカニズム。凍結LLMの埋込み行列のみ使用。

**Module C（テキスト統合層、凍結LLM使用）**
```
FOMC声明文 → 凍結LLMで埋込み → テキスト表現
テキスト表現 + 語彙空間の時系列表現 → Cross-Attention → 統合表現
```
RQ1のテキスト統合。

**アブレーション構造**：

| 構成 | 使用モジュール | 検証する問い |
|---|---|---|
| BVAR | なし | ベースライン |
| A のみ | CrossVar Attention + Head | 非線形アテンションはBVARに対してどこで有利か |
| A + B | + 語彙Reprogramming | 語彙アンカーの効果。解釈の品質 |
| A + B + C | + FOMC声明文統合 | テキスト情報は予測を改善するか |
| Time-LLM (CI) | 現行アーキテクチャ | 凍結LLMのtransformer層の追加的寄与 |

### 2026-03-27: Module Cの解釈可能性問題と構造化テキスト統合

Module Cにおいて、生のテキスト埋込みへのクロスアテンションでは「どのBPEトークンに注目したか」しか分からず、経済学的な解釈が得られない問題を指摘。

**解決策：2段階クロスアテンションによる概念構造化**

FOMC声明文を「経済概念 × 時間軸」のグリッドに分解し、このグリッドに対してクロスアテンションを行う。

- **経済概念**（~11カテゴリ）：実体経済・生産、消費、労働市場、賃金、インフレ、エネルギー・供給、住宅、設備投資、金融環境、対外・為替、金融政策スタンス
- **時間軸**（4区分）：直近の動向(recent)、現状評価(current)、先行き(outlook)、リスク評価(risks)
- → **11概念 × 4時間軸 = 44スロット**

**Stage 1（概念抽出）**: 44個の概念クエリ × FOMCテキスト埋込み → 概念表現 + 抽出アテンション
**Stage 2（時系列統合）**: 時系列表現 × 概念表現 → 統合表現 + 活用アテンション

**解釈のチェーン**：
```
FOMCテキストのどの文 → [Stage 1 抽出アテンション]
  → どの経済概念×時間軸に集約されたか → [Stage 2 活用アテンション]
  → どの変数のどのパッチの予測に影響したか
```

### 2026-03-27: FOMC声明文からの概念学習とLLMの役割再定義

概念タクソノミーを人手で定義するのではなく、FOMC声明文コーパスからデータ駆動で学習する方法を検討。

**課題**: 244声明文（~3,000-5,000文）は教師なし学習に小さい。

**LLMの役割 = 意味的事前知識（semantic prior）**:
- LLMの埋込み空間が「inflation expectations」と「price pressures」を近傍に配置する意味構造を提供
- この意味構造が、小コーパスからの概念クラスタリングを可能にする鍵

**概念学習パイプライン**:
1. 244声明文を文単位に分割
2. LLM埋込み（または sentence transformer）で各文をベクトル化
3. クラスタリングで概念発見（LLMの意味構造がクラスタの質を保証）
4. 時間軸分類（ルールベース: "has been"→recent, "is expected to"→outlook、または段落位置）
5. クラスタ重心が概念クエリベクトルになる

**人手定義 vs FOMC学習の比較**:

| 観点 | 人手定義 | FOMC文から学習 |
|---|---|---|
| 概念の根拠 | 経済学の事前知識 | FOMC声明文のデータ駆動 |
| Fedの認知フレームワークとの整合 | 推定 | 直接的 |
| 粒度の適切さ | 恣意的 | データが決定 |
| 再現性 | 定義者依存 | アルゴリズムで再現可能 |
| 未知の概念の発見 | 不可能 | 可能 |

→ FOMC文からの学習がより知的に誠実。概念タクソノミーがFed自身のコミュニケーション構造から導出される。

### 2026-03-27: 統一概念空間 — Module BとCの統合

FOMC文から学習された概念空間は、Module B（語彙Reprogramming）とModule C（テキスト統合）を統一できる。

**統一設計**:
```
時系列パッチ → Concept Projection → 概念空間 (44 slots)
                                        ↕ 同じ空間で比較・統合
FOMC声明文  → Concept Extraction  → 概念空間 (44 slots)
```

- 時系列パッチの概念写像：「CPIの直近上昇パッチ → "インフレ・直近" に高いウェイト」
- FOMC声明文の概念写像：「"Inflation has moved up" → "インフレ・直近" に高いウェイト」
- 両者が同じ概念にマッピングされることで、テキストと時系列の整合・乖離が直接比較可能

**利点**:
- 語彙Reprogramming（50,000トークンのメタ埋込み）よりコンパクト（44スロット）
- 概念ラベルが明示的で解釈が直接的
- 時系列とテキストのクロスモーダルアライメントが自然に実現
- テキストと時系列のミスマッチ（例：「データはインフレ加速を示すがFOMCは一時的と評価」）が予測に有用な情報を含む可能性

### 2026-03-27: LLMの役割の最終的再定義

| 機能 | LLMの使い方 | 使用タイミング |
|---|---|---|
| 概念空間の構築 | FOMC文の埋込み + クラスタリング | **前処理（1回のみ）** |
| 概念クエリの定義 | クラスタ重心ベクトル | **前処理（1回のみ）** |
| FOMC声明文の概念写像 | 文埋込み + 概念クエリへのアテンション | **推論時（文埋込みは前処理可能）** |
| 時系列の概念写像 | 学習可能な投射（LLM不要） | **訓練・推論時** |

凍結LLMのtransformer層をモデルのフォワードパスで走らせる必要がない。LLMの寄与は前処理段階での意味的事前知識の提供に限定。推論時の計算コストは大幅に低下。

## 選択肢の比較

### アーキテクチャ全体

| 観点 | 現行Time-LLM | モジュール分離設計 | 統一概念空間設計 |
|---|---|---|---|
| 変数間インタラクション | なし（CI） | Module A で捕捉 | Module A で捕捉 |
| テキスト統合 | PaPでプロンプト注入 | Module C で2段階クロスアテンション | 概念空間を介した統合 |
| 解釈可能性 | Reprogrammingアテンション（メタ埋込み） | 語彙Reprogram + 概念×時間軸アテンション | 概念×時間軸アテンション（44スロット、ラベル付き） |
| LLMの推論時使用 | 毎バッチ凍結LLMを走行 | Module Cのみ | **不要**（前処理済み） |
| パラメータ効率 | Reprogram + Head のみ学習 | 各モジュールの学習パラメータ | 概念投射 + CrossVar + Head |
| VARとの比較可能性 | 低い | 高い（モジュール追加で検証） | 高い |

## 結論

### 暫定方針

1. **統一概念空間設計を暫定採用**：FOMC声明文からデータ駆動で概念空間（~44スロット）を構築し、時系列とテキストの共通座標系とする
2. **LLMの役割**：意味的事前知識（semantic prior）の提供に限定。前処理段階でFOMC文を埋込み、概念クラスタを構築。推論時にLLMのtransformer層を走らせない
3. **モジュール構成**：
   - Module A（CrossVar Attention）：変数間インタラクション
   - 統一Module B+C（概念空間）：時系列とFOMCテキストの概念レベル統合・解釈
   - 予測ヘッド
4. **アブレーション**：BVAR → A のみ → A + 概念空間（時系列のみ） → A + 概念空間（+FOMC） → Time-LLM (CI) の段階的比較

### Time-LLMとの関係

現行Time-LLMアーキテクチャ（CI + Reprogramming + 凍結LLM全層）はアブレーション構成の一つとして位置づける。提案アーキテクチャはTime-LLMの「拡張」というより、Time-LLMの各コンポーネントの有用性を検証した上での「再構成」。

## 残課題・今後の検討事項

- [ ] 概念空間の具体的構築手順（クラスタリング手法、クラスタ数決定、時間軸分類実装）
- [ ] 概念クエリの初期化方法の詳細設計（sentence transformer vs LLM埋込み）
- [ ] 時系列→概念空間の投射層の設計（d_model→d_concept、学習可能パラメータ数）
- [ ] テキストと時系列の概念空間ミスマッチの活用方法
- [ ] 提案書（proposal/research_proposal.md）の再フレーミング
- [ ] 実装計画（proposal/implementation_plan.md）の改訂
- [ ] 既存実装コード（src/models/time_llm/）との関係整理

## 参考資料

- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *ICLR 2024*.
- Tan, M. et al. (2024). "Are Language Models Actually Useful for Time Series Forecasting?" *NeurIPS 2024*.
- Liu, Y. et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." *ICLR 2024*.
- Carriero, A. et al. (2025). "Macroeconomic Forecasting with Large Language Models." arXiv preprint.
- Shapiro, A.H. & Wilson, D.J. (2022). "Taking the Fed at its Word." *Review of Economic Studies*.
