# Prompt-as-Prefix再設計とLook-aheadバイアス対策

<!-- status: decided -->
<!-- created: 2026-02-18 -->
<!-- updated: 2026-02-18 -->
<!-- related: decisions.md#DEC-007, decisions.md#DEC-008, 003_preprocessing_pipeline.md -->
<!-- chat_refs: fec25999-527e-411c-878c-c73f665ac98b -->

## ステータス

- **状態**: decided
- **作成日**: 2026-02-18
- **最終更新**: 2026-02-18

## 背景

Time-LLMのPrompt-as-Prefix（PaP）は、汎用ベンチマーク（ETTh, Weather等）を対象とするため、データセットの一般的説明と記述統計（min, max, median, trend, lags）で構成されている。FRED-MDではより豊かなドメイン知識を活用できる可能性がある一方、凍結LLMの事前学習コーパスを経由するlook-aheadバイアスへの対処が不可欠である。

## 議論の経緯

### 2026-02-18: PaPの非効率性の分析

オリジナルTime-LLMのPaPの問題点を分析:

- **記述統計はパッチと冗長**: min/max/median/trendはパッチ埋込みが既にLLMに伝えている情報と重複。float→text→token→embeddingの変換を経ており数値精度も低い
- **変数固有情報の欠如**: チャネル独立処理のため、各チャネルが何の変数かをLLMが知らない
- **変換意味の欠如**: tcode=6（Δ²log）で変換済みのCPIは「インフレ率の月次変化」を表すが、この意味がLLMに伝わっていない

設計原則: **パッチから推論できない情報だけをプロンプトに載せる。**

### 2026-02-18: [UNPUB]トークンとPaPの関係

前処理パイプラインで設計した[UNPUB]トークン（学習可能な埋込み）に関して、PaPでも未公表ステータスを伝える必要があるかを検討。

結論: **不要。** [UNPUB]埋込みはend-to-endで予測タスクに最適化して学習されるため、未公表の意味論は埋込み自体に集約される。凍結LLMのテキストプロンプトで追加的に伝える必要はない。ただし、プロンプトの統計量は公表済みデータのみから計算する必要がある（データ衛生の問題）。

### 2026-02-18: Domain-Anchored PaPの提案

変数名・経済的役割・変換の意味をプロンプトに含める「Domain-Anchored PaP」を提案:

```
<|start_prompt|>
Variable: CPI All Items (CPIAUCSL). Headline consumer inflation.
Transformed: second difference of log (change in the monthly inflation rate).
Forecast the next 12 steps.
<|end_prompt|>
```

これにより凍結LLMの時代不変のドメイン知識を活性化。推定トークン数も現行の60–80から25–35に削減。

### 2026-02-18: 時期情報のlook-aheadリスク

当初「Period: Jan 2007 to Dec 2009」のような時期情報も含める案があったが、重大なlook-aheadバイアスの経路となることを確認:

1. 凍結LLMの事前学習コーパスは「2010年に何が起きたか」を含む
2. 学習可能な層（Reprogramming, FlattenHead）は訓練過程でこの漏洩信号を抽出するよう最適化されうる
3. LLMの学習データ打ち切り日の前後で性能が非連続に変化し、再現性のない結果になる

**時期情報は除外と決定。**

### 2026-02-18: FOMC声明文のlook-aheadリスクと提案書レビュー

FOMC声明文自体も時期を暗黙的に特定しうる（例: 特定のFF金利レンジは特定の期間に一意対応）。提案書のlook-ahead認識は「伝統的なデータ管理」にとどまっており、LLM内在的look-aheadへの対処が不足。

- **Step 2（Fed語彙Reprogramming）**: look-aheadリスク**低い**。語彙埋込みは概念の意味を符号化しており、時期を特定しない
- **Step 1b（FOMC声明文PaP）**: look-aheadリスク**高い**。統制実験による分離が必須
- **Step 1a（Domain-Anchored PaP）**: look-aheadリスク**極めて低い**。時代不変の定義のみ

提案書・実装計画にlook-ahead統制実験（時間的シャッフル検定、LLM打ち切り日テスト、GPT-2バックボーン検証）を追加。look-aheadの定量的分離を第3の方法論的貢献として位置づけ。

## 結論

### PaP設計

- **Phase 1**: Domain-Anchored PaP（変数名＋経済的役割＋変換意味）。統計ベースPaPを置換。look-ahead-free
- **Phase 2**: Domain-Anchored PaP + FOMC声明文。look-ahead統制実験と組み合わせて評価
- **統計量**: 原則除外。アブレーション候補として残す
- **時期情報**: 除外（look-aheadリスク）
- **[UNPUB]ステータス**: プロンプトでは伝達しない（埋込みで十分）。統計量は公表済みデータのみから計算

### Look-ahead対策

- 時間的シャッフル検定、LLM打ち切り日テスト、GPT-2バックボーン検証、Fedモデル仮想データ検証の4手法
- look-aheadの定量的分離自体を方法論的貢献として位置づけ

## 残課題・今後の検討事項

- [ ] 29変数のプロンプト辞書（変数名・経済的役割・変換意味）の具体的定義
- [ ] model.py のPaPモジュール改修
- [ ] look-ahead統制実験の実装（フェーズ4）

## 参考資料

- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *ICLR 2024*.
- Carriero, A. et al. (2025). "Macroeconomic Forecasting with Large Language Models." arXiv preprint.
