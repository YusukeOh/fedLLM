# 解釈可能性の確立とChannel Independenceを超える枠組み

<!-- status: active -->
<!-- created: 2026-03-27 -->
<!-- updated: 2026-03-27 -->
<!-- related: decisions.md#DEC-010, 004_pap_design_and_lookahead.md, 002_fred_md_variable_selection.md -->
<!-- chat_refs: THIS_SESSION -->

## ステータス

- **状態**: active
- **作成日**: 2026-03-27
- **最終更新**: 2026-03-27

## 背景

本プロジェクトのRQ2は「Fed語彙でExplainable AIを構築できるか」である。提案書Step 2では「CPIの上昇に対して、同時に失業率の低下が観測されれば『需要過熱』、生産の停滞が観測されれば『供給制約』というナラティブを抽出する」ことを構想している。しかし、現行のTime-LLMアーキテクチャは完全なChannel Independence（CI）であり、あらゆる処理段階で変数間のインタラクションが存在しない。この制約の下では、RQ2の中核的主張である多変量ナラティブの生成が原理的に不可能である。

本議論では以下の2つの密接に関連するイシューを扱う：
1. 29変数を扱う中でどのように解釈可能性を確立するか
2. 変数間インタラクションを明示的にモデル化するため、CIに代わる枠組みが必要か

## 議論の経緯

### 2026-03-27: 現行アーキテクチャのCI構造の確認

`FedTimeLLM.forecast()` の処理フローを精査し、**すべての段階で変数間のインタラクションが存在しない**ことを確認：

- **PatchEmbedding**: `(B, N, T)` → `(B*N, n_patches, d_model)` — 各変数が独立
- **ReprogrammingLayer**: `(B*N, ...)` に対して共通の `source_emb` で独立にクロスアテンション
- **凍結LLM**: バッチサイズ `B*N` — 29変数は29本の独立シーケンスとして処理
- **FlattenHead**: per-variable linear projection — 出力ヘッドにもクロス変数ミキシングなし

CPIの予測はCPIの過去の系列だけから生成され、同時期のUNRATEやFEDFUNDSの情報は一切使われない。

### 2026-03-27: CIの下でのRQ2の不成立

提案書Step 2の「需要過熱 vs 供給制約」の識別は、複数変数の同時的な状態を条件とした推論である。CIの下では、CPIのReprogrammingアテンションはCPIのパッチだけを見ているため、UNRATEの状態を知りようがない。同じCPIの上昇に対して常に同じアテンションパターンが出力され、ナラティブの出し分けは原理的に不可能。

**結論: CIを超える枠組みなしには、提案書が構想する多変量ナラティブ（RQ2の中核）が成立しない。**

### 2026-03-27: クロス変数インタラクションの設計空間

4つのアプローチを検討：

- **アプローチA（CI維持 + ポストホック解釈）**: per-variable attention weightsを事後的に人間が統合。モデル自体はナラティブを生成しない。RQ2への回答が弱い
- **アプローチB（LLM後にCross-Variable Attention追加）**: LLM出力 `(B, N, D)` に変数次元のアテンション層を挿入。iTransformerに近い
- **アプローチC（全変数を1シーケンスとしてLLMに投入）**: LLMのself-attentionで変数間関係を処理。GPT-2(1024 context)では物理的に不可能
- **アプローチD（Variable-Group処理）**: 経済カテゴリ単位でLLMに投入。グループ間インタラクションが欠落

### 2026-03-27: アプローチBの解釈可能性の限界

アプローチBではCross-Variable AttentionがLLM出力の抽象表現（`d_ff`次元）上で動作する。「UNRATEが寄与した」とは分かるが「失業率の3ヶ月連続低下というパターンが寄与した」とは言えない。**LLM後の抽象表現ではパターンレベルの解釈が失われる。**

→ クロス変数インタラクションは**LLMの前**に配置すべきとの結論に至る。

### 2026-03-27: LLM前のクロス変数インタラクション — 方向性1 vs 方向性2

**方向性1: パッチ → CrossVar Attention → Reprogramming → LLM**

- Reprogrammingの入力がクロス変数の文脈を含むため、語彙マッピングが直接多変量ナラティブを表現
- CI版との語彙マッピング差分がクロス変数の効果を直接可視化
- 課題: `d_model=16` が低次元すぎる → 投射層（d_model→d_crossvar=128）+ 残差接続で解決可能。パラメータ追加 ~70K

**方向性2: パッチ → Reprogramming → CrossVar Attention → LLM**

- per-variable語彙プロファイルがクリーンに保全される
- ナラティブ生成には2つのアテンションマップの合成が必要
- Reprogramming後にCrossVar Attentionが表現を混合するため、**語彙埋込み多様体から逸脱するリスク**（凍結LLMの入力品質劣化）
- d_llm空間での処理はパラメータ数が大きい（GPT-2: ~2.4M, LLaMA: ~67M）。圧縮層を追加しても方向性1より複雑

**方向性1を支持する3つの論拠:**

1. **ナラティブ生成の直接性**: Reprogrammingアテンションを読むだけで多変量ナラティブが得られる（RQ2との整合性が高い）
2. **LLMへの入力品質**: 語彙埋込み多様体上に留まる（Reprogrammingが適正に写像）
3. **パラメータ効率**: ~70K（投射層込み）。575ヶ月のサンプルで十分学習可能

### 2026-03-27: 凍結LLMの役割の再検討

クロス変数インタラクションの議論の中で、より根本的な問いが浮上：**変数間インタラクションを捉えるのに、Fed語彙によるReprogrammingも凍結LLMのtransformer層も機構的に不要ではないか？**

各コンポーネントの機能分析：

| コンポーネント | RQ1（予測精度） | RQ2（解釈可能性） | クロス変数 |
|---|---|---|---|
| Cross-Variable Attention | **主要な寄与** | 間接的（語彙写像に影響） | **本体** |
| 語彙埋込み行列 | 間接的（表現空間の構造） | **主要な寄与** | 無関係 |
| Reprogramming層 | 中（LLM向けの表現学習） | **主要な寄与** | 無関係 |
| Domain-Anchored PaP | 小〜中（条件付け） | 中（変数同定） | 無関係 |
| 凍結LLM transformer層 | **不確実** | 小（プロンプト理解） | 無関係 |
| 予測ヘッド | 最終射影 | なし | なし |

**凍結LLMのtransformer層の寄与が不確実である根拠:**

1. LLM出力の `d_llm` 次元のうち先頭 `d_ff=32` しか使用していない（GPT-2: 32/768, LLaMA: 32/4096）
2. 1変数あたりのシーケンス長が ~42 tokens と極めて短い
3. 先行研究（Tan et al., 2024 "Are Language Models Actually Useful for Time Series Forecasting?"）が、凍結LLMをランダム初期化Transformerや単純なMLPに置換しても性能がほぼ同等であることを報告

**凍結LLMが確実に価値を持つ経路:**

1. **プロンプトの言語理解**: Domain-Anchored PaPのテキストを意味的に処理し、パッチ処理を変数の意味に条件付けできる。ランダム初期化Transformerにはない能力
2. **語彙埋込み行列の意味構造**: Reprogrammingの解釈基盤。ただしこれはtransformer層ではなく埋込み行列の寄与であり、transformer層を通さなくても利用可能
3. **FOMC声明文の処理（Step 1b）**: 凍結LLMがFOMC声明文を意味的に理解することが前提。ここでは凍結LLMが不可欠

**各コンポーネントの役割の分離:**

- **Cross-Variable Attention**: 変数間インタラクションの学習（RQ1主導、RQ2にも間接寄与）
- **語彙埋込み行列 + Reprogramming**: 解釈可能な表現基盤（RQ2主導）
- **凍結LLM transformer層**: テキスト情報と時系列情報を同一空間で処理するインターフェース。「万能の時系列モデル」ではなく、テキスト注入チャネルとして正当化される

## 選択肢の比較

### クロス変数インタラクションの配置

| 観点 | 方向性1（パッチ→CrossVar→Reprogram→LLM） | 方向性2（パッチ→Reprogram→CrossVar→LLM） |
|---|---|---|
| ナラティブの直接性 | Reprogramming weightが直接多変量ナラティブを表現 | 2つのattention mapの合成が必要 |
| CI比較による効果分離 | 語彙マッピングの変化として可視化可能 | 間接的 |
| LLM入力品質 | 語彙多様体上に留まる | 多様体から逸脱するリスク |
| per-variable解釈の保全 | 残差接続で保持（完全な分離ではない） | 完全に保全 |
| パラメータ効率 | ~70K（投射層込み） | 圧縮しても方向性1より複雑 |
| d_model容量 | 要投射層（16→128） | d_llm空間で十分 |
| RQ2との整合性 | 高い | 中 |

### 凍結LLMの位置づけ

| 観点 | 凍結LLM維持 | 凍結LLM除去（埋込み行列のみ保持） |
|---|---|---|
| 予測精度 | 不確実（先行研究で疑問視） | 学習可能層に置換すれば同等以上の可能性 |
| テキスト注入（PaP） | 可能 | 不可能（ランダム初期化では言語理解なし） |
| FOMC声明文処理（Step 1b） | 可能 | 不可能 |
| 語彙ベース解釈（RQ2） | 可能 | 可能（埋込み行列で十分） |
| 計算コスト | 高（8Bパラメータの推論） | 大幅に低下 |
| 研究ポジショニング | "LLMベースの予測" | "語彙誘導型の解釈可能予測" |

## 結論

### 暫定方針

1. **クロス変数インタラクション**: 方向性1（パッチレベルCrossVar Attention → Reprogramming → LLM）を暫定採用。投射層（d_model=16 → d_crossvar=128）+ 残差接続で実装。パラメータ追加 ~70K
2. **凍結LLMの位置づけ**: 「テキスト情報と時系列情報を同一空間で処理するインターフェース」として維持。変数間インタラクションはCrossVar Attentionが担い、凍結LLMには依存しない
3. **アブレーション設計**: 凍結LLMのtransformer層の寄与を実験的に検証するため、以下の比較を設計に含める

### 提案するアブレーション構造

1. **CI版（現行Time-LLM）**: 凍結LLMのみ。ベースライン
2. **CI + CrossVar Attention**: 変数間インタラクションの追加効果
3. **CrossVar Attentionのみ（凍結LLMのtransformer層除去、語彙埋込み行列は保持）**: 凍結LLMのtransformer層が本当に必要かのアブレーション
4. **CI + CrossVar + FOMC PaP**: テキスト注入の効果（ここでは凍結LLMが不可欠）

## 残課題・今後の検討事項

- [ ] 方向性1のCross-Variable Attention層の詳細設計（投射次元、ヘッド数、残差接続の形式）
- [ ] アブレーション3（凍結LLM除去版）の具体的な代替アーキテクチャ（学習可能Transformer? MLP?）
- [ ] Cross-Variable Attentionの導入タイミング（Step 1aからか、Step 2からか）
- [ ] アプローチC（全変数1シーケンス）の再検討可能性（LLaMA-3.1-8Bの128kコンテキストでは十分実現可能）
- [ ] 方向性1でのアテンション可視化・ナラティブ抽出の具体的手法
- [ ] 575ヶ月のサンプルサイズでCrossVar Attentionの29変数間関係が学習可能かの検証計画

## 参考資料

- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *ICLR 2024*.
- Tan, M. et al. (2024). "Are Language Models Actually Useful for Time Series Forecasting?" *NeurIPS 2024*.
- Liu, Y. et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." *ICLR 2024*.
- Carriero, A. et al. (2025). "Macroeconomic Forecasting with Large Language Models." arXiv preprint.
