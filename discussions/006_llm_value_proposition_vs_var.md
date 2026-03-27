# LLMアーキテクチャのVARに対する有用性の根本的検討

<!-- status: active -->
<!-- created: 2026-03-27 -->
<!-- updated: 2026-03-27 -->
<!-- related: decisions.md#DEC-010, 005_interpretability_and_cross_variable.md -->
<!-- chat_refs: THIS_SESSION -->

## ステータス

- **状態**: active
- **作成日**: 2026-03-27
- **最終更新**: 2026-03-27

## 背景

005の議論（解釈可能性とChannel Independence）を通じて、各コンポーネント（Cross-Variable Attention、語彙埋込み行列、Reprogramming、凍結LLMのtransformer層）の役割を分析した結果、より根本的な問いが浮上した：**VARに比べて、cross-attention・reprogramming・LLMを取り入れることの有益性は十分に正当化できるか？**

## 議論の経緯

### 2026-03-27: LLMコンポーネントの有用性をゼロベースで再評価

005の議論で確立された知見を踏まえ、各コンポーネントの有用性を改めて精査した。

**変数間インタラクションの捕捉にLLMは不要**

Cross-Variable Attentionは語彙埋込みにも凍結LLMにも依存しない純粋なアテンション機構。VAR、DFM、GNN等、LLM不要の手法で変数間関係を捕捉可能。

**解釈可能性の3層構造**

| 層 | 内容 | LLM必要性 |
|---|---|---|
| 第1層: 変数レベルの寄与度 | どの変数がどの予測に寄与するか | 不要（CrossVar Attentionで直接得られる） |
| 第2層: 時間パターンの寄与 | どの時期のどのパターンが重要か | 不要（パッチレベルAttentionで得られる） |
| 第3層: 経済的ナラティブ | 概念レベルの解釈（「需要過熱」等） | 語彙埋込み行列が必要（transformer層は不要） |

**Reprogrammingの実態**

- mapping_layerが語彙全体を~1,000個のメタ埋込みに圧縮。パッチが写像されるのは個別の語彙トークンではなくメタ埋込み
- 解釈には事後的に最近傍の語彙トークンを調べる必要がある
- 提供するのは「ナラティブそのもの」ではなく「ナラティブの原材料（語彙空間での座標）」
- Fed語彙制約で改善の余地あるが、根本的な間接性は残る

**凍結LLM transformer層の寄与**

- 出力のd_ff=32/d_llm=768次元しか使用していない
- 入力シーケンスは~42トークンと極短
- Tan et al. (2024): ランダム初期化Transformerや単純MLPとの差がほぼない
- 確実に価値がある唯一の機能：プロンプトの言語理解（テキスト注入チャネル）

### 2026-03-27: VARとの正面比較

| 領域 | VAR | 提案アーキテクチャ | 判定 |
|---|---|---|---|
| 平常時の予測精度 | 強い（線形が十分、少サンプルに適合） | 勝てる見込みが薄い | VAR優位 |
| 転換点の予測 | 弱い（線形外挿の限界） | ポテンシャルあり（非線形+テキスト） | 未検証だが差別化の余地 |
| 変数間関係の定量的解釈 | IRF, FEVDが確立済み | Attentionウェイト（未確立） | VAR優位 |
| 動的・概念レベルの解釈 | 不可能（係数は固定的） | Reprogrammingが時点ごとに異なる語彙プロファイルを出力しうる | 提案アーキテクチャに固有 |
| テキスト情報の直接利用 | 不可能 | 凍結LLMがFOMC声明文を処理可能 | 提案アーキテクチャに固有 |
| サンプル効率 | Bayesian priorで高い | 575ヶ月で非線形関係の学習は困難 | VAR優位 |
| 理論的基盤 | 構造VAR、DSGE | 弱い | VAR優位 |

5勝2敗でVAR優位。

### 2026-03-27: 提案アーキテクチャの固有の価値

VARには不可能だが提案アーキテクチャに可能なことは2つ：

1. **FOMC声明文の前向き情報の直接統合**: フォワードガイダンス、リスク評価、政策意図の変化は数値データに先行するシグナル。VARはテキスト→センチメントスカラーへの圧縮でしかアクセスできない
2. **動的・語彙ベースのナラティブ**: 同じCPIの上昇でも時期によって異なる語彙プロファイル（e.g., "supply disruption" vs "demand pressure"）。VARの固定係数では表現不可能

### 2026-03-27: プロジェクトの再フレーミング

**正当化できない主張**: 「LLMベースのアーキテクチャがVAR（BVAR）を全般的に上回る」

**正当化できる主張**: 「VARが原理的にアクセスできない2つの能力——テキスト情報の直接統合と動的な語彙ベース解釈——が、マクロ経済予測においてどの程度有用かを、厳密な統制実験で検証する」

否定的な結果（LLMは有用でない）であっても方法論的分析として貢献。VARに全般的に勝つことは目標にすべきではない。

## 結論

- 予測精度の全般的改善を主張するのは困難。VARの壁は厚い
- テキスト統合（FOMC声明文）と動的ナラティブ（語彙Reprogramming）がVARにはない固有の価値
- プロジェクトの成否はこれらの「VARにはできないこと」の実証にかかる

## 残課題・今後の検討事項

- [ ] 上記の再フレーミングを踏まえたアーキテクチャの再設計
- [ ] VARベースラインとの公正な比較実験設計
- [ ] 転換点予測でのFOMCテキスト効果の検証方法
- [ ] 語彙ナラティブの品質評価基準の設計

## 参考資料

- Carriero, A. et al. (2025). "Macroeconomic Forecasting with Large Language Models." arXiv preprint.
- Tan, M. et al. (2024). "Are Language Models Actually Useful for Time Series Forecasting?" *NeurIPS 2024*.
- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *ICLR 2024*.
- Shapiro, A.H. & Wilson, D.J. (2022). "Taking the Fed at its Word: A New Approach to Estimating Central Bank Objectives using Text Analysis." *Review of Economic Studies*.
