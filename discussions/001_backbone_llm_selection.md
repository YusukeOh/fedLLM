# バックボーンLLMの選定

<!-- status: active -->
<!-- created: 2025-02-18 -->
<!-- updated: 2026-02-18 -->
<!-- related: decisions.md#DEC-002 -->
<!-- chat_refs: 65860807-18d1-44a9-96b4-f65961779c7a -->

## ステータス

- **状態**: active
- **作成日**: 2025-02-18
- **最終更新**: 2026-02-18

## 背景

Time-LLMフレームワークでは、事前学習済みLLMのパラメータを凍結し、Reprogramming層を通じて時系列データをLLMの語彙空間にマッピングする。そのため、バックボーンLLMの語彙サイズとアーキテクチャは、Reprogramming性能と経済用語のカバレッジに直接影響する。

マシンスペック: RTX A6000 x2（各48GB VRAM）、Intel Core i9-10980XE（18コア/36スレッド）、RAM 128GB、CUDA 12.8

## 議論の経緯

### 2025-02-18: 初期検討 — マシンスペックを踏まえた候補の洗い出し

- Time-LLMの元論文ではLLaMA-7B（32k語彙）を使用
- GPT-2は軽量だが語彙が50kと限定的
- LLaMA-3系は128k語彙でReprogramming空間が大幅に拡大
- A6000 1枚（48GB）でLLaMA-3.1-8Bが十分動作可能であることを確認

### 2025-02-18: フェーズ別の使い分け方針を策定

- フェーズごとに異なるモデルを使うことで、開発効率と実験品質を両立する方針に合意
- Reprogramming層は語彙空間の線形和を使うため、語彙サイズが実験の質に直結するという認識を共有

### 2026-02-18: 詳細レビュー — 元論文のコード確認に基づく再評価

元論文の`TimeLLM.py`を精査し、以下を確認:

- 元論文は**LLAMA / GPT2 / BERT**の3つをサポート（デフォルトはLLaMA-7B全32層）
- デフォルトスクリプト（`TimeLLM_ETTh1.sh`）は`num_process=8`（8 GPU）、`batch_size=24`、`llama_layers=32`
- LLMパラメータは完全凍結（`param.requires_grad = False`）のため、オプティマイザ状態のVRAM負担なし
- Reprogramming層の核心: `word_embeddings`（語彙数×埋め込み次元）→ `mapping_layer = nn.Linear(vocab_size, 1000)` で1000トークンに圧縮

**語彙サイズの重要性に関する追加分析:**

Reprogramming層は`mapping_layer`を通じて全語彙の埋め込みから1000個の「代表トークン」を学習する。語彙が豊かであるほど:
1. 経済用語が個別トークンとして存在する可能性が高く、Reprogramming空間の「分解能」が向上
2. Step 2でFed語彙に制限する際、サブセット抽出の精度が上がる
3. "inflation"、"tightening"、"contractionary"等が分割されずに保持される

**各候補の詳細VRAM見積もり（BF16、バッチサイズ24基準）:**

| モデル | 重み | 学習時推定 | A6000 1枚での余裕 |
|--------|------|-----------|------------------|
| GPT-2 (124M) | ~0.5GB | ~2GB | 46GB余裕 |
| LLaMA-3.2-3B | ~6.5GB | ~12-15GB | 33GB余裕 |
| LLaMA-7B | ~13.5GB | ~20-25GB | 23GB余裕 |
| Mistral-7B | ~14.5GB | ~22-27GB | 21GB余裕 |
| LLaMA-3.1-8B | ~16GB | ~25-30GB | 18GB余裕 |
| LLaMA-2-13B | ~26GB | ~35-42GB | 6GB余裕（タイト） |

**実装上の注意点:**

LLaMA-3系に切り替える際のコード変更点:
- トークナイザ: `LlamaTokenizer` → `AutoTokenizer`（tiktoken系のため）
- `mapping_layer`: `nn.Linear(32000, 1000)` → `nn.Linear(128256, 1000)`（パラメータ増加は軽微: ~96Mパラメータ追加）
- モデル本体 (`LlamaModel`) はLLaMA-3でもそのまま利用可能

## 選択肢の比較

| 観点 | GPT-2 (124M) | LLaMA-7B | LLaMA-3.2-3B | LLaMA-3.1-8B | Mistral-7B | LLaMA-2-13B |
|------|-------------|----------|--------------|--------------|------------|-------------|
| パラメータ数 | 124M | 6.7B | 3.2B | 8.0B | 7.2B | 13B |
| 語彙サイズ | 50,257 | 32,000 | 128,256 | 128,256 | 32,000 | 32,000 |
| VRAM（学習時） | ~2GB | ~20-25GB | ~12-15GB | ~25-30GB | ~22-27GB | ~35-42GB |
| 言語理解力 | 低 | 中 | 中-高 | 高 | 高 | 中-高 |
| 経済用語カバレッジ | 低 | 中 | 高 | 高 | 中 | 中 |
| Time-LLMとの互換性 | 高（実装済み） | 高（実装済み） | 要改修（軽微） | 要改修（軽微） | 要改修（軽微） | 要改修（軽微） |
| イテレーション速度 | 最速 | 中 | 速い | やや遅い | 中 | 遅い |
| 用途 | 開発・デバッグ | 再現実験 | 中間デバッグ | **本実験** | ロバスト性検証 | ロバスト性検証 |

## 結論

**フェーズ別の使い分け:**

| フェーズ | モデル | HuggingFace ID | 理由 |
|---------|--------|----------------|------|
| フェーズ0（再現） | LLaMA-7B | `huggyllama/llama-7b` | 元論文に準拠。結果の再現性を確保 |
| 開発・デバッグ | GPT-2 → LLaMA-3.2-3B | `openai-community/gpt2` → `meta-llama/Llama-3.2-3B` | 高速イテレーション。LLaMA-3.2-3Bは語彙空間がLLaMA-3.1-8Bと同一のため、Fed語彙制限ロジックの開発に有用 |
| フェーズ2-3（本実験） | **LLaMA-3.1-8B** | `meta-llama/Llama-3.1-8B` | 128k語彙による経済用語の豊富なカバレッジ。A6000 1枚で余裕を持って動作 |
| フェーズ4（検証） | Mistral-7B / LLaMA-2-13B | `mistralai/Mistral-7B-v0.1` / `meta-llama/Llama-2-13b-hf` | 異なるバックボーンでの汎化性能の確認。アブレーションとして論文に価値 |

## 残課題・今後の検討事項

- [ ] LLaMA-3.1-8Bへの切り替え実装（Time-LLMコードの改修）
- [ ] 128k語彙のうち、経済関連トークンの分布を事前調査
- [ ] フェーズ2-3の実験後、語彙サイズがReprogramming品質に与える影響を定量評価
- [ ] 必要に応じてLLaMA-3.1-8B-Instructとの比較も検討
- [ ] 元論文のスクリプトが8 GPU前提（`num_process=8`）のため、2 GPU環境に合わせたaccelerateの設定調整

## 参考資料

- Time-LLM元論文: Jin et al. (2024), ICLR 2024
- LLaMA-3 技術レポート: Meta AI (2024)
- HuggingFace: `meta-llama/Llama-3.1-8B`
- 元コード参照: `references/Time-LLM/models/TimeLLM.py`（LLAMA/GPT2/BERT対応箇所）
- 元スクリプト参照: `references/Time-LLM/scripts/TimeLLM_ETTh1.sh`（デフォルト設定）
