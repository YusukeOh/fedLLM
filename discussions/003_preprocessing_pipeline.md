# FRED-MD 前処理パイプラインの設計

<!-- status: decided -->
<!-- created: 2026-02-18 -->
<!-- updated: 2026-02-18 -->
<!-- related: decisions.md#DEC-005, decisions.md#DEC-006, 002_fred_md_variable_selection.md -->
<!-- chat_refs: THIS_SESSION -->

## ステータス

- **状態**: decided
- **作成日**: 2026-02-18
- **最終更新**: 2026-02-18

## 背景

FRED-MDコア29変数をTime-LLMのReprogramming層に入力するための前処理パイプラインを設計する必要がある。主な論点は以下の5点:

1. 変数の誤同定（CES0600000007）と差替え
2. 構造的欠損・公表ラグによるjagged edgesへの対処
3. McCracken & Ng変換コード（tcode）の妥当性
4. 外れ値処理方法の選定
5. 正規化手法とパイプライン順序

加えて、チャネル独立（CI）アーキテクチャの限界と、Phase 3での多変量拡張を見据えた設計が求められた。

## 議論の経緯

### 2026-02-18: 変数特性の全数精査

29変数の生データ・変換後データを1978年以降のサンプルで精査し、以下を確認:

- **CES0600000007の誤同定**: 「Avg Hourly Earnings（時間当たり賃金）」として選定したが、実際は「Avg Weekly Hours: Goods-Producing（週平均労働時間: 財生産部門）」。値域37–42で60年間ほぼ横ばい。AWHMAN（週平均労働時間: 製造業）との相関 r=0.965 で実質的に重複。CES0600000008（Avg Hourly Earnings, tcode=6, 1959年〜利用可能）に差替えることで賃金チャネルを復活。
- **29変数中の冗長性分析**: |r|>0.90の全ペアを点検。HOUST×PERMIT(0.97)、CPIAUCSL×CPIULFSL(0.95)、PAYEMS×UNRATE(-0.94)、INDPRO×CUMFNS(0.94) — いずれも経済学的に異なる役割（先行/一致、ヘッドライン/コア等）を持ち、除外不要と判断。
- **UMCSENTxの独自性**: 他28変数との最大相関が |r|=0.34 と極めて低く、代替不可能。始期制約（1978-02）のコストを払ってでも維持すべきと判断。

### 2026-02-18: 訓練始期の決定

29変数の構造的欠損パターンを分析し、始期候補を比較:

| 始期 | 月数 | 訓練サンプル/変数 | 欠損変数 | 備考 |
|------|------|-----------------|---------|------|
| 1959-03 | 802 | 514 | 4変数に構造的欠損 | フルサンプル |
| 1973-02 | 635 | 397 | UMCSENTx 9.4% | TWEXAFEGSMTHx利用可能 |
| 1978-02 | 575 | 355 | **なし** | 全29変数完全 |

**1978-02始期を採用**。根拠:
- 全29変数が構造的欠損なく揃う
- seq_len=36の制約により、1973始期でも石油危機（1973–75）はルックバック窓のみ（実効的損失 -10.6%）
- ボルカー引締め（1979–82）、GFC（2007–09）は完全にカバー
- 現代金融政策レジームの起点として経済学的にも自然

### 2026-02-18: 外れ値処理方針 — Winsorization

McCracken & Ng (2016) のNaN置換→補間方式ではなく、**Winsorization（打ち切り）** を採用。

根拠: FEDFUNDSのボルカー期（1979–84）に15ヶ月分が10×IQR超。NaN置換はこれらの金融政策シグナルを消去するが、Winsorization は方向と時期を保持しつつ StandardScaler の歪みを防止。予測タスクでは極端な事象の学習が重要であり、因子分析向けのMcCracken & Ng方式よりWinsorization が適切。

### 2026-02-18: Jagged edges対処 — NaN伝播の発見と [UNPUB] トークン方式

**NaN伝播問題**: 生データの散発的NaN穴（例: 2025-10月のCPIAUCSL）がtcode変換で増幅される。tcode=6（Δ²log）では1ヶ月のNaN穴が3ヶ月に伝播。

**対処**: 生データ上で内部NaN穴を線形補間してからtcode変換を適用（変換前補間）。これにより伝播を完全防止。

**末尾jagged edgesの根本的対処**: 切詰めでもffillでもなく、**[UNPUB] トークン方式** を採用:
- 末尾の未公表NaNはそのまま保持
- PatchEmbedding層で、未公表パッチを学習可能な [UNPUB] 埋込みに置換
- 訓練時: 既知の公表ラグ構造に基づくシミュレーション（データ拡張）
- プロンプトでも公表ステータスを伝達

この方式は「切詰め」（速い変数の情報を破棄）、「ffill」（偽情報を注入）、「per-variable end」（実装複雑）のいずれの欠点も回避し、モデルが不在を明示的に認知した上で予測を行う。

### 2026-02-18: CI vs CD アーキテクチャの議論

研究提案書のRQ2（「CPI↑ + UNRATE↓ → 需要過熱」等のクロス変数ナラティブ抽出）は、チャネル独立（CI）処理では原理的に不可能であることを確認。

段階的解決を設計:
1. **Phase 2**: FOMC声明文プロンプトでクロス変数文脈を注入（アーキテクチャ変更なし）
2. **Phase 3 前半**: 事後的ナラティブ統合（per-variable attention mapの並列可視化）
3. **Phase 3 核心**: 多変量パッチReprogramming（CI/CDハイブリッド、本研究の独自貢献候補）

前処理パイプラインはCI/CD両対応の有効性マスクを提供する設計とした。

### 2026-02-18: 設計のコード実装

DEC-005/006/007の設計をコードに反映:

- `src/data/fred_md.py`: `load_fred_md()` に `interpolate_internal_nans()`（変換前線形補間）、`winsorize()`（median±10×IQR）、`generate_publication_mask()`（公表ステータスマスク）、`start_date` パラメータを追加。`FREDMDDataset` がデフォルトで `start_date="1978-02"`, `winsorize_k=10.0` を使用
- `configs/default.yaml`: preprocessingセクションに始期、Winsorization閾値、公表ラグtier（tier_0/1/2）を定義
- CES0600000007→CES0600000008の差替えをconfigs/fred_md.py両方に反映
- 動作確認: 575ヶ月×29変数、全変数NA率0%

## 確定した前処理パイプライン

```
Raw FRED-MD CSV
  ↓ Step 1a: 内部NaN穴の線形補間（生データ上、変換前）
  ↓ Step 1b: tcode変換（McCracken & Ng定常化コード）
  ↓ Step 2:  始期制限（1978-02〜）、末尾は切り詰めない
  ↓ Step 3:  外れ値のWinsorization（median ± 10×IQR で打ち切り）
  ↓ Step 4:  StandardScaler（訓練セットでfit）
  ↓ Step 5:  公表ステータスマスク生成（NaN位置 → [UNPUB]対象）
  ↓ [Model] PatchEmbedding（[UNPUB]トークン注入）
  ↓ [Model] 訓練時: 公表ラグシミュレーション（データ拡張）
  ↓ [Model] RevIN → Reprogramming → LLM → Forecast
```

### 設定仕様

```yaml
preprocessing:
  start_date: "1978-02"
  end_date: auto              # 切り詰めなし。末尾NaNは[UNPUB]で処理
  pre_transform_imputation:
    method: linear
    scope: internal_only      # 末尾は補完しない
  tcode: fred_md_official
  tcode_override: {}          # アブレーション用
  outlier:
    method: winsorize
    threshold: 10             # ×IQR
  normalization: standard_scaler
  unpub_token:
    enabled: true
    training_simulation:
      enabled: true
      probability: 0.3        # 訓練サンプルの30%でラグをシミュレート
    publication_lags:
      tier_0: [FEDFUNDS, GS10, BAA, "S&P 500", INDPRO, CUMFNS, PAYEMS,
               UNRATE, CES0600000008, CLAIMSx, CLF16OV, AWHMAN,
               CPIAUCSL, CPIULFSL, OILPRICEx, WPSFD49207,
               TWEXAFEGSMTHx, M2SL, BUSLOANS, UMCSENTx]
      tier_1: [DPCERA3M086SBEA, W875RX1, PCEPI, DSERRG3M086SBEA, ANDENOx]
      tier_2: [CMRMTSPLx, HOUST, PERMIT, ISRATIOx]
```

## 選択肢の比較

### 外れ値処理

| 観点 | McCracken & Ng (NaN→補間) | Winsorization (打ち切り) | なし |
|------|--------------------------|------------------------|------|
| 極端値の情報 | 完全消失 | 方向・時期を保持 | 完全保持 |
| StandardScalerへの影響 | 小 | 小 | 大 |
| 文献的正当性 | FRED-MD公式 | 標準的手法 | — |
| 予測タスクへの適性 | 低（事象を消去） | **高** | 中（数値不安定） |

### Jagged edges

| 観点 | 切詰め | ffill | per-variable end | [UNPUB]トークン |
|------|--------|-------|-----------------|----------------|
| 速い変数の先行性 | 破棄 | 保持 | 保持 | **保持** |
| 偽情報の注入 | なし | あり | なし | **なし** |
| モデルの認知 | 不在を知らない | 不在を知らない | 不在を知らない | **不在を明示的に認知** |
| CI/CD両対応 | CI向き | CI向き | CI向き | **両方** |
| Phase 4拡張 | 再設計必要 | 再設計必要 | 部分的に可 | **そのまま拡張可能** |
| 実装複雑度 | 低 | 低 | 中 | 中 |

## 残課題・今後の検討事項

- [x] `configs/default.yaml` に前処理設定を反映
- [x] `src/data/fred_md.py` に変換前補間・Winsorization・公表マスクを実装
- [ ] `src/models/time_llm/layers.py` に [UNPUB] 埋込みを追加（Phase 2で実装）
- [ ] 訓練時の公表ラグシミュレーション実装（Phase 2で実装）
- [ ] tcode=4（HOUST, PERMIT）→ tcode=5 のアブレーション実験
- [ ] Winsorization閾値（10×IQR）の感度分析
- [ ] Phase 3: 多変量パッチReprogramming層の設計

## 参考資料

- McCracken, M.W. & Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *JBES*, 34(4). — tcode変換、外れ値処理の原典
- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *ICLR 2024*. — PatchEmbedding、ReprogrammingLayerの設計
- Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023*. — チャネル独立・マスクドパッチ事前学習
- Bok, B. et al. (2018). "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics*. — jagged edges、公表ラグの構造
