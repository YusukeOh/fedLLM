# FRED-MD変数選択

<!-- status: decided -->
<!-- created: 2026-02-18 -->
<!-- updated: 2026-02-18 -->
<!-- related: decisions.md#DEC-003 -->
<!-- chat_refs: THIS_SESSION -->

## ステータス

- **状態**: decided
- **作成日**: 2026-02-18
- **最終更新**: 2026-02-18

## 背景

FRED-MDは128系列のマクロ経済変数を含む月次データセット。Time-LLMのReprogrammingでは、多変量時系列をFed語彙の埋め込み空間に射影するため、入力変数の選定はモデルの解釈可能性と予測精度に直結する。全系列をそのまま使うか、意味のあるサブセットに絞るかが論点。

## 議論の経緯

### 2026-02-18: 全系列使用の問題点を整理（初期検討）

- 128系列すべてを入力に使う場合の問題を検討:
  - **ノイズの混入**: 地域別変数の重複（例: HOUSTNE/MW/S/W）
  - **次元の呪い**: 月次データの観測数（~800）に対して128変数は過大
  - **経済学的意味づけの希薄化**: Reprogrammingで得られるナラティブが不明瞭になるリスク
- FOMC声明文・議事録・SEPで言及される変数を基準に絞り込む方針で合意
- コア → 拡張 → フルの3段階で実験する設計に合意

### 2026-02-18: CES0600000007 の誤同定と差替え

前処理パイプライン設計時にデータを精査した結果、CES0600000007 を「Avg Hourly Earnings（時間当たり賃金）」として選定していたが、実際は「Avg Weekly Hours: Goods-Producing（週平均労働時間: 財生産部門）」であることが判明。値域37–42で60年間ほぼ横ばい。AWHMAN（週平均労働時間: 製造業）との相関 r=0.965 で実質重複しており、賃金変数がコア29に一つも含まれていない状態だった。

**対応**: CES0600000008（Avg Hourly Earnings: Goods-Producing, tcode=6）に差替え。1959年から完全利用可能であり、始期制約を追加しない。これにより wage-price dynamics の学習に必要な賃金チャネルが復活した。

→ 詳細は [003_preprocessing_pipeline.md](003_preprocessing_pipeline.md) を参照。

### 2026-02-18: 多角的レビューによる最終選定

4つの視点から系列を精査し、段階的に変数リストを確定した。

#### (1) Fed対外コミュニケーション — 直近10年（2015–2025）

FOMC声明文、議事録、SEP、議長記者会見を精査。以下の重要テーマに対応する変数を特定:

- **デュアルマンデート**: 雇用（PAYEMS, UNRATE）と物価（CPIAUCSL, PCEPI）は必須
- **労働供給制約**（2020–2025の主要テーマ）: `CLF16OV`の追加が必要と判断。既存の労働変数は需要側のみであり、参加率/労働力人口という供給側が欠落していた
- **財 vs サービスインフレの峻別**（2022年〜パウエル議長が三分法を導入）: `DSERRG3M086SBEA`（PCEサービス物価）の追加が必要と判断

#### (2) Fed対外コミュニケーション — 全期間（1959–2026）

FRED-MDの67年間を通史的に精査。FOMC議事録（初期は"Record of Policy Actions"）における各変数の言及パターンを検証:

- **PPI（生産者物価/卸売物価）の一貫した重要性**: 1959–1990年代のFOMC記録では、卸売物価（現PPI）がCPIに先行して議論されるのが常。ボルカー期のディスインフレーションでもPPI最終財の下落がCPI鈍化の先行指標として極めて重視された。2021–2022年のサプライチェーン危機でもPPIが復活。サンプル67年のうち40年以上にわたりFedの主要指標であった → `WPSFD49207`の追加
- **データ可用性の問題**: 一部系列（PERMIT, ANDENOx, TWEXAFEGSMTHx, UMCSENTx）が1959年初期に欠損。実装時の対処が必要

#### (3) マクロ経済予測の学術文献

- **McCracken & Ng (2016)**: FRED-MDの8因子構造に対し、29変数が全因子をカバーしていることを確認
- **Boivin & Ng (2006)**: 「More is not always better」— 変数のスクリーニングが因子推定を改善。128→29への厳選はこの知見と整合
- **Conference Board LEI**: 10構成要素のうちFRED-MDで対応可能な8項目中7項目をカバー。欠落していた`AWHMAN`（製造業週平均労働時間）を追加 — 先行指標として景気転換点を他の労働指標より数ヶ月早く捕捉
- **ターゲット別の充足度**: GDP予測・インフレ予測・失業率予測のいずれに対しても、Stock & Watson (2003), Faust & Wright (2013) 等が特定した重要変数がカバーされていることを確認

#### (4) ナウキャスティング文献（NY Fed Staff Nowcast）

- **Bok et al. (2018)**: NY Fed Nowcastの「ニュース・インパクト」ランキングと照合。上位カテゴリ（雇用統計、鉱工業生産、住宅、耐久財受注、所得・消費）は網羅済み
- **NBER一致指標の完備**: NBERビジネスサイクル日付委員会の4つの月次一致指標のうち3つ（W875RX1, PAYEMS, INDPRO）はカバー済みだが、4番目の`CMRMTSPLx`（実質製造業・商業売上高）が欠落 → 追加。生産（INDPRO）と売上（CMRMTSPLx）の乖離が在庫ダイナミクスを直接反映
- **ソフトデータのギャップ**: UMCSENTxのみ（ISM PMI等はFRED-MD非収録）。FOMC声明文がソフトデータを集約する設計で緩和

## 確定したコア系列：29変数

### 実体経済・生産（5系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `INDPRO` | Industrial Production Index | 5 | NBER一致指標。FOMC "Staff Review" 定番 |
| `DPCERA3M086SBEA` | Real Personal Consumption Expenditures | 5 | GDP最大構成要素（~68%）。声明文 "consumer spending" |
| `CUMFNS` | Capacity Utilization: Manufacturing | 2 | 供給制約の指標。提案書「供給制約」ナラティブに直結 |
| `W875RX1` | Real Personal Income ex Transfers | 5 | NBER一致指標。移転所得除くことで有機的所得成長を捕捉 |
| `CMRMTSPLx` | Real Mfg & Trade Industries Sales | 5 | NBER一致指標（4番目）。GDPの月次プロキシ。INDPRO（生産）との乖離＝在庫変動 |

### 労働市場（6系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `PAYEMS` | Total Nonfarm Payrolls | 5 | NBER一致指標。雇用統計の最重要指標 |
| `UNRATE` | Unemployment Rate | 2 | SEP予測変数。デュアルマンデート |
| `CES0600000008` | Avg Hourly Earnings: Goods-Producing | 6 | 賃金上昇率。wage-price spiralの経路。CES0600000007（週平均労働時間、AWHMANと重複）から差替え |
| `CLAIMSx` | Initial Unemployment Claims | 5 | 先行指標（解雇の流れ）。LEI構成要素 |
| `CLF16OV` | Civilian Labor Force | 5 | 労働供給側。2020年以降「参加率」がFedの主要テーマに |
| `AWHMAN` | Avg Weekly Hours: Manufacturing | 1 | 先行指標（intensive margin）。LEI構成要素。1959年〜完全データ |

### 物価（6系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `CPIAUCSL` | CPI All Items | 6 | ヘッドライン消費者物価 |
| `CPIULFSL` | CPI Less Food & Energy | 6 | コアインフレ。声明文 "core inflation" |
| `PCEPI` | PCE Price Index | 6 | Fedの最重視指標。SEP予測変数 |
| `OILPRICEx` | Crude Oil Prices (WTI) | 6 | 供給ショック識別。「需要過熱 vs 供給制約」弁別の鍵 |
| `DSERRG3M086SBEA` | PCE Services Price Index | 6 | 2022年〜Fedの三分法（財/住宅サービス/非住宅サービス）の中核 |
| `WPSFD49207` | PPI: Finished Goods | 6 | 1959–1990年代FOMCの主要指標。パイプライン・インフレの先行指標 |

### 住宅（2系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `HOUST` | Housing Starts | 4 | 住宅活動の実績指標。声明文 "housing activity" |
| `PERMIT` | Building Permits | 4 | 住宅活動の先行指標。LEI構成要素 |

### 設備投資・在庫（2系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `ANDENOx` | New Orders: Nondefense Capital Goods ex Aircraft | 5 | 設備投資の先行指標。LEI構成要素。データ欠損時は`AMDMNOx`で代替 |
| `ISRATIOx` | Total Business: Inventories/Sales Ratio | 2 | 在庫循環の指標。需給バランスの直接測定 |

### 金融環境（4系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `FEDFUNDS` | Federal Funds Rate | 2 | SEP予測変数。金融政策スタンス |
| `GS10` | 10-Year Treasury Rate | 2 | 長期金利。イールドカーブ（GS10-FEDFUNDS）暗黙的に利用可能 |
| `BAA` | Moody's Baa Corporate Bond Yield | 2 | 信用リスク環境。議事録 "credit conditions" |
| `S&P 500` | S&P 500 Index | 5 | 金融環境・資産効果。LEI構成要素 |

### 為替（1系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `TWEXAFEGSMTHx` | Trade-Weighted USD Index | 5 | 議事録 "the dollar"。純輸出・輸入物価の代理。1973年〜（Bretton Woods前は概念的に不要） |

### マネー・信用（2系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `M2SL` | M2 Money Stock | 6 | 1970s–80sは金融政策の中間目標。COVID期に再注目 |
| `BUSLOANS` | Commercial & Industrial Loans | 6 | 信用チャネル。金融引締めの実体経済への波及を測定 |

### センチメント（1系列）

| FRED-MD ID | 系列名 | tcode | 選定根拠 |
|-----------|--------|-------|---------|
| `UMCSENTx` | U. Michigan Consumer Sentiment | 2 | Fedが最も参照するサーベイ。インフレ期待の代理変数。月次は1978年〜 |

### カテゴリ別サマリー

| カテゴリ | 系列数 | 系列 |
|---------|-------|------|
| 実体経済・生産 | 5 | INDPRO, DPCERA3M086SBEA, CUMFNS, W875RX1, CMRMTSPLx |
| 労働市場 | 6 | PAYEMS, UNRATE, CES0600000008, CLAIMSx, CLF16OV, AWHMAN |
| 物価 | 6 | CPIAUCSL, CPIULFSL, PCEPI, OILPRICEx, DSERRG3M086SBEA, WPSFD49207 |
| 住宅 | 2 | HOUST, PERMIT |
| 設備投資・在庫 | 2 | ANDENOx, ISRATIOx |
| 金融環境 | 4 | FEDFUNDS, GS10, BAA, S&P 500 |
| 為替 | 1 | TWEXAFEGSMTHx |
| マネー・信用 | 2 | M2SL, BUSLOANS |
| センチメント | 1 | UMCSENTx |
| **合計** | **29** | |

## 選択肢の比較

| 観点 | フル系列 (128変数) | コア系列 (29変数) |
|------|------------------|------------------|
| 情報量 | 最大（冗長を含む） | FOMC・学術文献・ナウキャスティングの知見に基づき精選 |
| ノイズ | 高（地域別変数、セクター別変数の重複） | 低（各カテゴリで代表的な系列のみ） |
| 因子カバレッジ | 全8因子（McCracken & Ng, 2016） | 全8因子をカバー |
| NBER一致指標 | 4/4 | 4/4（CMRMTSPLx追加により完備） |
| LEIカバレッジ | 10/10 | 8/10（ISM, Leading Credit IndexはFRED-MD非収録） |
| 解釈可能性 | 低（Reprogrammingのナラティブが拡散） | **高**（Fed語彙との対応が明確） |
| 計算コスト | 高 | 低〜中 |
| 文献的支持 | Boivin & Ng (2006)が課題を指摘 | Boivin & Ng (2006), Ng (2013)の変数スクリーニングと整合 |

## 残課題・今後の検討事項（実装フェーズ）

- [x] `configs/default.yaml` にコア29系列リストを定義
- [x] `src/data/fred_md.py` の `FREDMDDataset` に系列フィルタリングロジックを実装
- [x] データ可用性の検証: TWEXAFEGSMTHx (20.8% NA), UMCSENTx (28.3% NA), ANDENOx (13.6% NA) → `keep_columns` パラメータで保護し `ffill/bfill` で補完。全29変数利用可能を確認
- [x] `data/prompt_bank/fred_md.txt` のプロンプト記述を29系列に合わせて更新
- [ ] コア(29変数) vs フル(128変数) のアブレーション実験設計
- [ ] VIXCLSx の将来的追加を検討（FOMC声明文の利用可能期間[1994年〜]にサンプルを限定する場合、データ欠損の問題が解消）

## 参考資料

### データソース
- McCracken, M.W. & Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4).
- FRED-MD更新データ: https://research.stlouisfed.org/econ/mccracken/fred-databases/

### Fed対外コミュニケーション
- FOMC Statements: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- FOMC Minutes: https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm
- Summary of Economic Projections (SEP): https://www.federalreserve.gov/monetarypolicy/fomcprojtabl.htm

### マクロ経済予測文献
- Boivin, J. & Ng, S. (2006). "Are More Data Always Better for Factor Analysis?" *Journal of Econometrics*, 132(1).
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *JASA*, 97(460).
- Bai, J. & Ng, S. (2008). "Forecasting Economic Time Series Using Targeted Predictors." *Journal of Econometrics*, 146(2).
- Faust, J. & Wright, J.H. (2013). "Forecasting Inflation." *Handbook of Economic Forecasting*, Vol. 2A.
- Ng, S. (2013). "Variable Selection in Predictive Regressions." *Handbook of Economic Forecasting*, Vol. 2A.

### ナウキャスティング
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A. & Tambalotti, A. (2018). "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics*, 10.
- Giannone, D., Reichlin, L. & Small, D. (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics*, 55(4).
- Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). "Now-Casting and the Real-Time Data Flow." *Handbook of Economic Forecasting*, Vol. 2A.

### 本プロジェクト提案書
- Carriero, A., Pettenuzzo, D. & Shekhar, S. (2025). "Macroeconomic Forecasting with Large Language Models." arXiv preprint.
- Jin, M. et al. (2024). "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." *Proceedings of ICLR 2024*.
