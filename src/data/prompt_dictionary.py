"""
Domain-Anchored Prompt-as-Prefix dictionary for 29 FRED-MD core variables.

DEC-007: Each prompt contains only information that cannot be inferred from
the patch embeddings: variable name, economic role, and the economic meaning
of the tcode transformation applied.  No summary statistics, no date/period
information (look-ahead risk), no [UNPUB] status (handled by embedding).

tcode semantic mapping:
  1: level                      → "in levels"
  2: first difference           → "monthly change"
  4: log level                  → "in log levels"
  5: first difference of log    → "monthly growth rate"
  6: second difference of log   → "change in the monthly growth rate"
"""

from __future__ import annotations

TCODE_DESCRIPTIONS: dict[int, str] = {
    1: "in levels",
    2: "monthly change",
    3: "change in the monthly change",
    4: "in log levels",
    5: "monthly growth rate",
    6: "change in the monthly growth rate",
    7: "monthly percent change",
}

VARIABLE_PROMPTS: dict[str, dict[str, str]] = {
    # ── Real economy (5) ──
    "INDPRO": {
        "name": "Industrial Production Index",
        "role": "Broad measure of real output in manufacturing, mining, and utilities. NBER coincident indicator.",
        "tcode": 5,
    },
    "DPCERA3M086SBEA": {
        "name": "Real Personal Consumption Expenditures",
        "role": "Largest component of GDP (~68%). Measures real consumer spending.",
        "tcode": 5,
    },
    "CUMFNS": {
        "name": "Capacity Utilization: Manufacturing",
        "role": "Supply constraint indicator. High values signal inflationary pressure from tight capacity.",
        "tcode": 2,
    },
    "W875RX1": {
        "name": "Real Personal Income excluding Transfers",
        "role": "Organic income growth excluding government transfers. NBER coincident indicator.",
        "tcode": 5,
    },
    "CMRMTSPLx": {
        "name": "Real Manufacturing and Trade Industries Sales",
        "role": "Monthly GDP proxy. Divergence from INDPRO reflects inventory dynamics. NBER coincident indicator.",
        "tcode": 5,
    },
    # ── Labor market (6) ──
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "role": "Headline employment measure. NBER coincident indicator and dual-mandate variable.",
        "tcode": 5,
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "role": "Dual-mandate variable. SEP projection target.",
        "tcode": 2,
    },
    "CES0600000008": {
        "name": "Average Hourly Earnings: Goods-Producing",
        "role": "Wage growth indicator. Key input to wage-price spiral dynamics.",
        "tcode": 6,
    },
    "CLAIMSx": {
        "name": "Initial Unemployment Claims",
        "role": "Leading indicator of labor market turning points. Captures layoff flows.",
        "tcode": 5,
    },
    "CLF16OV": {
        "name": "Civilian Labor Force",
        "role": "Labor supply side. Participation rate became a key Fed theme post-2020.",
        "tcode": 5,
    },
    "AWHMAN": {
        "name": "Average Weekly Hours: Manufacturing",
        "role": "Leading indicator at the intensive margin. Employers adjust hours before headcount.",
        "tcode": 1,
    },
    # ── Prices (6) ──
    "CPIAUCSL": {
        "name": "CPI All Items",
        "role": "Headline consumer inflation.",
        "tcode": 6,
    },
    "CPIULFSL": {
        "name": "CPI Less Food and Energy",
        "role": "Core inflation. Strips volatile components to reveal underlying price trend.",
        "tcode": 6,
    },
    "PCEPI": {
        "name": "PCE Price Index",
        "role": "The Fed's preferred inflation measure. SEP projection target.",
        "tcode": 6,
    },
    "OILPRICEx": {
        "name": "Crude Oil Prices (WTI)",
        "role": "Supply shock identifier. Distinguishes demand-pull from cost-push inflation.",
        "tcode": 6,
    },
    "DSERRG3M086SBEA": {
        "name": "PCE Services Price Index",
        "role": "Core of Fed's three-way inflation decomposition (goods/housing services/non-housing services) since 2022.",
        "tcode": 6,
    },
    "WPSFD49207": {
        "name": "PPI: Finished Goods",
        "role": "Pipeline inflation. Leads CPI. Key FOMC indicator from 1959 through the 2021-22 supply crisis.",
        "tcode": 6,
    },
    # ── Housing (2) ──
    "HOUST": {
        "name": "Housing Starts",
        "role": "Realized housing activity. Interest-rate sensitive sector.",
        "tcode": 4,
    },
    "PERMIT": {
        "name": "Building Permits",
        "role": "Leading indicator of housing activity. Conference Board LEI component.",
        "tcode": 4,
    },
    # ── Investment & Inventory (2) ──
    "ANDENOx": {
        "name": "New Orders: Nondefense Capital Goods excluding Aircraft",
        "role": "Leading indicator of business fixed investment. Conference Board LEI component.",
        "tcode": 5,
    },
    "ISRATIOx": {
        "name": "Total Business Inventories to Sales Ratio",
        "role": "Inventory cycle indicator. Directly measures supply-demand balance.",
        "tcode": 2,
    },
    # ── Financial conditions (4) ──
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "role": "Monetary policy stance. SEP projection target.",
        "tcode": 2,
    },
    "GS10": {
        "name": "10-Year Treasury Rate",
        "role": "Long-term interest rate. Yield curve (GS10 minus FEDFUNDS) signals recession risk.",
        "tcode": 2,
    },
    "BAA": {
        "name": "Moody's Baa Corporate Bond Yield",
        "role": "Credit risk environment. Spread over Treasuries measures financial stress.",
        "tcode": 2,
    },
    "S&P 500": {
        "name": "S&P 500 Index",
        "role": "Financial conditions and wealth effect. Conference Board LEI component.",
        "tcode": 5,
    },
    # ── Exchange rate (1) ──
    "TWEXAFEGSMTHx": {
        "name": "Trade-Weighted US Dollar Index",
        "role": "External competitiveness. Affects net exports and imported inflation.",
        "tcode": 5,
    },
    # ── Money & Credit (2) ──
    "M2SL": {
        "name": "M2 Money Stock",
        "role": "Broad money supply. Intermediate monetary policy target in 1970s-80s, re-emphasized during COVID.",
        "tcode": 6,
    },
    "BUSLOANS": {
        "name": "Commercial and Industrial Loans",
        "role": "Credit channel transmission. Measures how monetary tightening propagates to the real economy.",
        "tcode": 6,
    },
    # ── Sentiment (1) ──
    "UMCSENTx": {
        "name": "University of Michigan Consumer Sentiment",
        "role": "Most-cited survey by the Fed. Proxy for inflation expectations.",
        "tcode": 2,
    },
}


def build_domain_anchored_prompt(
    variable_id: str,
    pred_len: int,
    seq_len: int,
) -> str:
    """Build a Domain-Anchored PaP string for a single variable.

    DEC-007 template:
      <|start_prompt|>
      Variable: {name} ({id}). {role}
      Transformed: {tcode_description}.
      Forecast the next {pred_len} steps.
      <|end_prompt|>
    """
    info = VARIABLE_PROMPTS.get(variable_id)
    if info is None:
        return (
            f"<|start_prompt|>"
            f"Forecast the next {pred_len} steps "
            f"given {seq_len} months of macroeconomic data."
            f"<|end_prompt|>"
        )

    tcode_desc = TCODE_DESCRIPTIONS.get(info["tcode"], "transformed")
    return (
        f"<|start_prompt|>"
        f"Variable: {info['name']} ({variable_id}). {info['role']} "
        f"Transformed: {tcode_desc}. "
        f"Forecast the next {pred_len} steps."
        f"<|end_prompt|>"
    )
