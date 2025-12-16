# 📈 Halal Stock AI Analysis Pipeline

A sophisticated AI-powered stock analysis system that combines halal compliance screening, LSTM neural networks, backtest validation, and multi-factor risk analysis to identify the most promising investment opportunities.

## 🎯 System Overview

```
┌─────────────────────────────────────────────┐
│  Stage 1: Universe Builder                  │
│  build_halal_universe.jl                    │
│  ↓ Zoya API Screening                       │
│  ↓ 526 Halal-Compliant Stocks               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Stage 2: AI Analysis & Validation          │
│  stock_picker.jl (Julia)                    │
│  ↓ LSTM Neural Network (20 epochs)          │
│  ↓ 30-Day Backtest Validation               │
│  ↓ Risk Metrics (Sharpe, Drawdown)          │
│  ↓ Analyst Ratings                          │
│  ↓ News Sentiment Analysis                  │
│  ↓ Champion Formula Scoring                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Output: Multi-Sheet Excel Report           │
│  • All Results with Risk Categories         │
│  • Champions Only (70%+ backtest, <30% DD)  │
│  • Detailed Investment Explanations         │
└─────────────────────────────────────────────┘
```

---

## 🕌 Stage 1: Halal Universe Builder

**Script:** `build_halal_universe.jl`

### Purpose
Fetches halal-compliant stocks from the Zoya API to create a universe of investment candidates that meet Islamic financial principles.

### Features
- 🔍 **Zoya API Integration** - Comprehensive halal compliance database
- 🏛️ **Exchange Filtering** - Major US exchanges (NYSE, NASDAQ, AMEX, ARCA)
- 💰 **Price Screening** - Stocks ≥ $5 for liquidity
- 🔄 **Retry Logic** - Exponential backoff for API limits
- 💾 **Progress Saving** - Checkpoints every 50 stocks

### Output Files

| File | Format | Content |
|------|--------|---------|
| `halal_universe.txt` | Line-separated | 526 halal stocks |
| `halal_universe_comma.txt` | Comma-separated | Same, easier to copy |

### How to Run

```powershell
julia build_halal_universe.jl
```

### Configuration

```julia
API_KEY = "live-03e8bf0f-6bda-40b5-9d0e-ec884e8c6c9b"
MIN_PRICE = 5.00
EXCHANGES = ["XNYS", "XNAS", "XASE", "ARCX"]
```

---

## 🚀 Stage 2: AI Stock Analyzer

**Script:** `stock_picker.jl`

### Purpose
Performs comprehensive AI analysis with neural network predictions, backtest validation, risk metrics, analyst ratings, and news sentiment to identify champion investment opportunities.

## 🧠 Analysis Components

### 1. LSTM Neural Network (40% weight)
- **Architecture**: 3-layer LSTM (20→64→32→1)
- **Training Data**: 730 days (2 years) of historical prices
- **Training**: 20 epochs with Adam optimizer
- **Method**: 20-day sliding windows for supervised learning
- **Output**: Price movement prediction

### 2. Backtest Validation (Multiplier)
- **Period**: 30-day out-of-sample validation
- **Metric**: Directional accuracy percentage
- **Multiplier**: Maps 30%→0.3x, 70%→1.0x, 110%→1.5x
- **Effect**: Amplifies or dampens base score based on proven accuracy

### 3. Risk Metrics (Adjustment)
- **Sharpe Ratio**: Risk-adjusted return quality (±10% bonus)
- **Max Drawdown**: Worst peak-to-trough loss (up to -25% penalty)
- **Volatility**: Price stability measurement
- **Combined**: Risk adjustment added to final score

### 4. Analyst Ratings (12% weight)
- **Source**: Finnhub professional recommendations
- **Data**: Strong Buy, Buy, Hold, Sell, Strong Sell counts
- **Score Range**: -2 (bearish) to +2 (bullish)
- **Coverage**: Major financial institutions

### 5. News Sentiment (8% weight)
- **Source**: Finnhub company news (last 7 days)
- **Analysis**: Advanced sentiment scoring (auto-selects best available)
- **Options**:
  - **TextAnalysis.jl** (Julia's VADER equivalent) - if installed
  - **Custom VADER-like** (fallback) - built-in implementation
- **Features**:
  - Weighted sentiment lexicon (1.0-3.0 intensity scores)
  - Negation handling ("not good" flips sentiment)
  - Intensity boosters ("very good" amplifies score)
  - Capitalization emphasis (ALL CAPS = stronger)
  - Punctuation emphasis (!!! increases intensity)
- **Score Range**: -1.0 (very negative) to +1.0 (very positive)
- **Install TextAnalysis**: `julia -e 'using Pkg; Pkg.add("TextAnalysis")'`

## 🏆 Champion Formula

```
Base Score = (40% AI Model + 12% Analysts + 8% Sentiment)

Backtest Multiplier = Map accuracy (30%→0.3x, 70%→1.0x, 110%→1.5x)

Risk Adjustment = (Sharpe Bonus ±10% - DD Penalty up to -25%) × 0.25

Final Score = (Base Score × Backtest Multiplier) + Risk Adjustment
```

### Champion Criteria (Excel Sheet 2)
✅ **70%+ Backtest Accuracy** - Proven predictive power  
✅ **<30% Max Drawdown** - Controlled risk  
✅ **Positive Combined Score** - Net bullish signal  

## 🎯 Risk Categories

| Category | Max Drawdown | Badge | Description |
|----------|-------------|-------|-------------|
| **SAFE** | <15% | 🛡️ | Very low risk |
| **LOW RISK** | 15-25% | ✅ | Acceptable risk |
| **MEDIUM RISK** | 25-40% | ⚠️ | Moderate risk |
| **HIGH RISK** | 40-60% | ⚠️⚠️ | Significant risk |
| **EXTREME RISK** | >60% | ❌ | Dangerous volatility |

## 🔧 Technical Features

### Performance Optimization
- ⚡ **12-Thread Parallel Processing** - Concurrent training
- 📦 **Sequential Downloads** - Prevents HTTP threading crashes
- 🔄 **Retry Logic** - Exponential backoff (max 5 attempts)
- 🎯 **Rate Limiting** - Respects API limits

### Data Sources

| Source | Purpose | Authentication |
|--------|---------|----------------|
| Yahoo Finance | Historical prices (730 days) | None (free) |
| Finnhub API | Analyst ratings + news | `d4kp2j1r01qvpdollej0` |

### Output Formats
1. 📊 **CSV** - Complete dataset with all metrics
2. 📈 **Excel (2 sheets)**
   - Sheet 1: All Results with risk categories
   - Sheet 2: Champions Only with detailed explanations
3. 📉 **PNG Chart** - Top 10 visual ranking (1400×900)

---

## 🚀 How to Run

### Quick Start (15 stocks from tickers.txt)

```powershell
$env:JULIA_NUM_THREADS=12; julia stock_picker.jl
```

⏱️ **~5-10 minutes**

### Full Universe Analysis (526 halal stocks)

```powershell
Copy-Item halal_universe.txt tickers.txt
$env:JULIA_NUM_THREADS=12; julia stock_picker.jl
```

⏱️ **~2-3 hours**

---

## 📁 Output Files

All outputs saved to `export/` directory with timestamp: `YYYYMMDD_HHMMSS`

| File | Description |
|------|-------------|
| `stock_predictions_*.csv` | Complete dataset (all metrics) |
| `stock_predictions_*.xlsx` | **2-Sheet Excel Report**<br>• Sheet 1: All results<br>• Sheet 2: Champions only |
| `stock_predictions_*.png` | Top 10 bar chart with badges |

---

## ⚙️ Configuration

```julia
# Training Parameters
const HISTORY_DAYS = 730        # 2 years of historical data
const WINDOW_SIZE = 20          # Days per training sample
const EPOCHS = 20               # Training iterations per stock
const BATCH_SIZE = 64           # Neural network batch size
const BACKTEST_DAYS = 30        # Validation period

# Performance Settings
const MAX_PARALLEL = 12         # Concurrent stock processing
const USE_GPU = false           # CPU mode (CUDA disabled)

# Champion Formula Weights
const AI_WEIGHT = 0.40          # Neural network prediction
const ANALYST_WEIGHT = 0.12     # Professional ratings
const NEWS_WEIGHT = 0.08        # Sentiment analysis
# Note: Backtest multiplier and risk adjustment applied separately
```

---

## 📊 Sample Output

### Terminal Output

```
[INFO] Processing 15 stocks with 12 parallel workers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📥 DOWNLOAD PHASE (Sequential)
  [OK] AAPL: HTTP 200, got 730 rows
  [OK] MSFT: HTTP 200, got 730 rows
  ...

🧠 TRAINING PHASE (Parallel - 12 workers)
Stock AAPL:
  Epoch 1/20 - Train: 0.0234, Test: 0.0198
  ...
  Epoch 20/20 - Train: 0.0098, Test: 0.0089
  Backtest Accuracy: 72.5% ✅
  Max Drawdown: 18.3% (LOW RISK)
  Sharpe Ratio: 1.89 (EXCELLENT)
  Analyst: 15 strong buy, 8 buy (score: 1.65)
  Sentiment: 0.75 (Positive news bias)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 FINAL RANKINGS
  1. ⭐ AAPL - 8.45% | 72% backtest | 18% DD | 1.89 Sharpe | LOW RISK
     Why: Strong backtest + excellent Sharpe + Wall Street loves it
  
  2. ⭐ MSFT - 7.82% | 75% backtest | 16% DD | 2.15 Sharpe | LOW RISK
     Why: Outstanding backtest + exceptional Sharpe + consistent growth
  ...

💰 INVESTMENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⭐ CHAMPIONS (70%+ Backtest, <30% Drawdown, Positive Score):
   5 stocks meet ALL champion criteria:
   • AAPL - 8.45% | 72% backtest | 18% DD | 1.89 Sharpe
   • MSFT - 7.82% | 75% backtest | 16% DD | 2.15 Sharpe
   ...

💡 Portfolio Statistics:
   • Stocks analyzed: 15
   • Average backtest accuracy: 65.3%
   • Average max drawdown: 24.7%
   • Average Sharpe ratio: 1.23
   
   • Champions (all 3 criteria): 5
   • High accuracy stocks (70%+ backtest): 8
   • Low risk stocks (<25% drawdown): 10
```

---

## 💻 Technical Requirements

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **Julia** | 1.12+ |
| **RAM** | 8GB+ (16GB recommended for 526 stocks) |
| **CPU** | Multi-core (12 threads recommended) |
| **GPU** | Optional (NVIDIA RTX 4090, currently disabled) |

### Required Julia Packages

```julia
HTTP          # API requests
JSON3         # JSON parsing  
DataFrames    # Data manipulation
Flux          # Neural networks
Random        # Randomization
Statistics    # Statistical functions
Dates         # Timestamps
Base.Threads  # Parallel processing
CSV           # CSV export
XLSX          # Excel export
Plots         # Chart generation
```

### Optional Packages (Recommended)

```julia
TextAnalysis  # Advanced NLP & sentiment analysis (Julia's VADER equivalent)
Languages     # Language processing utilities
```

### Quick Installation

**Essential packages:**
```powershell
julia -e 'import Pkg; Pkg.add(["HTTP", "JSON3", "DataFrames", "Flux", "CSV", "XLSX", "Plots"])'
```

**With advanced sentiment analysis:**
```powershell
julia -e 'import Pkg; Pkg.add(["HTTP", "JSON3", "DataFrames", "Flux", "CSV", "XLSX", "Plots", "TextAnalysis", "Languages"])'
```

---

## 🔧 Troubleshooting

### ⚠️ API Rate Limits

✅ **Built-in protection:**
- Exponential backoff retry (max 5 attempts)
- Sequential downloads (prevents threading crashes)
- Random delays between requests

### ⚠️ Memory Issues

If you experience crashes:

```julia
const MAX_PARALLEL = 6    # Reduce from 12
const HISTORY_DAYS = 365  # Reduce from 730
```

### ⚠️ GPU Support (Currently Disabled)

To re-enable CUDA acceleration:

```powershell
julia -e 'import Pkg; Pkg.update("CUDA"); Pkg.build("CUDA")'
```

Then in `stock_picker.jl`:

```julia
using CUDA
const USE_GPU = true
```

---

## 🎓 Understanding Results

### 🏆 Champion Stocks

These meet **ALL three criteria:**
1. ✅ 70%+ backtest accuracy (proven predictions)
2. ✅ <30% max drawdown (controlled risk)  
3. ✅ Positive combined score (bullish signal)

### 📊 Score Interpretation

| Score | Signal | Action |
|-------|--------|--------|
| **>6%** | Strong Buy | High confidence entry |
| **3-6%** | Buy | Good opportunity |
| **0-3%** | Weak Buy | Monitor for better entry |
| **<0%** | Avoid | Bearish signal |

### 🎯 Excel Sheet 2 Columns Explained

| Column | Meaning |
|--------|---------|
| **Backtest_Rating** | How accurate past predictions were |
| **Why_Trust_It** | Plain English explanation of accuracy |
| **Risk_Level** | SAFE/LOW/MEDIUM risk category |
| **Worst_Case_Loss** | Dollar impact on $10,000 investment |
| **AI_Outlook** | STRONG BUY / BUY / NEUTRAL |
| **Bang_For_Buck** | Return per $100 of risk (Sharpe) |
| **Wall_Street_Says** | Analyst consensus summary |
| **News_Quality** | Sentiment analysis result |
| **Why_Champion** | Comprehensive explanation |

---

## 📁 Project Structure

```
Julia_Predicto_Test/
├── 📄 README.md                       # Documentation (this file)
├── 🧩 build_halal_universe.jl        # Stage 1: Halal universe builder
├── 🤖 stock_picker.jl                # Stage 2: AI analyzer
├── 📋 halal_universe.txt             # 526 halal stocks (line-separated)
├── 📋 halal_universe_comma.txt       # 526 halal stocks (comma-separated)
├── 📋 tickers.txt                    # Input tickers to analyze
├── 🚫 .gitignore                     # Git exclusions
├── 📦 export/                        # All output files (CSV, Excel, PNG)
│   ├── stock_predictions_*.csv       # Complete datasets
│   ├── stock_predictions_*.xlsx      # 2-sheet Excel reports
│   └── stock_predictions_*.png       # Top 10 charts
└── 📦 archive/                       # Historical runs (git ignored)
```

---

## 🔑 API Configuration

### Zoya API (Halal Compliance)

```julia
API_KEY = "live-03e8bf0f-6bda-40b5-9d0e-ec884e8c6c9b"
FILE: build_halal_universe.jl
```

### Finnhub API (Analyst Ratings & News Sentiment)

```julia
FINNHUB_KEY = "d4kp2j1r01qvpdollej0d4kp2j1r01qvpdollejg"
FILE: stock_picker.jl
```

### Yahoo Finance API
- **Authentication**: None required (free public access)
- **Rate Limits**: Generous (built-in retry logic)

---

## 🎯 Best Practices

### ✅ DO
- Focus on **Champion stocks** (Sheet 2 in Excel)
- Prioritize stocks with 70%+ backtest accuracy
- Consider risk levels (prefer SAFE or LOW RISK)
- Check Sharpe ratios (>1.5 is excellent)
- Verify against your own research

### ❌ DON'T
- Ignore max drawdown percentages
- Rely solely on AI predictions
- Invest in stocks with <50% backtest accuracy
- Ignore negative news sentiment
- Use predictions as sole investment basis

---

## 🌟 Roadmap

- [ ] GPU acceleration (re-enable CUDA)
- [ ] Portfolio optimization (Modern Portfolio Theory)
- [ ] Stop-loss recommendations
- [ ] Price target calculations
- [ ] Real-time monitoring dashboard
- [ ] Automated trading integration
- [ ] Risk-adjusted position sizing

---

## ⚖️ Disclaimer

**For Educational Purposes Only**

This software provides AI-based stock analysis for educational and research purposes. Stock predictions are based on historical data and mathematical models, which **do not guarantee future performance**. 

⚠️ **Important:**
- Always conduct your own research
- Consult qualified financial advisors
- Verify halal compliance with your own Islamic scholars
- Past performance does not indicate future results
- Invest only what you can afford to lose

The halal compliance data is sourced from Zoya API and should be independently verified according to your personal Islamic financial principles.

---

## 👨‍💻 Author

**Built with ❤️ using Julia AI & Deep Learning**

📧 Questions? Open an issue on GitHub  
⭐ Like this project? Give it a star!

---

**Predicto Julia ALT** - *Advanced Stock Analysis with AI-Powered Insights*
