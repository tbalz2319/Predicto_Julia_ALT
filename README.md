# 📈 Halal Stock AI Analysis Pipeline

A sophisticated multi-stage AI-powered stock analysis system that combines halal compliance screening, primary AI predictions, and secondary neural network validation to identify the most promising investment opportunities.

## 🎯 System Overview

This pipeline performs three sequential stages of analysis:

```
┌─────────────────────────┐
│  Stage 1: Universe      │
│  build_halal_universe.jl│
│  ↓ Zoya API Screening   │
│  ↓ 526 Halal Stocks     │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Stage 2: Primary AI    │
│  predicto.py (Python)   │
│  ↓ Initial Predictions  │
│  ↓ Top 15 Candidates    │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Stage 3: Validation    │
│  stock_picker.jl (Julia)│
│  ↓ Neural Network       │
│  ↓ Analyst Ratings      │
│  ↓ News Sentiment       │
│  ↓ Final Rankings       │
└─────────────────────────┘
```

---

## 🕌 Stage 1: Halal Universe Builder

**Script:** `build_halal_universe.jl`

### Purpose
Fetches and filters halal-compliant stocks from the Zoya API to create a universe of investment candidates that meet Islamic financial principles.

### Features
- **Zoya API Integration**: Queries comprehensive halal compliance database
- **Exchange Filtering**: Focuses on major US exchanges (NYSE, NASDAQ, AMEX, ARCA)
- **Price Screening**: Filters stocks ≥ $5 to ensure liquidity
- **Retry Logic**: Exponential backoff for API rate limits
- **Progress Saving**: Checkpoints every 50 stocks processed

### Output Files
- `halal_universe.txt`: One ticker per line (526 stocks)
- `halal_universe_comma.txt`: Comma-separated format for easy copying

### How to Run
```powershell
julia build_halal_universe.jl
```

### Configuration
- **Zoya API Key**: `live-03e8bf0f-6bda-40b5-9d0e-ec884e8c6c9b`
- **Minimum Price**: $5.00
- **Target Exchanges**: XNYS, XNAS, XASE, ARCX

---

## 🐍 Stage 2: Primary AI Predictor

**Script:** `predicto.py` (Python)

### Purpose
Performs initial AI-based predictions on the halal stock universe to identify the most promising candidates for deeper analysis.

### Process
1. Ingests the 526 halal-compliant tickers from Stage 1
2. Runs proprietary AI prediction models
3. Ranks stocks by predicted performance
4. Outputs top 15 candidates to `tickers.txt`

### Output
- **Top 15 Stock Tickers**: Saved to `tickers.txt` for Stage 3 processing

### How to Run
```powershell
python predicto.py
```

*Note: Predicto.py is a separate Python-based prediction system maintained independently.*

---

## 🚀 Stage 3: Neural Network Validator

**Script:** `stock_picker.jl` (Julia)

### Purpose
Performs comprehensive secondary analysis on the top 15 candidates using neural networks, analyst ratings, and news sentiment to produce final investment rankings.

### Multi-Factor Analysis

#### 1️⃣ Neural Network Predictions (60% weight)
- **Architecture**: 3-layer network (20→64→32→1)
- **Training Data**: 365 days of historical prices from Yahoo Finance
- **Method**: 20-day sliding windows for supervised learning
- **Training**: 3 epochs with Adam optimizer
- **Metrics**: Direction accuracy and price movement prediction

#### 2️⃣ Analyst Ratings (25% weight)
- **Source**: Finnhub professional analyst recommendations
- **Data Points**: Strong Buy, Buy, Hold, Sell, Strong Sell counts
- **Scoring**: Buy score ranging from -2 (bearish) to +2 (bullish)
- **Coverage**: Consensus from major financial institutions

#### 3️⃣ News Sentiment (15% weight)
- **Source**: Finnhub company news API
- **Window**: Last 7 days of news articles
- **Metric**: Buzz score (0.0 to 1.0) based on article volume
- **Calculation**: news_count / 10.0, capped at 1.0

### Technical Features

#### Performance Optimization
- **Parallel Processing**: 12-thread concurrent execution
- **Batch Processing**: MAX_PARALLEL=12 stocks simultaneously
- **Rate Limiting**: Random delays (0.5-1s) to respect API limits
- **Retry Logic**: Exponential backoff for 429 errors (max 5 attempts)

#### Data Sources
- **Yahoo Finance**: Historical price data (free, no authentication)
- **Finnhub API**: Analyst ratings & news sentiment
  - API Key: `d4kp2j1r01qvpdollej0d4kp2j1r01qvpdollejg`

#### Output Formats
1. **CSV**: Detailed spreadsheet with all metrics
2. **Excel**: Formatted XLSX with rankings
3. **PNG Chart**: Visual bar chart (1400x900) with color-coded predictions
   - Green bars: Positive predictions
   - Red bars: Negative predictions

### How to Run

#### Basic Execution
```powershell
$env:JULIA_NUM_THREADS=12; julia stock_picker.jl
```

#### Run on Full Universe (526 stocks)
```powershell
Copy-Item halal_universe.txt tickers.txt
$env:JULIA_NUM_THREADS=12; julia stock_picker.jl
```
*Estimated time: 45-60 minutes*

### Output Files
All outputs are timestamped with format: `stock_predictions_YYYYMMDD_HHMMSS`

- `stock_predictions_20251207_231249.csv`: Complete analysis data
- `stock_predictions_20251207_231249.xlsx`: Excel-formatted results
- `stock_predictions_20251207_231249.png`: Visual ranking chart

### Configuration Options

```julia
# Training Parameters
const HISTORY_DAYS = 365    # Days of historical data
const WINDOW_SIZE = 20       # Days per training sample
const EPOCHS = 3             # Training iterations per stock
const BATCH_SIZE = 64        # Neural network batch size

# Performance Settings
const MAX_PARALLEL = 12      # Concurrent stock processing
const USE_GPU = false        # CPU mode (CUDA temporarily disabled)

# Scoring Weights
const AI_WEIGHT = 0.60       # Neural network prediction
const ANALYST_WEIGHT = 0.25  # Professional analyst ratings
const NEWS_WEIGHT = 0.15     # News sentiment/buzz
```

---

## 📊 Sample Output

### Terminal Output
```
[INFO] Processing 15 stocks with 12 parallel workers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Download Phase:
  [OK] LRN: HTTP 200, got 249 rows
  [OK] GEV: HTTP 200, got 249 rows
  [OK] PAR: HTTP 200, got 249 rows
  ...

Training Phase:
Stock LRN:
  Epoch 1/3 - Train: 0.0234, Test: 0.0198
  Epoch 2/3 - Train: 0.0189, Test: 0.0165
  Epoch 3/3 - Train: 0.0156, Test: 0.0142
  Direction Accuracy: 54.35%
  Analyst: 4 strong buy, 4 buy, 3 hold (buy_score: 1.09)
  News Buzz: 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final Rankings:
  1. LRN    : 42.519% (Model: 0.41%, Analyst: 1.09, Buzz: 1.0)
  2. GEV    : 34.999% (Model: -0.66%, Analyst: 0.82, Buzz: 1.0)
  3. PAR    : 34.568% (Model: 1.54%, Analyst: 1.29, Buzz: 0.1)
  ...

[OK] Results saved to stock_predictions_20251207_231249.csv
[OK] Results saved to stock_predictions_20251207_231249.xlsx
[OK] Chart saved to stock_predictions_20251207_231249.png

All done.
```

### CSV Format
```csv
Rank,Symbol,Combined_Score,Model_Prediction,Analyst_Score,News_Buzz
1,LRN,42.519,0.411,1.09,1.0
2,GEV,34.999,-0.663,0.82,1.0
3,PAR,34.568,1.542,1.29,0.1
```

---

## 🛠️ Technical Requirements

### Julia Environment
- **Version**: Julia 1.12+
- **Hardware**: NVIDIA RTX 4090 GPU (optional, currently in CPU mode)
- **Threads**: 12 recommended for optimal performance

### Required Packages
```julia
using HTTP          # API requests
using JSON3         # JSON parsing
using DataFrames    # Data manipulation
using Flux          # Neural networks
using Random        # Randomization
using Statistics    # Statistical functions
using Dates         # Timestamp handling
using Base.Threads  # Parallel processing
using CSV           # CSV export
using XLSX          # Excel export
using Plots         # Chart generation
```

### Installation
```powershell
julia -e 'import Pkg; Pkg.add(["HTTP", "JSON3", "DataFrames", "Flux", "CSV", "XLSX", "Plots"])'
```

---

## 🔧 Troubleshooting

### CUDA GPU Support
Currently disabled due to version mismatch (CUDA 13.1 vs 13.0). To re-enable:
```powershell
julia -e 'import Pkg; Pkg.update("CUDA"); Pkg.build("CUDA")'
```
Then uncomment `using CUDA` in `stock_picker.jl` and set `USE_GPU = true`.

### API Rate Limits
- Built-in retry logic with exponential backoff
- Random delays between requests (0.5-1s)
- If persistent issues occur, increase delay times in code

### Memory Usage
- Processing 526 stocks requires ~4-8 GB RAM
- Reduce `MAX_PARALLEL` if experiencing memory constraints
- Each stock processes ~365 days of historical data

---

## 📈 Performance Optimization Tips

1. **Increase Training Depth**
   ```julia
   const EPOCHS = 10          # More training iterations
   const HISTORY_DAYS = 730   # 2 years of data
   ```

2. **Adjust Parallelization**
   ```julia
   const MAX_PARALLEL = 16    # If you have more CPU cores
   ```

3. **Enable GPU** (10-50x faster training)
   ```julia
   const USE_GPU = true       # After fixing CUDA
   ```

---

## 📝 File Structure

```
Julia_Predicto_Test/
├── README.md                          # This file
├── build_halal_universe.jl           # Stage 1: Universe builder
├── predicto.py                        # Stage 2: Primary AI (Python)
├── stock_picker.jl                    # Stage 3: Neural validator
├── halal_universe.txt                 # 526 halal stocks (line-separated)
├── halal_universe_comma.txt          # 526 halal stocks (comma-separated)
├── tickers.txt                        # Top 15 from predicto.py
├── stock_predictions_YYYYMMDD_HHMMSS.csv   # Latest results (CSV)
├── stock_predictions_YYYYMMDD_HHMMSS.xlsx  # Latest results (Excel)
├── stock_predictions_YYYYMMDD_HHMMSS.png   # Latest chart
└── archive/                           # Historical prediction runs
```

---

## 🎓 Understanding the Results

### Combined Score Interpretation
- **> 30%**: Strong buy signal with multiple confirming factors
- **10-30%**: Moderate buy signal, consider entry points
- **0-10%**: Weak signal, monitor for better opportunities
- **< 0%**: Bearish signal, avoid or consider shorting

### Component Analysis
- **Model Prediction**: Raw AI-predicted price movement
- **Analyst Score**: Professional consensus (-2 to +2 scale)
- **News Buzz**: Market attention/sentiment (0.0 to 1.0 scale)

### Best Practices
1. Focus on stocks with positive scores across all three factors
2. High analyst scores indicate institutional confidence
3. High buzz scores suggest market momentum
4. Verify predictions against your own research

---

## 📞 API Keys & Configuration

### Zoya API (Halal Compliance)
- **Key**: `live-03e8bf0f-6bda-40b5-9d0e-ec884e8c6c9b`
- **Location**: `build_halal_universe.jl`

### Finnhub API (Analyst Ratings & News)
- **Key**: `d4kp2j1r01qvpdollej0d4kp2j1r01qvpdollejg`
- **Location**: `stock_picker.jl`

### Yahoo Finance API
- **Authentication**: None required (free public access)
- **Rate Limits**: Generous, built-in retry logic handles any issues

---

## 🌟 Future Enhancements

- [ ] GPU acceleration (re-enable CUDA support)
- [ ] Historical backtesting validation
- [ ] Stop-loss recommendations
- [ ] Price target calculations
- [ ] Portfolio optimization across top picks
- [ ] Real-time monitoring and alerts
- [ ] Integration with trading platforms

---

## ⚖️ Disclaimer

This software is provided for educational and research purposes only. Stock predictions are based on historical data and AI models, which do not guarantee future performance. Always conduct your own research and consult with qualified financial advisors before making investment decisions. The halal compliance data is sourced from Zoya and should be verified according to your own Islamic financial guidelines.

---

**Built with ❤️ using Julia, Python, and AI**
#   P r e d i c t o _ J u l i a _ A L T  
 