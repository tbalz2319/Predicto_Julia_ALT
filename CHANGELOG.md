# Changelog

## [Recent Updates] - December 2025

### 🎯 Terminal Summary Alignment
- **Fixed**: Terminal summary now uses same champion criteria as Excel Sheet 2
- **Champion Filter**: 70%+ backtest accuracy + <30% max drawdown + positive score
- **Enhanced Display**: Shows detailed breakdown of stocks meeting each criterion
- **Added Stats**: Portfolio statistics including average Sharpe ratio

### 📊 Enhanced Sentiment Analysis (VADER-like)
- **Upgraded**: From simple keyword counting to sophisticated VADER-like analysis
- **Weighted Lexicon**: Sentiment words scored 1.0-3.0 (positive) or -1.0 to -3.0 (negative)
- **Negation Handling**: Detects "not good", "don't like" and flips sentiment
- **Intensity Modifiers**: 
  - Boosters: "very", "extremely", "highly" amplify by 30%
  - Dampeners: "somewhat", "slightly" reduce by 30%
- **Emphasis Detection**:
  - ALL CAPS increases intensity by 20%
  - Exclamation marks (!!!) boost by 10% each (max 3)
- **VADER Normalization**: Uses compound score formula for nuanced results

### 📦 TextAnalysis.jl Integration
- **Added Support**: Julia's native NLP library (VADER equivalent)
- **Auto-Detection**: Code automatically uses TextAnalysis.jl if installed
- **Fallback**: Uses custom VADER-like implementation if not available
- **Optional Install**: `julia -e 'using Pkg; Pkg.add("TextAnalysis")'`

### 📁 File Organization
- **Created**: `export/` directory for all output files
- **Moved**: CSV, Excel, PNG files to export directory
- **Updated**: `.gitignore` to exclude output files
- **Ignored**: `*.csv`, `*.xlsx`, `*.png` in root directory
- **Preserved**: `archive/` and `export/` remain git-ignored

### 📖 Documentation
- **Fixed**: README.md formatting for proper GitHub rendering
- **Added**: TextAnalysis.jl installation instructions
- **Enhanced**: Sentiment analysis section with VADER features
- **Updated**: Package installation with optional TextAnalysis
- **Improved**: Project structure visualization

### 🔧 Technical Improvements
- **Startup Message**: Shows which sentiment engine is active
- **Better Error Handling**: TextAnalysis gracefully falls back to custom implementation
- **Modular Design**: Separate functions for TextAnalysis vs custom VADER

---

## Previous Features

### Phase 1 Enhancements
- 20 epochs (increased from 3)
- 730 days historical data (2 years)
- 30-day backtest validation
- Risk metrics (Sharpe ratio, max drawdown, volatility)
- Champion Formula with backtest multiplier
- Beautiful terminal output with badges
- Two-sheet Excel workbook (All Results + Champions Only)
- Sequential downloads + parallel training (prevents crashes)

### Core Features
- LSTM neural network predictions (3-layer: 20→64→32→1)
- Multi-factor analysis (AI + Analysts + Sentiment)
- Risk-adjusted scoring
- Halal compliance screening (526 stocks)
- Parallel processing (12 threads)
- Rate limiting and retry logic
- Comprehensive Excel reports with detailed explanations
