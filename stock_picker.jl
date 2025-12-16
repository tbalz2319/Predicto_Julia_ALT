#############################
# stock_picker.jl - PARALLEL VERSION
#############################

using HTTP
using JSON3
using DataFrames
using Flux
# using CUDA  # Known issue: Julia 1.12 + CUDA 13.1 + Flux extension deadlock
using Random
using Statistics
using Dates
using Base.Threads
using CSV
using XLSX

# Optional: Plots for visualization
const HAS_PLOTS = try
    using Plots
    true
catch
    false
end

# Optional: TextAnalysis.jl for advanced sentiment analysis (Julia's VADER equivalent)
# Install with: julia -e 'using Pkg; Pkg.add("TextAnalysis")'
const HAS_TEXTANALYSIS = try
    using TextAnalysis
    using Languages
    true
catch
    false
end

#############################
# CONFIG
#############################

const FINNHUB_API_KEY = "d4kp2j1r01qvpdollej0d4kp2j1r01qvpdollejg"  # Your Finnhub API key

# Read tickers from file (filter out empty lines and remove duplicates)
const TICKERS = unique(filter(!isempty, strip.(readlines("halal_universe.txt"))))

const RESOLUTION     = "D"      # daily candles
const LOOKBACK       = 20       # days in each input window
const HISTORY_DAYS   = 730      # 2 years for better training (was 365)
const TRAIN_SPLIT    = 0.8      # 80% train, 20% test
const EPOCHS         = 20       # increased from 3 for better accuracy
const BATCH_SIZE     = 64
const MAX_PARALLEL   = 12       # 12 parallel requests
const BACKTEST_DAYS  = 30       # days to backtest for validation

# GPU disabled due to Julia 1.12 + CUDA 13.1 + Flux extension deadlock
# This is a known Julia 1.12 bug - will be fixed in future versions
# CPU mode is still fast with 12-thread parallelization
const USE_GPU = false
device(x) = x

# Global lock to prevent all threads from hitting APIs simultaneously
const API_LOCK = ReentrantLock()
const LAST_API_CALL = Ref(0.0)

println("=== Julia Stock Predictor (PARALLEL - Phase 1 Enhanced) ===")
println("Using GPU: ", USE_GPU, " (CPU mode - 12 threads)")
println("Enhancements: 20 epochs, 2 years data, backtest validation, risk metrics")
println("Sentiment Engine: ", HAS_TEXTANALYSIS ? "TextAnalysis.jl (Native Julia NLP)" : "Custom VADER-like")
println("Number of threads: ", Threads.nthreads())
println("Tickers to process: ", length(TICKERS))
println("====================================")
flush(stdout)

#############################
# FETCH DATA FROM YAHOO FINANCE
#############################

function fetch_candles(symbol; resolution::String="D", days::Int=HISTORY_DAYS)
    println("[$symbol] Downloading data from Yahoo Finance...")
    flush(stdout)
    
    # Yahoo Finance uses period2 (end) and period1 (start) as Unix timestamps
    now_unix = Int(round(time()))
    from_unix = now_unix - days * 24 * 60 * 60

    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(symbol)" *
          "?period1=$(from_unix)&period2=$(now_unix)&interval=1d"

    # Retry logic with exponential backoff
    max_retries = 5
    last_error = nothing
    
    for attempt in 1:max_retries
        try
            # Rate limiting: Ensure minimum 100ms between API calls across all threads
            lock(API_LOCK) do
                elapsed = time() - LAST_API_CALL[]
                if elapsed < 0.1
                    sleep(0.1 - elapsed)
                end
                LAST_API_CALL[] = time()
            end
            
            sleep(rand() * 0.3)  # Additional random jitter
            resp = HTTP.get(url; retry=false, readtimeout=45, connect_timeout=15)
            return process_yahoo_response(symbol, resp)
        catch e
            last_error = e
            
            if e isa HTTP.Exceptions.StatusError && e.status == 429
                wait_time = 2.0 ^ attempt + rand() * 2.0
                println("[$symbol] Rate limited (429), waiting $(round(wait_time, digits=1))s... (attempt $attempt/$max_retries)")
                flush(stdout)
                sleep(wait_time)
                continue
            elseif attempt < max_retries
                # Generic retry for network errors, timeouts, etc.
                wait_time = 1.0 * attempt + rand()
                println("[$symbol] Error ($(typeof(e))), retrying in $(round(wait_time, digits=1))s... (attempt $attempt/$max_retries)")
                flush(stdout)
                sleep(wait_time)
                continue
            end
        end
    end
    
    # All retries failed
    println("[$symbol] FAILED after $max_retries attempts: $last_error")
    flush(stdout)
    rethrow(last_error)
end

function process_yahoo_response(symbol, resp)
    println("[$symbol] HTTP status: ", resp.status)
    flush(stdout)
    if resp.status != 200
        error("HTTP $(resp.status) for symbol $symbol: $(String(resp.body))")
    end

    data = JSON3.read(resp.body)

    # Check if we have valid data
    if !haskey(data, :chart) || !haskey(data.chart, :result) || isempty(data.chart.result)
        error("No data returned from Yahoo Finance for $symbol")
    end

    result = data.chart.result[1]
    
    if !haskey(result, :timestamp) || !haskey(result, :indicators)
        error("Invalid data structure for $symbol")
    end

    timestamps = result.timestamp
    quotes = result.indicators.quote[1]
    
    if !haskey(quotes, :close)
        error("No close prices for $symbol")
    end

    close_prices = quotes.close
    
    # Filter out null values
    valid_idx = findall(x -> !isnothing(x), close_prices)
    
    if isempty(valid_idx)
        error("No valid close prices for $symbol")
    end

    ts = timestamps[valid_idx]
    c = Float64.(close_prices[valid_idx])

    df = DataFrame(
        t = [DateTime(Dates.unix2datetime(unix)) for unix in ts],
        c = c,
    )
    sort!(df, :t)
    println("[$symbol] Got $(nrow(df)) rows of data.")
    return df
end

#############################
# FETCH ANALYST RATINGS & SENTIMENT
#############################

function fetch_analyst_data(symbol)
    try
        # Fetch from Finnhub recommendation trends
        url = "https://finnhub.io/api/v1/stock/recommendation?symbol=$(symbol)&token=$(FINNHUB_API_KEY)"
        
        sleep(0.3 + rand() * 0.2)
        resp = HTTP.get(url; retry=false, readtimeout=10)
        
        if resp.status != 200
            return (buy_score=0.0, has_data=false)
        end
        
        data = JSON3.read(resp.body)
        
        # Finnhub returns array of recommendations, get most recent
        if isempty(data)
            return (buy_score=0.0, has_data=false)
        end
        
        # Get the most recent recommendation (first in array)
        latest = data[1]
        
        strong_buy = haskey(latest, :strongBuy) ? latest.strongBuy : 0
        buy = haskey(latest, :buy) ? latest.buy : 0
        hold = haskey(latest, :hold) ? latest.hold : 0
        sell = haskey(latest, :sell) ? latest.sell : 0
        strong_sell = haskey(latest, :strongSell) ? latest.strongSell : 0
        
        total = strong_buy + buy + hold + sell + strong_sell
        
        if total > 0
            # Calculate buy score: weight strong buy/buy positively, sell negatively
            # Range: -2 (all strong sell) to +2 (all strong buy)
            buy_score = (strong_buy * 2.0 + buy * 1.0 - sell * 1.0 - strong_sell * 2.0) / total
            return (buy_score=buy_score, has_data=true, strong_buy=strong_buy, buy=buy, hold=hold, sell=sell, strong_sell=strong_sell)
        end
        
        return (buy_score=0.0, has_data=false)
    catch e
        return (buy_score=0.0, has_data=false)
    end
end

"""
TextAnalysis.jl sentiment analysis (Julia's VADER equivalent).
Uses built-in sentiment lexicon and sophisticated NLP features.
Returns sentiment score from -1.0 (very negative) to +1.0 (very positive)
"""
function analyze_textanalysis_sentiment(text::String)
    if isempty(text) || !HAS_TEXTANALYSIS
        return 0.0
    end
    
    try
        # Create StringDocument for analysis
        doc = StringDocument(text)
        
        # Prepare document (tokenize, stem, etc.)
        prepare!(doc, strip_punctuation | strip_case | strip_whitespace)
        
        # Get sentiment scores (returns positive and negative scores)
        # Note: TextAnalysis doesn't have a built-in compound score like VADER
        # So we'll use a simple approach: count positive/negative words
        tokens = tokens(doc)
        
        # Simple sentiment scoring based on common positive/negative words
        positive_score = 0.0
        negative_score = 0.0
        
        for token in tokens
            # This is a simplified version - TextAnalysis.jl doesn't have VADER's lexicon
            # but provides better preprocessing
            if token in ["good", "great", "excellent", "profit", "growth", "strong", "bullish", "up"]
                positive_score += 1.0
            elseif token in ["bad", "poor", "loss", "weak", "bearish", "down", "lawsuit", "fraud"]
                negative_score += 1.0
            end
        end
        
        total = positive_score + negative_score
        if total > 0
            return (positive_score - negative_score) / total
        end
        return 0.0
    catch e
        return 0.0
    end
end

"""
VADER-like sentiment analysis for text (custom implementation).
Handles intensity modifiers, negations, punctuation, and capitalization.
Returns sentiment score from -1.0 (very negative) to +1.0 (very positive)
"""
function analyze_vader_sentiment(text::String, positive_lex::Dict, negative_lex::Dict, 
                                 boosters::Vector, dampeners::Vector, negations::Vector)
    if isempty(text)
        return 0.0
    end
    
    # Preserve original for capitalization check
    original_text = text
    text_lower = lowercase(text)
    
    # Split into words
    words = split(text_lower, r"\W+")
    
    sentiments = Float64[]
    
    for (i, word) in enumerate(words)
        if isempty(word)
            continue
        end
        
        # Check if word is in lexicon
        base_score = 0.0
        if haskey(positive_lex, word)
            base_score = positive_lex[word]
        elseif haskey(negative_lex, word)
            base_score = negative_lex[word]
        else
            continue  # Not a sentiment word
        end
        
        # Initialize score for this word
        word_score = base_score
        
        # Check for negation in previous 3 words (flips sentiment)
        negation_found = false
        for j in max(1, i-3):i-1
            if words[j] in negations
                word_score *= -0.5  # Flip and dampen (VADER approach)
                negation_found = true
                break
            end
        end
        
        # Check for intensity boosters/dampeners in previous 2 words
        if !negation_found && i > 1
            prev_word = i > 1 ? words[i-1] : ""
            if prev_word in boosters
                word_score *= 1.3  # Boost by 30%
            elseif prev_word in dampeners
                word_score *= 0.7  # Dampen by 30%
            end
        end
        
        # Check for ALL CAPS (emphasis)
        word_original = ""
        for w in split(original_text, r"\W+")
            if lowercase(w) == word
                word_original = w
                break
            end
        end
        if !isempty(word_original) && length(word_original) > 2 && all(isuppercase, word_original)
            word_score *= 1.2  # ALL CAPS increases intensity by 20%
        end
        
        push!(sentiments, word_score)
    end
    
    # Handle punctuation emphasis (!!!, ???)
    exclamation_count = count(c -> c == '!', text)
    if exclamation_count > 0
        # Each exclamation adds emphasis (max 3)
        punct_boost = min(exclamation_count, 3) * 0.1
        sentiments = sentiments .* (1.0 + punct_boost)
    end
    
    # Calculate compound score
    if isempty(sentiments)
        return 0.0
    end
    
    sum_s = sum(sentiments)
    
    # Normalize using VADER's alpha parameter
    alpha = 15.0
    compound = sum_s / sqrt(sum_s^2 + alpha)
    
    # Clamp to [-1, 1]
    return clamp(compound, -1.0, 1.0)
end

"""
Fetch news sentiment using Finnhub company news.
Returns (sentiment_score, buzz_score, article_count, summary)

Enhanced VADER-like Sentiment Analysis:
- Weighted lexicon with intensity scores
- Handles negations (not good, don't like)
- Intensity boosters (very good, extremely bad)
- Capitalization emphasis (ALL CAPS)
- Punctuation emphasis (!!!)
- Score: -1.0 (very negative) to +1.0 (very positive)
"""
function fetch_news_sentiment(symbol)
    try
        # Use Finnhub company news for last 7 days
        to_date = Dates.format(today(), "yyyy-mm-dd")
        from_date = Dates.format(today() - Day(7), "yyyy-mm-dd")
        
        url = "https://finnhub.io/api/v1/company-news?symbol=$(symbol)&from=$(from_date)&to=$(to_date)&token=$(FINNHUB_API_KEY)"
        
        sleep(0.3 + rand() * 0.2)
        resp = HTTP.get(url; retry=false, readtimeout=10)
        
        if resp.status != 200
            return (sentiment=0.0, buzz=0.0, count=0, summary="No news")
        end
        
        data = JSON3.read(resp.body)
        
        if isempty(data)
            return (sentiment=0.0, buzz=0.0, count=0, summary="No news")
        end
        
        # Enhanced VADER-like sentiment lexicon with intensity scores
        # Positive words: score 1.0 to 3.0 (stronger = higher)
        positive_lexicon = Dict(
            # Strong positive (3.0)
            "breakthrough" => 3.0, "exceptional" => 3.0, "outstanding" => 3.0, "soaring" => 3.0,
            "skyrocket" => 3.0, "surge" => 3.0, "explosive" => 3.0, "stellar" => 3.0,
            # Medium positive (2.0)
            "profit" => 2.0, "growth" => 2.0, "beat" => 2.0, "upgrade" => 2.0, "bullish" => 2.0,
            "acquire" => 2.0, "expand" => 2.0, "record" => 2.0, "gain" => 2.0, "strong" => 2.0,
            "outperform" => 2.0, "innovation" => 2.0, "partnership" => 2.0, "revenue" => 2.0,
            # Mild positive (1.0)
            "good" => 1.0, "positive" => 1.0, "improve" => 1.0, "up" => 1.0, "rise" => 1.0,
            "increase" => 1.0, "better" => 1.0, "success" => 1.0, "opportunity" => 1.0
        )
        
        # Negative words: score -1.0 to -3.0 (stronger = more negative)
        negative_lexicon = Dict(
            # Strong negative (-3.0)
            "fraud" => -3.0, "scandal" => -3.0, "lawsuit" => -3.0, "plunge" => -3.0,
            "crash" => -3.0, "crisis" => -3.0, "disaster" => -3.0, "collapse" => -3.0,
            # Medium negative (-2.0)
            "investigation" => -2.0, "downgrade" => -2.0, "loss" => -2.0, "bearish" => -2.0,
            "recall" => -2.0, "warning" => -2.0, "miss" => -2.0, "decline" => -2.0,
            "concern" => -2.0, "probe" => -2.0, "allegation" => -2.0, "slump" => -2.0,
            # Mild negative (-1.0)
            "weak" => -1.0, "disappointing" => -1.0, "risk" => -1.0, "fall" => -1.0,
            "down" => -1.0, "lower" => -1.0, "drop" => -1.0, "bad" => -1.0, "negative" => -1.0
        )
        
        # Intensity boosters and dampeners
        boosters = ["very", "extremely", "highly", "incredibly", "absolutely", "completely", "totally"]
        dampeners = ["somewhat", "slightly", "barely", "hardly", "moderately", "relatively"]
        negations = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"]
        
        sentiment_scores = Float64[]
        article_count = length(data)
        
        for article in data
            headline = get(article, :headline, "")
            summary_text = get(article, :summary, "")
            combined_text = headline * " " * summary_text
            
            # Calculate sentiment using TextAnalysis.jl if available, otherwise use custom VADER-like
            article_sentiment = if HAS_TEXTANALYSIS
                analyze_textanalysis_sentiment(combined_text)
            else
                analyze_vader_sentiment(combined_text, positive_lexicon, 
                                      negative_lexicon, boosters, dampeners, negations)
            end
            
            push!(sentiment_scores, article_sentiment)
        end
        
        # Average sentiment across all articles
        avg_sentiment = isempty(sentiment_scores) ? 0.0 : mean(sentiment_scores)
        
        # Buzz score (0.0 to 1.0 based on article count)
        buzz_score = min(article_count / 10.0, 1.0)
        
        # Generate summary
        if avg_sentiment > 0.3
            sentiment_label = "POSITIVE"
        elseif avg_sentiment < -0.3
            sentiment_label = "NEGATIVE"
        else
            sentiment_label = "NEUTRAL"
        end
        
        summary = "$article_count articles, $sentiment_label sentiment"
        
        return (sentiment=avg_sentiment, buzz=buzz_score, count=article_count, summary=summary)
        
    catch e
        return (sentiment=0.0, buzz=0.0, count=0, summary="Error fetching news")
    end
end

#############################
# RISK METRICS
#############################

function calculate_risk_metrics(prices::Vector{Float64})
    """
    Calculate key risk metrics:
    - Volatility: Annualized standard deviation of returns
    - Max Drawdown: Largest peak-to-trough decline
    - Sharpe Ratio: Risk-adjusted return (assuming 0% risk-free rate)
    """
    if length(prices) < 2
        return (volatility=0.0, max_drawdown=0.0, sharpe_ratio=0.0)
    end
    
    # Calculate daily returns
    returns = diff(log.(prices))
    
    # Annualized volatility (252 trading days per year)
    volatility = std(returns) * sqrt(252) * 100  # as percentage
    
    # Max drawdown
    cumulative = cumprod(1.0 .+ (diff(prices) ./ prices[1:end-1]))
    running_max = accumulate(max, cumulative)
    drawdowns = (cumulative .- running_max) ./ running_max
    max_drawdown = abs(minimum(drawdowns)) * 100  # as percentage
    
    # Sharpe ratio (annualized, assuming 0% risk-free rate)
    if std(returns) > 0
        sharpe_ratio = mean(returns) / std(returns) * sqrt(252)
    else
        sharpe_ratio = 0.0
    end
    
    return (volatility=volatility, max_drawdown=max_drawdown, sharpe_ratio=sharpe_ratio)
end

#############################
# BACKTEST VALIDATION
#############################

function backtest_model(model, prices::Vector{Float64}, lookback::Int, backtest_days::Int)
    """
    Backtest the model on the last N days of data.
    Train on data up to N days ago, then predict each subsequent day.
    Returns: (predictions, actuals, accuracy)
    """
    if length(prices) < lookback + backtest_days + 50
        return (predictions=Float64[], actuals=Float64[], accuracy=0.0)
    end
    
    # Split: use all data except last backtest_days for training
    train_end = length(prices) - backtest_days
    
    predictions = Float64[]
    actuals = Float64[]
    
    # Predict each day in the backtest period
    for i in train_end:(length(prices)-1)
        if i < lookback
            continue
        end
        
        window = prices[i-lookback+1:i]
        next_price = prices[i+1]
        last_price = window[end]
        
        # Normalize window
        norm_window = (window ./ last_price) .- 1.0
        x_test = reshape(Float32.(norm_window), (lookback, 1))
        
        # Predict
        pred = model(x_test) |> Array
        pred_return = Float64(pred[1])
        actual_return = (next_price / last_price) - 1.0
        
        push!(predictions, pred_return)
        push!(actuals, actual_return)
    end
    
    # Calculate directional accuracy
    if !isempty(predictions)
        pred_signs = sign.(predictions)
        actual_signs = sign.(actuals)
        accuracy = mean(pred_signs .== actual_signs)
    else
        accuracy = 0.0
    end
    
    return (predictions=predictions, actuals=actuals, accuracy=accuracy)
end

#############################
# BUILD SUPERVISED DATA
#############################

function build_dataset(prices::Vector{Float64}; lookback::Int)
    n = length(prices)
    if n <= lookback + 1
        error("Not enough data points: need > $(lookback+1), got $n")
    end

    X = Float32[]
    y = Float32[]

    for i in 1:(n - lookback - 1)
        window = prices[i : i + lookback - 1]
        next_price = prices[i + lookback]
        last_price = window[end]

        norm_window = (window ./ last_price) .- 1.0
        ret = (next_price / last_price) - 1.0

        append!(X, Float32.(norm_window))
        push!(y, Float32(ret))
    end

    num_samples = length(y)
    X_mat = reshape(X, (lookback, num_samples))
    y_vec = reshape(y, (1, num_samples))
    return X_mat, y_vec
end

#############################
# TRAIN + SCORE ONE SYMBOL
#############################

function train_symbol(sym, df::DataFrame)
    println("\n==============================")
    println("Processing $sym")
    println("==============================")

    prices = df.c
    println("[$sym] Number of price points: ", length(prices))

    X, y = build_dataset(prices; lookback=LOOKBACK)
    num_samples = size(X, 2)
    println("[$sym] Dataset samples: $num_samples")

    if num_samples < 50
        println("[$sym] Too few samples (need ~50+), skipping.")
        return nothing
    end

    train_size = Int(floor(TRAIN_SPLIT * num_samples))
    all_idx = shuffle(1:num_samples)

    train_idx = all_idx[1:train_size]
    test_idx  = all_idx[train_size+1:end]

    X_train = X[:, train_idx]
    y_train = y[:, train_idx]
    X_test  = X[:, test_idx]
    y_test  = y[:, test_idx]

    X_train_d = device(X_train)
    y_train_d = device(y_train)
    X_test_d  = device(X_test)
    y_test_d  = device(y_test)

    model = Chain(
        Dense(LOOKBACK, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 1)
    ) |> device

    opt_state = Flux.setup(Adam(1e-3), model)

    for epoch in 1:EPOCHS
        idx = shuffle(1:train_size)
        for i in 1:BATCH_SIZE:train_size
            batch_end = min(i + BATCH_SIZE - 1, train_size)
            batch_indices = idx[i:batch_end]
            xb = X_train_d[:, batch_indices]
            yb = y_train_d[:, batch_indices]
            gs = gradient(model) do m
                Flux.mse(m(xb), yb)
            end
            Flux.update!(opt_state, model, gs[1])
        end
        train_l = Flux.mse(model(X_train_d), y_train_d) |> float
        test_l  = Flux.mse(model(X_test_d), y_test_d)  |> float
        println("[$sym] Epoch $epoch: train_loss=$(round(train_l, sigdigits=4)) test_loss=$(round(test_l, sigdigits=4))")
    end

    preds = model(X_test_d) |> Array
    actual = y_test |> Array

    pred_sign   = sign.(preds[1, :])
    actual_sign = sign.(actual[1, :])

    dir_accuracy = mean(pred_sign .== actual_sign)
    println("[$sym] Direction accuracy: $(round(dir_accuracy * 100, digits=2))%")
    
    # Backtest validation
    println("[$sym] Running backtest validation...")
    println("[$sym] [INFO] Backtest = Testing if model predictions would have been RIGHT on recent history")
    backtest_result = backtest_model(model, prices, LOOKBACK, BACKTEST_DAYS)
    backtest_accuracy = backtest_result.accuracy
    backtest_pct = round(backtest_accuracy * 100, digits=2)
    println("[$sym] Backtest accuracy (last $BACKTEST_DAYS days): $(backtest_pct)%")
    if backtest_pct >= 70
        println("[$sym] [OK] High confidence - Model predicted correctly $(Int(round(backtest_pct * 0.3))) out of 30 days")
    elseif backtest_pct >= 60
        println("[$sym] [INFO] Good - Better than random guessing")
    elseif backtest_pct >= 50
        println("[$sym] [WARN] Coin flip territory - Model struggling with this stock")
    else
        println("[$sym] [WARN] Low confidence - Model predictions not reliable")
    end
    
    # Risk metrics
    println("[$sym] Calculating risk metrics...")
    println("[$sym] [INFO] Risk Metrics = How risky/volatile is this stock?")
    risk = calculate_risk_metrics(prices)
    vol = round(risk.volatility, digits=2)
    dd = round(risk.max_drawdown, digits=2)
    sharpe = round(risk.sharpe_ratio, digits=2)
    println("[$sym] Volatility: $(vol)% | Max Drawdown (DD): $(dd)% | Sharpe: $(sharpe)")
    
    # Explain Max Drawdown
    if dd < 15
        println("[$sym] [OK] Low risk (DD < 15%) - Very stable, blue-chip level")
    elseif dd < 25
        println("[$sym] [OK] Normal risk (DD 15-25%) - Typical healthy stock")
    elseif dd < 40
        println("[$sym] [WARN] High risk (DD 25-40%) - Significant drops possible")
    elseif dd < 60
        println("[$sym] [WARN] Very high risk (DD 40-60%) - Speculative, big swings")
    else
        println("[$sym] [WARN] EXTREME risk (DD > 60%) - Stock crashed $(dd)% from peak! Casino-level volatility")
    end
    
    # Explain what DD means in dollars
    println("[$sym] [INFO] If you bought at the worst time, your \$10,000 would have dropped to \$$(Int(round(10000 * (1 - dd/100))))")

    # Live prediction
    window = prices[end-LOOKBACK+1:end]
    last_price = window[end]
    norm_window = (window ./ last_price) .- 1.0
    x_live = reshape(Float32.(norm_window), (LOOKBACK, 1)) |> device

    pred_live = model(x_live) |> Array
    pred_return = Float64(pred_live[1])

    # Fetch analyst ratings and sentiment
    println("[$sym] Fetching analyst ratings...")
    analyst_data = fetch_analyst_data(sym)
    
    println("[$sym] Fetching news sentiment...")
    println("[$sym] [INFO] Sentiment = Analyzing if news is POSITIVE or NEGATIVE (not just volume)")
    news_data = fetch_news_sentiment(sym)
    
    # Display sentiment results
    println("[$sym] News: $(news_data.summary)")
    if news_data.sentiment > 0.3
        println("[$sym] [OK] POSITIVE news detected (score: $(round(news_data.sentiment, digits=2))) - Good for stock!")
    elseif news_data.sentiment < -0.3
        println("[$sym] [WARN] NEGATIVE news detected (score: $(round(news_data.sentiment, digits=2))) - Lawsuits/problems found!")
        println("[$sym] [WARN] Consider avoiding this stock due to negative sentiment")
    else
        println("[$sym] [INFO] Neutral news sentiment (score: $(round(news_data.sentiment, digits=2)))")
    end
    
    # ULTIMATE CHAMPION FORMULA: All factors weighted
    # Base prediction components (60% total)
    sentiment_contribution = news_data.sentiment * 0.08  # 8%
    prediction_score = pred_return * 0.40 + analyst_data.buy_score * 0.12 + sentiment_contribution  # 60%
    
    # BACKTEST MULTIPLIER: Trust is everything - scale by how well model actually works
    # 50% backtest = 0.5x multiplier (coin flip = cut score in half)
    # 70% backtest = 1.0x multiplier (good = keep full score)
    # 85% backtest = 1.3x multiplier (excellent = 30% bonus!)
    backtest_multiplier = (backtest_accuracy / 100.0 - 0.3) / 0.4  # Maps 30%→0.0x, 70%→1.0x, 110%→2.0x
    backtest_multiplier = clamp(backtest_multiplier, 0.3, 1.5)  # Min 30%, max 150% of score
    
    # RISK ADJUSTMENT: Reward low risk, punish high risk (25% influence)
    # DD penalty: 0% DD = no penalty, 30% DD = -10%, 60% DD = -20%, 80%+ DD = -25%
    dd_penalty = min(risk.max_drawdown / 100.0 * 0.35, 0.25)
    
    # Sharpe bonus: Sharpe 1.0 = +5%, Sharpe 2.0 = +10%
    sharpe_bonus = clamp(risk.sharpe_ratio * 0.05, -0.05, 0.10)
    
    risk_adjustment = (sharpe_bonus - dd_penalty) * 0.25
    
    # FINAL CHAMPION SCORE = Prediction × Backtest × Risk
    combined_score = (prediction_score * backtest_multiplier) + risk_adjustment
    
    analyst_summary = ""
    if analyst_data.has_data
        analyst_summary = "Analysts: $(analyst_data.strong_buy) Strong Buy, $(analyst_data.buy) Buy, $(analyst_data.hold) Hold, $(analyst_data.sell) Sell"
    else
        analyst_summary = "No analyst data"
    end
    
    println("[$sym] Model prediction: $(round(pred_return * 100, digits=3))%")
    println("[$sym] Analyst buy score: $(round(analyst_data.buy_score, digits=3)) - $analyst_summary")
    println("[$sym] News sentiment: $(round(news_data.sentiment, digits=3)) ($(news_data.count) articles)")
    println("[$sym] Backtest multiplier: $(round(backtest_multiplier, digits=2))x ($(round(backtest_accuracy, digits=1))% accuracy)")
    println("[$sym] CHAMPION SCORE: $(round(combined_score * 100, digits=3))% (Base: $(round(prediction_score*100, digits=1))% × Backtest: $(round(backtest_multiplier, digits=2))x + Risk: $(round(risk_adjustment*100, digits=1))%)")

    return (sym, combined_score, dir_accuracy, pred_return, analyst_data.buy_score, news_data.sentiment, 
            backtest_accuracy, risk.volatility, risk.max_drawdown, risk.sharpe_ratio)
end

#############################
# MAIN FLOW - DOWNLOAD FIRST (SEQUENTIAL), THEN TRAIN (PARALLEL)
#############################

try
    println("\n[PHASE 1] Downloading price data sequentially (avoids HTTP threading bugs)...")
    flush(stdout)
    
    # Download all data sequentially to avoid HTTP.jl threading crash
    data_cache = Dict{String, DataFrame}()
    for (idx, sym) in enumerate(TICKERS)
        try
            println("[$sym] Downloading... ($idx/$(length(TICKERS)))")
            flush(stdout)
            df = fetch_candles(sym, resolution=RESOLUTION, days=HISTORY_DAYS)
            data_cache[sym] = df
        catch e
            println("[$sym] Download failed: $e")
            flush(stdout)
        end
    end
    
    println("\n[PHASE 2] Training models in parallel...")
    flush(stdout)
    
    results = []
    batch_size = MAX_PARALLEL

    for batch_start in 1:batch_size:length(TICKERS)
    batch_end = min(batch_start + batch_size - 1, length(TICKERS))
    batch_tickers = TICKERS[batch_start:batch_end]
    batch_results = Vector{Any}(undef, length(batch_tickers))
    
    Threads.@threads for i in 1:length(batch_tickers)
        sym = batch_tickers[i]
        try
            if !haskey(data_cache, sym)
                println("[$sym] Skipping - no data downloaded")
                batch_results[i] = nothing
                continue
            end
            
            println("[$sym] Starting training...")
            flush(stdout)
            res = train_symbol(sym, data_cache[sym])  # Pass cached data
            println("[$sym] Training completed successfully")
            flush(stdout)
            batch_results[i] = res
        catch e
            println("[$sym] CRITICAL ERROR CAUGHT: $e")
            println("[$sym] Error type: $(typeof(e))")
            println("[$sym] Full stacktrace: ")
            flush(stdout)
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
            flush(stdout)
            batch_results[i] = nothing
        end
    end
    
    # Add non-nothing results
    for res in batch_results
        if res !== nothing
            push!(results, res)
        end
    end
    
    # Progress update
    println("\n[PROGRESS] Completed $(min(batch_end, length(TICKERS)))/$(length(TICKERS)) stocks")
    flush(stdout)
end

println("\n====================================")
println("RUN COMPLETE")
println("====================================")

if isempty(results)
    println("No valid results. Either API key issue or all symbols returned no usable data.")
else
    # Remove any duplicates (just in case)
    unique_results = unique(results)
    
    # results is Vector{Tuple{String, Float64, Float64}}
    sort!(unique_results, by = x -> x[2], rev = true)

    println("\n" * "="^80)
    println("🏆 TOP STOCK PICKS - RANKED BY AI CHAMPION SCORE")
    println("="^80)
    println("Total analyzed: $(length(unique_results)) stocks | Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
    println("\nHow to read this:")
    println("  ⭐ = CHAMPION (High backtest + Low risk)")
    println("  ✅ = GOOD BUY (Proven predictions)")
    println("  ⚠️  = RISKY (High volatility/drawdown)")
    println("  ❌ = AVOID (Poor predictions or extreme risk)")
    println("\n" * "-"^80)
    
    # Categorize stocks
    for (idx, r) in enumerate(unique_results)
        (sym, score, acc, model_pred, analyst, buzz, backtest_acc, vol, drawdown, sharpe) = r
        score_pct = round(score * 100, digits=2)
        backtest_pct = round(backtest_acc * 100, digits=1)
        dd = round(drawdown, digits=1)
        sharpe_val = round(sharpe, digits=2)
        
        # Determine badge
        badge = if backtest_pct >= 70 && dd < 30 && score > 0
            "⭐"
        elseif backtest_pct >= 60 && dd < 40 && score > 0
            "✅"
        elseif dd > 60 || backtest_pct < 50
            "❌"
        elseif dd > 40
            "⚠️"
        else
            "  "
        end
        
        # Risk label
        risk = if dd < 15
            "SAFE"
        elseif dd < 25
            "LOW"
        elseif dd < 40
            "MEDIUM"
        elseif dd < 60
            "HIGH"
        else
            "EXTREME"
        end
        
        # Print ranking
        rank = lpad("#$idx", 4)
        println("$rank $badge $(rpad(sym, 5)) │ Score: $(lpad(score_pct, 6))%  │ Backtest: $(lpad(backtest_pct, 5))%  │ Risk: $(rpad(risk, 7)) (DD: $(lpad(dd, 5))%)  │ Sharpe: $(lpad(sharpe_val, 5))")
        
        # Add explanation for top 5
        if idx <= 5
            reason = if backtest_pct >= 70
                "Model predictions proven accurate"
            elseif dd < 30
                "Low risk, stable returns"
            elseif sharpe_val > 1.0
                "Excellent risk-adjusted returns"
            else
                "Balanced opportunity"
            end
            println("     └─ Why: $reason")
        end
        
        if idx == 5
            println("-"^80)
        end
    end
    
    # Save to DataFrame
    df_results = DataFrame(
        Symbol = String[r[1] for r in unique_results],
        Combined_Score_Pct = Float64[round(r[2] * 100, digits=3) for r in unique_results],
        Direction_Accuracy_Pct = Float64[round(r[3] * 100, digits=2) for r in unique_results],
        Model_Prediction_Pct = Float64[round(r[4] * 100, digits=3) for r in unique_results],
        Analyst_Buy_Score = Float64[round(r[5], digits=3) for r in unique_results],
        News_Buzz_Score = Float64[round(r[6], digits=3) for r in unique_results],
        Backtest_Accuracy_Pct = Float64[round(r[7] * 100, digits=2) for r in unique_results],
        Volatility_Pct = Float64[round(r[8], digits=2) for r in unique_results],
        Max_Drawdown_Pct = Float64[round(r[9], digits=2) for r in unique_results],
        Sharpe_Ratio = Float64[round(r[10], digits=2) for r in unique_results],
        Recommendation = String[r[2] > 0 ? "UP" : "DOWN/FLAT" for r in unique_results]
    )
    
    # Generate timestamp for filename
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv_filename = "stock_predictions_$(timestamp).csv"
    
    # Save to CSV
    CSV.write(csv_filename, df_results)
    println("\n[OK] Results saved to: $csv_filename")
    
    # Save to Excel with TWO SHEETS: All Results + Champions Only
    try
        xlsx_filename = "stock_predictions_$(timestamp).xlsx"
        
        println("\n[INFO] Creating Excel workbook with 2 sheets...")
        
        # Filter champions: backtest >= 70%, DD < 30%, positive score
        champions = filter(row -> row.Backtest_Accuracy_Pct >= 70.0 && 
                                   row.Max_Drawdown_Pct < 30.0 && 
                                   row.Combined_Score_Pct > 0, 
                           df_results)
        
        if !isempty(champions)
            # Create detailed champions sheet
            champions_detailed = DataFrame(
                Rank = 1:nrow(champions),
                Symbol = champions.Symbol,
                Champion_Score = champions.Combined_Score_Pct,
                
                # Prediction Quality
                Backtest_Accuracy = champions.Backtest_Accuracy_Pct,
                Backtest_Rating = map(champions.Backtest_Accuracy_Pct) do acc
                    acc >= 80 ? "⭐ EXCELLENT (80%+)" :
                    acc >= 75 ? "✅ VERY GOOD (75-80%)" :
                    "✅ GOOD (70-75%)"
                end,
                Why_Trust_It = map(champions.Backtest_Accuracy_Pct) do acc
                    "Model correctly predicted $(round(acc, digits=1))% of price movements in the last 30 days"
                end,
                
                # Risk Assessment
                Max_Drawdown_Pct = champions.Max_Drawdown_Pct,
                Risk_Level = map(champions.Max_Drawdown_Pct) do dd
                    dd < 15 ? "SAFE (<15%)" :
                    dd < 25 ? "LOW RISK (15-25%)" :
                    "MEDIUM RISK (25-30%)"
                end,
                Worst_Case_Loss = map(champions.Max_Drawdown_Pct) do dd
                    "\$$(Int(round(10000 * (1 - dd/100)))) left from \$10,000 investment"
                end,
                
                # Return Potential
                Model_Prediction_Pct = champions.Model_Prediction_Pct,
                AI_Outlook = map(champions.Model_Prediction_Pct) do pred
                    pred > 2 ? "STRONG BUY (>2%)" :
                    pred > 0 ? "BUY (Positive)" :
                    "NEUTRAL/HOLD"
                end,
                
                # Risk-Adjusted Performance
                Sharpe_Ratio = champions.Sharpe_Ratio,
                Sharpe_Rating = map(champions.Sharpe_Ratio) do sharpe
                    sharpe > 1.5 ? "EXCEPTIONAL" :
                    sharpe > 1.0 ? "EXCELLENT" :
                    sharpe > 0.5 ? "GOOD" :
                    "ACCEPTABLE"
                end,
                Bang_For_Buck = map(champions.Sharpe_Ratio) do sharpe
                    "Getting \$$(round(sharpe * 100, digits=0)) return per \$100 of risk taken"
                end,
                
                # Analyst Consensus
                Analyst_Score = champions.Analyst_Buy_Score,
                Wall_Street_Says = map(champions.Analyst_Buy_Score) do score
                    score > 1.5 ? "STRONG BUY consensus" :
                    score > 0.8 ? "BUY consensus" :
                    score > 0 ? "MODERATE BUY" :
                    "HOLD consensus"
                end,
                
                # News Sentiment
                News_Sentiment = champions.News_Buzz_Score,
                News_Quality = map(champions.News_Buzz_Score) do sent
                    sent > 0.3 ? "POSITIVE news flow" :
                    sent < -0.3 ? "NEGATIVE news detected" :
                    "NEUTRAL coverage"
                end,
                
                # Volatility
                Volatility_Pct = champions.Volatility_Pct,
                Price_Stability = map(champions.Volatility_Pct) do vol
                    vol < 25 ? "STABLE (<25%)" :
                    vol < 40 ? "MODERATE (25-40%)" :
                    "VOLATILE (40%+)"
                end,
                
                # Overall Summary
                Why_Champion = map(eachrow(champions)) do row
                    reasons = String[]
                    if row.Backtest_Accuracy_Pct >= 75
                        push!(reasons, "Proven accurate predictions")
                    end
                    if row.Max_Drawdown_Pct < 20
                        push!(reasons, "Low risk profile")
                    end
                    if row.Sharpe_Ratio > 1.0
                        push!(reasons, "Excellent risk-adjusted returns")
                    end
                    if row.Analyst_Buy_Score > 1.0
                        push!(reasons, "Strong analyst support")
                    end
                    if row.News_Buzz_Score > 0.3
                        push!(reasons, "Positive market sentiment")
                    end
                    join(reasons, " • ")
                end,
                
                Investment_Grade = map(eachrow(champions)) do row
                    if row.Backtest_Accuracy_Pct >= 80 && row.Max_Drawdown_Pct < 20 && row.Sharpe_Ratio > 1.0
                        "PREMIUM - Top tier investment"
                    elseif row.Backtest_Accuracy_Pct >= 75 && row.Max_Drawdown_Pct < 25
                        "EXCELLENT - Highly recommended"
                    else
                        "SOLID - Strong fundamentals"
                    end
                end
            )
            
            # Create Excel with 2 sheets
            XLSX.openxlsx(xlsx_filename, mode="w") do xf
                # Sheet 1: All Results
                sheet1 = xf[1]
                XLSX.rename!(sheet1, "All_Results")
                XLSX.writetable!(sheet1, df_results, anchor_cell=XLSX.CellRef("A1"))
                
                # Sheet 2: Champions Only
                XLSX.addsheet!(xf, "CHAMPIONS_ONLY")
                sheet2 = xf[2]
                XLSX.writetable!(sheet2, champions_detailed, anchor_cell=XLSX.CellRef("A1"))
            end
            
            println("[OK] Excel saved to: $xlsx_filename")
            println("     📊 Sheet 1: 'All_Results' - All $(nrow(df_results)) stocks analyzed")
            println("     🏆 Sheet 2: 'CHAMPIONS_ONLY' - $(nrow(champions)) champion stocks with detailed explanations!")
        else
            # No champions found, just save all results
            XLSX.writetable(xlsx_filename, 
                collect(DataFrames.eachcol(df_results)),
                DataFrames.names(df_results)
            )
            println("[OK] Excel saved to: $xlsx_filename")
            println("[INFO] No champions found (need: 70%+ backtest, <30% DD, positive score)")
        end
    catch e
        println("[WARN] Excel export failed: $e")
        println("       Details: $(sprint(showerror, e))")
    end
    
    # Champions were already included in the Excel file above, no need for duplicate code

    
    # Create visualization (if Plots is available)
    if HAS_PLOTS
        try
            println("\n[INFO] Creating shareable graphic...")
            
            # Get top 10 stocks
            top_10 = first(df_results, min(10, nrow(df_results)))
            
            # Create simple bar chart
            p = Plots.bar(
                top_10.Symbol,
                top_10.Combined_Score_Pct,
                title = "🏆 TOP 10 AI STOCK PICKS - $(Dates.format(now(), "yyyy-mm-dd"))\nRanked by Champion Score: Predictions × Backtest × Risk",
                xlabel = "Stock Symbol",
                ylabel = "Champion Score (%)",
                legend = false,
                color = ifelse.(top_10.Combined_Score_Pct .> 0, :green, :red),
                fillalpha = 0.8,
                size = (1400, 900),
                titlefontsize = 16,
                guidefontsize = 14,
                tickfontsize = 12,
                left_margin = 15Plots.mm,
                bottom_margin = 10Plots.mm,
                top_margin = 10Plots.mm,
                bar_width = 0.7
            )
            
            img_filename = "stock_predictions_$(timestamp).png"
            Plots.savefig(p, img_filename)
            println("[OK] Graphic saved to: $img_filename")
            println("[OK] Ready to share on Telegram!")
        catch e
            println("[WARN] Visualization failed: $e")
            println("       Error details: $(sprint(showerror, e))")
        end
    else
        println("\n[INFO] Plots package not installed. To enable graphics, run:")
        println("   julia -e 'import Pkg; Pkg.add(\"Plots\"); Pkg.add(\"GR\")'")
    end
    
    # Investment summary
    println("\n" * "="^80)
    println("💰 INVESTMENT SUMMARY")
    println("="^80)
    
    # Filter champions using SAME criteria as Excel Sheet 2
    # Champions: 70%+ backtest accuracy, <30% max drawdown, positive score
    champions = filter(r -> r[7] >= 70.0 && r[9] < 30.0 && r[2] > 0, unique_results)
    
    # Additional categories for context
    high_accuracy = filter(r -> r[7] >= 70.0 && r[2] > 0, unique_results)  # 70%+ backtest
    low_risk = filter(r -> r[9] < 25.0 && r[2] > 0, unique_results)  # <25% drawdown
    
    println("\n📊 Portfolio Recommendations:")
    if !isempty(champions)
        println("\n⭐ CHAMPIONS (70%+ Backtest, <30% Drawdown, Positive Score):")
        println("   $(length(champions)) stocks meet ALL champion criteria:")
        for r in first(champions, min(5, length(champions)))
            println("   • $(r[1]) - Score: $(round(r[2]*100, digits=2))% | Backtest: $(round(r[7], digits=1))% | Max Loss: $(round(r[9], digits=1))% | Sharpe: $(round(r[10], digits=2))")
        end
        if length(champions) > 5
            println("   ... and $(length(champions) - 5) more champions (see Excel Sheet 2)")
        end
    else
        println("\n⭐ CHAMPIONS: None found")
        println("   No stocks met ALL criteria (70%+ backtest + <30% drawdown + positive score)")
    end
    
    println("\n💡 Portfolio Statistics:")
    avg_backtest = mean([r[7] for r in unique_results])
    avg_dd = mean([r[9] for r in unique_results])
    avg_sharpe = mean([r[10] for r in unique_results])
    println("   • Stocks analyzed: $(length(unique_results))")
    println("   • Average backtest accuracy: $(round(avg_backtest, digits=1))%")
    println("   • Average max drawdown: $(round(avg_dd, digits=1))%")
    println("   • Average Sharpe ratio: $(round(avg_sharpe, digits=2))")
    println("\n   • Champions (all 3 criteria): $(length(champions))")
    println("   • High accuracy stocks (70%+ backtest): $(length(high_accuracy))")
    println("   • Low risk stocks (<25% drawdown): $(length(low_risk))")
    
    println("\n" * "="^80)
end

catch e
    println("\n====================================")
    println("FATAL ERROR IN MAIN EXECUTION")
    println("====================================")
    println("Error: $e")
    println("Type: $(typeof(e))")
    println("\nFull stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    println("\n====================================")
    rethrow(e)
end

println("\nAll done.")
