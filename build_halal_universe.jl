#!/usr/bin/env julia

using HTTP
using JSON3

# ==============================
# CONFIG – PUT YOUR KEYS HERE
# ==============================
const ZOYA_API_KEY    = "live-03e8bf0f-6bda-40b5-9d0e-ec884e8c6c9b"
const FINNHUB_API_KEY = "d4kp2j1r01qvpdollej0d4kp2j1r01qvpdollejg"  # Your Finnhub API key

# Minimum price to NOT be considered a penny stock
const MIN_PRICE = 5.0

# Output file (one ticker per line)
const OUTPUT_FILE = "halal_universe.txt"

# Major US exchanges to keep
const MAJOR_EXCHANGES = Set([
    "XNYS",  # NYSE
    "XNAS",  # NASDAQ
    "XASE",  # NYSE American
    "ARCX",  # NYSE Arca (lots of ETFs)
])

# ==============================
# Zoya GraphQL (Basic Data API)
# ==============================
const ZOYA_ENDPOINT = "https://api.zoya.finance/graphql"

# NOTE:
# If you get an error with basicCompliance, try basicComplianceLegacy
const ZOYA_QUERY = raw"""
query ListCompliantStocks($nextToken: String) {
  basicCompliance {
    reports(input: {
      filters: { status: COMPLIANT }
      nextToken: $nextToken
      limit: 1000
    }) {
      items {
        symbol
        name
        exchange
      }
      nextToken
    }
  }
}
"""

function fetch_zoya_page(next_token::Union{Nothing,String})
    vars = Dict{String,Any}("nextToken" => next_token)

    body = Dict(
        "query" => ZOYA_QUERY,
        "variables" => vars,
    )

    resp = HTTP.post(
        ZOYA_ENDPOINT;
        headers = [
            "Authorization" => ZOYA_API_KEY,
            "Content-Type"  => "application/json",
        ],
        body = JSON3.write(body),
    )

    resp.status == 200 || error("Zoya API error: HTTP $(resp.status) — body: $(String(resp.body))")

    data = JSON3.read(String(resp.body))
    reports = data["data"]["basicCompliance"]["reports"]

    items = reports["items"]
    next_token = reports["nextToken"]

    return items, (next_token === nothing ? nothing : String(next_token))
end

# ==============================
# Finnhub price lookup
# ==============================
const FINNHUB_QUOTE_ENDPOINT = "https://finnhub.io/api/v1/quote"

function get_price_finnhub(symbol::String; max_retries::Int = 5)
    url = "$FINNHUB_QUOTE_ENDPOINT?symbol=$(symbol)&token=$(FINNHUB_API_KEY)"
    
    for attempt in 1:max_retries
        try
            resp = HTTP.get(url)
            
            resp.status == 200 || return nothing
            
            data = JSON3.read(String(resp.body))
            
            # "c" is current price (float or 0)
            haskey(data, "c") || return nothing
            
            price_val = try
                Float64(data["c"])
            catch
                return nothing
            end
            
            price_val == 0.0 && return nothing
            
            return price_val
            
        catch e
            if e isa HTTP.Exceptions.StatusError && e.status == 429
                # Rate limited - wait and retry with exponential backoff
                wait_time = 2.0 ^ attempt
                @warn "Rate limited on $(symbol), waiting $(wait_time)s before retry $(attempt)/$(max_retries)"
                sleep(wait_time)
                continue
            else
                # Other error - skip this symbol
                @warn "Error fetching price for $(symbol): $(e)"
                return nothing
            end
        end
    end
    
    @warn "Max retries exceeded for $(symbol)"
    return nothing
end

# ==============================
# Build filtered universe
# ==============================
function build_universe(; min_price::Float64 = MIN_PRICE)
    symbols = String[]
    seen    = Set{String}()

    next_token = nothing
    processed_count = 0
    save_interval = 50  # Save progress every 50 stocks

    while true
        items, next_token = fetch_zoya_page(next_token)

        for item in items
            symbol   = String(item["symbol"])
            exchange = String(item["exchange"])

            # Only keep major US exchanges
            exchange ∈ MAJOR_EXCHANGES || continue

            # Avoid duplicates across pages
            symbol ∈ seen && continue
            push!(seen, symbol)

            # Lookup price via Finnhub with retry logic
            price = get_price_finnhub(symbol)

            if price !== nothing && price >= min_price
                push!(symbols, symbol)
                @info "Added $(symbol) (exchange=$(exchange), price=$(price))"
            else
                @info "Skipped $(symbol) (no price or price < $(min_price))"
            end

            processed_count += 1
            
            # Save progress periodically
            if processed_count % save_interval == 0
                temp_file = OUTPUT_FILE * ".tmp"
                open(temp_file, "w") do io
                    for sym in sort(symbols)
                        println(io, sym)
                    end
                end
                @info "Progress saved: $(length(symbols)) tickers after processing $(processed_count) symbols"
            end

            # Rate limit: 60 requests/min = 1 per second to be safe
            sleep(1.0)
        end

        next_token === nothing && break
    end

    sort!(symbols)
    return symbols
end

# ==============================
# Main
# ==============================
function main()
    println("Building halal US universe (major exchanges, price ≥ $(MIN_PRICE))...")
    tickers = build_universe()

    # Output 1: One ticker per line
    open(OUTPUT_FILE, "w") do io
        for sym in tickers
            println(io, sym)
        end
    end

    # Output 2: All tickers on one line, comma-separated
    comma_file = replace(OUTPUT_FILE, ".txt" => "_comma.txt")
    open(comma_file, "w") do io
        println(io, join(tickers, ","))
    end

    println("Done.")
    println("Wrote $(length(tickers)) tickers to $(OUTPUT_FILE)")
    println("Wrote comma-separated format to $(comma_file)")
end

main()
