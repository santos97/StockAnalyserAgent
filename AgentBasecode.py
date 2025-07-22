
# news: https://newsapi.org/account
# https://console.groq.com/playground
# https://smith.langchain.com/

# Requirements:
# pip install groq yfinance requests pandas_ta langgraph langsmith python-dotenv matplotlib networkx


# !pip install --upgrade langgraph
# !pip install groq
# !pip install dotenv
# !pip install numpy==1.23.5 pandas_ta==0.3.14b0
# !pip install matplotlib networkx
# !pip install pandas_ta
# pip install matplotlib networkx




import os
from typing import TypedDict, Optional
from datetime import datetime
import yfinance as yf
import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langsmith import traceable
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import Image, display

def visualize_langgraph_mermaid():
    try:
        g = stock_graph.get_graph()
        display(Image(g.draw_mermaid_png()))
    except Exception as e:
        print("Diagram rendering failed:", e)
# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    from groq import Groq
    GROQ_API_KEY ="key"
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    raise ImportError("Please install groq: pip install groq")
except Exception as e:
    print(f"Groq initialization error: {e}")
    client = None

# News API Config
NEWS_API_KEY = "key"
LANGSMITH_API_KEY = "key"
# -------------------------------
# State Definition
# -------------------------------
class StockState(TypedDict):
    ticker: str
    price: Optional[float]
    history: Optional[dict]
    pe_ratio: Optional[float]
    eps: Optional[float]
    technical: Optional[dict]
    fundamentals: Optional[dict]
    sentiment: Optional[dict]
    final_decision: Optional[str]
    justification: Optional[str]
    report: Optional[str]

# -------------------------------
# Node Functions
# -------------------------------
@traceable(name="Fetch Stock Data")
def fetch_stock_data(state: StockState) -> StockState:
    ticker = state["ticker"]
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="60d")  # More data for better indicators
        info = stock.info
        return {
            "price": info.get("currentPrice"),
            "history": hist.to_dict() if not hist.empty else None,
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps")
        }
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return {
            "price": None,
            "history": None,
            "pe_ratio": None,
            "eps": None
        }

@traceable(name="Technical Analysis")
def analyze_technical(state: StockState) -> StockState:
    if not state.get("history"):
        return {"technical": None}
    try:
        close_prices = pd.Series(state["history"]["Close"])
        hist_df = pd.DataFrame(state["history"])

        ma_5 = close_prices.rolling(window=5).mean().iloc[-1]
        ma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else ma_5
        rsi = ta.rsi(close_prices, length=14).iloc[-1]
        macd_df = ta.macd(close_prices)
        macd = macd_df["MACD_12_26_9"].iloc[-1] if "MACD_12_26_9" in macd_df.columns else None
        bbands_df = ta.bbands(close_prices)
        if not bbands_df.empty:
            lower = bbands_df["BBL_20_2.0"].iloc[-1]
            middle = bbands_df["BBM_20_2.0"].iloc[-1]
            upper = bbands_df["BBU_20_2.0"].iloc[-1]
        else:
            lower = middle = upper = None

        current = close_prices.iloc[-1]

        signals = []
        if current > ma_5:
            signals.append("Price above 5-day MA")
        else:
            signals.append("Price below 5-day MA")

        if current > ma_20:
            signals.append("Price above 20-day MA")
        else:
            signals.append("Price below 20-day MA")

        if rsi is not None:
            if rsi < 30:
                signals.append(f"RSI low ({rsi:.1f}): Oversold, potential buy")
            elif rsi > 70:
                signals.append(f"RSI high ({rsi:.1f}): Overbought, potential sell")
            else:
                signals.append(f"RSI neutral ({rsi:.1f})")

        if macd is not None:
            if macd > 0:
                signals.append("MACD positive: bullish momentum")
            else:
                signals.append("MACD negative: bearish momentum")

        if lower is not None and upper is not None:
            if current < lower:
                signals.append("Price below lower Bollinger Band: oversold")
            elif current > upper:
                signals.append("Price above upper Bollinger Band: overbought")
            else:
                signals.append("Price within Bollinger Bands")

        buy_votes = sum("above" in s or "Oversold" in s or "bullish" in s for s in signals)
        sell_votes = sum("below" in s or "Overbought" in s or "bearish" in s for s in signals)

        final_signal = "Buy" if buy_votes > sell_votes else "Sell" if sell_votes > buy_votes else "Hold"

        return {
            "technical": {
                "5_day_MA": round(ma_5, 2),
                "20_day_MA": round(ma_20, 2),
                "RSI": round(rsi, 2) if rsi is not None else None,
                "MACD": round(macd, 4) if macd is not None else None,
                "BollingerBands": {
                    "lower": round(lower, 2) if lower is not None else None,
                    "middle": round(middle, 2) if middle is not None else None,
                    "upper": round(upper, 2) if upper is not None else None
                },
                "signals": signals,
                "final_signal": final_signal,
                "updated": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Enhanced Technical analysis error: {e}")
        return {"technical": None}

@traceable(name="Fundamental Analysis")
def analyze_fundamentals(state: StockState) -> StockState:
    pe = state.get("pe_ratio")
    eps = state.get("eps")
    if pe is None or eps is None:
        signal = "Hold (Missing Data)"
    else:
        signal = "Buy" if pe < 25 and eps > 2 else "Hold"
    return {
        "fundamentals": {
            "PE Ratio": pe,
            "EPS": eps,
            "signal": signal,
            "updated": datetime.now().isoformat()
        }
    }

@traceable(name="Sentiment Analysis with Groq")
def analyze_sentiment(state: StockState) -> StockState:
    if not client:
        return {
            "sentiment": {
                "score": 0,
                "signal": "Hold",
                "reasoning": "Groq client not available",
                "updated": datetime.now().isoformat()
            }
        }

    ticker = state["ticker"]

    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10).json()
        headlines = [a["title"] for a in response.get("articles", [])[:3]]
    except Exception as e:
        print(f"News API error: {e}")
        headlines = ["No recent news available"]

    try:
        prompt = f"""Analyze these stock market headlines about {ticker}:

{chr(10).join(headlines)}

Provide:
1. Sentiment (Positive/Neutral/Negative)
2. Brief reasoning (1-2 sentences)
3. Recommended action (Buy/Hold/Sell)"""

        result = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        reply = result.choices[0].message.content
        signal = "Hold"
        if "buy" in reply.lower():
            signal = "Buy"
            score = 1
        elif "sell" in reply.lower():
            signal = "Sell"
            score = -1
        else:
            signal = "Hold"
            score = 0

        return {
            "sentiment": {
                "score": score,
                "signal": signal,
                "reasoning": reply,
                "headlines": headlines,
                "updated": datetime.now().isoformat()
            }
        }

    except Exception as e:
        print(f"Groq API error: {e}")
        return {
            "sentiment": {
                "score": 0,
                "signal": "Hold",
                "reasoning": "Analysis unavailable",
                "headlines": headlines
            }
        }

@traceable(name="Decision Aggregator")
def aggregate_decision(state: StockState) -> StockState:
    signals = []
    justifications = []

    if state.get("technical"):
        tech_signal = state["technical"]["final_signal"]
        signals.append(tech_signal)
        justifications.append(f"Technical Analysis: {tech_signal} because " +
                              "; ".join(state["technical"]["signals"]))

    if state.get("fundamentals"):
        fund_signal = state["fundamentals"]["signal"]
        signals.append(fund_signal)
        justifications.append(f"Fundamental Analysis: {fund_signal} based on P/E {state['fundamentals']['PE Ratio']} and EPS {state['fundamentals']['EPS']}")

    if state.get("sentiment"):
        sent_signal = state["sentiment"]["signal"]
        signals.append(sent_signal)
        justifications.append(f"Sentiment Analysis: {sent_signal} because {state['sentiment']['reasoning']}")

    if not signals:
        final_decision = "Hold (No Data)"
    else:
        final_decision = max(set(signals), key=signals.count)

    justification_text = "\n".join(justifications)

    return {
        "final_decision": final_decision,
        "justification": justification_text
    }

@traceable(name="Generate Report")
def generate_report(state: StockState) -> StockState:
    report_lines = [
        f"Stock Report: {state['ticker'].upper()}",
        "--------------------------",
        f"Current Price: ${state.get('price')}",
        "",
        "Technical Analysis:",
    ]

    if state.get("technical"):
        tech = state["technical"]
        report_lines.extend([
            f"- 5-Day MA: {tech['5_day_MA']}",
            f"- 20-Day MA: {tech['20_day_MA']}",
            f"- RSI: {tech['RSI']}",
            f"- MACD: {tech['MACD']}",
            f"- Bollinger Bands: Lower={tech['BollingerBands']['lower']}, Middle={tech['BollingerBands']['middle']}, Upper={tech['BollingerBands']['upper']}",
            f"- Signals: " + "; ".join(tech['signals']),
            f"- Final Technical Signal: {tech['final_signal']}",
            f"- Updated: {tech.get('updated', 'N/A')}"
        ])
    else:
        report_lines.append("- No technical data available")

    report_lines.extend([
        "",
        "📊 Fundamental Analysis:"
    ])

    if state.get("fundamentals"):
        fund = state["fundamentals"]
        report_lines.extend([
            f"- P/E Ratio: {fund['PE Ratio']}",
            f"- EPS: {fund['EPS']}",
            f"- Signal: {fund['signal']}",
            f"- Updated: {fund.get('updated', 'N/A')}"
        ])
    else:
        report_lines.append("- No fundamental data available")

    report_lines.extend([
        "",
        "📰 Sentiment Analysis:"
    ])

    if state.get("sentiment"):
        sent = state["sentiment"]
        report_lines.extend([
            f"- Score: {sent['score']}",
            f"- Signal: {sent['signal']}",
            f"- Updated: {sent.get('updated', 'N/A')}",
            f"- Reasoning: {sent['reasoning']}"
        ])
    else:
        report_lines.append("- No sentiment data available")

    report_lines.extend([
        "",
        "🧠 Final Recommendation: " + state.get('final_decision', 'Unknown'),
        "",
        "📝 Justifications:",
        state.get("justification", "No justification available")
    ])

    return {"report": "\n".join(report_lines)}

# -------------------------------
# Graph Assembly
# -------------------------------
graph_builder = StateGraph(StockState)

graph_builder.add_node("fetch_data", fetch_stock_data)
graph_builder.add_node("technical", analyze_technical)
graph_builder.add_node("fundamentals", analyze_fundamentals)
graph_builder.add_node("sentiment", analyze_sentiment)
graph_builder.add_node("decision", aggregate_decision)
graph_builder.add_node("report", generate_report)

graph_builder.set_entry_point("fetch_data")
graph_builder.add_edge("fetch_data", "technical")
graph_builder.add_edge("fetch_data", "fundamentals")
graph_builder.add_edge("fetch_data", "sentiment")
graph_builder.add_edge("technical", "decision")
graph_builder.add_edge("fundamentals", "decision")
graph_builder.add_edge("sentiment", "decision")
graph_builder.add_edge("decision", "report")
graph_builder.add_edge("report", END)

stock_graph = graph_builder.compile()

# -------------------------------
# Visualization Function (Optional)
# -------------------------------
def visualize_graph():
    graph_obj = stock_graph.get_graph()
    G = nx.DiGraph()
    for node in graph_obj.nodes:
        G.add_node(node)
    for edge in graph_obj.edges:
        G.add_edge(edge[0], edge[1])
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        G,
        with_labels=True,
        node_color='skyblue',
        node_size=2000,
        font_size=10,
        arrowsize=20,
        arrowstyle='-|>'
    )
    plt.title("LangGraph Agent Flow", fontsize=14)
    plt.axis('off')
    plt.show()

# -------------------------------
# Execution Function
# -------------------------------
@traceable(name="Run Stock Analysis")
def run_stock_analysis(ticker: str):
    input_state: StockState = {
        "ticker": ticker.upper(),
        "price": None,
        "history": None,
        "pe_ratio": None,
        "eps": None,
        "technical": None,
        "fundamentals": None,
        "sentiment": None,
        "final_decision": None,
        "justification": None,
        "report": None
    }
    try:
        final_state = stock_graph.invoke(input_state)
        print(final_state["report"])
    except Exception as e:
        print(f"Analysis failed: {e}")


print("Stock Analysis Assistant")
print("-----------------------")
ticker = input("Enter stock ticker (e.g. AAPL): ").strip()
if ticker:
    run_stock_analysis(ticker)
    visualize_langgraph_mermaid()
else:
    print("No ticker provided. Running demo with AAPL...")
    run_stock_analysis("AAPL")
    visualize_langgraph_mermaid()
