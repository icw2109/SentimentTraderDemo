Trading Bot with News Sentiment & Automated Order Execution
This repository contains a Python-based trading bot that integrates multiple data sources and APIs—including Interactive Brokers (IB), Alpaca, and OpenAI's GPT-3—to analyze news headlines and automatically place orders based on predicted market movements for the S&P E-mini Futures.

Features
Real-Time News Streaming:
Connects to Alpaca’s news websocket to receive breaking market headlines.

Sentiment Analysis & Prediction:
Uses OpenAI GPT-3 to analyze headlines and produce a market prediction (up/down) along with a confidence probability.

Market Data Retrieval:
Fetches historical market data via IB API to compare recent price movements and assess direction.

Automated Order Execution:
Places market orders on IB based on the sentiment analysis, configurable by prediction probability.

Success Rate Calculation:
Tracks predictions and market movements over various intervals (1, 5, 15, and 30 minutes) to calculate the prediction success rate.

Requirements
Python 3.7+

Libraries:

ibapi, ib_insync

alpaca_trade_api

openai

nltk (for VADER sentiment analysis)

asyncio, websockets

pandas, requests

Valid API keys for:

Interactive Brokers (IB)

Alpaca (API key and secret)

OpenAI
