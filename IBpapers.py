import json
import time
import asyncio
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ib_insync import *

import openai
openai.api_key = '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'

import alpaca_trade_api as tradeapi
alpaca_api_key = 'AKSR8N1N9O2YP12J9USP'
alpaca_api_secret = '3AmrHYhL0EbTSVLbwGo1QpHXiEumlwpQvhVPo4uF'

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import websockets
import asyncio
from datetime import datetime, timedelta


# from gnews import 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distutils")



class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)


    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} {errorCode} {errorString}")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print(f"OrderStatus. Id: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}, LastFillPrice: {lastFillPrice}")

    def openOrder(self, orderId, contract, order, orderState):
        print(f"OpenOrder. ID: {orderId}, Symbol: {contract.symbol}, Type: {contract.secType}, Exchange: {contract.exchange}, Order: {order.action} {order.orderType} {order.totalQuantity}, Status: {orderState.status}")

    def execDetails(self, reqId, contract, execution):
        print(f"ExecDetails. ReqId: {reqId}, Symbol: {contract.symbol}, Execution: {execution.execId}, OrderId: {execution.orderId}, Shares: {execution.shares}, Liquidity: {execution.lastLiquidity}")


from ibapi.client import EClient
from ibapi.wrapper import EWrapper
#from ibapi.contract import 
import requests

from datetime import date, timedelta


def create_order(prediction, quantity=1):
    order = Order()
    order.action = 'BUY' if prediction == 'up' else 'SELL'
    order.totalQuantity = quantity
    order.orderType = 'MKT'
    return order


async def place_order(ib, contract, order):
    print(f"Placing order: {order.action} {contract.symbol} {order.totalQuantity}")
    trade = await ib.placeOrderAsync(contract, order)
    print(f"Order status: {trade.orderStatus.status}")




def analyze_with_gpt(headline):
    while True:
        try:
            prompt = f"The headline is: {headline}. You are a quant, make an educated guess based on the headline, give a probability betwee 0 and 100 which way the S&P E-Mini Futures "
            response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)
            print(f"Response: {response.choices[0].text}")  # print the response

            probability = [int(s) for s in response.choices[0].text.split() if s.isdigit()]

            if probability:
                return ("up" if probability[0] > 50 else "down", probability[0])
            else:
                return ("undetermined", 0)
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Waiting for 60 seconds before retrying.")
            time.sleep(5)





def calculate_success_rate(predictions, market_movements):
    if len(predictions) == 0:
        print("No predictions were made.")
        return 0  
    correct_predictions = sum(pred == move for pred, move in zip(predictions, market_movements))
    return correct_predictions / len(predictions)



async def get_market_movement(ib, contract):
    bars = await ib.run_async(
        ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
    )

    if not bars:
        print(f"No historical data returned for contract {contract.symbol}")
        return "undetermined"

    last_two_bars = bars[-2:]
    close_prices = [bar.close for bar in last_two_bars]
    return "up" if close_prices[1] > close_prices[0] else "down"


async def retrieve_and_process_news(api_key, api_secret):
    # Connect to the Alpaca news stream
    print(f"Subscribing to all news updates...")
    url = 'wss://stream.data.alpaca.markets/v1beta1/news'
    async with websockets.connect(url) as ws:
        
        auth_data = {
            "action": "auth",
            "key": api_key,
            "secret": api_secret
        }
        await ws.send(json.dumps(auth_data))

        sub_data = {
            "action": "subscribe",
            "news": ["*"]
        }
        await ws.send(json.dumps(sub_data))

        while True:
            print("Waiting for news update...")

            message = await ws.recv()
            data = json.loads(message)

            if data.get("T") == "n":
                print(f"Received news update: {data.get('headline')}")
                headline = data.get("headline")

                prediction, probability = analyze_with_gpt(headline)
                timestamp = datetime.strptime(data.get('created_at'), "%Y-%m-%dT%H:%M:%SZ")
                print(f"Prediction: {prediction}, Probability: {probability}, Market movement: {market_movement}")

                market_movement = await get_market_movement(ib, contract)

                store_prediction_and_market_movement(prediction, market_movement, timestamp)

                if probability > 80:
                    order = create_order(prediction)
                    place_order(ib, contract, order)

            elif data.get("T") == "i":
                print("Interval update received.")
                calculate_and_store_success_rates()

            print("News processing loop completed, waiting for next news update.")


predictions_1m = {}
predictions_5m = {}
predictions_15m = {}
predictions_30m = {}

market_movements_1m = {}
market_movements_5m = {}
market_movements_15m = {}
market_movements_30m = {}

success_rates_1m = {}
success_rates_5m = {}
success_rates_15m = {}
success_rates_30m = {}

def store_prediction_and_market_movement(prediction, market_movement, timestamp):
    if timestamp not in predictions_1m:
        predictions_1m[timestamp] = prediction
        market_movements_1m[timestamp] = market_movement
    if timestamp not in predictions_5m:
        predictions_5m[timestamp] = prediction
        market_movements_5m[timestamp] = market_movement
    if timestamp not in predictions_15m:
        predictions_15m[timestamp] = prediction
        market_movements_15m[timestamp] = market_movement
    if timestamp not in predictions_30m:
        predictions_30m[timestamp] = prediction
        market_movements_30m[timestamp] = market_movement


def calculate_and_store_success_rates():
    now = datetime.now()
    success_rates_1m[now] = calculate_success_rate(list(predictions_1m.values()), list(market_movements_1m.values()))
    success_rates_5m[now] = calculate_success_rate(list(predictions_5m.values()), list(market_movements_5m.values()))
    success_rates_15m[now] = calculate_success_rate(list(predictions_15m.values()), list(market_movements_15m.values()))
    success_rates_30m[now] = calculate_success_rate(list(predictions_30m.values()), list(market_movements_30m.values()))



# Use OpenAI GPT-3 model to analyze a headline



def calculate_success_rate(predictions, market_movements):
    if len(predictions) == 0:
        print("No predictions were made.")
        return 0  
    correct_predictions = sum(pred == move for pred, move in zip(predictions, market_movements))
    return correct_predictions / len(predictions)


async def store_market_movements(ib, contract):
    while True:
        market_movement = await get_market_movement(ib, contract)
        market_movements_1m[datetime.now()] = market_movement
        await asyncio.sleep(60)


async def main(api_key, api_secret, symbol, duration):
    contract = Contract()
    #...

    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=1)
    print(f"Connected to IB with client ID {ib.client.clientId}.")

    news_task = asyncio.create_task(retrieve_and_process_news(api_key, api_secret))
    market_movement_task = asyncio.create_task(store_market_movements(ib, contract))

    await asyncio.sleep(duration * 60)  
    print(f"Time elapsed: {duration} minutes. Cancelling tasks...")

    news_task.cancel()
    market_movement_task.cancel()

    print("Success rates for each interval:")
    print("1 minute:", success_rates_1m)
    #...

    print("Disconnecting from IB.")
    await ib.disconnectAsync()


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    loop.run_until_complete(main(alpaca_api_key, alpaca_api_secret, "ES", 60))
finally:
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True)) # added missing closing parenthesis
    print("Closing event loop.")
    loop.close()






############################################################################
############################################################################



"""
def trade_based_on_sentiment(ib, contract, prediction, timestamp):
    # Define the order type (MARKET, BUY/SELL, 1 contract)
    if prediction == "up":
        action = 'BUY'
    else:
        action = 'SELL'
        
    order = MarketOrder(action, 1)

    # Submit the order
    ib.placeOrder(contract, order)

    print(f'{timestamp}: Placed a {action} order based on sentiment analysis.')

    return
"""



"""

def main():
    api_key = 'AKSR8N1N9O2YP12J9USP'
    api_secret = '3AmrHYhL0EbTSVLbwGo1QpHXiEumlwpQvhVPo4uF'
    symbols = ['AAPL', 'TSLA']

    asyncio.run(retrieve_and_process_news(api_key, api_secret, symbols))

"""

"""

def retrieve_news_headlines(session):

    news_service = session.getService('//blp/news')
    request = news_service.createRequest('RetrieveHeadlinesRequest')
    request.set('subject', 'AAPL')  # Set the subject to 'AAPL' for Apple news
    request.set('startTime', '2021-01-01T00:00:00')
    request.set('endTime', '2021-12-31T23:59:59')

    session.sendRequest(request)
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                print(msg)
            break

try:
    # Initialize the session
    options = blpapi.SessionOptions()
    options.setServerHost('localhost')
    options.setServerPort(8194)
    session = blpapi.Session(options)
    session.start()

    # Create a request for historical data
    ref_data_service = session.getService('//blp/refdata')
    request = ref_data_service.createRequest('HistoricalDataRequest')
    request.getElement('securities').appendValue('AAPL US Equity')
    request.getElement('fields').appendValue('PX_LAST')
    request.set('startDate', '20210101')
    request.set('endDate', '20211231')

    # Send the request and process the response
    session.sendRequest(request)
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                print(msg)
            break

    # Retrieve news headlines
    retrieve_news_headlines(session)

finally:
    # Stop the session
    session.stop()

"""
"""

def retrieve_news_headlines(session):
    # Create a request for news headlines
    news_service = session.getService('//blp/news')
    request = news_service.createRequest('RetrieveHeadlinesRequest')
    request.set('subject', 'AAPL')  # Set the subject to 'AAPL' for Apple news
    request.set('startTime', '2021-01-01T00:00:00')
    request.set('endTime', '2021-12-31T23:59:59')
    session.sendRequest(request)
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                print(msg)
            break

try:
    options = blpapi.SessionOptions()
    options.setServerHost('localhost')
    options.setServerPort(8194)
    session = blpapi.Session(options)
    session.start()

    # Create a request for historical data
    ref_data_service = session.getService('//blp/refdata')
    request = ref_data_service.createRequest('HistoricalDataRequest')
    request.getElement('securities').appendValue('AAPL US Equity')
    request.getElement('fields').appendValue('PX_LAST')
    request.set('startDate', '20210101')
    request.set('endDate', '20211231')

    # Send the request and process the response
    session.sendRequest(request)
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                print(msg)
            break

    # Retrieve news headlines
    retrieve_news_headlines(session)

finally:
    # Stop the session
    session.stop()

"""



####################################
########### Google News Headline #####################
############################################

#def retrieve_news_headlines():
#    google_news = GNews(language='en', country='US', period='7d')
#    relevant_news = google_news.get_news('Federal Reserve OR economy OR finance OR politics')#

    # Print the headlines and their timestamps
#    for index, article in enumerate(relevant_news):
#        headline = article['title']
#        timestamp = datetime.strptime(article['published date'], "%a, %d %b %Y %H:%M:%S %Z")  
#        print(f"{index+1}. {headline}, published at {timestamp}")
#
#    return relevant_news



####################################
########### Google News Headline #####################
############################################

#def retrieve_news_headlines():
#    google_news = GNews(language='en', country='US', period='7d')
#    relevant_news = google_news.get_news('Federal Reserve OR economy OR finance OR politics')#

    # Print the headlines and their timestamps
#    for index, article in enumerate(relevant_news):
#        headline = article['title']
#        timestamp = datetime.strptime(article['published date'], "%a, %d %b %Y %H:%M:%S %Z")  
#        print(f"{index+1}. {headline}, published at {timestamp}")
#
#    return relevant_news

##############################################
#############################################3

##############################################
#############################################3

#def main():
#    retrieve_news_headlines()

##if __name__ == "__main__":
#    main()


