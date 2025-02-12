import logging
import numpy as np
import ta
import pandas as pd
import requests
import discord
from discord.ext import commands
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pybit.unified_trading import HTTP
import os

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

COMMAND_PREFIX = "!"
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=discord.Intents.default())
TESTNET = False  # True ако тестваш на testnet, False за реален акаунт

session = HTTP(
    testnet=TESTNET,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

# Trade Configuration
risk_percentage = 0.05  # 5% от баланса за сделка
default_leverage = 10  # По подразбиране 10x ливъридж


def send_discord_message(message):
    """Изпраща съобщение в Discord"""
    data = {"content": message}
    response = requests.post(DISCORD_WEBHOOK_URL, json=data)
    return response.status_code


def get_balance():
    """Извлича текущия баланс от Bybit"""
    response = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    if response['retCode'] == 0:
        return float(response['result']['list'][0]['coin'][0]['walletBalance'])
    logging.error("Failed to get balance from Bybit")
    return 0


def calculate_trade_size(symbol):
    """Изчислява размера на сделката според 5% риск и минималния допустим ордер на Bybit."""

    balance = get_balance()
    if balance == 0:
        return 0

    price = float(session.get_tickers(category="linear", symbol=symbol)['result']['list'][0]['lastPrice'])
    risk_amount = balance * risk_percentage
    position_size = (risk_amount * default_leverage) / price  # Лот размер

    response = session.get_instruments_info(category="linear", symbol=symbol)

    if response['retCode'] != 0:
        logging.error(f"Error fetching instrument info: {response['retMsg']}")
        return 0

    min_order_qty = float(response['result']['list'][0]['lotSizeFilter']['minOrderQty'])

    if position_size < min_order_qty:
        logging.warning(f"Position size too small ({position_size}), adjusting to minimum allowed: {min_order_qty}")
        position_size = min_order_qty

    return round(position_size, 3)


def calculate_dynamic_tp_sl(symbol):
    """Определя Take Profit (TP) и Stop Loss (SL) на база анализ."""
    klines = session.get_kline(symbol=symbol, interval=5, limit=500)['result']['list']
    if not klines:
        return None, None

    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)

    # Индикатори
    bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["Bollinger_Upper"] = bollinger.bollinger_hband()
    df["Bollinger_Lower"] = bollinger.bollinger_lband()
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    latest_price = df["close"].iloc[-1]
    tp = df["Bollinger_Upper"].iloc[-1] + df["ATR"].iloc[-1] * 1.5
    sl = df["Bollinger_Lower"].iloc[-1] - df["ATR"].iloc[-1] * 1.5

    return round(tp, 2), round(sl, 2)


@bot.command(name="balance")
async def balance(ctx):
    balance = get_balance()
    await ctx.send(f"💰 Текущ баланс: {balance} USDT")


@bot.command(name="trade")
async def trade(ctx, symbol: str):
    position_size = calculate_trade_size(symbol)
    tp, sl = calculate_dynamic_tp_sl(symbol)

    if position_size == 0 or tp is None or sl is None:
        await ctx.send("⚠️ Неуспешен анализ, няма вход в сделка.")
        return

    response = session.place_order(
        category="linear",
        symbol=symbol,
        side="Buy",
        orderType="Market",
        qty=position_size,
        timeInForce="GTC",
        takeProfit=tp,
        stopLoss=sl,
        tpslMode="Full"
    )

    if response['retCode'] == 0:
        await ctx.send(f"✅ Сделка отворена: {position_size} {symbol} с TP: {tp} и SL: {sl}")
    else:
        await ctx.send(f"❌ Грешка при отваряне на сделка: {response['retMsg']}")


@bot.command(name="close")
async def close(ctx):
    await ctx.send("🔴 Всички сделки са затворени!")


@bot.event
async def on_ready():
    print(f"✅ Ботът {bot.user} е стартиран!")


bot.run(DISCORD_BOT_TOKEN)
