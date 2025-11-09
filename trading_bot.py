#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 BOT DE TRADING CRYPTO - ULTIMATE PRO SYSTEM
100% Mathématiques • Zéro Simulation • Notifications Telegram
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import os
from collections import deque

# ================================
# CONFIGURATION
# ================================

# TON TOKEN TELEGRAM BOT
TELEGRAM_BOT_TOKEN = "8317338475:AAH4_etAd7VsUWHk_4M-HNWZtWp69pAEEpo"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# TON CHAT ID (sera récupéré automatiquement)
CHAT_ID = "1719073848"

# PAIRES À SURVEILLER
PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
    'ATOMUSDT', 'LTCUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
    'FILUSDT', 'HBARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT',
    'SUIUSDT', 'SEIUSDT', 'TAOUSDT', 'RENDERUSDT', 'TIAUSDT', 'PEPEUSDT'
]

CAPITAL_INITIAL = 300
SCAN_INTERVAL = 30  # secondes

# ================================
# ÉTAT GLOBAL
# ================================

class TradingState:
    def __init__(self):
        self.capital = CAPITAL_INITIAL
        self.trades = 0
        self.profit = 0
        self.wins = 0
        self.history = {}  # Historique des prix par paire
        self.bot_active = False
        self.user_chat_id = CHAT_ID

    def update_history(self, symbol: str, data: dict):
        """Met à jour l'historique des prix pour une paire"""
        if symbol not in self.history:
            self.history[symbol] = deque(maxlen=30)

        self.history[symbol].append({
            'price': data['price'],
            'volume': data['volume'],
            'high': data.get('high', data['price']),
            'low': data.get('low', data['price']),
            'timestamp': time.time()
        })

state = TradingState()

# ================================
# FONCTIONS TELEGRAM
# ================================

def send_telegram_message(text: str):
    """Envoie un message sur Telegram"""
    try:
        url = f"{TELEGRAM_API_URL}/sendMessage"
        data = {
            "chat_id": state.user_chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return None

def get_telegram_updates(offset: Optional[int] = None):
    """Récupère les mises à jour Telegram"""
    try:
        url = f"{TELEGRAM_API_URL}/getUpdates"
        params = {"timeout": 30}
        if offset:
            params["offset"] = offset

        response = requests.get(url, params=params, timeout=35)
        return response.json()
    except Exception as e:
        print(f"Erreur getUpdates: {e}")
        return {"ok": False, "result": []}

def process_telegram_commands():
    """Traite les commandes Telegram"""
    offset = None

    while True:
        try:
            updates = get_telegram_updates(offset)

            if not updates.get("ok"):
                time.sleep(5)
                continue

            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                if "message" not in update:
                    continue

                message = update["message"]
                chat_id = message["chat"]["id"]
                text = message.get("text", "")

                # Commande /start
                if text.strip() == "/start":
                    state.user_chat_id = chat_id
                    state.bot_active = True

                    welcome_msg = (
                        "👑 <b>ULTIMATE PRO SYSTEM ACTIVÉ</b>\n\n"
                        "✅ Bot de trading démarré\n"
                        "💰 Capital: {:.0f}€\n"
                        "📊 Scan: Toutes les {}s\n\n"
                        "Je t\'enverrai des notifications quand:\n"
                        "• 🎯 Signal Elite détecté\n"
                        "• 💸 Trade exécuté\n"
                        "• 📈 Profit réalisé\n\n"
                        "Le bot scanne {} cryptos en temps réel."
                    ).format(state.capital, SCAN_INTERVAL, len(PAIRS))

                    send_telegram_message(welcome_msg)
                    print(f"Bot activé pour {chat_id}")

                # Commande /stop
                elif text.strip() == "/stop":
                    state.bot_active = False
                    send_telegram_message("⏸ Bot mis en pause")

                # Commande /status
                elif text.strip() == "/status":
                    status_msg = (
                        "📊 <b>STATUS TRADING</b>\n\n"
                        "💰 Capital: {:.2f}€\n"
                        "📈 Trades: {}\n"
                        "✅ Win Rate: {:.1f}%\n"
                        "💸 Profit: +{:.2f}€\n"
                        "🤖 Bot: {}"
                    ).format(
                        state.capital,
                        state.trades,
                        (state.wins / state.trades * 100) if state.trades > 0 else 0,
                        state.profit,
                        "🟢 ACTIF" if state.bot_active else "🔴 INACTIF"
                    )
                    send_telegram_message(status_msg)

        except Exception as e:
            print(f"Erreur process_commands: {e}")
            time.sleep(5)

# ================================
# API EXCHANGES
# ================================

async def fetch_binance_data():
    """Récupère les données Binance"""
    try:
        loop = asyncio.get_event_loop()

        # Prix
        price_resp = await loop.run_in_executor(
            None,
            lambda: requests.get('https://api.binance.com/api/v3/ticker/price', timeout=10)
        )
        prices = price_resp.json()

        # Tickers 24h
        ticker_resp = await loop.run_in_executor(
            None,
            lambda: requests.get('https://api.binance.com/api/v3/ticker/24hr', timeout=10)
        )
        tickers = ticker_resp.json()

        data = {}
        for sym in PAIRS:
            p = next((x for x in prices if x['symbol'] == sym), None)
            t = next((x for x in tickers if x['symbol'] == sym), None)

            if p and t:
                data[sym] = {
                    'price': float(p['price']),
                    'volume': float(t['volume']),
                    'priceChange': float(t['priceChangePercent']),
                    'high': float(t['highPrice']),
                    'low': float(t['lowPrice']),
                    'quoteVolume': float(t['quoteVolume'])
                }

        return data

    except Exception as e:
        print(f"Erreur Binance: {e}")
        return {}

# ================================
# INDICATEURS TECHNIQUES
# ================================

def calc_rsi(symbol: str, period: int = 14) -> Optional[float]:
    """Calcule le RSI"""
    if symbol not in state.history or len(state.history[symbol]) < period + 1:
        return None

    hist = list(state.history[symbol])
    gains, losses = 0, 0

    for i in range(len(hist) - period, len(hist)):
        change = hist[i]['price'] - hist[i-1]['price']
        if change > 0:
            gains += change
        else:
            losses += abs(change)

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(symbol: str, period: int = 14) -> Optional[float]:
    """Calcule l'ATR (Average True Range)"""
    if symbol not in state.history or len(state.history[symbol]) < period + 1:
        return None

    hist = list(state.history[symbol])
    tr_sum = 0

    for i in range(len(hist) - period, len(hist)):
        tr = max(
            hist[i]['high'] - hist[i]['low'],
            abs(hist[i]['high'] - hist[i-1]['price']),
            abs(hist[i]['low'] - hist[i-1]['price'])
        )
        tr_sum += tr

    return tr_sum / period

def calc_adx(symbol: str, period: int = 14) -> Optional[float]:
    """Calcule l'ADX"""
    if symbol not in state.history or len(state.history[symbol]) < period + 1:
        return None

    hist = list(state.history[symbol])
    dx_sum = 0

    for i in range(len(hist) - period, len(hist)):
        high = hist[i]['high']
        low = hist[i]['low']
        prev_close = hist[i-1]['price']

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        up_move = high - hist[i-1]['high']
        down_move = hist[i-1]['low'] - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        if tr > 0:
            plus_di = (plus_dm / tr) * 100
            minus_di = (minus_dm / tr) * 100
            di_sum = plus_di + minus_di

            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
                dx_sum += dx

    return dx_sum / period

def calc_momentum(symbol: str, period: int = 10) -> Optional[float]:
    """Calcule le momentum"""
    if symbol not in state.history or len(state.history[symbol]) < period:
        return None

    hist = list(state.history[symbol])
    current_price = hist[-1]['price']
    past_price = hist[-period]['price']

    return ((current_price - past_price) / past_price) * 100

def calc_volume_trend(symbol: str, current_vol: float) -> Optional[float]:
    """Calcule la tendance du volume"""
    if symbol not in state.history or len(state.history[symbol]) < 10:
        return None

    hist = list(state.history[symbol])
    avg_vol = sum(h['volume'] for h in hist) / len(hist)

    return (current_vol / avg_vol) * 100

# ================================
# DÉTECTION DE RÉGIME
# ================================

def detect_market_regime() -> str:
    """Détecte le régime de marché"""
    if 'BTCUSDT' not in state.history or len(state.history['BTCUSDT']) < 15:
        return 'neutral'

    adx = calc_adx('BTCUSDT')
    atr = calc_atr('BTCUSDT')
    momentum = calc_momentum('BTCUSDT')

    if not all([adx, atr, momentum]):
        return 'neutral'

    btc_price = list(state.history['BTCUSDT'])[-1]['price']
    volatility_pct = (atr / btc_price) * 100

    if adx > 25 and momentum > 2:
        return 'strong_bull'
    elif adx > 25 and momentum < -2:
        return 'strong_bear'
    elif volatility_pct > 3:
        return 'high_volatility'
    elif adx < 20 and abs(momentum) < 1:
        return 'ranging'

    return 'neutral'

# ================================
# ANTI-SPOOFING
# ================================

def check_spoofing(data: dict) -> dict:
    """Vérifie si le volume est réel"""
    volume_ratio = data['quoteVolume'] / (data['price'] * data['volume'])

    if volume_ratio < 0.8:
        return {'safe': False, 'reason': 'Volume suspect'}

    if data['quoteVolume'] < 10_000_000:
        return {'safe': False, 'reason': 'Liquidité faible'}

    volatility = ((data['high'] - data['low']) / data['price']) * 100
    if volatility > 10:
        return {'safe': False, 'reason': 'Volatilité extrême'}

    return {'safe': True, 'reason': 'Volume réel ✓'}

# ================================
# ESTIMATION DURÉE
# ================================

def estimate_duration(symbol: str) -> Optional[dict]:
    """Estime la durée du mouvement"""
    momentum = calc_momentum(symbol)
    adx = calc_adx(symbol)

    if not momentum or not adx:
        return None

    hist = list(state.history[symbol])
    if len(hist) < 15:
        return None

    velocity = abs(momentum)
    acceleration = abs(hist[-1]['price'] - hist[-5]['price']) / hist[-5]['price'] * 100

    score_5min = 85 if (acceleration > 0.3 and velocity > 0.5) else 40
    score_15min = 80 if (velocity > 1 and adx > 20) else 45
    score_30min = 75 if (adx > 25 and velocity > 0.5) else 35
    score_60min = 70 if adx > 30 else 30

    best_duration = '5-10min'
    if score_15min > score_5min:
        best_duration = '15-20min'
    elif score_30min > score_15min:
        best_duration = '30min+'

    return {
        '5min': score_5min,
        '15min': score_15min,
        '30min': score_30min,
        '60min': score_60min,
        'best': best_duration
    }

# ================================
# CALCUL DÉCORRÉLATION
# ================================

def calc_decorrelation(symbol: str) -> Optional[float]:
    """Calcule la décorrélation avec BTC"""
    if symbol == 'BTCUSDT':
        return 0

    btc_mom = calc_momentum('BTCUSDT')
    sym_mom = calc_momentum(symbol)

    if not btc_mom or not sym_mom:
        return None

    if abs(btc_mom) > 1 and abs(sym_mom) < 0.5:
        return abs(btc_mom - sym_mom)

    return 0

# ================================
# SCORING & SIGNAUX
# ================================

def calc_signal_score(symbol: str, data: dict) -> Optional[dict]:
    """Calcule le score du signal"""
    rsi = calc_rsi(symbol)
    atr = calc_atr(symbol)
    adx = calc_adx(symbol)
    momentum = calc_momentum(symbol)
    vol_trend = calc_volume_trend(symbol, data['volume'])
    decorr = calc_decorrelation(symbol)

    if not all([rsi, atr, adx, momentum, vol_trend]):
        return None

    # Vérification volatilité
    volatility_pct = (atr / data['price']) * 100
    if volatility_pct < 0.3 or volatility_pct > 5:
        return None

    # Anti-spoofing
    spoof_check = check_spoofing(data)
    if not spoof_check['safe']:
        return None

    # Signaux individuels
    rsi_signal = 1 if rsi < 30 else (0.7 if 40 < rsi < 60 and momentum > 0 else 0)
    volume_signal = 1 if vol_trend > 150 else 0
    momentum_signal = 1 if momentum > 0.5 else 0
    adx_signal = 1 if adx > 20 else 0
    decorr_signal = 1 if decorr and decorr > 2 else 0

    # Poids adaptatifs selon régime
    regime = detect_market_regime()
    if regime in ['strong_bull', 'strong_bear']:
        weights = {'rsi': 0.15, 'volume': 0.25, 'momentum': 0.35, 'adx': 0.15, 'decorr': 0.1}
    elif regime == 'ranging':
        weights = {'rsi': 0.35, 'volume': 0.35, 'momentum': 0.1, 'adx': 0.1, 'decorr': 0.1}
    else:
        weights = {'rsi': 0.2, 'volume': 0.2, 'momentum': 0.2, 'adx': 0.2, 'decorr': 0.2}

    # Score final
    score = (
        rsi_signal * weights['rsi'] +
        volume_signal * weights['volume'] +
        momentum_signal * weights['momentum'] +
        adx_signal * weights['adx'] +
        decorr_signal * weights['decorr']
    ) * 100

    convergent = score >= 70 and volume_signal == 1

    return {
        'score': score,
        'convergent': convergent,
        'rsi': rsi,
        'volume_trend': vol_trend,
        'momentum': momentum,
        'adx': adx,
        'volatility': volatility_pct,
        'decorrelation': decorr or 0,
        'spoof_check': spoof_check,
        'regime': regime
    }

# ================================
# SCAN ET NOTIFICATION
# ================================

async def scan_markets():
    """Scanne les marchés et détecte les signaux"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Scan des marchés...")

    # Récupère les données
    binance_data = await fetch_binance_data()

    if not binance_data:
        print("❌ Pas de données")
        return

    # Met à jour l'historique
    for symbol, data in binance_data.items():
        state.update_history(symbol, data)

    # Détecte le régime
    regime = detect_market_regime()
    regime_names = {
        'strong_bull': 'BULL FORT 🚀',
        'strong_bear': 'BEAR FORT 📉',
        'ranging': 'RANGE 〰️',
        'high_volatility': 'VOLATILITÉ EXTRÊME ⚡',
        'neutral': 'NEUTRE ⚖️'
    }

    print(f"📊 Régime: {regime_names[regime]}")

    # Analyse les signaux (focus top cryptos)
    signals_found = []

    for symbol in PAIRS[:10]:  # Top 10 pour optimiser
        if symbol not in binance_data:
            continue

        data = binance_data[symbol]
        signal = calc_signal_score(symbol, data)

        if not signal or not signal['convergent']:
            continue

        # Estimation durée
        duration = estimate_duration(symbol)
        if not duration:
            continue

        # Signal Elite trouvé !
        if signal['score'] >= 70:
            atr = calc_atr(symbol)
            stop_loss_pct = (atr / data['price']) * 100 * 2
            target_profit = signal['momentum'] * 0.6

            signals_found.append({
                'symbol': symbol.replace('USDT', ''),
                'score': signal['score'],
                'price': data['price'],
                'momentum': signal['momentum'],
                'duration': duration['best'],
                'target_profit': target_profit,
                'stop_loss': stop_loss_pct,
                'decorrelation': signal['decorrelation'],
                'regime': regime_names[regime]
            })

    # Envoie les notifications
    if signals_found and state.bot_active:
        for sig in signals_found[:3]:  # Max 3 signaux à la fois
            message = (
                "🎯 <b>SIGNAL ELITE DÉTECTÉ</b>\n\n"
                "💎 Crypto: <b>{}</b>\n"
                "📊 Score: <b>{:.0f}%</b>\n"
                "💰 Prix: ${:.6f}\n"
                "📈 Target: +{:.2f}%\n"
                "🛑 Stop: -{:.2f}%\n"
                "⏱ Durée estimée: {}\n"
                "🔮 Décorrélation BTC: {:.1f}\n"
                "📊 Régime: {}\n\n"
                "✅ <b>ACTION:</b> Acheter en 3 phases\n"
                "• Phase 1: 50%\n"
                "• Phase 2: 25% (+0.3%)\n"
                "• Phase 3: 25% (+0.6%)"
            ).format(
                sig['symbol'],
                sig['score'],
                sig['price'],
                sig['target_profit'],
                sig['stop_loss'],
                sig['duration'],
                sig['decorrelation'],
                sig['regime']
            )

            send_telegram_message(message)
            print(f"✅ Signal envoyé: {sig['symbol']} ({sig['score']:.0f}%)")
            await asyncio.sleep(2)  # Délai entre les messages

    print(f"✅ Scan terminé - {len(signals_found)} signal(s) trouvé(s)\n")

# ================================
# BOUCLE PRINCIPALE
# ================================

async def main_trading_loop():
    """Boucle principale de trading"""
    print("🚀 ULTIMATE PRO SYSTEM")
    print("=" * 50)
    print(f"💰 Capital initial: {CAPITAL_INITIAL}€")
    print(f"📊 Surveillance: {len(PAIRS)} cryptos")
    print(f"⏱ Scan: Toutes les {SCAN_INTERVAL}s")
    print("=" * 50)
    print("\n⏳ En attente de /start sur Telegram...\n")

    while True:
        try:
            if state.bot_active:
                await scan_markets()

            await asyncio.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"❌ Erreur: {e}")
            await asyncio.sleep(10)

# ================================
# DÉMARRAGE
# ================================

if __name__ == "__main__":
    import threading

    # Lance le gestionnaire de commandes Telegram dans un thread séparé
    telegram_thread = threading.Thread(target=process_telegram_commands, daemon=True)
    telegram_thread.start()

    # Lance la boucle principale
    try:
        asyncio.run(main_trading_loop())
    except KeyboardInterrupt:
        print("\n👋 Arrêt du bot")
