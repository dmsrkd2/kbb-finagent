import pandas as pd
import yfinance as yf
from typing import Dict
from datetime import datetime, timedelta

    


class StockDataAnalyzer:
    """주식 데이터 분석기"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """yfinance를 통해 주식 데이터 조회"""
        try:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if not ('.' in symbol):
                if symbol.isdigit():
                    symbol += '.KS'
                    
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"주식 데이터 조회 오류 ({symbol}): {e}")
            return pd.DataFrame()
    
    def calculate_financial_metrics(self, symbol: str, date: str) -> Dict:
        """PER, PBR, ROE 계산"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            per = info.get('trailingPE', 0)
            pbr = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0)
            
            return {
                'PER': per if per else 0,
                'PBR': pbr if pbr else 0,
                'ROE': roe if roe else 0
            }
            
        except Exception as e:
            print(f"재무 지표 계산 오류 ({symbol}): {e}")
            return {'PER': 0, 'PBR': 0, 'ROE': 0}
    
    def analyze_price_position(self, symbol: str, trade_date: str, period: int = 252) -> Dict:
        """매매 시점의 주가 위치 분석"""
        try:
            end_date = datetime.strptime(trade_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=period)
            
            data = self.get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), trade_date)
            
            if data.empty:
                return {'position': 'unknown', 'percentile': 0}
            
            trade_price = data.loc[trade_date, 'Close'] if trade_date in data.index else data['Close'].iloc[-1]
            
            high_price = data['High'].max()
            low_price = data['Low'].min()
            
            percentile = (trade_price - low_price) / (high_price - low_price) * 100
            
            if percentile >= 80:
                position = 'high'
            elif percentile <= 20:
                position = 'low'
            else:
                position = 'middle'
                
            return {
                'position': position,
                'percentile': percentile,
                'trade_price': trade_price,
                'high_price': high_price,
                'low_price': low_price
            }
            
        except Exception as e:
            print(f"가격 위치 분석 오류 ({symbol}): {e}")
            return {'position': 'unknown', 'percentile': 0}
