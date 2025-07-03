import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class NewsRAGDB:
    """뉴스 데이터를 저장하고 관리하는 RAG 데이터베이스"""
    
    def __init__(self, db_path: str = "news_rag.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                relevance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_date_symbol ON news(date, symbol)
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_news_from_naver(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """네이버 뉴스에서 종목 관련 뉴스 수집"""
        news_list = []
        
        # 한국 종목코드 처리 (예: 005930.KS -> 삼성전자)
        if symbol.endswith('.KS') or symbol.endswith('.KQ'):
            symbol_code = symbol.split('.')[0]
            search_query = f"{symbol_code} 주식"
        else:
            search_query = f"{symbol} stock"
            
        try:
            # 실제 구현에서는 네이버 뉴스 API나 크롤링 로직 필요
            # 여기서는 샘플 데이터로 대체
            sample_news = [
                {
                    'date': start_date,
                    'title': f"{symbol} 관련 주요 뉴스",
                    'content': f"{symbol} 종목에 대한 긍정적/부정적 뉴스 내용...",
                    'source': 'naver_news'
                }
            ]
            news_list.extend(sample_news)
            
        except Exception as e:
            print(f"뉴스 수집 중 오류 발생: {e}")
            
        return news_list
    
    def calculate_relevance_score(self, news_content: str, symbol: str) -> float:
        """뉴스 내용의 주가 변동 관련성 점수 계산"""
        # 주가 변동 관련 키워드
        positive_keywords = ['상승', '급등', '호재', '매수', '투자', '성장', '수익', '증가', '실적', '좋은']
        negative_keywords = ['하락', '급락', '악재', '매도', '손실', '감소', '부진', '위험', '하향', '나쁜']
        
        content_lower = news_content.lower()
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        # 종목명 포함 여부
        symbol_mention = 2 if symbol.lower() in content_lower else 0
        
        return (positive_score + negative_score + symbol_mention) / 10
    
    def store_news(self, news_data: List[Dict]):
        """뉴스 데이터를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for news in news_data:
            relevance_score = self.calculate_relevance_score(news['content'], news.get('symbol', ''))
            
            cursor.execute('''
                INSERT INTO news (date, symbol, title, content, source, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                news['date'],
                news.get('symbol', ''),
                news['title'],
                news['content'],
                news['source'],
                relevance_score
            ))
        
        conn.commit()
        conn.close()
    
    def get_relevant_news(self, symbol: str, date: str, threshold: float = 0.3) -> List[Dict]:
        """특정 종목과 날짜에 대한 관련 뉴스 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, content, source, relevance_score
            FROM news
            WHERE symbol = ? AND date = ? AND relevance_score >= ?
            ORDER BY relevance_score DESC
            LIMIT 5
        ''', (symbol, date, threshold))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'title': row[0],
                'content': row[1],
                'source': row[2],
                'relevance_score': row[3]
            }
            for row in results
        ]

class StockDataAnalyzer:
    """주식 데이터 분석기"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """yfinance를 통해 주식 데이터 조회"""
        try:
            # 캐시 확인
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 종목 코드 변환 (한국 주식)
            if not ('.' in symbol):
                if symbol.isdigit():
                    symbol += '.KS'  # 코스피
                    
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            # 캐시 저장
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
            
            # 재무 지표 추출
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
        """매매 시점의 주가 위치 분석 (고점/저점 판단)"""
        try:
            end_date = datetime.strptime(trade_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=period)
            
            data = self.get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), trade_date)
            
            if data.empty:
                return {'position': 'unknown', 'percentile': 0}
            
            trade_price = data.loc[trade_date, 'Close'] if trade_date in data.index else data['Close'].iloc[-1]
            
            # 기간 내 최고가, 최저가 대비 위치 계산
            high_price = data['High'].max()
            low_price = data['Low'].min()
            
            # 백분위 계산
            percentile = (trade_price - low_price) / (high_price - low_price) * 100
            
            # 위치 판단
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

class InvestmentAnalysisAgent:
    """투자 매매내역 분석 AI Agent"""
    
    def __init__(self, hyperclova_api_key: str):
        self.hyperclova_api_key = hyperclova_api_key
        self.news_db = NewsRAGDB()
        self.stock_analyzer = StockDataAnalyzer()
        
    def load_trading_data(self, csv_file_path: str) -> pd.DataFrame:
        """CSV 파일에서 매매 데이터 로드"""
        try:
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return pd.DataFrame()
    
    def prepare_news_data(self, trading_data: pd.DataFrame):
        """매매 데이터 기반 뉴스 수집 및 저장"""
        symbols = trading_data['Symbol'].unique()
        date_range = pd.date_range(
            start=trading_data['Date'].min(),
            end=trading_data['Date'].max(),
            freq='D'
        )
        
        for symbol in symbols:
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                news_data = self.news_db.fetch_news_from_naver(
                    symbol, date_str, date_str
                )
                
                # 뉴스에 심볼 정보 추가
                for news in news_data:
                    news['symbol'] = symbol
                    
                if news_data:
                    self.news_db.store_news(news_data)
    
    def analyze_single_trade(self, trade_row: pd.Series) -> Dict:
        """단일 매매 건 분석"""
        symbol = trade_row['Symbol']
        date = trade_row['Date'].strftime('%Y-%m-%d')
        trade_type = trade_row['TradeType']
        price = trade_row['Price']
        quantity = trade_row['Quantity']
        
        # 1. 재무 지표 계산
        financial_metrics = self.stock_analyzer.calculate_financial_metrics(symbol, date)
        
        # 2. 주가 위치 분석
        price_analysis = self.stock_analyzer.analyze_price_position(symbol, date)
        
        # 3. 관련 뉴스 조회
        relevant_news = self.news_db.get_relevant_news(symbol, date)
        
        return {
            'symbol': symbol,
            'date': date,
            'trade_type': trade_type,
            'price': price,
            'quantity': quantity,
            'financial_metrics': financial_metrics,
            'price_analysis': price_analysis,
            'relevant_news': relevant_news
        }
    
    def generate_analysis_prompt(self, trade_analysis: Dict) -> str:
        """HyperCLOVA X를 위한 분석 프롬프트 생성"""
        base_prompt = """당신은 사용자의 매매내역을 토대로 투자 습관을 분석하는 전문가입니다. 사용자의 매매내역과 해당 시점의 PER,PBR,ROE, 주가 데이터, 뉴스를 토대로 사용자가 해당 종목을 매수, 매도한 근거를 추론해서 투자자의 매매 습관을 분석하고 해당 매매 방식의 장점, 약점, 리스크, 보완방법 등을 분석해서 리포트 형태로 제출해줘."""
        
        # 매매 정보
        trade_info = f"""
## 매매 정보
- 종목: {trade_analysis['symbol']}
- 날짜: {trade_analysis['date']}
- 매매 유형: {trade_analysis['trade_type']}
- 가격: {trade_analysis['price']:,.0f}원
- 수량: {trade_analysis['quantity']}주
"""
        
        # 재무 지표
        metrics = trade_analysis['financial_metrics']
        financial_info = f"""
## 재무 지표 (매매 시점)
- PER: {metrics['PER']:.2f}
- PBR: {metrics['PBR']:.2f}
- ROE: {metrics['ROE']:.2f}%
"""
        
        # 주가 분석
        price_analysis = trade_analysis['price_analysis']
        price_info = f"""
## 주가 분석
- 가격 위치: {price_analysis['position']} (백분위: {price_analysis['percentile']:.1f}%)
- 매매가: {price_analysis.get('trade_price', 0):,.0f}원
- 기간 최고가: {price_analysis.get('high_price', 0):,.0f}원
- 기간 최저가: {price_analysis.get('low_price', 0):,.0f}원
"""
        
        # 관련 뉴스
        news_info = "## 관련 뉴스\n"
        for i, news in enumerate(trade_analysis['relevant_news'], 1):
            news_info += f"{i}. {news['title']} (관련도: {news['relevance_score']:.2f})\n"
            news_info += f"   내용: {news['content'][:200]}...\n\n"
        
        if not trade_analysis['relevant_news']:
            news_info += "해당 날짜에 관련 뉴스가 없습니다.\n"
        
        return base_prompt + trade_info + financial_info + price_info + news_info
    
    def call_hyperclova_api(self, prompt: str) -> str:
        """HyperCLOVA X API 호출"""
        # 실제 API 호출 구현 필요
        # 여기서는 샘플 응답으로 대체
        
        try:
            # HyperCLOVA X API 호출 로직
            # headers = {
            #     'Authorization': f'Bearer {self.hyperclova_api_key}',
            #     'Content-Type': 'application/json'
            # }
            # 
            # payload = {
            #     'prompt': prompt,
            #     'max_tokens': 2000,
            #     'temperature': 0.7
            # }
            # 
            # response = requests.post('HYPERCLOVA_X_API_URL', 
            #                         headers=headers, 
            #                         json=payload)
            # 
            # return response.json()['generated_text']
            
            # 샘플 응답
            return f"""
## 투자 습관 분석 리포트

### 매매 근거 추론
사용자의 매매는 다음과 같은 근거에 기반한 것으로 추정됩니다:
- 기술적 분석 기반의 매매 (주가 위치 고려)
- 재무 지표를 통한 가치 투자 접근
- 뉴스 기반의 이벤트 드리븐 투자

### 투자 스타일 분석
- **장점**: 다양한 정보를 종합적으로 고려하는 체계적 접근
- **약점**: 감정적 판단 요소 개입 가능성
- **리스크**: 시장 타이밍 위험, 정보 해석 오류 가능성

### 보완 방법
1. 손절매 규칙 설정 및 준수
2. 분할 매수/매도 전략 도입
3. 포트폴리오 분산 투자 확대
4. 정기적인 투자 성과 리뷰

### 권장사항
향후 투자 시 위험 관리를 더욱 강화하고, 장기적 관점에서의 투자 전략 수립을 권장합니다.
"""
            
        except Exception as e:
            print(f"HyperCLOVA API 호출 오류: {e}")
            return "API 호출 중 오류가 발생했습니다."
    
    def analyze_trading_patterns(self, csv_file_path: str) -> str:
        """전체 매매 패턴 분석"""
        # 1. 매매 데이터 로드
        trading_data = self.load_trading_data(csv_file_path)
        if trading_data.empty:
            return "매매 데이터를 로드할 수 없습니다."
        
        # 2. 뉴스 데이터 준비
        print("뉴스 데이터 수집 중...")
        self.prepare_news_data(trading_data)
        
        # 3. 각 매매 건별 분석
        analysis_results = []
        
        for idx, trade_row in trading_data.iterrows():
            print(f"분석 중: {trade_row['Symbol']} ({idx+1}/{len(trading_data)})")
            
            # 단일 매매 분석
            trade_analysis = self.analyze_single_trade(trade_row)
            
            # 프롬프트 생성
            prompt = self.generate_analysis_prompt(trade_analysis)
            
            # HyperCLOVA API 호출
            analysis_result = self.call_hyperclova_api(prompt)
            
            analysis_results.append({
                'trade_info': trade_analysis,
                'analysis': analysis_result
            })
        
        # 4. 종합 분석 결과 생성
        comprehensive_analysis = self.generate_comprehensive_report(analysis_results)
        
        return comprehensive_analysis
    
    def generate_comprehensive_report(self, analysis_results: List[Dict]) -> str:
        """종합 분석 리포트 생성"""
        report = """
# 투자 습관 종합 분석 리포트

## 개요
4개월간의 매매 내역을 종합적으로 분석한 결과입니다.

## 매매 패턴 분석
"""
        
        # 매매 통계
        total_trades = len(analysis_results)
        buy_count = sum(1 for r in analysis_results if r['trade_info']['trade_type'] == 'Buy')
        sell_count = total_trades - buy_count
        
        report += f"""
### 매매 통계
- 총 매매 건수: {total_trades}건
- 매수: {buy_count}건 ({buy_count/total_trades*100:.1f}%)
- 매도: {sell_count}건 ({sell_count/total_trades*100:.1f}%)
"""
        
        # 개별 분석 결과 요약
        report += "\n## 개별 매매 분석\n"
        for i, result in enumerate(analysis_results, 1):
            trade_info = result['trade_info']
            report += f"""
### {i}. {trade_info['symbol']} - {trade_info['date']} ({trade_info['trade_type']})
{result['analysis']}
---
"""
        
        return report

# 사용 예시
def main():
    """메인 실행 함수"""
    
    # API 키 설정 (실제 키로 교체 필요)
    HYPERCLOVA_API_KEY = "your_hyperclova_api_key_here"
    
    # AI Agent 초기화
    agent = InvestmentAnalysisAgent(HYPERCLOVA_API_KEY)
    
    # 매매 데이터 파일 경로
    csv_file_path = "trading_data.csv"
    
    # 분석 실행
    print("투자 습관 분석을 시작합니다...")
    analysis_report = agent.analyze_trading_patterns(csv_file_path)
    
    # 결과 출력
    print("\n" + "="*50)
    print("분석 완료!")
    print("="*50)
    print(analysis_report)
    
    # 결과를 파일로 저장
    with open("investment_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(analysis_report)
    
    print("\n분석 결과가 'investment_analysis_report.txt' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()