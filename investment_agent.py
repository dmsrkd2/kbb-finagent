import news_vector_store
import hyperclova_llm
import pandas as pd
from typing import Dict, List
from datetime import datetime
import numpy as np
import time


from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter



class InvestmentAnalysisAgent:
    """랭체인 기반 투자 매매내역 분석 AI Agent"""
    
    def __init__(self, hyperclova_api_key: str, google_api_key: str, google_search_engine_id: str, pbr_csv_path: str):
        self.hyperclova_llm = hyperclova_llm.HyperCLOVAXLLM(api_key = hyperclova_api_key)
        self.news_vectorstore = news_vector_store.NewsVectorStore(api_key = hyperclova_api_key)
        self.google_searcher = news_vector_store.GoogleNewsSearcher(google_api_key, google_search_engine_id)
        self.news_crawler = news_vector_store.NewsContentCrawler()
        self.pbr_csv_path = pbr_csv_path
        self.pbr_data = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # 다음 청크와 겹치는 부분 길이
            length_function=len
        )

        self.load_pbr_data()
        
        # 분석 프롬프트 템플릿
        self.analysis_prompt = PromptTemplate(
            input_variables=["trade_info", "financial_metrics", "price_analysis", "relevant_news"],
            template="""당신은 사용자의 매매내역을 토대로 투자 습관을 분석하는 전문가입니다. 
사용자의 매매내역과 해당 시점의 PER,PBR,ROE, 주가 데이터, 뉴스를 토대로 사용자가 해당 종목을 매수, 매도한 근거를 추론해서 투자자의 매매 습관을 분석하고 해당 매매 방식의 장점, 약점, 리스크, 보완방법 등을 분석해서 리포트 형태로 제출해줘.

## 상위 3개 종목 분석
{top_symbols_analysis}

## 매매 내역 상세
{trading_records}

## PBR 분석
{pbr_analysis}

## 관련 뉴스
{relevant_news}

위 정보를 바탕으로 상세한 투자 습관 분석을 제공해주세요."""
        )

    def load_pbr_data(self):
        """PBR CSV 파일 로드"""
        try:
            self.pbr_data = pd.read_csv(self.pbr_csv_path)
            print(f"✅ PBR 데이터 로드 완료: {len(self.pbr_data)}개 종목")
        except Exception as e:
            print(f"❌ PBR 데이터 로드 오류: {e}")
            self.pbr_data = None

    def load_trading_data(self, csv_file_path: str) -> pd.DataFrame:
        """CSV 파일에서 매매 데이터 로드"""
        try:
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date']) # 문자열 형식의 날짜를 datetime으로 변환.
            return df
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return pd.DataFrame()

    def get_top_3_symbols_by_volume(self, trading_data: pd.DataFrame) -> List[str]:
        """4개월간 총 거래량 기준 상위 3개 종목 선정"""
        # 종목별 총 거래량 계산 (매수량 + 매도량)
        symbol_volumes = {}
        
        for _, row in trading_data.iterrows():  # iterrows -> (index, row Series) 형태의 튜플 반환
            symbol = row['Symbol']
            quantity = row['Quantity']
            
            if symbol not in symbol_volumes:
                symbol_volumes[symbol] = 0
            symbol_volumes[symbol] += quantity   

        # 거래량 기준 상위 3개 종목 선정
        top_symbols = sorted(symbol_volumes.items(), key=lambda x: x[1], reverse=True)[:3]  # sorted(iterable, key = 기준함수, reverse)
        top_symbol_names = [symbol for symbol, volume in top_symbols]
        
        print(f"📊 거래량 상위 3개 종목:")
        for i, (symbol, volume) in enumerate(top_symbols, 1):
            print(f"  {i}. {symbol}: {volume:,}주")
        
        return top_symbol_names
    
    def calculate_average_price(self, symbol: str, trading_data: pd.DataFrame) -> Dict:
        """종목별 평단가 계산"""
        symbol_trades = trading_data[trading_data['Symbol'] == symbol].copy()
        symbol_trades = symbol_trades.sort_values('Date')
        
        total_buy_amount = 0
        total_buy_quantity = 0
        total_sell_amount = 0
        total_sell_quantity = 0
        
        trade_records = []
        
        for _, row in symbol_trades.iterrows():
            trade_type = row['TradeType']
            price = row['Price']
            quantity = row['Quantity']
            amount = price * quantity
            
            if trade_type == 'Buy':
                total_buy_amount += amount
                total_buy_quantity += quantity
            else:  # Sell
                total_sell_amount += amount
                total_sell_quantity += quantity
            
            # 현재 시점 평단가 계산
            current_avg_price = total_buy_amount / total_buy_quantity if total_buy_quantity > 0 else 0
            
            trade_records.append({
                'date': row['Date'],
                'trade_type': trade_type,
                'price': price,
                'quantity': quantity,
                'amount': amount,
                'avg_price': current_avg_price
            })
        
        buy_avg_price = total_buy_amount / total_buy_quantity if total_buy_quantity > 0 else 0
        sell_avg_price = total_sell_amount / total_sell_quantity if total_sell_quantity > 0 else 0
        
        return {
            'symbol': symbol,
            'buy_avg_price': buy_avg_price,
            'sell_avg_price': sell_avg_price,
            'total_buy_quantity': total_buy_quantity,
            'total_sell_quantity': total_sell_quantity,
            'trade_records': trade_records
        }

    def get_pbr_data_for_symbol(self, symbol: str, first_trade_date: str) -> Dict:
        """종목의 PBR 데이터 및 동종업계 중앙값 계산"""
        if self.pbr_data is None:
            return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
        
        try:
            # 첫 매매 날짜를 기준으로 분기 결정
            date_obj = datetime.strptime(first_trade_date, '%Y-%m-%d')
            
            # 분기 매핑 (예: 2024-01-01 -> 2024Q1)
            quarter = f"{date_obj.year}Q{(date_obj.month - 1) // 3 + 1}"
            
            # 해당 종목의 PBR 데이터 찾기
            symbol_data = self.pbr_data[self.pbr_data['Symbol'] == symbol]
            
            if symbol_data.empty:
                print(f"⚠️  {symbol} PBR 데이터 없음")
                return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
            
            symbol_row = symbol_data.iloc[0]
            
            # 해당 분기의 PBR 값 가져오기
            symbol_pbr = symbol_row.get(quarter, 0)
            
            # 한국 기업인지 나스닥 기업인지 구분
            if symbol.endswith('.KS') or symbol.endswith('.KQ'):
                industry_column = 'WICS_MidCap'  # 한국 기업 - WICS 중분류
            else:
                industry_column = 'GICS_MidCap'  # 나스닥 기업 - GICS 중분류
            
            industry = symbol_row.get(industry_column, 'Unknown')
            
            # 동종업계 PBR 중앙값 계산
            industry_data = self.pbr_data[self.pbr_data[industry_column] == industry]
            industry_pbr_values = industry_data[quarter].dropna()
            industry_median_pbr = industry_pbr_values.median() if not industry_pbr_values.empty else 0
            
            return {
                'symbol_pbr': symbol_pbr,
                'industry_median_pbr': industry_median_pbr,
                'industry': industry,
                'quarter': quarter
            }
            
        except Exception as e:
            print(f"❌ PBR 데이터 처리 오류 ({symbol}): {e}")
            return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
        
    def get_pbr_data_for_symbol(self, symbol: str, first_trade_date: str) -> Dict:
        """종목의 PBR 데이터 및 동종업계 중앙값 계산"""
        if self.pbr_data is None:
            return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
        
        try:
            # 첫 매매 날짜를 기준으로 분기 결정
            date_obj = datetime.strptime(first_trade_date, '%Y-%m-%d')
            
            # 분기 매핑 (예: 2024-01-01 -> 2024Q1)
            quarter = f"{date_obj.year}Q{(date_obj.month - 1) // 3 + 1}"
            
            # 해당 종목의 PBR 데이터 찾기
            symbol_data = self.pbr_data[self.pbr_data['Symbol'] == symbol]
            
            if symbol_data.empty:
                print(f"⚠️  {symbol} PBR 데이터 없음")
                return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
            
            symbol_row = symbol_data.iloc[0]
            
            # 해당 분기의 PBR 값 가져오기
            symbol_pbr = symbol_row.get(quarter, 0)
            
            # 한국 기업인지 나스닥 기업인지 구분
            if symbol.endswith('.KS') or symbol.endswith('.KQ'):    # csv 저장 형식에 따라 달라짐. KS 안 붙으면 isdigit 써도 될 듯
                industry_column = 'WICS_MidCap'  # 한국 기업 - WICS 중분류
            else:
                industry_column = 'GICS_MidCap'  # 나스닥 기업 - GICS 중분류
            
            industry = symbol_row.get(industry_column, 'Unknown')
            
            # 동종업계 PBR 중앙값 계산
            industry_data = self.pbr_data[self.pbr_data[industry_column] == industry]
            industry_pbr_values = industry_data[quarter].dropna()   # 결측치 제거
            industry_median_pbr = industry_pbr_values.median() if not industry_pbr_values.empty else 0
            
            return {
                'symbol_pbr': symbol_pbr,
                'industry_median_pbr': industry_median_pbr,
                'industry': industry,
                'quarter': quarter
            }
            
        except Exception as e:
            print(f"❌ PBR 데이터 처리 오류 ({symbol}): {e}")
            return {'symbol_pbr': 0, 'industry_median_pbr': 0, 'industry': 'Unknown'}
        
    def get_limited_news_for_symbol(self, symbol: str, trading_data: pd.DataFrame) -> List[Dict]:
        """종목당 2개의 뉴스 수집"""
        symbol_trades = trading_data[trading_data['Symbol'] == symbol].sort_values('Date')
        
        if symbol_trades.empty:
            return []
        
        # 첫 번째 매매 날짜 기준으로 뉴스 검색
        first_trade_date = symbol_trades.iloc[0]['Date']
        
        # 기존 뉴스 검색
        query = f"{symbol} 주식 뉴스"
        existing_news = self.news_vectorstore.search_existing_news(symbol, first_trade_date, query, k=2)
        
        if len(existing_news) >= 2:
            return existing_news[:2]
        
        # 부족한 경우 새 뉴스 수집
        print(f"🔍 {symbol} 뉴스 수집 중...")
        news_results = self.google_searcher.search_news(symbol, first_trade_date)
        
        processed_news = []
        for news in news_results[:2]:  # 최대 2개만
            content = self.news_crawler.crawl_news_content(news['url'])
            if content:
                chunks = self.text_splitter.split_text(content)
                
                news_data = {
                    'symbol': symbol,
                    'date': first_trade_date,
                    'title': news['title'],
                    'url': news['url'],
                    'source': news['source'],
                    'content': content
                }
                
                if self.news_vectorstore.store_news_chunks(news_data, chunks):
                    processed_news.append({
                        'title': news['title'],
                        'content': ' '.join(chunks[:2]),  # 상위 2개 청크만
                        'relevance_score': self.calculate_relevance_score(content, symbol)
                    })
                
                time.sleep(1)
        
        return (existing_news + processed_news)[:2]
    
    def analyze_trading_patterns(self, csv_file_path: str) -> str:
        """상위 3개 종목 기준 매매 패턴 분석"""
        
        # 매매 데이터 로드
        trading_data = self.load_trading_data(csv_file_path)
        if trading_data.empty:
            return "매매 데이터를 로드할 수 없습니다."
        
        # 상위 3개 종목 선정
        top_symbols = self.get_top_3_symbols_by_volume(trading_data)
        
        # 상위 3개 종목만 필터링
        filtered_data = trading_data[trading_data['Symbol'].isin(top_symbols)]
        
        print(f"📈 상위 3개 종목 ({len(filtered_data)}건) 분석 시작...")
        
        # 종목별 분석
        symbol_analyses = {}
        all_news = []       # 이거 왜 있는지 모르겠음
        
        for symbol in top_symbols:
            symbol_data = filtered_data[filtered_data['Symbol'] == symbol]
            
            # 평단가 계산
            avg_price_info = self.calculate_average_price(symbol, symbol_data)
            
            # PBR 분석
            first_trade_date = symbol_data.sort_values('Date').iloc[0]['Date']
            pbr_info = self.get_pbr_data_for_symbol(symbol, first_trade_date)
            
            # 뉴스 수집 (종목당 2개)
            news_info = self.get_limited_news_for_symbol(symbol, symbol_data)
            
            symbol_analyses[symbol] = {
                'avg_price_info': avg_price_info,
                'pbr_info': pbr_info,
                'news_info': news_info,
                'first_trade_date': first_trade_date
            }
            
            all_news.extend(news_info)
        
        # 분석 리포트 생성
        analysis_report = self.generate_comprehensive_analysis_report(
            top_symbols, symbol_analyses, filtered_data
        )
        
        return analysis_report
    
    def generate_comprehensive_analysis_report(self, top_symbols: List[str], 
                                             symbol_analyses: Dict, 
                                             trading_data: pd.DataFrame) -> str:
        """종합 분석 리포트 생성"""
        
        # 상위 3개 종목 분석 요약
        top_symbols_analysis = self.format_top_symbols_analysis(top_symbols, symbol_analyses)
        
        # 매매 내역 상세 포맷팅
        trading_records = self.format_trading_records(trading_data, symbol_analyses)
        
        # PBR 분석 포맷팅
        pbr_analysis = self.format_pbr_analysis(symbol_analyses)
        
        # 뉴스 정보 포맷팅
        relevant_news = self.format_news_analysis(symbol_analyses)
        
        # 프롬프트 생성 및 실행
        prompt = self.analysis_prompt.format(
            top_symbols_analysis=top_symbols_analysis,
            trading_records=trading_records,
            pbr_analysis=pbr_analysis,
            relevant_news=relevant_news
        )
        
        return self.hyperclova_llm._call(prompt)
    
    def format_top_symbols_analysis(self, top_symbols: List[str], symbol_analyses: Dict) -> str:
        """상위 3개 종목 분석 포맷팅"""
        analysis = "## 거래량 상위 3개 종목 개요\n\n"
        
        for i, symbol in enumerate(top_symbols, 1):
            info = symbol_analyses[symbol]
            avg_info = info['avg_price_info']
            
            analysis += f"### {i}. {symbol}\n"
            analysis += f"- 총 매수량: {avg_info['total_buy_quantity']:,}주\n"
            analysis += f"- 총 매도량: {avg_info['total_sell_quantity']:,}주\n"
            analysis += f"- 매수 평단가: {avg_info['buy_avg_price']:,.0f}원\n"
            analysis += f"- 매도 평단가: {avg_info['sell_avg_price']:,.0f}원\n"
            analysis += f"- 첫 매매일: {info['first_trade_date']}\n\n"
        
        return analysis
    
    def format_trading_records(self, trading_data: pd.DataFrame, symbol_analyses: Dict) -> str:
        """매매 내역 상세 포맷팅"""
        records = "## 매매 내역 상세\n\n"
        
        for symbol, info in symbol_analyses.items():
            records += f"### {symbol} 매매 내역\n"
            
            for trade in info['avg_price_info']['trade_records']:
                records += f"- {trade['date']} | {trade['trade_type']} | "
                records += f"가격: {trade['price']:,.0f}원 | "
                records += f"거래대금: {trade['amount']:,.0f}원 | "
                records += f"평단가: {trade['avg_price']:,.0f}원\n"
            
            records += "\n"
        
        return records
    
    def format_pbr_analysis(self, symbol_analyses: Dict) -> str:
        """PBR 분석 포맷팅"""
        analysis = "## PBR 분석\n\n"
        
        for symbol, info in symbol_analyses.items():
            pbr_info = info['pbr_info']
            
            analysis += f"### {symbol}\n"
            analysis += f"- 종목 PBR: {pbr_info['symbol_pbr']:.2f}\n"
            analysis += f"- 동종업계 PBR 중앙값: {pbr_info['industry_median_pbr']:.2f}\n"
            analysis += f"- 업종: {pbr_info['industry']}\n"
            analysis += f"- 기준 분기: {pbr_info.get('quarter', 'N/A')}\n"
            
            # PBR 상대적 위치 분석
            if pbr_info['symbol_pbr'] > 0 and pbr_info['industry_median_pbr'] > 0:
                ratio = pbr_info['symbol_pbr'] / pbr_info['industry_median_pbr']
                if ratio > 1.2:
                    analysis += f"- 평가: 업종 대비 고평가 (비율: {ratio:.2f})\n"
                elif ratio < 0.8:
                    analysis += f"- 평가: 업종 대비 저평가 (비율: {ratio:.2f})\n"
                else:
                    analysis += f"- 평가: 업종 대비 적정 수준 (비율: {ratio:.2f})\n"
            
            analysis += "\n"
        
        return analysis
    
    def format_news_analysis(self, symbol_analyses: Dict) -> str:
        """뉴스 분석 포맷팅"""
        analysis = "## 관련 뉴스 분석\n\n"
        
        for symbol, info in symbol_analyses.items():
            news_list = info['news_info']
            
            analysis += f"### {symbol} 관련 뉴스\n"
            
            if not news_list:
                analysis += "- 관련 뉴스 없음\n\n"
                continue
            
            for i, news in enumerate(news_list, 1):
                analysis += f"{i}. **{news['title']}**\n"
                analysis += f"   - 관련도: {news.get('relevance_score', 0):.2f}\n"
                analysis += f"   - 요약: {news['content'][:200]}...\n\n"
        
        return analysis
        
    
    def get_or_fetch_news(self, symbol: str, date: str) -> List[Dict]:
        """뉴스 검색 또는 수집"""
        
        # 1. 기존 벡터 DB에서 검색
        query = f"{symbol} 주식 뉴스"
        existing_news = self.news_vectorstore.search_existing_news(symbol, date, query)
        
        if existing_news:
            print(f"✅ 기존 뉴스 발견: {len(existing_news)}개")
            return existing_news
        
        # 2. 구글 검색 API로 새 뉴스 수집
        print(f"🔍 새 뉴스 검색: {symbol} - {date}")
        news_results = self.google_searcher.search_news(symbol, date)
        
        if not news_results:
            print("❌ 관련 뉴스를 찾을 수 없습니다.")
            return []
        
        # 3. 뉴스 본문 크롤링 및 처리
        processed_news = []
        for news in news_results[:3]:  # 상위 3개만 처리
            print(f"📰 크롤링 중: {news['title'][:50]}...")
            
            # 본문 크롤링
            content = self.news_crawler.crawl_news_content(news['url'])
            if not content:
                continue
            
            # 텍스트 분할
            chunks = self.text_splitter.split_text(content)

            
            # 뉴스 데이터 구성
            news_data = {
                'symbol': symbol,
                'date': date,
                'title': news['title'],
                'url': news['url'],
                'source': news['source'],
                'content': content
            }
            
            # 벡터 DB에 저장
            if self.news_vectorstore.store_news_chunks(news_data, chunks):
                processed_news.append({
                    'title': news['title'],
                    'content': ' '.join(chunks[:3]),  # 상위 3개 청크만
                    'relevance_score': self.calculate_relevance_score(content, symbol)
                })
            
            time.sleep(1)  # 크롤링 간격 조절
        
        return processed_news
    
    def calculate_relevance_score(self, content: str, symbol: str) -> float:    # TODO relevance_score 로직 변경
        """뉴스 관련성 점수 계산"""
        positive_keywords = ['상승', '급등', '호재', '매수', '투자', '성장', '수익', '증가']
        negative_keywords = ['하락', '급락', '악재', '매도', '손실', '감소', '부진', '위험']
        
        content_lower = content.lower()
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in content_lower)
        symbol_mention = 2 if symbol.lower() in content_lower else 0
        
        return (positive_score + negative_score + symbol_mention) / 10
    
   


    def save_analysis_report(self, report: str, filename: str = None) -> str:
        """분석 리포트 저장"""
        
        if filename is None:
            filename = f"investment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ 분석 리포트 저장 완료: {filename}")
            return filename
        except Exception as e:
            print(f"❌ 리포트 저장 오류: {e}")
            return ""
