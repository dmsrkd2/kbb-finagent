import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Tuple, Optional
import re
import warnings
import hashlib
import os
from urllib.parse import urljoin, urlparse
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# Chroma
import chromadb
from chromadb.config import Settings

warnings.filterwarnings('ignore')

class HyperCLOVAXLLM(LLM):
    """HyperCLOVA X API를 위한 커스텀 LLM 클래스"""
    
    api_key: str
    api_url: str = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
    
    @property
    def _llm_type(self) -> str:
        return "hyperclova_x"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """HyperCLOVA X API 호출"""
        try:
            headers = {
                'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
                'X-NCP-APIGW-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "topP": 0.8,
                "topK": 0,
                "maxTokens": 2048,
                "temperature": 0.3,
                "repeatPenalty": 1.2,
                "stopBefore": stop or [],
                "includeAiFilters": True
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['result']['message']['content']
            else:
                print(f"API 호출 실패: {response.status_code}")
                return "API 호출 중 오류가 발생했습니다."
                
        except Exception as e:
            print(f"HyperCLOVA API 호출 오류: {e}")
            return "API 호출 중 오류가 발생했습니다."

class NewsVectorStore:
    """Chroma 벡터 DB를 사용한 뉴스 저장소"""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "news_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Chroma 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # LangChain Chroma 벡터스토어 초기화
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def generate_content_hash(self, content: str) -> str:
        """콘텐츠 해시 생성 (중복 방지용)"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_content_duplicate(self, content_hash: str) -> bool:
        """콘텐츠 중복 여부 확인"""
        try:
            results = self.collection.get(
                where={"content_hash": content_hash},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def search_existing_news(self, symbol: str, date: str, query: str, k: int = 5) -> List[Dict]:
        """기존 뉴스 검색"""
        try:
            # 벡터 유사도 검색
            docs = self.vectorstore.similarity_search_with_score(
                query=f"{symbol} {date} {query}",
                k=k,
                filter={
                    "symbol": symbol,
                    "date": date
                }
            )
            
            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                })
            
            return results
            
        except Exception as e:
            print(f"뉴스 검색 오류: {e}")
            return []
    
    def store_news_chunks(self, news_data: Dict, chunks: List[str]) -> bool:
        """뉴스 청크를 벡터 DB에 저장"""
        try:
            documents = []
            
            for i, chunk in enumerate(chunks):
                content_hash = self.generate_content_hash(chunk)
                
                # 중복 체크
                if self.is_content_duplicate(content_hash):
                    print(f"중복 콘텐츠 스킵: {content_hash[:8]}...")
                    continue
                
                # 메타데이터 생성
                metadata = {
                    'symbol': news_data['symbol'],
                    'date': news_data['date'],
                    'title': news_data['title'],
                    'url': news_data['url'],
                    'source': news_data['source'],
                    'chunk_index': i,
                    'content_hash': content_hash,
                    'keywords': news_data.get('keywords', []),
                    'created_at': datetime.now().isoformat()
                }
                
                # Document 객체 생성
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                
                documents.append(doc)
            
            # 벡터 DB에 저장
            if documents:
                self.vectorstore.add_documents(documents)
                print(f"✅ {len(documents)}개 청크 저장 완료")
                return True
            else:
                print("저장할 새로운 청크가 없습니다.")
                return False
                
        except Exception as e:
            print(f"뉴스 저장 오류: {e}")
            return False

class GoogleNewsSearcher:
    """구글 검색 API를 사용한 뉴스 검색기"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search_news(self, symbol: str, date: str, num_results: int = 10) -> List[Dict]:
        """구글 검색 API로 뉴스 검색"""
        try:
            # 검색 쿼리 생성
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y년 %m월 %d일')
            
            # 한국 종목인지 확인
            if symbol.endswith('.KS') or symbol.endswith('.KQ'):
                symbol_code = symbol.split('.')[0]
                query = f"{symbol_code} 주식 뉴스 {date_str}"
            else:
                query = f"{symbol} stock news {date}"
            
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': num_results,
                'dateRestrict': 'd7',  # 7일 이내
                'sort': 'date'
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                results = response.json()
                news_list = []
                
                for item in results.get('items', []):
                    news_list.append({
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': urlparse(item.get('link', '')).netloc,
                        'date': date,
                        'symbol': symbol
                    })
                
                return news_list
                
            else:
                print(f"구글 검색 API 오류: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"뉴스 검색 오류: {e}")
            return []

class NewsContentCrawler:
    """뉴스 본문 크롤링"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def crawl_news_content(self, url: str) -> Optional[str]:
        """뉴스 본문 크롤링"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 일반적인 뉴스 본문 태그들
            content_selectors = [
                'div.news_content',
                'div.article_body',
                'div.article-content',
                'div.entry-content',
                'div.post-content',
                'article',
                'div.content',
                'div.text'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for elem in elements:
                        content += elem.get_text(strip=True) + " "
                    break
            
            # 기본 본문 추출이 실패한 경우
            if not content:
                # p 태그들로 본문 구성
                paragraphs = soup.find_all('p')
                content = " ".join([p.get_text(strip=True) for p in paragraphs])
            
            # 텍스트 정리
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            return content if len(content) > 100 else None
            
        except Exception as e:
            print(f"크롤링 오류 ({url}): {e}")
            return None

class NewsContentSummarizer:
    """뉴스 본문 요약기"""
    
    def __init__(self, llm):
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_keywords(self, content: str, symbol: str) -> List[str]:
        """뉴스 본문에서 키워드 추출"""
        # 주가 관련 키워드
        stock_keywords = ['주가', '주식', '매수', '매도', '상승', '하락', '급등', '급락', 
                         '호재', '악재', '실적', '수익', '손실', '투자', '거래량']
        
        # 기업 관련 키워드
        company_keywords = ['기업', '회사', '사업', '매출', '영업이익', '순이익', 
                           '성장', '확장', '계약', '제품', '서비스']
        
        all_keywords = stock_keywords + company_keywords
        found_keywords = []
        
        for keyword in all_keywords:
            if keyword in content:
                found_keywords.append(keyword)
        
        # 종목 코드나 이름 추가
        if symbol:
            found_keywords.append(symbol)
        
        return found_keywords
    
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """청크별 요약"""
        summarized_chunks = []
        
        for chunk in chunks:
            if len(chunk) > 500:  # 길이가 충분한 경우만 요약
                summary_prompt = f"""
다음 뉴스 내용을 주요 포인트 중심으로 요약해주세요:

{chunk}

요약:
"""
                try:
                    summary = self.llm._call(summary_prompt)
                    summarized_chunks.append(summary)
                except:
                    summarized_chunks.append(chunk)  # 요약 실패 시 원본 사용
            else:
                summarized_chunks.append(chunk)
        
        return summarized_chunks

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

class InvestmentAnalysisAgent:
    """랭체인 기반 투자 매매내역 분석 AI Agent"""
    
    def __init__(self, hyperclova_api_key: str, google_api_key: str, google_search_engine_id: str):
        self.hyperclova_llm = HyperCLOVAXLLM(api_key=hyperclova_api_key)
        self.news_vectorstore = NewsVectorStore()
        self.google_searcher = GoogleNewsSearcher(google_api_key, google_search_engine_id)
        self.news_crawler = NewsContentCrawler()
        self.news_summarizer = NewsContentSummarizer(self.hyperclova_llm)
        self.stock_analyzer = StockDataAnalyzer()
        
        # 분석 프롬프트 템플릿
        self.analysis_prompt = PromptTemplate(
            input_variables=["trade_info", "financial_metrics", "price_analysis", "relevant_news"],
            template="""당신은 사용자의 매매내역을 토대로 투자 습관을 분석하는 전문가입니다. 
사용자의 매매내역과 해당 시점의 PER,PBR,ROE, 주가 데이터, 뉴스를 토대로 사용자가 해당 종목을 매수, 매도한 근거를 추론해서 투자자의 매매 습관을 분석하고 해당 매매 방식의 장점, 약점, 리스크, 보완방법 등을 분석해서 리포트 형태로 제출해줘.

## 매매 정보
{trade_info}

## 재무 지표
{financial_metrics}

## 주가 분석
{price_analysis}

## 관련 뉴스
{relevant_news}

위 정보를 바탕으로 상세한 투자 습관 분석을 제공해주세요."""
        )
    
    def load_trading_data(self, csv_file_path: str) -> pd.DataFrame:
        """CSV 파일에서 매매 데이터 로드"""
        try:
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return pd.DataFrame()
    
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
        for news in news_results[:5]:  # 상위 5개만 처리
            print(f"📰 크롤링 중: {news['title'][:50]}...")
            
            # 본문 크롤링
            content = self.news_crawler.crawl_news_content(news['url'])
            if not content:
                continue
            
            # 키워드 추출
            keywords = self.news_summarizer.extract_keywords(content, symbol)
            
            # 텍스트 분할
            chunks = self.news_summarizer.text_splitter.split_text(content)
            
            # 청크 요약
            summarized_chunks = self.news_summarizer.summarize_chunks(chunks)
            
            # 뉴스 데이터 구성
            news_data = {
                'symbol': symbol,
                'date': date,
                'title': news['title'],
                'url': news['url'],
                'source': news['source'],
                'keywords': keywords,
                'content': content
            }
            
            # 벡터 DB에 저장
            if self.news_vectorstore.store_news_chunks(news_data, summarized_chunks):
                processed_news.append({
                    'title': news['title'],
                    'content': ' '.join(summarized_chunks[:3]),  # 상위 3개 청크만
                    'keywords': keywords,
                    'relevance_score': self.calculate_relevance_score(content, symbol)
                })
            
            time.sleep(1)  # 크롤링 간격 조절
        
        return processed_news
    
    def calculate_relevance_score(self, content: str, symbol: str) -> float:
        """뉴스 관련성 점수 계산"""
        positive_keywords = ['상승', '급등', '호재', '매수', '투자', '성장', '수익', '증가']
        negative_keywords = ['하락', '급락', '악재', '매도', '손실', '감소', '부진', '위험']
        
        content_lower = content.lower()
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in content_lower)
        symbol_mention = 2 if symbol.lower() in content_lower else 0
        
        return (positive_score + negative_score + symbol_mention) / 10
    
    def analyze_single_trade(self, trade_row: pd.Series) -> Dict:
        """단일 매매 건 분석"""
        symbol = trade_row['Symbol']
        date = trade_row['Date'].strftime('%Y-%m-%d')
        trade_type = trade_row['TradeType']
        price = trade_row['Price']
        quantity = trade_row['Quantity']
        
        print(f"\n📊 분석 중: {symbol} - {date} ({trade_type})")
        
        # 1. 재무 지표 계산
        financial_metrics = self.stock_analyzer.calculate_financial_metrics(symbol, date)
        
        # 2. 주가 위치 분석
        price_analysis = self.stock_analyzer.analyze_price_position(symbol, date)
        
        # 3. 관련 뉴스 수집
        relevant_news = self.get_or_fetch_news(symbol, date)
        
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
    
    def generate_analysis_report(self, trade_analysis: Dict) -> str:
        """LangChain을 사용한 분석 리포트 생성"""
        
        # 입력 데이터 포맷팅
        trade_info = f"""
- 종목: {trade_analysis['symbol']}
- 날짜: {trade_analysis['date']}
- 매매 유형: {trade_analysis['trade_type']}
- 가격: {trade_analysis['price']:,.0f}원
- 수량: {trade_analysis['quantity']}주
"""
        
        metrics = trade_analysis['financial_metrics']
        financial_metrics = f"""
- PER: {metrics['PER']:.2f}
- PBR: {metrics['PBR']:.2f}
- ROE: {metrics['ROE']:.2f}%
"""
        
        price_analysis = trade_analysis['price_analysis']
        price_info = f"""
- 가격 위치: {price_analysis['position']} (백분위: {price_analysis['percentile']:.1f}%)
- 매매가: {price_analysis.get('trade_price', 0):,.0f}원
- 기간 최고가: {price_analysis.get('high_price', 0):,.0f}원
- 기간 최저가: {price_analysis.get('low_price', 0):,.0f}원
"""
        
        news_info = ""
        for i, news in enumerate(trade_analysis['relevant_news'][:3], 1):
            news_info += f"{i}. {news['title']}\n"
            news_info += f"   키워드: {', '.join(news.get('keywords', [])[:5])}\n"
            news_info += f"   관련도: {news.get('relevance_score', 0):.2f}\n\n"
        
        if not trade_analysis['relevant_news']:
            news_info = "해당 날짜에 관련 뉴스가 없습니다."
        
        # 프롬프트 생성 및 실행
        prompt = self.analysis_prompt.format(
            trade_info=trade_info,
            financial_metrics=financial_metrics,
            price_analysis=price_info,
            relevant_news=news_info
        )
        
        return self.hyperclova_llm._call(prompt)
    
    def analyze_trading_patterns(self, csv_file_path: str) -> str:
        """전체 매매 패턴 분석"""
        
        # 매매 데이터 로드
        trading_data = self.load_trading_data(csv_file_path)
        if trading_data.empty:
            return "매매 데이터를 로드할 수 없습니다."
        
        print(f"📈 총 {len(trading_data)}건의 매매 데이터 분석 시작...")
        
        # 각 매매 건별 분석
        analysis_results = []
        
        for idx, trade_row in trading_data.iterrows():
            try:
                # 단일 매매 분석
                trade_analysis = self.analyze_single_trade(trade_row)
                
                # 분석 리포트 생성
                analysis_report = self.generate_analysis_report(trade_analysis)
                
                analysis_results.append({
                    'trade_info': trade_analysis,
                    'analysis_report': analysis_report
                })
                
                print(f"✅ 완료: {idx+1}/{len(trading_data)}")
                
            except Exception as e:
                print(f"❌ 분석 오류 ({idx+1}): {e}")
                continue
        
        # 종합 분석 리포트 생성
        comprehensive_report = self.generate_comprehensive_report(analysis_results)
        
        return comprehensive_report
    
    def generate_comprehensive_report(self, analysis_results: List[Dict]) -> str:
        """종합 분석 리포트 생성"""
        
        if not analysis_results:
            return "분석할 데이터가 없습니다."
        
        report = f"""
# 🎯 투자 습관 종합 분석 리포트

## 📊 개요
- 분석 기간: {len(analysis_results)}건의 매매 내역
- 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 매매 패턴 분석
"""
        
        # 매매 통계
        total_trades = len(analysis_results)
        buy_count = sum(1 for r in analysis_results if r['trade_info']['trade_type'] == 'Buy')
        sell_count = total_trades - buy_count
        
        # 종목 분석
        symbols = [r['trade_info']['symbol'] for r in analysis_results]
        unique_symbols = list(set(symbols))
        symbol_counts = {symbol: symbols.count(symbol) for symbol in unique_symbols}
        
        report += f"""
### 매매 현황
- 총 매매 건수: {total_trades}건
- 매수: {buy_count}건 ({buy_count/total_trades*100:.1f}%)
- 매도: {sell_count}건 ({sell_count/total_trades*100:.1f}%)

### 거래 종목 현황
- 거래 종목 수: {len(unique_symbols)}개
- 주요 거래 종목:
"""
        
        # 상위 거래 종목
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_symbols[:5]:
            report += f"  - {symbol}: {count}건\n"
        
        # 가격 위치 분석
        price_positions = [r['trade_info']['price_analysis']['position'] for r in analysis_results]
        high_count = price_positions.count('high')
        middle_count = price_positions.count('middle')
        low_count = price_positions.count('low')
        
        report += f"""
### 매매 시점 분석
- 고가권 매매: {high_count}건 ({high_count/total_trades*100:.1f}%)
- 중간권 매매: {middle_count}건 ({middle_count/total_trades*100:.1f}%)
- 저가권 매매: {low_count}건 ({low_count/total_trades*100:.1f}%)
"""
        
        # 개별 매매 분석 요약
        report += "\n## 📝 개별 매매 분석\n\n"
        
        for i, result in enumerate(analysis_results, 1):
            trade_info = result['trade_info']
            report += f"### {i}. {trade_info['symbol']} - {trade_info['date']} ({trade_info['trade_type']})\n"
            report += f"**가격:** {trade_info['price']:,.0f}원 / **수량:** {trade_info['quantity']}주\n"
            report += f"**위치:** {trade_info['price_analysis']['position']} (백분위: {trade_info['price_analysis']['percentile']:.1f}%)\n"
            report += f"**재무지표:** PER {trade_info['financial_metrics']['PER']:.2f}, "
            report += f"PBR {trade_info['financial_metrics']['PBR']:.2f}, "
            report += f"ROE {trade_info['financial_metrics']['ROE']:.2f}%\n"
            
            if trade_info['relevant_news']:
                report += f"**관련 뉴스:** {len(trade_info['relevant_news'])}건\n"
            
            report += "\n**분석 결과:**\n"
            report += result['analysis_report']
            report += "\n" + "="*50 + "\n\n"
        
        # 종합 투자 습관 분석
        comprehensive_analysis = self.generate_final_comprehensive_analysis(analysis_results)
        report += f"\n## 🎯 종합 투자 습관 분석\n\n{comprehensive_analysis}"
        
        return report
    
    def generate_final_comprehensive_analysis(self, analysis_results: List[Dict]) -> str:
        """최종 종합 분석"""
        
        # 전체 매매 패턴 요약
        patterns_summary = self.summarize_trading_patterns(analysis_results)
        
        # 종합 분석 프롬프트
        comprehensive_prompt = f"""
당신은 투자 전문가로서 다음과 같은 매매 패턴 분석 결과를 토대로 투자자의 전반적인 투자 습관을 종합 분석해주세요:

## 매매 패턴 요약
{patterns_summary}

## 개별 매매 분석 결과
{self.extract_key_insights(analysis_results)}

위 정보를 종합하여 다음 관점에서 분석해주세요:

1. **투자 스타일 분석** - 가치투자, 성장투자, 기술적 분석 등 어떤 스타일인지
2. **매매 타이밍 분석** - 고점/저점 매매 패턴, 시장 타이밍 능력
3. **종목 선택 기준** - 어떤 기준으로 종목을 선택하는지 추론
4. **리스크 관리** - 포트폴리오 분산, 손절/익절 패턴
5. **강점과 약점** - 현재 투자 방식의 장단점
6. **개선 방안** - 구체적인 개선 제안

분석 결과를 체계적이고 실용적으로 제시해주세요.
"""
        
        return self.hyperclova_llm._call(comprehensive_prompt)
    
    def summarize_trading_patterns(self, analysis_results: List[Dict]) -> str:
        """매매 패턴 요약"""
        
        total_trades = len(analysis_results)
        
        # 매매 타입별 분석
        buy_trades = [r for r in analysis_results if r['trade_info']['trade_type'] == 'Buy']
        sell_trades = [r for r in analysis_results if r['trade_info']['trade_type'] == 'Sell']
        
        # 가격 위치 분석
        buy_positions = [r['trade_info']['price_analysis']['position'] for r in buy_trades]
        sell_positions = [r['trade_info']['price_analysis']['position'] for r in sell_trades]
        
        # 재무지표 분석
        avg_per = np.mean([r['trade_info']['financial_metrics']['PER'] for r in analysis_results if r['trade_info']['financial_metrics']['PER'] > 0])
        avg_pbr = np.mean([r['trade_info']['financial_metrics']['PBR'] for r in analysis_results if r['trade_info']['financial_metrics']['PBR'] > 0])
        avg_roe = np.mean([r['trade_info']['financial_metrics']['ROE'] for r in analysis_results if r['trade_info']['financial_metrics']['ROE'] > 0])
        
        summary = f"""
### 매매 패턴 통계
- 총 매매: {total_trades}건
- 매수: {len(buy_trades)}건, 매도: {len(sell_trades)}건

### 매수 패턴
- 고가권 매수: {buy_positions.count('high')}건
- 중간권 매수: {buy_positions.count('middle')}건  
- 저가권 매수: {buy_positions.count('low')}건

### 매도 패턴
- 고가권 매도: {sell_positions.count('high')}건
- 중간권 매도: {sell_positions.count('middle')}건
- 저가권 매도: {sell_positions.count('low')}건

### 평균 재무지표
- 평균 PER: {avg_per:.2f}
- 평균 PBR: {avg_pbr:.2f}
- 평균 ROE: {avg_roe:.2f}%
"""
        
        return summary
    
    def extract_key_insights(self, analysis_results: List[Dict]) -> str:
        """핵심 인사이트 추출"""
        
        insights = []
        
        for result in analysis_results:
            trade_info = result['trade_info']
            analysis = result['analysis_report']
            
            # 주요 특징 추출
            key_features = {
                'symbol': trade_info['symbol'],
                'trade_type': trade_info['trade_type'],
                'price_position': trade_info['price_analysis']['position'],
                'per': trade_info['financial_metrics']['PER'],
                'news_count': len(trade_info['relevant_news'])
            }
            
            insights.append(key_features)
        
        # 인사이트 요약
        summary = "주요 매매 특징:\n"
        for i, insight in enumerate(insights, 1):
            summary += f"{i}. {insight['symbol']} {insight['trade_type']} - "
            summary += f"{insight['price_position']}가권, PER {insight['per']:.1f}, "
            summary += f"뉴스 {insight['news_count']}건\n"
        
        return summary
    
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

# 사용 예시
def main():
    """메인 실행 함수"""
    
    # API 키 설정 (실제 사용 시 환경변수나 설정 파일에서 로드)
    HYPERCLOVA_API_KEY = "your_hyperclova_api_key"
    GOOGLE_API_KEY = "your_google_api_key"
    GOOGLE_SEARCH_ENGINE_ID = "your_google_search_engine_id"
    
    # OpenAI API 키 설정 (embeddings용)
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    
    # AI Agent 초기화
    agent = InvestmentAnalysisAgent(
        hyperclova_api_key=HYPERCLOVA_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_search_engine_id=GOOGLE_SEARCH_ENGINE_ID
    )
    
    # 매매 데이터 파일 경로
    csv_file_path = "trading_data.csv"
    
    try:
        print("🚀 투자 매매내역 분석 AI Agent 시작")
        print("="*60)
        
        # 분석 실행
        analysis_report = agent.analyze_trading_patterns(csv_file_path)
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 분석 완료!")
        print("="*60)
        print(analysis_report)
        
        # 리포트 저장
        saved_file = agent.save_analysis_report(analysis_report)
        
        print(f"\n✅ 분석 완료! 리포트 파일: {saved_file}")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 추가 유틸리티 함수들
class TradingDataValidator:
    """매매 데이터 검증"""
    
    @staticmethod
    def validate_csv_format(df: pd.DataFrame) -> bool:
        """CSV 형식 검증"""
        required_columns = ['Date', 'Symbol', 'TradeType', 'Price', 'Quantity']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ 필수 컬럼 누락: {col}")
                return False
        
        # 데이터 타입 검증
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price'] = pd.to_numeric(df['Price'])
            df['Quantity'] = pd.to_numeric(df['Quantity'])
        except Exception as e:
            print(f"❌ 데이터 타입 오류: {e}")
            return False
        
        # TradeType 검증
        valid_trade_types = ['Buy', 'Sell', 'buy', 'sell']
        if not df['TradeType'].isin(valid_trade_types).all():
            print("❌ TradeType은 'Buy' 또는 'Sell'이어야 합니다.")
            return False
        
        print("✅ CSV 형식 검증 완료")
        return True
    
    @staticmethod
    def clean_symbol_format(symbol: str) -> str:
        """종목 코드 형식 정리"""
        symbol = symbol.upper().strip()
        
        # 한국 종목 처리
        if symbol.isdigit():
            if len(symbol) == 6:
                # 6자리 숫자인 경우 KS/KQ 자동 판별 (임시)
                return f"{symbol}.KS"
            
        return symbol

class PerformanceTracker:
    """성과 추적"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.processing_times = {}
    
    def start_tracking(self, task_name: str):
        """추적 시작"""
        self.processing_times[task_name] = time.time()
    
    def end_tracking(self, task_name: str):
        """추적 종료"""
        if task_name in self.processing_times:
            elapsed = time.time() - self.processing_times[task_name]
            print(f"⏱️ {task_name}: {elapsed:.2f}초")
            return elapsed
        return 0
    
    def get_summary(self) -> str:
        """성과 요약"""
        total_time = sum(self.processing_times.values())
        summary = f"총 처리 시간: {total_time:.2f}초\n"
        
        for task, duration in self.processing_times.items():
            percentage = (duration / total_time) * 100
            summary += f"- {task}: {duration:.2f}초 ({percentage:.1f}%)\n"
        
        return summary

# 설정 파일 템플릿
CONFIG_TEMPLATE = """
# 투자 매매내역 분석 AI Agent 설정 파일

# API 키 설정
HYPERCLOVA_API_KEY = "your_hyperclova_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
GOOGLE_SEARCH_ENGINE_ID = "your_google_search_engine_id_here"
OPENAI_API_KEY = "your_openai_api_key_here"

# 데이터베이스 설정
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "investment_news"

# 분석 설정
ANALYSIS_PERIOD_DAYS = 252  # 1년
MAX_NEWS_PER_SYMBOL = 5
NEWS_RELEVANCE_THRESHOLD = 0.3

# 크롤링 설정
CRAWLING_DELAY = 1  # 초
REQUEST_TIMEOUT = 10  # 초
MAX_RETRIES = 3
"""

def create_config_file(filename: str = "config.py"):
    """설정 파일 생성"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(CONFIG_TEMPLATE)
    print(f"✅ 설정 파일 생성: {filename}")

def create_sample_data(filename: str = "sample_trading_data.csv"):
    """샘플 데이터 생성"""
    sample_data = {
        'Date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-02-15', '2024-03-01'],
        'Symbol': ['AAPL', '005930.KS', 'GOOGL', '000660.KS', 'TSLA'],
        'TradeType': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell'],
        'Price': [150.00, 75000, 2800.00, 45000, 220.00],
        'Quantity': [100, 50, 25, 80, 150]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(filename, index=False)
    print(f"✅ 샘플 데이터 생성: {filename}")

# 실행 도우미
def setup_environment():
    """환경 설정 도우미"""
    print("🔧 환경 설정 시작...")
    
    # 필요한 디렉토리 생성
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    
    # 설정 파일 생성 (없는 경우)
    if not os.path.exists("config.py"):
        create_config_file()
    
    # 샘플 데이터 생성 (없는 경우)
    if not os.path.exists("sample_trading_data.csv"):
        create_sample_data()
    
    print("✅ 환경 설정 완료!")
    print("📝 config.py 파일에 API 키를 설정하세요.")
    print("📊 sample_trading_data.csv 파일을 참고하여 데이터를 준비하세요.")

if __name__ == "__main__":
    # 환경 설정
    setup_environment()
    
    # 메인 실행
    main()