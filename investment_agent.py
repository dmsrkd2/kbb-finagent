import news_vector_store
import stock_data_analyzer
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
    
    def __init__(self, hyperclova_api_key: str, google_api_key: str, google_search_engine_id: str):
        self.hyperclova_llm = hyperclova_llm.HyperCLOVAXLLM(api_key = hyperclova_api_key)
        self.news_vectorstore = news_vector_store.NewsVectorStore(api_key = hyperclova_api_key)
        self.google_searcher = news_vector_store.GoogleNewsSearcher(google_api_key, google_search_engine_id)
        self.news_crawler = news_vector_store.NewsContentCrawler()
        self.stock_analyzer = stock_data_analyzer.StockDataAnalyzer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # 다음 청크와 겹치는 부분 길이
            length_function=len
        )
        
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
    
    def analyze_single_trade(self, trade_row: pd.Series) -> Dict:
        """단일 매매 건 분석"""
        symbol = trade_row['Symbol']
        date = datetime.strptime(trade_row['Date'], '%Y-%m-%d')
        date = date.strftime('%Y-%m-%d')
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
    
    def generate_analysis_report(self, trade_analysis: Dict) -> str:    # 여기 나중에 오류 생길만 한데 어짜피 바꿔야해서 냅둘게
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
