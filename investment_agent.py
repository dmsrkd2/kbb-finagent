import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import time
from RAGsearcher import NaverBlogSearcher, NewsContentCrawler



class FinancialDataProcessor:
    """재무 데이터 처리 클래스"""
    
    def __init__(self, stock_pbr_csv_path: str, sector_pbr_csv_path: str):
        self.stock_pbr_csv_path = stock_pbr_csv_path
        self.sector_pbr_csv_path = sector_pbr_csv_path
        self.stock_data = None
        self.sector_data = None
        self.load_financial_data()
    
    def load_financial_data(self):
        """재무 데이터 로드"""
        try:
            # 종목별 PBR/PER 데이터 로드
            self.stock_data = pd.read_csv(self.stock_pbr_csv_path)
            print(f"✅ 종목별 재무 데이터 로드 완료: {len(self.stock_data)}개 종목")
            
            # 섹터별 평균 PBR/PER 데이터 로드
            self.sector_data = pd.read_csv(self.sector_pbr_csv_path)
            print(f"✅ 섹터별 재무 데이터 로드 완료: {len(self.sector_data)}개 섹터")
            
        except Exception as e:
            print(f"❌ 재무 데이터 로드 오류: {e}")
            self.stock_data = None
            self.sector_data = None
    
    def get_quarter_column(self, date_str: str, metric: str) -> str:
        """날짜를 기준으로 분기 컬럼명 생성"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            quarter = (date_obj.month - 1) // 3 + 1
            return f"{year}년 {quarter}분기_{metric}"
        except:
            return f"24년 1분기_{metric}"  # 기본값
        
    def symbol_to_company_name(self, symbol: str) -> str:
        if self.stock_data is not None:
            match = self.stock_data[self.stock_data["Ticker"] == symbol]
            if not match.empty:
                return match.iloc[0]["Name"]
        return symbol  # fallback
    
    def get_stock_financial_data(self, symbol: str, first_trade_date: str) -> Dict:
        """종목의 PBR/PER 데이터 조회"""
        if self.stock_data is None:
            return {'pbr': 0, 'per': 0, 'sector': 'Unknown', 'sector_pbr': 0, 'sector_per': 0}
        
        try:
            # 종목명으로 데이터 찾기 (Ticker 컬럼 사용)
            stock_row = self.stock_data[self.stock_data['Ticker'] == symbol]
            
            if stock_row.empty:
                print(f"⚠️ {symbol} 재무 데이터 없음")
                return {'pbr': 0, 'per': 0, 'sector': 'Unknown', 'sector_pbr': 0, 'sector_per': 0}
            
            stock_row = stock_row.iloc[0]
            sector = stock_row.get('SECTOR', 'Unknown')
            
            # 해당 분기의 PBR/PER 컬럼명 생성
            pbr_column = self.get_quarter_column(first_trade_date, 'PBR')
            per_column = self.get_quarter_column(first_trade_date, 'PER')
            
            # 종목 PBR/PER 값
            stock_pbr = stock_row.get(pbr_column, 0)
            stock_per = stock_row.get(per_column, 0)
            
            # 섹터 평균 PBR/PER 값
            sector_pbr, sector_per = self.get_sector_averages(sector, pbr_column, per_column)
            
            return {
                'pbr': stock_pbr,
                'per': stock_per,
                'sector': sector,
                'sector_pbr': sector_pbr,
                'sector_per': sector_per,
                'quarter': pbr_column.split('_')[0]
            }
            
        except Exception as e:
            print(f"❌ 재무 데이터 처리 오류 ({symbol}): {e}")
            return {'pbr': 0, 'per': 0, 'sector': 'Unknown', 'sector_pbr': 0, 'sector_per': 0}
    
    def get_sector_averages(self, sector: str, pbr_column: str, per_column: str) -> tuple:
        """섹터 평균 PBR/PER 조회"""
        if self.sector_data is None:
            return 0, 0
        
        try:
            sector_row = self.sector_data[self.sector_data['SECTOR'] == sector]
            
            if sector_row.empty:
                return 0, 0
            
            sector_row = sector_row.iloc[0]
            sector_pbr = sector_row.get(pbr_column, 0)
            sector_per = sector_row.get(per_column, 0)
            
            return sector_pbr, sector_per
            
        except Exception as e:
            print(f"❌ 섹터 평균 조회 오류: {e}")
            return 0, 0


class NewsAnalyzer:
    """뉴스 분석 클래스"""
    
    def __init__(self, naver_client_id: str, naver_client_secret: str):
        self.news_searcher = NaverBlogSearcher(naver_client_id, naver_client_secret)
        self.news_crawler = NewsContentCrawler()
    
    def get_relevant_news(self, symbol: str, first_trade_date: str, max_news: int = 2) -> str:
        """관련 뉴스 수집 및 분석"""
        try:
            # 날짜 포맷 변환 (2024-03-12 -> 2024년 3월 12일)
            date_obj = datetime.strptime(first_trade_date, '%Y-%m-%d')
            formatted_date = f"{date_obj.year}년 {date_obj.month}월 {date_obj.day}일"
            
            # 뉴스 검색
            news_list = self.news_searcher.search_news(symbol, formatted_date)
            
            newscrawler = NewsContentCrawler()
            prompt_content = ""
            if news_list:
                successful_crawls = 0
                for i, news in enumerate(news_list, 1):
                    print(f"\n=== {i}. {news['title']} ===")
                    print(f"URL: {news['url']}")
                    
                    content = newscrawler.crawl_news_content(news["url"])
                    if content:
                        prompt_content += content +"\n"
                        print(f"내용 (처음 200자): {content[:200]}...")
                        successful_crawls += 1
                    else:
                        print("내용을 가져올 수 없습니다.")
                    
                    # 요청 간 딜레이
                    if i < len(news_list):
                        time.sleep(2)
                
                print(f"\n총 {len(news_list)}개 중 {successful_crawls}개 성공적으로 크롤링")
                return prompt_content
            else:
                print("검색 결과가 없습니다.")
                return ""
            
        except Exception as e:
            print(f"❌ 뉴스 분석 오류 ({symbol}): {e}")
            return []
    

class InvestmentAnalysisAgent:
    """개선된 투자 매매내역 분석 AI Agent"""
    
    def __init__(self, naver_client_id: str, naver_client_secret: str, 
                 stock_pbr_csv_path: str, sector_pbr_csv_path: str):
        
        # 데이터 처리 클래스 초기화
        self.financial_processor = FinancialDataProcessor(stock_pbr_csv_path, sector_pbr_csv_path)
        self.news_analyzer = NewsAnalyzer(naver_client_id, naver_client_secret)
    
    def load_trading_data(self, csv_file_path: str) -> pd.DataFrame:
        """CSV 파일에서 매매 데이터 로드"""
        try:
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"✅ 매매 데이터 로드 완료: {len(df)}건")
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            return pd.DataFrame()
    
    def get_top_symbols_by_volume(self, trading_data: pd.DataFrame, top_n: int = 3) -> List[str]:
        """거래량 기준 상위 종목 선정"""
        symbol_volumes = trading_data.groupby('Symbol')['Quantity'].sum().sort_values(ascending=False)
        top_symbols = symbol_volumes.head(top_n).index.tolist()
        
        print(f"📊 거래량 상위 {top_n}개 종목:")
        for i, symbol in enumerate(top_symbols, 1):
            volume = symbol_volumes[symbol]
            print(f"  {i}. {symbol}: {volume:,}주")
        
        return top_symbols
    

    def calculate_trading_metrics(self, symbol: str, trading_data: pd.DataFrame) -> Dict:
        """종목별 매매 지표 계산"""
        symbol_data = trading_data[trading_data['Symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('Date')
        
        buy_trades = symbol_data[symbol_data['TradeType'] == 'BUY']
        sell_trades = symbol_data[symbol_data['TradeType'] == 'SELL']
        
        # 기본 통계
        total_buy_amount = (buy_trades['Price'] * buy_trades['Quantity']).sum()
        total_buy_quantity = buy_trades['Quantity'].sum()
        total_sell_amount = (sell_trades['Price'] * sell_trades['Quantity']).sum()
        total_sell_quantity = sell_trades['Quantity'].sum()
        
        # 평단가 계산
        buy_avg_price = total_buy_amount / total_buy_quantity if total_buy_quantity > 0 else 0
        sell_avg_price = total_sell_amount / total_sell_quantity if total_sell_quantity > 0 else 0
        
        # 수익률 계산
        profit_loss = total_sell_amount - total_buy_amount
        profit_rate = (profit_loss / total_buy_amount * 100) if total_buy_amount > 0 else 0
        
        # 보유 기간 계산
        if not symbol_data.empty:
            first_date = symbol_data.iloc[0]['Date']
            last_date = symbol_data.iloc[-1]['Date']
            holding_period = (last_date - first_date).days
        else:
            holding_period = 0
        
        return {
            'symbol': symbol,
            'total_trades': len(symbol_data),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'total_buy_quantity': total_buy_quantity,
            'total_sell_quantity': total_sell_quantity,
            'buy_avg_price': buy_avg_price,
            'sell_avg_price': sell_avg_price,
            'total_buy_amount': total_buy_amount,
            'total_sell_amount': total_sell_amount,
            'profit_loss': profit_loss,
            'profit_rate': profit_rate,
            'holding_period': holding_period,
            'first_trade_date': first_date.strftime('%Y-%m-%d'),
            'last_trade_date': last_date.strftime('%Y-%m-%d'),
            'trade_records': symbol_data.to_dict('records')
        }
    
    def analyze_trading_patterns(self, csv_file_path: str) -> str:
        """매매 패턴 종합 분석"""
        
        # 1. 매매 데이터 로드
        trading_data = self.load_trading_data(csv_file_path)
        if trading_data.empty:
            return "❌ 매매 데이터를 로드할 수 없습니다."
        
        # 2. 상위 3개 종목 선정
        top_symbols = self.get_top_symbols_by_volume(trading_data, 3)
        
        if not top_symbols:
            return "❌ 분석할 종목이 없습니다."
        
        print(f"🔍 상위 3개 종목 분석 시작...")
        
        # 3. 종목별 상세 분석
        symbol_analyses = {}
        
        for symbol in top_symbols:
            print(f"\n📈 {symbol} 분석 중...")
            
            # 매매 지표 계산
            trading_metrics = self.calculate_trading_metrics(symbol, trading_data)
            
            # 재무 데이터 조회
            financial_data = self.financial_processor.get_stock_financial_data(
                symbol, trading_metrics['first_trade_date']
            )
            
            # 뉴스 분석
            company_name = self.financial_processor.symbol_to_company_name(symbol)
            news_data = self.news_analyzer.get_relevant_news(
                company_name, trading_metrics['first_trade_date'], max_news=2
            )
            
            symbol_analyses[symbol] = {
                'trading_metrics': trading_metrics,
                'financial_data': financial_data,
                'news_data': news_data
            }
        
        # 4. 종합 리포트 생성
        analysis_report = self.generate_analysis_report(symbol_analyses)
        
        return analysis_report
    
    def generate_analysis_report(self, symbol_analyses: Dict) -> str:
        """종합 분석 리포트 생성"""
        
        report = []
        report.append("# 🔍 투자 매매 습관 분석 리포트")
        report.append("=" * 80)
        report.append(f"📅 분석 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
        report.append("")
        
        # 1. 전체 개요
        report.append("## 📊 분석 개요")
        report.append("")
        
        total_symbols = len(symbol_analyses)
        total_profit = sum(data['trading_metrics']['profit_loss'] for data in symbol_analyses.values())
        total_investment = sum(data['trading_metrics']['total_buy_amount'] for data in symbol_analyses.values())
        overall_return = (total_profit / total_investment * 100) if total_investment > 0 else 0
        
        report.append(f"- **분석 대상 종목**: {total_symbols}개")
        report.append(f"- **총 투자금액**: {total_investment:,.0f}원")
        report.append(f"- **총 손익**: {total_profit:,.0f}원")
        report.append(f"- **전체 수익률**: {overall_return:.2f}%")
        report.append("")
        
        # 2. 종목별 상세 분석
        report.append("## 📈 종목별 상세 분석")
        report.append("")
        
        for i, (symbol, data) in enumerate(symbol_analyses.items(), 1):
            tm = data['trading_metrics']
            fd = data['financial_data']
            nd = data['news_data']
            
            report.append(f"### {i}. {symbol}")
            report.append("")
            
            # 매매 현황
            report.append("**📊 매매 현황**")
            report.append(f"- 총 거래 횟수: {tm['total_trades']}회 (매수 {tm['buy_count']}회, 매도 {tm['sell_count']}회)")
            report.append(f"- 매수 평단가: {tm['buy_avg_price']:,.0f}원")
            report.append(f"- 매도 평단가: {tm['sell_avg_price']:,.0f}원")
            report.append(f"- 투자 기간: {tm['holding_period']}일")
            report.append(f"- 손익: {tm['profit_loss']:,.0f}원 ({tm['profit_rate']:+.2f}%)")
            report.append("")
            
            # 재무 분석
            report.append("**💰 재무 분석**")
            report.append(f"- 섹터: {fd['sector']}")
            report.append(f"- 종목 PBR: {fd['pbr']:.2f} vs 섹터 평균: {fd['sector_pbr']:.2f}")
            report.append(f"- 종목 PER: {fd['per']:.2f} vs 섹터 평균: {fd['sector_per']:.2f}")
            
            # PBR/PER 평가
            if fd['pbr'] > 0 and fd['sector_pbr'] > 0:
                pbr_ratio = fd['pbr'] / fd['sector_pbr']
                if pbr_ratio > 1.2:
                    report.append(f"- PBR 평가: 섹터 대비 고평가 ({pbr_ratio:.2f}배)")
                elif pbr_ratio < 0.8:
                    report.append(f"- PBR 평가: 섹터 대비 저평가 ({pbr_ratio:.2f}배)")
                else:
                    report.append(f"- PBR 평가: 섹터 대비 적정 수준 ({pbr_ratio:.2f}배)")
            report.append("")
            
            # 뉴스 분석
            report.append("**📰 관련 뉴스**")
            if nd:
                report.append(f"{nd}")
                report.append("")
            else:
                report.append("- 관련 뉴스 없음")
                report.append("")
        

        # 수익률 패턴 분석
        profit_symbols = [s for s, d in symbol_analyses.items() if d['trading_metrics']['profit_rate'] > 0]
        loss_symbols = [s for s, d in symbol_analyses.items() if d['trading_metrics']['profit_rate'] < 0]
        
        report.append("**📊 수익 패턴**")
        report.append(f"- 수익 종목: {len(profit_symbols)}개 ({len(profit_symbols)/total_symbols*100:.1f}%)")
        report.append(f"- 손실 종목: {len(loss_symbols)}개 ({len(loss_symbols)/total_symbols*100:.1f}%)")
        report.append("")
        
        # 보유 기간 분석
        avg_holding = np.mean([d['trading_metrics']['holding_period'] for d in symbol_analyses.values()])
        report.append(f"**⏰ 보유 기간**")
        report.append(f"- 평균 보유 기간: {avg_holding:.1f}일")
        

        return "\n".join(report)
