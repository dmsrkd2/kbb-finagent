import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacrotrendsDataCollector:
    def __init__(self):
        self.base_url = "https://www.macrotrends.net"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.company_names = {
            'AAPL': 'apple',
            'MSFT': 'microsoft',
            'GOOGL': 'alphabet',
            'GOOG': 'alphabet',
            'AMZN': 'amazon',
            'TSLA': 'tesla',
            'NVDA': 'nvidia',
            'META': 'meta-platforms',
            'NFLX': 'netflix',
            'ADBE': 'adobe'
        }
    

    
    def get_financial_ratios(self, ticker, metric_type='pe-ratio'):
        """
        특정 티커의 금융 비율 데이터 가져오기
        metric_type: 'pe-ratio' 또는 'price-book-ratio'
        """
        # Macrotrends URL 패턴
        company_name = self.company_names.get(ticker,"알 수 없음")
        url = f"{self.base_url}/stocks/charts/{ticker}/{company_name}/{metric_type}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # JavaScript에서 데이터 추출
            scripts = soup.find_all('script')
            data = None
            
            for script in scripts:
                if script.string and 'chart_data' in script.string:
                    # 정규식으로 차트 데이터 추출
                    pattern = r'var chart_data = (\[.*?\]);'
                    match = re.search(pattern, script.string, re.DOTALL)
                    if match:
                        data_str = match.group(1)
                        data = json.loads(data_str)
                        break
            
            if not data:
                logger.warning(f"{ticker} {metric_type} 데이터를 찾을 수 없습니다.")
                return None
            
            # 데이터 파싱
            parsed_data = []
            for item in data:
                if len(item) >= 2:
                    # timestamp를 날짜로 변환 (밀리초 단위)
                    timestamp = item[0] / 1000
                    date = datetime.fromtimestamp(timestamp)
                    value = item[1]
                    
                    parsed_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'quarter': f"{date.year}Q{(date.month-1)//3 + 1}",
                        'value': value
                    })
            
            return parsed_data
            
        except requests.RequestException as e:
            logger.error(f"HTTP 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"데이터 파싱 실패: {e}")
            return None
    
    def get_per_pbr_data(self, ticker, target_date=None):
        """
        특정 티커의 PER, PBR 데이터 가져오기
        target_date: 'YYYY-MM-DD' 형식 또는 None (최신 데이터)
        """
        logger.info(f"{ticker} 데이터 수집 시작...")
        
        # PER 데이터 가져오기
        per_data = self.get_financial_ratios(ticker, 'pe-ratio')
        time.sleep(1)  # API 호출 간격 조절
        
        # PBR 데이터 가져오기
        pbr_data = self.get_financial_ratios(ticker, 'price-book-ratio')
        
        if not per_data or not pbr_data:
            logger.error(f"{ticker} 데이터 수집 실패")
            return None
        
        # 데이터 병합
        result = []
        per_dict = {item['date']: item for item in per_data}
        pbr_dict = {item['date']: item for item in pbr_data}
        
        # 공통 날짜 찾기
        common_dates = set(per_dict.keys()) & set(pbr_dict.keys())
        
        for date in sorted(common_dates):
            per_item = per_dict[date]
            pbr_item = pbr_dict[date]
            
            result.append({
                'ticker': ticker,
                'date': date,
                'quarter': per_item['quarter'],
                'per': per_item['value'],
                'pbr': pbr_item['value']
            })
        
        # 특정 날짜가 지정된 경우 필터링
        if target_date:
            target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
            # 가장 가까운 날짜 찾기
            closest_item = min(result, 
                             key=lambda x: abs(datetime.strptime(x['date'], '%Y-%m-%d') - target_datetime))
            return [closest_item]
        
        return result

def main():
    """사용 예시"""
    collector = MacrotrendsDataCollector()
    
    # 단일 종목 데이터 수집
    ticker = "AAPL"  # Apple Inc.
    target_date = "2023-12-31"  # 특정 날짜 (선택사항)
    
    print(f"\n=== {ticker} PER/PBR 데이터 수집 ===")
    data = collector.get_per_pbr_data(ticker, target_date)
    
    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # CSV 저장
        filename = f"{ticker}_per_pbr_data.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n데이터가 {filename}에 저장되었습니다.")
    else:
        print("데이터 수집에 실패했습니다.")
    
    # 여러 종목 처리 예시
    tickers = ["MSFT", "GOOGL", "AMZN"]
    all_data = []
    
    print(f"\n=== 여러 종목 데이터 수집 ===")
    for ticker in tickers:
        print(f"{ticker} 처리 중...")
        data = collector.get_per_pbr_data(ticker)
        if data:
            # 최신 데이터만 가져오기
            latest_data = max(data, key=lambda x: x['date'])
            all_data.append(latest_data)
        
        time.sleep(2)  # 요청 간격 조절
    
    if all_data:
        df_all = pd.DataFrame(all_data)
        print("\n=== 수집된 데이터 ===")
        print(df_all.to_string(index=False))
        
        # 전체 데이터 CSV 저장
        df_all.to_csv("nasdaq_per_pbr_summary.csv", index=False, encoding='utf-8-sig')
        print("\n전체 데이터가 nasdaq_per_pbr_summary.csv에 저장되었습니다.")

def get_quarterly_data(ticker, year, quarter):
    """
    특정 연도와 분기의 데이터 가져오기
    year: 연도 (예: 2023)
    quarter: 분기 (1, 2, 3, 4)
    """
    collector = MacrotrendsDataCollector()
    
    # 분기별 대략적인 날짜 매핑
    quarter_dates = {
        1: f"{year}-03-31",
        2: f"{year}-06-30", 
        3: f"{year}-09-30",
        4: f"{year}-12-31"
    }
    
    target_date = quarter_dates.get(quarter)
    if not target_date:
        print("올바르지 않은 분기입니다. 1, 2, 3, 4 중 선택하세요.")
        return None
    
    print(f"{ticker} {year}년 {quarter}분기 데이터 수집...")
    data = collector.get_per_pbr_data(ticker, target_date)
    
    if data:
        return data[0]  # 가장 가까운 데이터 반환
    return None

if __name__ == "__main__":
    # 기본 실행
    main()
    
    # 특정 분기 데이터 예시
    print("\n=== 특정 분기 데이터 예시 ===")
    quarterly_data = get_quarterly_data("AAPL", 2023, 4)
    if quarterly_data:
        print(f"AAPL 2023년 4분기 데이터:")
        for key, value in quarterly_data.items():
            print(f"{key}: {value}")