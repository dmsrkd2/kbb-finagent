import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_dummy_trading_data():
    """실제 투자 패턴을 반영한 더미 매매 데이터 생성"""
    
    # 한국 주식 (코스피, 코스닥)
    korean_stocks = [
        '005930.KS',  # 삼성전자
        '000660.KS',  # SK하이닉스
        '035420.KS',  # NAVER
        '051910.KS',  # LG화학
        '006400.KS',  # 삼성SDI
        '035720.KS',  # 카카오
        '207940.KS',  # 삼성바이오로직스
        '373220.KS',  # LG에너지솔루션
        '068270.KS',  # 셀트리온
        '012330.KS',  # 현대모비스
        '028260.KS',  # 삼성물산
        '096770.KS',  # SK이노베이션
        '323410.KS',  # 카카오뱅크
        '086520.KS',  # 에코프로
        '247540.KS',  # 에코프로비엠
        '042700.KS',  # 한미반도체
        '403870.KS',  # HPSP
        '450080.KS',  # 에코프로머티
        '196170.KQ',  # 알테오젠
        '091990.KQ',  # 셀트리온헬스케어
        '141080.KQ',  # 레고켐바이오
        '091700.KQ',  # 파트론
        '058470.KQ',  # 리노공업
        '060310.KQ',  # 3S
        '064550.KQ',  # 바이오니아
        '900140.KQ',  # 엘브이엠씨홀딩스
        '317330.KQ',  # 덕산테코피아
        '101490.KQ',  # 에스앤에스텍
        '950140.KQ',  # 잉글우드랩
        '222800.KQ',  # 심텍
    ]
    
    # 나스닥 주식
    nasdaq_stocks = [
        'AAPL',    # 애플
        'MSFT',    # 마이크로소프트
        'GOOGL',   # 구글
        'AMZN',    # 아마존
        'TSLA',    # 테슬라
        'NVDA',    # 엔비디아
        'META',    # 메타
        'NFLX',    # 넷플릭스
        'ADBE',    # 어도비
        'INTC',    # 인텔
        'AMD',     # AMD
        'QCOM',    # 퀄컴
        'AVGO',    # 브로드컴
        'CSCO',    # 시스코
        'ORCL',    # 오라클
        'CRM',     # 세일즈포스
        'PYPL',    # 페이팔
        'UBER',    # 우버
        'ZOOM',    # 줌
        'ROKU',    # 로쿠
    ]
    
    # 전체 종목 리스트
    all_stocks = korean_stocks + nasdaq_stocks
    
    # 4개월 기간 설정 (2024년 1월 ~ 4월)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 4, 30)
    
    # 매매 데이터 생성
    trading_data = []
    
    # 각 종목별로 1-3개의 매매 기록 생성
    for stock in all_stocks:
        num_trades = random.randint(1, 3)
        
        for _ in range(num_trades):
            # 랜덤 거래 날짜 생성 (주말 제외)
            trade_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            
            # 주말이면 월요일로 조정
            if trade_date.weekday() >= 5:  # 토요일(5) 또는 일요일(6)
                trade_date = trade_date + timedelta(days=(7 - trade_date.weekday()))
            
            # 매매 타입 (Buy/Sell 비율 조정)
            trade_type = random.choice(['Buy', 'Buy', 'Buy', 'Sell', 'Sell'])  # 매수 비중 높게
            
            # 가격 설정 (종목별 실제 가격대 반영)
            if stock.endswith('.KS') or stock.endswith('.KQ'):
                # 한국 주식 가격 (원)
                if stock in ['005930.KS', '207940.KS', '373220.KS']:  # 고가 종목
                    price = random.randint(60000, 120000)
                elif stock in ['035420.KS', '035720.KS', '068270.KS']:  # 중고가 종목
                    price = random.randint(30000, 80000)
                else:  # 일반 종목
                    price = random.randint(10000, 50000)
            else:
                # 나스닥 주식 가격 (달러)
                if stock in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:  # 고가 종목
                    price = round(random.uniform(150, 300), 2)
                elif stock in ['TSLA', 'NVDA', 'META', 'NFLX']:  # 중고가 종목
                    price = round(random.uniform(100, 250), 2)
                else:  # 일반 종목
                    price = round(random.uniform(50, 150), 2)
            
            # 수량 설정
            if stock.endswith('.KS') or stock.endswith('.KQ'):
                # 한국 주식 - 가격에 반비례하여 수량 조정
                if price > 80000:
                    quantity = random.randint(1, 10)
                elif price > 40000:
                    quantity = random.randint(5, 20)
                else:
                    quantity = random.randint(10, 50)
            else:
                # 나스닥 주식
                if price > 200:
                    quantity = random.randint(1, 10)
                elif price > 100:
                    quantity = random.randint(5, 25)
                else:
                    quantity = random.randint(10, 50)
            
            trading_data.append({
                'Date': trade_date.strftime('%Y-%m-%d'),
                'Symbol': stock,
                'TradeType': trade_type,
                'Price': price,
                'Quantity': quantity
            })
    
    # DataFrame 생성 및 날짜순 정렬
    df = pd.DataFrame(trading_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def add_realistic_trading_patterns(df):
    """실제 투자 패턴을 반영한 추가 매매 기록"""
    
    # 동일 종목 매수 후 매도 패턴 추가
    additional_trades = []
    
    # 일부 종목에 대해 매수 후 매도 패턴 생성
    popular_stocks = ['005930.KS', 'AAPL', 'TSLA', '035420.KS', 'NVDA']
    
    for stock in popular_stocks:
        # 매수 날짜
        buy_date = datetime(2024, 2, 15) + timedelta(days=random.randint(0, 30))
        
        # 매도 날짜 (매수 후 5-30일 후)
        sell_date = buy_date + timedelta(days=random.randint(5, 30))
        
        # 가격 설정 (매수 < 매도 또는 매수 > 매도)
        if stock.endswith('.KS') or stock.endswith('.KQ'):
            buy_price = random.randint(50000, 90000)
            # 70% 확률로 수익, 30% 확률로 손실
            if random.random() < 0.7:
                sell_price = int(buy_price * random.uniform(1.02, 1.15))  # 2-15% 수익
            else:
                sell_price = int(buy_price * random.uniform(0.85, 0.98))  # 2-15% 손실
        else:
            buy_price = round(random.uniform(100, 200), 2)
            if random.random() < 0.7:
                sell_price = round(buy_price * random.uniform(1.02, 1.15), 2)
            else:
                sell_price = round(buy_price * random.uniform(0.85, 0.98), 2)
        
        quantity = random.randint(10, 30)
        
        # 매수 기록
        additional_trades.append({
            'Date': buy_date.strftime('%Y-%m-%d'),
            'Symbol': stock,
            'TradeType': 'Buy',
            'Price': buy_price,
            'Quantity': quantity
        })
        
        # 매도 기록
        additional_trades.append({
            'Date': sell_date.strftime('%Y-%m-%d'),
            'Symbol': stock,
            'TradeType': 'Sell',
            'Price': sell_price,
            'Quantity': quantity
        })
    
    # 추가 매매 기록을 DataFrame에 병합
    if additional_trades:
        additional_df = pd.DataFrame(additional_trades)
        df = pd.concat([df, additional_df], ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def create_csv_file():
    """CSV 파일 생성"""
    
    # 더미 데이터 생성
    print("더미 매매 데이터 생성 중...")
    df = create_dummy_trading_data()
    
    # 실제 투자 패턴 추가
    print("실제 투자 패턴 추가 중...")
    df = add_realistic_trading_patterns(df)
    
    # CSV 파일로 저장
    filename = 'trading_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ 테스트용 CSV 파일이 생성되었습니다: {filename}")
    print(f"📊 총 매매 기록: {len(df)}건")
    
    # 데이터 요약 정보 출력
    print("\n📈 데이터 요약:")
    print(f"- 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"- 종목 수: {df['Symbol'].nunique()}개")
    print(f"- 매수 건수: {len(df[df['TradeType'] == 'Buy'])}건")
    print(f"- 매도 건수: {len(df[df['TradeType'] == 'Sell'])}건")
    
    # 상위 5개 종목 매매 현황
    print("\n🔝 상위 5개 종목 매매 현황:")
    top_stocks = df['Symbol'].value_counts().head()
    for stock, count in top_stocks.items():
        print(f"- {stock}: {count}건")
    
    # 샘플 데이터 출력
    print("\n📋 샘플 데이터 (최근 10건):")
    print(df.head(10).to_string(index=False))
    
    return df

# 실행
if __name__ == "__main__":
    df = create_csv_file()