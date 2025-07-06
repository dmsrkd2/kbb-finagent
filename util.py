import pandas as pd
import time
import os
import sys


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

def check_env_file():
    if not os.path.exists(".env"):
        print("[ERROR] .env 파일이 존재하지 않습니다. .env.example을 복사하여 .env를 생성해주세요.")
        sys.exit(1)


# 실행 도우미
def setup_environment():
    """환경 설정 도우미"""
    print("🔧 환경 설정 시작...")
    
    # 필요한 디렉토리 생성
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    
    check_env_file()
    
    print("✅ 환경 설정 완료!")


