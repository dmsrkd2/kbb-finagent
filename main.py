import util
import investment_agent
import os
from dotenv import load_dotenv


def main():
    """메인 실행 함수"""
    
    # API 키 설정
    load_dotenv()
    HYPERCLOVA_API_KEY = os.getenv("HYPERCLOVA_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    # 매매 데이터 파일 경로
    csv_file_path = "trading_data.csv"
    pbr_csv_path = "pbr_data.csv"

    # AI Agent 초기화
    agent = investment_agent.InvestmentAnalysisAgent(
        hyperclova_api_key=HYPERCLOVA_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_search_engine_id=GOOGLE_SEARCH_ENGINE_ID
        pbr_csv_path = pbr_csv_path
    )
    

    
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
    # 환경 설정
    util.setup_environment()
    
    # 메인 실행
    main()