import os
import sys
from datetime import datetime
from investment_agent import InvestmentAnalysisAgent
from langchain_naver import ChatClovaX
from langchain.schema import SystemMessage, HumanMessage



def main():
    """메인 실행 함수"""
    
    print("🚀 투자 매매내역 분석 AI Agent 시작")
    print("=" * 60)
    
    # API 키 및 설정 정보
    NAVER_CLIENT_ID = "{여기에 API 클라이언트 ID}"  # 네이버 검색 API 클라이언트 ID
    NAVER_CLIENT_SECRET = "{여기에 API 클라이언트 시크릿}"  # 네이버 검색 API 클라이언트 시크릿
    
    # 데이터 파일 경로
    TRADING_CSV_PATH = "모멘텀 투자_분할매수매도O_보유기간5일_수익률+10%.csv"  # 매매 데이터
    STOCK_PBR_CSV_PATH = "종목별 per pbr.csv"  # 종목별 PBR/PER 데이터
    SECTOR_PBR_CSV_PATH = "섹터별 pbr 중앙값.csv"  # 섹터별 평균 PBR/PER 데이터
    
    # 파일 존재 여부 확인
    required_files = [TRADING_CSV_PATH, STOCK_PBR_CSV_PATH, SECTOR_PBR_CSV_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ 다음 파일들이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n필요한 파일들을 현재 디렉토리에 배치한 후 다시 실행해주세요.")
        return
    
    try:
        # AI Agent 초기화
        print("🔧 AI Agent 초기화 중...")
        agent = InvestmentAnalysisAgent(
            naver_client_id=NAVER_CLIENT_ID,
            naver_client_secret=NAVER_CLIENT_SECRET,
            stock_pbr_csv_path=STOCK_PBR_CSV_PATH,
            sector_pbr_csv_path=SECTOR_PBR_CSV_PATH
        )
        prompt = ""
        
        chat = ChatClovaX(
            model = "HCX-005", # 모델명 입력
            api_key = "{클로바 API 키 직접 넣으면 됩니다.}}"
        )
        messages = [
        SystemMessage(content = 
                "당신은 매매내역을 기반으로 투자 습관을 분석해주는 전문가입니다. 주어진 자료들을 가지고 사용자가 투자한 근거를 추론하여 해당 매매 방식의 장점, 단점, 개선방안 등을"
                 "리포트로 작성하세요."),
        HumanMessage(content="내 매매내역을 분석해줘")
        ]

        
        print("✅ 초기화 완료!")
        print("-" * 60)
        
        # 매매 데이터 분석 실행
        print(f"📊 매매 데이터 분석 시작: {TRADING_CSV_PATH}")
        analysis_report = agent.analyze_trading_patterns(TRADING_CSV_PATH)
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("📊 분석 완료!")
        print("=" * 60)
        print(analysis_report)
        
        # 리포트 저장
        print("\n" + "-" * 60)
        print(analysis_report)
        print("\n" + "-" * 60)
        ai_msg = chat.invoke(messages+[HumanMessage(content=analysis_report)])
        print(ai_msg)
        
        print("\n🎉 모든 작업이 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        print("\n🔍 오류 상세 정보:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
