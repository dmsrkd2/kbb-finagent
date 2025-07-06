import requests
from typing import List, Optional
import warnings

# LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun



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