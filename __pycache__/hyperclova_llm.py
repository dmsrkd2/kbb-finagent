from langchain.llms.base import LLM
from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests

class HyperCLOVAXLLM(LLM):
    api_key: str
    api_url: str = "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005"


    @property
    def _llm_type(self) -> str:
        return "hyperclova_x"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                'Content-Type': 'application/json'
                'request_id'='44490bc020f14ec1819b12b86745b579'
            }

            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "topP": 0.8,
                "topK": 0,
                "maxTokens": 1000,
                "temperature": 0.4,
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
