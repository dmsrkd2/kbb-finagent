from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import warnings
import hashlib
from urllib.parse import urlparse

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_naver import ClovaXEmbeddings


# Chroma
import chromadb

warnings.filterwarnings('ignore')





class NewsVectorStore:
    """Chroma 벡터 DB를 사용한 뉴스 저장소"""
    
    def __init__(self, api_key, persist_directory: str = "./chroma_db", collection_name: str = "news_collection"):
        self.persist_directory = persist_directory  # 벡터 DB 데이터 저장할 로컬 폴더 경로
        self.collection_name = collection_name
        self.embeddings = ClovaXEmbeddings(
            model = "bge-m3",
            api_key = (api_key)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # 다음 청크와 겹치는 부분 길이
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
                metadata={"hnsw:space": "cosine"}   # 벡터간 유사도 측정 방식은 코사인 유사도를 사용
            )
        
        # LangChain Chroma 벡터스토어 초기화
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings, 
            persist_directory=persist_directory
        )
        
    
    def generate_content_hash(self, content: str) -> str:   # TODO 중복 방지 메서드 코사인 유사도나 TF-IDF 사용으로 바꾸기 
        """콘텐츠 해시 생성 (중복 방지용)"""
        return hashlib.md5(content.encode('utf-8')).hexdigest() # content 문자열을 MD5 해시값으로 변환
    
    def is_content_duplicate(self, content_hash: str) -> bool:
        """콘텐츠 중복 여부 확인"""
        try:
            results = self.collection.get(  # 메타데이터 기반 검색
                where={"content_hash": content_hash},
                limit=1 # 1개만 찾음
            )
            return len(results['ids']) > 0  # 검색 결과 1개 이상이면 중복
        except:
            return False
    
    def search_existing_news(self, symbol: str, date: str, query: str, k: int = 5) -> List[Dict]:
        """기존 뉴스 검색"""
        try:
            # 벡터 유사도 검색
            docs = self.vectorstore.similarity_search_with_score(
                query=f"{symbol} {date} {query}",   # 벡터 검색을 위한 쿼리를 종목, 날짜, 키워드로 구성
                k=k,
                filter={                # 메타데이터 필터링
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
        self.base_url = "https://www.googleapis.com/customsearch/v1"    # TODO 이거 맞는지 확인 필요
    
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
                        'snippet': item.get('snippet', ''), # 요약 본문
                        'source': urlparse(item.get('link', '')).netloc,    # 뉴스 출처
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
            response = requests.get(url, headers=self.headers, timeout=10)  # 10초 동안 서버 응답 기다림
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')   # html.parser 파싱 엔진으로 HTML 파싱
            
            # 일반적인 뉴스 본문 태그들
            content_selectors = [   # html 태그
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
            content = re.sub(r'\s+', ' ', content)  # 정규표현식 사용. 하나 이상의 공백문자에 대해서 ' '로 변환
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
    
    def extract_keywords(self, content: str, symbol: str) -> List[str]: # TODO 요약 로직도 생각해보기
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
