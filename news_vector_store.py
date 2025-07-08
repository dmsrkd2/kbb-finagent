from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import warnings
from urllib.parse import urlparse

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_naver import ClovaXEmbeddings
from langchain.schema import Document


# Chroma
import chromadb

warnings.filterwarnings('ignore')



class NewsVectorStore:
    """Chroma 벡터 DB를 사용한 뉴스 저장소"""
    
    def __init__(self, api_key, persist_directory: str = "./chroma_db", collection_name: str = "news_collection"):
        self.persist_directory = persist_directory  # 벡터 DB 데이터 저장할 로컬 폴더 경로
        self.collection_name = collection_name
        self.api_key = api_key
        self.embeddings = ClovaXEmbeddings(
            model = "bge-m3",
            api_key = (self.api_key)
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
            client = self.chroma_client,
            collection_name = collection_name,
            embedding_function = None,
            persist_directory = persist_directory
        )
        
    
    def is_content_duplicate(self, query_vec, symbol: str, date: str) -> bool: # 같은 종목, 같은 날짜에 대해서 코사인 유사도가 높은 뉴스의 경우 중복 처리
        """콘텐츠 중복 여부 확인"""
        # 벡터 DB 내에 중복 뉴스 있는지 확인
        try:
            result = self.vectorstore.similarity_search_with_vector(
                query_embedding = query_vec,
                k = 1,      # 가장 유사도 높은거 하나만 가져옴
                filter ={
                    "symbol": symbol,
                    "date": date}
            )

            if not result: # 필터에 해당하는 데이터가 없음
                return False
            
            doc, score = result[0]

            if score >= 0.9:   # 코사인 유사도 점수 0.9 이상인 경우 중복으로 판정
                return True
            else:
                return False

        except:
            print("중복 처리 불가")
            return False

    
    def search_existing_news(self, symbol: str, date: str, query: str, k: int = 3) -> List[Dict]:
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
        

    def chunk_vectorize(self, chunk: str) -> List:
        """청크 벡터화"""
        try:
            embed_vec = self.embeddings.embed_query(chunk) # 벡터화
                
            return embed_vec
        
        except:
            print("청크 벡터화 실패")
            return None
    

    def store_news_chunks(self, news_data: Dict, chunks: List[str]) -> bool:
        """ 뉴스 중복 확인 후 벡터 DB 저장"""
        try:
            documents = []
            for i, chunk in enumerate(chunks):

                embed_vec = self.chunk_vectorize(chunk)

                if embed_vec is None:
                    return False

                # 중복 체크
                if self.is_content_duplicate(embed_vec, news_data['symbol'],news_data['date']):
                    print(f"중복 콘텐츠 스킵")
                    continue
                
            # 메타데이터 생성
            
                metadata = {
                    'symbol': news_data['symbol'],
                    'date': news_data['date'],
                    'title': news_data['title'],
                    'url': news_data['url'],
                    'source': news_data['source'],
                    'chunk_index': i,
                    'keywords': news_data.get('keywords', []),
                    'created_at': datetime.now().isoformat()
                }
            

                # 벡터 DB에 저장
                doc = Document(
                    page_content = chunk,
                    metadata = metadata
                )
                documents.append(doc)      

                if documents:          
                    self.vectorstore.add_documents(documents)
            return True
            
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
