import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import time
import urllib.parse

class NaverBlogSearcher:
    """네이버 검색 API를 사용한 RAG 검색기"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def search_news(self, symbol: str, date: str) -> List[Dict]:
        """네이버 검색 API로 블로그 검색"""
        try:
            # URL 인코딩 추가
            query = urllib.parse.quote(f"{date} {symbol}")
            url = f"https://openapi.naver.com/v1/search/blog.json?query={query}&display=3"

            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                print(f"검색 성공: {symbol}")
            else:
                print("Error Code:", response.status_code)
                return []

            news_list = []
            data = response.json()
            for item in data.get('items', []):
                # HTML 태그 제거
                title = re.sub(r'<[^>]+>', '', item.get('title', ''))
                news_list.append({
                    'title': title,
                    'url': item.get('link', ''),
                    'symbol': symbol
                })
            
            return news_list
                          
        except Exception as e:
            print(f"뉴스 검색 오류: {e}")
            return []

class NewsContentCrawler:
    """블로그 본문 크롤링"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def crawl_news_content(self, url: str) -> Optional[str]:
        """블로그 본문 크롤링"""
        try:
            print(f"크롤링 시작: {url}")
            
            # 네이버 블로그인지 확인
            if "blog.naver.com" not in url:
                return self._crawl_general_content(url)
            
            # 네이버 블로그 크롤링
            return self._crawl_naver_blog(url)
            
        except Exception as e:
            print(f"크롤링 오류 ({url}): {e}")
            return None

    def _crawl_naver_blog(self, url: str) -> Optional[str]:
        """네이버 블로그 전용 크롤링"""
        try:
            # Step 1: 첫 번째 요청
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Step 2: iframe 찾기 (여러 방법 시도)
            iframe = None
            iframe_selectors = [
                "iframe#mainFrame",
                "iframe[name='mainFrame']",
                "iframe.blog_content_frame"
            ]
            
            for selector in iframe_selectors:
                iframe = soup.select_one(selector)
                if iframe:
                    break
            
            if not iframe:
                print("iframe을 찾을 수 없습니다.")
                return self._crawl_general_content(url)
            
            # iframe URL 구성
            iframe_src = iframe.get("src", "")
            if iframe_src.startswith("//"):
                iframe_url = "https:" + iframe_src
            elif iframe_src.startswith("/"):
                iframe_url = "https://blog.naver.com" + iframe_src
            else:
                iframe_url = iframe_src
            
            print(f"iframe URL: {iframe_url}")
            
            # Step 3: iframe URL로 재요청
            time.sleep(1)  # 요청 간격 조절
            res2 = self.session.get(iframe_url, timeout=15)
            res2.raise_for_status()
            
            soup2 = BeautifulSoup(res2.text, "html.parser")
            
            # Step 4: 본문 추출 (다양한 셀렉터 시도)
            content_selectors = [
                "div.se-main-container",
                "div.se-component-content",
                "div.__se_component_area",
                "div.post-view",
                "div.blog-content",
                "div#viewTypeSelector",
                "div.post_ct",
                "div.se-viewer",
                "div.smartOutput"
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup2.select(selector)
                if elements:
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if text and len(text) > 50:  # 충분한 텍스트가 있는 경우
                            content += text + " "
                    if content:
                        break
            
            # 기본 추출이 실패한 경우 p 태그 시도
            if not content:
                paragraphs = soup2.find_all('p')
                content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # 여전히 내용이 없으면 모든 텍스트 추출
            if not content:
                # 스크립트, 스타일 태그 제거
                for tag in soup2(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                content = soup2.get_text(strip=True)
            
            # 텍스트 정리
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            # 너무 짧은 내용 필터링
            if len(content) < 100:
                print(f"내용이 너무 짧습니다: {len(content)}자")
                return None
            
            print(f"크롤링 완료: {len(content)}자")
            return content
            
        except Exception as e:
            print(f"네이버 블로그 크롤링 오류: {e}")
            return None

    def _crawl_general_content(self, url: str) -> Optional[str]:
        """일반 웹사이트 크롤링"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 불필요한 태그 제거
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # 본문 추출 시도
            content_selectors = [
                "article",
                "main",
                "div.content",
                "div.post-content",
                "div.entry-content",
                "div.article-content"
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = " ".join([elem.get_text(strip=True) for elem in elements])
                    break
            
            if not content:
                # p 태그들로 본문 구성
                paragraphs = soup.find_all('p')
                content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # 텍스트 정리
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            return content if len(content) > 100 else None
            
        except Exception as e:
            print(f"일반 크롤링 오류: {e}")
            return None
        


if __name__ == '__main__':
    # 테스트 실행
    client_id = "ydTKk3TTnytTEkyEfKhS"  # 실제 client_id 입력
    client_secret = "gOzF8p0re5"  # 실제 client_secret 입력
    
    naverblogsearcher = NaverBlogSearcher(client_id, client_secret)
    news_list = naverblogsearcher.search_news('sk하이닉스', '2025년 4월 3일')
    
    newscrawler = NewsContentCrawler()
    
    if news_list:
        successful_crawls = 0
        for i, news in enumerate(news_list, 1):
            print(f"\n=== {i}. {news['title']} ===")
            print(f"URL: {news['url']}")
            
            content = newscrawler.crawl_news_content(news["url"])
            if content:
                print(f"내용 (처음 200자): {content[:200]}...")
                successful_crawls += 1
            else:
                print("내용을 가져올 수 없습니다.")
            
            # 요청 간 딜레이
            if i < len(news_list):
                time.sleep(2)
        
        print(f"\n총 {len(news_list)}개 중 {successful_crawls}개 성공적으로 크롤링")
    else:
        print("검색 결과가 없습니다.")

