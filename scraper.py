import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

KEY_PATHS = ["/about", "/about-us", "/team", "/products", "/services", "/careers", "/technology", "/platform"]
MAX_TOTAL_CHARS = 15000


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    return " ".join(text.split())


async def scrape_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    pages_text: list[str] = []
    visited: set[str] = set()
    total_chars = 0

    urls_to_scrape = [url] + [urljoin(base, path) for path in KEY_PATHS]

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (compatible; AIReadinessScanner/1.0)"}) as client:
        for page_url in urls_to_scrape:
            if page_url in visited or total_chars >= MAX_TOTAL_CHARS:
                break
            visited.add(page_url)

            try:
                resp = await client.get(page_url)
                if resp.status_code != 200:
                    continue
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                text = _clean_text(soup)

                if len(text) < 50:
                    continue

                remaining = MAX_TOTAL_CHARS - total_chars
                chunk = text[:remaining]
                pages_text.append(f"--- Page: {page_url} ---\n{chunk}")
                total_chars += len(chunk)

            except (httpx.HTTPError, Exception):
                continue

    if not pages_text:
        raise ValueError(f"Could not scrape any content from {url}")

    return "\n\n".join(pages_text)
