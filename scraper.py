import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

KEY_PATHS = ["/about", "/about-us", "/team", "/products", "/services", "/careers", "/technology", "/platform"]
MAX_TOTAL_CHARS = 15000

TECH_PATTERNS = {
    "React": [r"react", r"__next", r"_next/static"],
    "Next.js": [r"_next/", r"__next"],
    "Vue.js": [r"vue\.js", r"vue\.min\.js", r"nuxt"],
    "Angular": [r"ng-version", r"angular"],
    "Google Analytics": [r"google-analytics", r"gtag", r"googletagmanager"],
    "Segment": [r"segment\.com/analytics", r"analytics\.js"],
    "Mixpanel": [r"mixpanel"],
    "HubSpot": [r"hubspot", r"hs-scripts"],
    "Intercom": [r"intercom", r"widget\.intercom"],
    "Drift": [r"drift\.com", r"driftt"],
    "Zendesk": [r"zendesk", r"zdassets"],
    "Salesforce": [r"salesforce", r"pardot"],
    "Stripe": [r"stripe\.com/v", r"js\.stripe"],
    "Cloudflare": [r"cloudflare"],
    "Fastly": [r"fastly"],
}


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _detect_tech_stack(html: str, headers: dict) -> list[str]:
    detected = []
    html_lower = html.lower()
    for tech, patterns in TECH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, html_lower):
                detected.append(tech)
                break

    server = headers.get("server", "").lower()
    if "cloudflare" in server:
        detected.append("Cloudflare CDN")
    if "nginx" in server:
        detected.append("Nginx")

    if headers.get("strict-transport-security"):
        detected.append("HSTS")
    if headers.get("content-security-policy"):
        detected.append("CSP Headers")

    return list(set(detected))


def _detect_structured_data(soup: BeautifulSoup) -> list[str]:
    signals = []
    json_ld = soup.find_all("script", type="application/ld+json")
    if json_ld:
        signals.append(f"JSON-LD structured data ({len(json_ld)} blocks)")

    if soup.find_all(attrs={"itemtype": True}):
        signals.append("Schema.org microdata")

    og_tags = soup.find_all("meta", property=re.compile(r"^og:"))
    if og_tags:
        signals.append("OpenGraph meta tags")

    return signals


def _check_chatbot_widgets(html: str) -> list[str]:
    widgets = []
    html_lower = html.lower()
    checks = {
        "Intercom chat widget": ["intercom-frame", "intercom-container"],
        "Drift chat widget": ["drift-frame", "drift-widget"],
        "Zendesk chat widget": ["zopim", "zendesk-chat"],
        "HubSpot chat widget": ["hubspot-messages-iframe"],
        "Crisp chat widget": ["crisp-client"],
        "LiveChat widget": ["livechat"],
        "Tidio chat widget": ["tidio"],
    }
    for name, patterns in checks.items():
        for p in patterns:
            if p in html_lower:
                widgets.append(name)
                break
    return widgets


async def check_technical_signals(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    signals = {
        "llms_txt": False,
        "llms_txt_content": "",
        "robots_txt": False,
        "robots_txt_permissive": False,
        "sitemap_xml": False,
        "rss_feed": False,
        "structured_data": [],
        "tech_stack": [],
        "chatbot_widgets": [],
        "has_api_docs": False,
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0, headers={"User-Agent": "Mozilla/5.0 (compatible; AIReadinessScanner/1.0)"}) as client:
        # Check llms.txt
        try:
            resp = await client.get(f"{base}/llms.txt")
            if resp.status_code == 200 and len(resp.text.strip()) > 10:
                signals["llms_txt"] = True
                signals["llms_txt_content"] = resp.text[:500]
        except Exception:
            pass

        # Check robots.txt
        try:
            resp = await client.get(f"{base}/robots.txt")
            if resp.status_code == 200:
                signals["robots_txt"] = True
                content = resp.text.lower()
                disallow_count = content.count("disallow:")
                allow_count = content.count("allow:")
                has_blanket_disallow = "disallow: /" in content and "disallow: /\n" not in content
                signals["robots_txt_permissive"] = not has_blanket_disallow and disallow_count < 10
        except Exception:
            pass

        # Check sitemap.xml
        try:
            resp = await client.get(f"{base}/sitemap.xml")
            if resp.status_code == 200 and "xml" in resp.headers.get("content-type", ""):
                signals["sitemap_xml"] = True
        except Exception:
            pass

        # Check homepage for tech signals
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                signals["tech_stack"] = _detect_tech_stack(resp.text, dict(resp.headers))
                signals["structured_data"] = _detect_structured_data(soup)
                signals["chatbot_widgets"] = _check_chatbot_widgets(resp.text)

                # Check for RSS
                rss_link = soup.find("link", type=re.compile(r"rss|atom"))
                if rss_link:
                    signals["rss_feed"] = True

                # Check for API/developer docs links
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "").lower()
                    text = a.get_text().lower()
                    if any(k in href or k in text for k in ["api", "developer", "docs", "documentation"]):
                        signals["has_api_docs"] = True
                        break
        except Exception:
            pass

    return signals


def format_technical_signals(signals: dict) -> str:
    lines = ["=== TECHNICAL SIGNALS DETECTED ==="]

    lines.append(f"\n[Agent/AI Readiness]")
    lines.append(f"- llms.txt present: {'YES' if signals['llms_txt'] else 'NO'}")
    if signals["llms_txt"]:
        lines.append(f"  Content preview: {signals['llms_txt_content'][:200]}")
    lines.append(f"- robots.txt present: {'YES' if signals['robots_txt'] else 'NO'}")
    if signals["robots_txt"]:
        lines.append(f"  Permissive to crawlers: {'YES' if signals['robots_txt_permissive'] else 'NO'}")
    lines.append(f"- sitemap.xml present: {'YES' if signals['sitemap_xml'] else 'NO'}")
    lines.append(f"- RSS/Atom feed: {'YES' if signals['rss_feed'] else 'NO'}")
    lines.append(f"- API/Developer docs: {'YES' if signals['has_api_docs'] else 'NO'}")
    if signals["structured_data"]:
        lines.append(f"- Structured data: {', '.join(signals['structured_data'])}")
    else:
        lines.append(f"- Structured data: NONE detected")

    lines.append(f"\n[Digital Maturity]")
    if signals["tech_stack"]:
        lines.append(f"- Tech stack detected: {', '.join(signals['tech_stack'])}")
    else:
        lines.append(f"- Tech stack: No major frameworks/tools detected")

    lines.append(f"\n[Existing Automation]")
    if signals["chatbot_widgets"]:
        lines.append(f"- Chat widgets: {', '.join(signals['chatbot_widgets'])}")
    else:
        lines.append(f"- Chat widgets: NONE detected")

    return "\n".join(lines)


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
