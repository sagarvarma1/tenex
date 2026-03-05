import json
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

AI_BOTS = ["gptbot", "claudebot", "anthropic-ai", "chatgpt-user", "google-extended", "bingbot", "ccbot", "perplexitybot"]


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


# ── v2.0 Detection Functions ──────────────────────────────────────────


async def _check_llms_txt_compliance(base: str, client: httpx.AsyncClient) -> dict:
    result = {"present": False, "content": "", "compliant": False, "links_live": 0, "links_total": 0}

    for path in ["/llms.txt", "/.well-known/llms.txt"]:
        try:
            resp = await client.get(f"{base}{path}")
            if resp.status_code == 200 and len(resp.text.strip()) > 10:
                result["present"] = True
                result["content"] = resp.text[:500]

                # Check compliance: /full/ vs /brief/ convention
                if "/full/" in resp.text or "/brief/" in resp.text:
                    result["compliant"] = True

                # Check links
                urls = re.findall(r'https?://\S+', resp.text)
                result["links_total"] = len(urls)
                live = 0
                for u in urls[:10]:  # Check up to 10 links
                    try:
                        r = await client.head(u, timeout=5.0)
                        if r.status_code < 400:
                            live += 1
                    except Exception:
                        pass
                result["links_live"] = live
                break
        except Exception:
            pass

    return result


def _check_robots_ai_bots(robots_content: str) -> list[str]:
    rules = []
    content_lower = robots_content.lower()
    for bot in AI_BOTS:
        if bot in content_lower:
            # Find the relevant section
            pattern = rf"user-agent:\s*{bot}.*?(?=user-agent:|\Z)"
            match = re.search(pattern, content_lower, re.DOTALL)
            if match:
                section = match.group()
                if "disallow: /" in section:
                    rules.append(f"{bot}: BLOCKED")
                else:
                    rules.append(f"{bot}: ALLOWED")
            else:
                rules.append(f"{bot}: MENTIONED")
    return rules


async def _check_sitemap_quality(base: str, client: httpx.AsyncClient) -> dict:
    result = {"present": False, "has_lastmod": False}
    try:
        resp = await client.get(f"{base}/sitemap.xml")
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "xml" in content_type or "<urlset" in resp.text[:500] or "<sitemapindex" in resp.text[:500]:
                result["present"] = True
                result["has_lastmod"] = "<lastmod>" in resp.text
    except Exception:
        pass
    return result


async def _check_rss_content(soup: BeautifulSoup, client: httpx.AsyncClient) -> dict:
    result = {"present": False, "full_content": False}
    rss_link = soup.find("link", type=re.compile(r"rss|atom"))
    if rss_link:
        result["present"] = True
        href = rss_link.get("href", "")
        if href:
            try:
                resp = await client.get(href, timeout=10.0)
                if resp.status_code == 200:
                    # Full content feeds typically have long <content:encoded> or <content> tags
                    result["full_content"] = bool(
                        re.search(r"<content:encoded>|<content[^>]*>.{500,}", resp.text)
                    )
            except Exception:
                pass
    return result


def _measure_token_density(html: str, clean_text: str) -> float:
    if not html:
        return 0.0
    return round(len(clean_text) / len(html), 3)


def _check_markdown_availability(soup: BeautifulSoup) -> bool:
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").lower()
        text = a.get_text().lower()
        if href.endswith(".md") or "markdown" in text or "reader mode" in text or "raw" in href:
            return True
    return False


def _validate_json_ld(soup: BeautifulSoup) -> dict:
    result = {"valid": False, "count": 0, "has_breadcrumb": False}
    blocks = soup.find_all("script", type="application/ld+json")
    result["count"] = len(blocks)
    all_valid = True
    for block in blocks:
        try:
            data = json.loads(block.string or "")
            if isinstance(data, dict):
                if data.get("@type") == "BreadcrumbList":
                    result["has_breadcrumb"] = True
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") == "BreadcrumbList":
                        result["has_breadcrumb"] = True
        except (json.JSONDecodeError, TypeError):
            all_valid = False
    if blocks:
        result["valid"] = all_valid
    return result


def _check_alt_text_coverage(soup: BeautifulSoup) -> float:
    images = soup.find_all("img")
    if not images:
        return 1.0  # No images = no issue
    with_alt = sum(1 for img in images if img.get("alt", "").strip())
    return round(with_alt / len(images), 2)


def _check_aria_landmarks(soup: BeautifulSoup) -> int:
    landmark_roles = ["banner", "navigation", "main", "complementary", "contentinfo", "search", "form", "region"]
    count = 0
    for role in landmark_roles:
        count += len(soup.find_all(attrs={"role": role}))
    # Also count semantic HTML5 elements that serve as landmarks
    for tag in ["main", "nav", "aside", "header", "footer"]:
        count += len(soup.find_all(tag))
    return count


def _check_mcp_support(soup: BeautifulSoup, headers: dict, html: str) -> bool:
    html_lower = html.lower()
    if "mcp-server" in html_lower or "mcp" in headers.get("x-mcp", "").lower():
        return True
    for a in soup.find_all("a", href=True):
        if "mcp" in a.get("href", "").lower():
            return True
    # Check for MCP-related meta tags
    for meta in soup.find_all("meta"):
        if "mcp" in (meta.get("name", "") + meta.get("content", "")).lower():
            return True
    return False


def _check_form_quality(soup: BeautifulSoup) -> dict:
    result = {"total_forms": 0, "introspectable_ratio": 0.0}
    forms = soup.find_all("form")
    result["total_forms"] = len(forms)
    if not forms:
        return result

    total_fields = 0
    semantic_fields = 0
    non_semantic = re.compile(r"^(field_?\d+|input_?\d+|f\d+|q\d+)$", re.IGNORECASE)

    for form in forms:
        inputs = form.find_all(["input", "select", "textarea"])
        for inp in inputs:
            input_type = inp.get("type", "").lower()
            if input_type in ("hidden", "submit", "button"):
                continue
            total_fields += 1
            name = inp.get("name", "") or inp.get("id", "")
            if name and not non_semantic.match(name):
                semantic_fields += 1

    if total_fields > 0:
        result["introspectable_ratio"] = round(semantic_fields / total_fields, 2)
    return result


def _check_friction(soup: BeautifulSoup, html: str) -> dict:
    html_lower = html.lower()
    result = {
        "has_captcha": False,
        "has_sales_gate": False,
        "uses_shadow_dom": False,
        "uses_infinite_scroll": False,
        "headless_friendly": True,
    }

    # CAPTCHA detection
    captcha_signals = ["recaptcha", "hcaptcha", "captcha", "turnstile"]
    result["has_captcha"] = any(s in html_lower for s in captcha_signals)

    # Sales gate detection
    gate_patterns = ["talk to sales", "contact sales", "request a demo", "book a demo", "schedule a call", "get in touch"]
    gate_count = sum(1 for p in gate_patterns if p in html_lower)
    result["has_sales_gate"] = gate_count >= 2  # Multiple gate signals = likely gated

    # Shadow DOM detection
    result["uses_shadow_dom"] = "shadowroot" in html_lower or "attachshadow" in html_lower

    # Infinite scroll detection
    scroll_signals = ["infinite-scroll", "infinitescroll", "loadmore", "load-more", "intersection-observer"]
    result["uses_infinite_scroll"] = any(s in html_lower for s in scroll_signals)

    # Overall headless friendliness
    if result["has_captcha"] or result["uses_shadow_dom"] or result["uses_infinite_scroll"]:
        result["headless_friendly"] = False

    return result


async def _check_api_specs(base: str, client: httpx.AsyncClient) -> dict:
    result = {"has_spec": False, "spec_type": ""}
    spec_paths = [
        ("/swagger.json", "Swagger/OpenAPI"),
        ("/openapi.json", "OpenAPI"),
        ("/openapi.yaml", "OpenAPI"),
        ("/.well-known/api-configuration", "API Configuration"),
        ("/api/docs", "API Docs"),
        ("/api/v1/docs", "API Docs"),
    ]
    for path, spec_type in spec_paths:
        try:
            resp = await client.get(f"{base}{path}", timeout=5.0)
            if resp.status_code == 200:
                result["has_spec"] = True
                result["spec_type"] = spec_type
                break
        except Exception:
            pass
    return result


def _check_sdk_references(soup: BeautifulSoup, html: str) -> list[str]:
    sdks = []
    html_lower = html.lower()
    checks = {
        "Python": ["pip install", "import ", "pypi.org"],
        "TypeScript/JavaScript": ["npm install", "yarn add", "pnpm add", "npmjs.com"],
        "Go": ["go get", "go install"],
        "Ruby": ["gem install", "rubygems.org"],
        "Java": ["maven", "gradle", "mvnrepository"],
    }
    for lang, patterns in checks.items():
        for p in patterns:
            if p in html_lower:
                sdks.append(lang)
                break
    return sdks


def _check_rate_limits(headers: dict) -> bool:
    rate_headers = ["retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-rate-limit-limit", "ratelimit-limit"]
    return any(h in headers for h in rate_headers)


# ── Main Detection Functions ──────────────────────────────────────────


async def check_technical_signals(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    signals = {
        # v1 signals
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

        # v2 Agent & Crawler Discoverability
        "llms_txt_compliant": False,
        "llms_txt_links_live": 0,
        "llms_txt_links_total": 0,
        "robots_txt_ai_bot_rules": [],
        "sitemap_has_lastmod": False,
        "rss_full_content": False,

        # v2 Semantic Digestibility
        "token_density": 0.0,
        "markdown_available": False,
        "json_ld_valid": False,
        "json_ld_count": 0,
        "json_ld_has_breadcrumb": False,
        "alt_text_coverage": 0.0,
        "aria_landmarks": 0,

        # v2 Actionability & Friction
        "mcp_support": False,
        "forms_total": 0,
        "forms_introspectable_ratio": 0.0,
        "has_captcha": False,
        "has_sales_gate": False,
        "headless_friendly": True,
        "uses_shadow_dom": False,
        "uses_infinite_scroll": False,

        # v2 Developer & API Maturity
        "has_openapi_spec": False,
        "openapi_spec_type": "",
        "sdk_languages": [],
        "rate_limit_transparent": False,
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (compatible; AIReadinessScanner/1.0)"}) as client:

        # llms.txt deep check
        llms_result = await _check_llms_txt_compliance(base, client)
        signals["llms_txt"] = llms_result["present"]
        signals["llms_txt_content"] = llms_result["content"]
        signals["llms_txt_compliant"] = llms_result["compliant"]
        signals["llms_txt_links_live"] = llms_result["links_live"]
        signals["llms_txt_links_total"] = llms_result["links_total"]

        # robots.txt
        try:
            resp = await client.get(f"{base}/robots.txt")
            if resp.status_code == 200:
                signals["robots_txt"] = True
                content = resp.text.lower()
                disallow_count = content.count("disallow:")
                has_blanket_disallow = "disallow: /" in content and "disallow: /\n" not in content
                signals["robots_txt_permissive"] = not has_blanket_disallow and disallow_count < 10
                signals["robots_txt_ai_bot_rules"] = _check_robots_ai_bots(resp.text)
        except Exception:
            pass

        # Sitemap quality
        sitemap = await _check_sitemap_quality(base, client)
        signals["sitemap_xml"] = sitemap["present"]
        signals["sitemap_has_lastmod"] = sitemap["has_lastmod"]

        # Homepage analysis
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                headers_dict = {k.lower(): v for k, v in resp.headers.items()}

                # v1 signals
                signals["tech_stack"] = _detect_tech_stack(resp.text, headers_dict)
                signals["structured_data"] = _detect_structured_data(soup)
                signals["chatbot_widgets"] = _check_chatbot_widgets(resp.text)

                # RSS check
                rss = await _check_rss_content(soup, client)
                signals["rss_feed"] = rss["present"]
                signals["rss_full_content"] = rss["full_content"]

                # API docs link
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "").lower()
                    text = a.get_text().lower()
                    if any(k in href or k in text for k in ["api", "developer", "docs", "documentation"]):
                        signals["has_api_docs"] = True
                        break

                # Semantic Digestibility
                clean = _clean_text(BeautifulSoup(resp.text, "html.parser"))  # Fresh soup since _clean_text modifies it
                signals["token_density"] = _measure_token_density(resp.text, clean)
                signals["markdown_available"] = _check_markdown_availability(soup)
                json_ld = _validate_json_ld(soup)
                signals["json_ld_valid"] = json_ld["valid"]
                signals["json_ld_count"] = json_ld["count"]
                signals["json_ld_has_breadcrumb"] = json_ld["has_breadcrumb"]
                signals["alt_text_coverage"] = _check_alt_text_coverage(soup)
                signals["aria_landmarks"] = _check_aria_landmarks(soup)

                # Actionability & Friction
                signals["mcp_support"] = _check_mcp_support(soup, headers_dict, resp.text)
                form_quality = _check_form_quality(soup)
                signals["forms_total"] = form_quality["total_forms"]
                signals["forms_introspectable_ratio"] = form_quality["introspectable_ratio"]
                friction = _check_friction(soup, resp.text)
                signals["has_captcha"] = friction["has_captcha"]
                signals["has_sales_gate"] = friction["has_sales_gate"]
                signals["uses_shadow_dom"] = friction["uses_shadow_dom"]
                signals["uses_infinite_scroll"] = friction["uses_infinite_scroll"]
                signals["headless_friendly"] = friction["headless_friendly"]

                # Developer & API Maturity
                signals["sdk_languages"] = _check_sdk_references(soup, resp.text)
                signals["rate_limit_transparent"] = _check_rate_limits(headers_dict)
        except Exception:
            pass

        # API spec discovery
        api_specs = await _check_api_specs(base, client)
        signals["has_openapi_spec"] = api_specs["has_spec"]
        signals["openapi_spec_type"] = api_specs["spec_type"]

    return signals


def format_technical_signals(signals: dict) -> str:
    lines = ["=== TECHNICAL SIGNALS DETECTED ==="]

    # ── Dimension 1: Agent/AI Readiness (v1) ──
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

    # ── Dimension 2: Digital Maturity (v1) ──
    lines.append(f"\n[Digital Maturity]")
    if signals["tech_stack"]:
        lines.append(f"- Tech stack detected: {', '.join(signals['tech_stack'])}")
    else:
        lines.append(f"- Tech stack: No major frameworks/tools detected")

    # ── Dimension 3: Data Richness (v1 — assessed by AI from content) ──
    lines.append(f"\n[Data Richness]")
    lines.append(f"- (Assessed from scraped content below)")

    # ── Dimension 4: Existing Automation (v1) ──
    lines.append(f"\n[Existing Automation]")
    if signals["chatbot_widgets"]:
        lines.append(f"- Chat widgets: {', '.join(signals['chatbot_widgets'])}")
    else:
        lines.append(f"- Chat widgets: NONE detected")

    # ── Dimension 5: Agent & Crawler Discoverability (v2) ──
    lines.append(f"\n[Agent & Crawler Discoverability]")
    lines.append(f"- llms.txt compliant (full/brief convention): {'YES' if signals['llms_txt_compliant'] else 'NO'}")
    lines.append(f"- llms.txt links: {signals['llms_txt_links_live']}/{signals['llms_txt_links_total']} live")
    if signals["robots_txt_ai_bot_rules"]:
        lines.append(f"- AI bot rules in robots.txt: {', '.join(signals['robots_txt_ai_bot_rules'])}")
    else:
        lines.append(f"- AI bot rules in robots.txt: NONE (no specific AI bot rules)")
    lines.append(f"- Sitemap has lastmod timestamps: {'YES' if signals['sitemap_has_lastmod'] else 'NO'}")
    lines.append(f"- RSS has full content: {'YES' if signals['rss_full_content'] else 'NO' if signals['rss_feed'] else 'N/A (no feed)'}")

    # ── Dimension 6: Semantic Digestibility (v2) ──
    lines.append(f"\n[Semantic Digestibility]")
    lines.append(f"- Token density (content/HTML ratio): {signals['token_density']:.1%}")
    lines.append(f"- Markdown availability: {'YES' if signals['markdown_available'] else 'NO'}")
    lines.append(f"- JSON-LD blocks: {signals['json_ld_count']} ({'valid' if signals['json_ld_valid'] else 'invalid or none'})")
    lines.append(f"- Breadcrumb structured data: {'YES' if signals['json_ld_has_breadcrumb'] else 'NO'}")
    lines.append(f"- Alt text coverage: {signals['alt_text_coverage']:.0%}")
    lines.append(f"- ARIA landmarks: {signals['aria_landmarks']}")

    # ── Dimension 7: Actionability & Friction (v2) ──
    lines.append(f"\n[Actionability & Friction]")
    lines.append(f"- MCP support: {'YES' if signals['mcp_support'] else 'NO'}")
    lines.append(f"- Forms found: {signals['forms_total']}")
    if signals["forms_total"] > 0:
        lines.append(f"  Semantic field naming: {signals['forms_introspectable_ratio']:.0%}")
    lines.append(f"- CAPTCHA detected: {'YES' if signals['has_captcha'] else 'NO'}")
    lines.append(f"- Sales gate detected: {'YES' if signals['has_sales_gate'] else 'NO'}")
    lines.append(f"- Shadow DOM: {'YES' if signals['uses_shadow_dom'] else 'NO'}")
    lines.append(f"- Infinite scroll: {'YES' if signals['uses_infinite_scroll'] else 'NO'}")
    lines.append(f"- Headless-friendly: {'YES' if signals['headless_friendly'] else 'NO'}")

    # ── Dimension 8: Developer & API Maturity (v2) ──
    lines.append(f"\n[Developer & API Maturity]")
    lines.append(f"- OpenAPI/Swagger spec: {'YES (' + signals['openapi_spec_type'] + ')' if signals['has_openapi_spec'] else 'NO'}")
    lines.append(f"- API docs detected: {'YES' if signals['has_api_docs'] else 'NO'}")
    if signals["sdk_languages"]:
        lines.append(f"- SDK languages: {', '.join(signals['sdk_languages'])}")
    else:
        lines.append(f"- SDK languages: NONE detected")
    lines.append(f"- Rate limit headers: {'YES' if signals['rate_limit_transparent'] else 'NO'}")

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
