import json
import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

KEY_PATHS = ["/about", "/about-us", "/team", "/products", "/services", "/careers", "/technology", "/platform"]
MAX_TOTAL_CHARS = 15000

AI_BOTS = ["gptbot", "claudebot", "anthropic-ai", "chatgpt-user", "google-extended", "ccbot", "perplexitybot"]


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


# ── Dimension 1: Agent Discoverability ────────────────────────────────


async def _check_llms_txt(base: str, client: httpx.AsyncClient) -> dict:
    result = {"present": False, "content": "", "compliant": False, "links_live": 0, "links_total": 0}

    for path in ["/llms.txt", "/.well-known/llms.txt"]:
        try:
            resp = await client.get(f"{base}{path}")
            if resp.status_code == 200 and len(resp.text.strip()) > 10:
                result["present"] = True
                result["content"] = resp.text[:500]
                if "/full/" in resp.text or "/brief/" in resp.text:
                    result["compliant"] = True
                urls = re.findall(r'https?://\S+', resp.text)
                result["links_total"] = len(urls)
                live = 0
                for u in urls[:10]:
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


def _detect_structured_data(soup: BeautifulSoup) -> list[str]:
    signals = []
    json_ld = soup.find_all("script", type="application/ld+json")
    if json_ld:
        signals.append(f"JSON-LD ({len(json_ld)} blocks)")
    if soup.find_all(attrs={"itemtype": True}):
        signals.append("Schema.org microdata")
    og_tags = soup.find_all("meta", property=re.compile(r"^og:"))
    if og_tags:
        signals.append("OpenGraph meta tags")
    return signals


# ── Dimension 2: Semantic Digestibility ───────────────────────────────


def _measure_token_density(html: str, clean_text: str) -> float:
    if not html:
        return 0.0
    return round(len(clean_text) / len(html), 3)


def _validate_json_ld(soup: BeautifulSoup) -> dict:
    result = {"valid": False, "count": 0}
    blocks = soup.find_all("script", type="application/ld+json")
    result["count"] = len(blocks)
    all_valid = True
    for block in blocks:
        try:
            json.loads(block.string or "")
        except (json.JSONDecodeError, TypeError):
            all_valid = False
    if blocks:
        result["valid"] = all_valid
    return result


def _check_alt_text_coverage(soup: BeautifulSoup) -> float:
    images = soup.find_all("img")
    if not images:
        return 1.0
    with_alt = sum(1 for img in images if img.get("alt", "").strip())
    return round(with_alt / len(images), 2)


# ── Dimension 3: Actionability & Friction ─────────────────────────────


def _check_mcp_support(soup: BeautifulSoup, headers: dict, html: str) -> bool:
    html_lower = html.lower()
    if "mcp-server" in html_lower or "mcp" in headers.get("x-mcp", "").lower():
        return True
    for a in soup.find_all("a", href=True):
        if "mcp" in a.get("href", "").lower():
            return True
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


def _check_captcha(html: str) -> bool:
    html_lower = html.lower()
    return any(s in html_lower for s in ["recaptcha", "hcaptcha", "captcha", "turnstile"])


# ── Dimension 5: Developer & API Maturity ─────────────────────────────


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


def _check_api_docs_link(soup: BeautifulSoup) -> bool:
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").lower()
        text = a.get_text().lower()
        if any(k in href or k in text for k in ["api", "developer", "docs", "documentation"]):
            return True
    return False


# ── Main Detection ────────────────────────────────────────────────────


async def check_technical_signals(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    signals = {
        # Dimension 1: Agent Discoverability
        "llms_txt": False,
        "llms_txt_content": "",
        "llms_txt_compliant": False,
        "llms_txt_links_live": 0,
        "llms_txt_links_total": 0,
        "robots_txt": False,
        "robots_txt_permissive": False,
        "robots_txt_ai_bot_rules": [],
        "structured_data": [],

        # Dimension 2: Semantic Digestibility
        "token_density": 0.0,
        "json_ld_valid": False,
        "json_ld_count": 0,
        "alt_text_coverage": 0.0,

        # Dimension 3: Actionability & Friction
        "mcp_support": False,
        "forms_total": 0,
        "forms_introspectable_ratio": 0.0,
        "has_captcha": False,

        # Dimension 4: Data Richness (assessed by AI from content)

        # Dimension 5: Developer & API Maturity
        "has_openapi_spec": False,
        "openapi_spec_type": "",
        "has_api_docs": False,
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (compatible; AIReadinessScanner/1.0)"}) as client:

        # llms.txt
        llms = await _check_llms_txt(base, client)
        signals["llms_txt"] = llms["present"]
        signals["llms_txt_content"] = llms["content"]
        signals["llms_txt_compliant"] = llms["compliant"]
        signals["llms_txt_links_live"] = llms["links_live"]
        signals["llms_txt_links_total"] = llms["links_total"]

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

        # Homepage analysis
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                headers_dict = {k.lower(): v for k, v in resp.headers.items()}

                # Dimension 1
                signals["structured_data"] = _detect_structured_data(soup)

                # Dimension 2
                clean = _clean_text(BeautifulSoup(resp.text, "html.parser"))
                signals["token_density"] = _measure_token_density(resp.text, clean)
                json_ld = _validate_json_ld(soup)
                signals["json_ld_valid"] = json_ld["valid"]
                signals["json_ld_count"] = json_ld["count"]
                signals["alt_text_coverage"] = _check_alt_text_coverage(soup)

                # Dimension 3
                signals["mcp_support"] = _check_mcp_support(soup, headers_dict, resp.text)
                form_quality = _check_form_quality(soup)
                signals["forms_total"] = form_quality["total_forms"]
                signals["forms_introspectable_ratio"] = form_quality["introspectable_ratio"]
                signals["has_captcha"] = _check_captcha(resp.text)

                # Dimension 5
                signals["has_api_docs"] = _check_api_docs_link(soup)
        except Exception:
            pass

        # API spec discovery
        api_specs = await _check_api_specs(base, client)
        signals["has_openapi_spec"] = api_specs["has_spec"]
        signals["openapi_spec_type"] = api_specs["spec_type"]

    return signals


def format_technical_signals(signals: dict) -> str:
    lines = ["=== TECHNICAL SIGNALS DETECTED ==="]

    lines.append(f"\n[Agent Discoverability]")
    lines.append(f"- llms.txt present: {'YES' if signals['llms_txt'] else 'NO'}")
    if signals["llms_txt"]:
        lines.append(f"  Compliant (full/brief): {'YES' if signals['llms_txt_compliant'] else 'NO'}")
        lines.append(f"  Links: {signals['llms_txt_links_live']}/{signals['llms_txt_links_total']} live")
        lines.append(f"  Preview: {signals['llms_txt_content'][:200]}")
    lines.append(f"- robots.txt present: {'YES' if signals['robots_txt'] else 'NO'}")
    if signals["robots_txt"]:
        lines.append(f"  Permissive: {'YES' if signals['robots_txt_permissive'] else 'NO'}")
        if signals["robots_txt_ai_bot_rules"]:
            lines.append(f"  AI bot rules: {', '.join(signals['robots_txt_ai_bot_rules'])}")
        else:
            lines.append(f"  AI bot rules: NONE (no specific AI bot rules)")
    if signals["structured_data"]:
        lines.append(f"- Structured data: {', '.join(signals['structured_data'])}")
    else:
        lines.append(f"- Structured data: NONE")

    lines.append(f"\n[Semantic Digestibility]")
    lines.append(f"- Token density (content/HTML): {signals['token_density']:.1%}")
    lines.append(f"- JSON-LD: {signals['json_ld_count']} blocks ({'valid' if signals['json_ld_valid'] else 'invalid or none'})")
    lines.append(f"- Alt text coverage: {signals['alt_text_coverage']:.0%}")

    lines.append(f"\n[Actionability & Friction]")
    lines.append(f"- MCP support: {'YES' if signals['mcp_support'] else 'NO'}")
    lines.append(f"- Forms: {signals['forms_total']}")
    if signals["forms_total"] > 0:
        lines.append(f"  Semantic field naming: {signals['forms_introspectable_ratio']:.0%}")
    lines.append(f"- CAPTCHA: {'YES (friction)' if signals['has_captcha'] else 'NO'}")

    lines.append(f"\n[Data Richness]")
    lines.append(f"- (Assessed from scraped content below)")

    lines.append(f"\n[Developer & API Maturity]")
    lines.append(f"- OpenAPI/Swagger spec: {'YES (' + signals['openapi_spec_type'] + ')' if signals['has_openapi_spec'] else 'NO'}")
    lines.append(f"- API docs link: {'YES' if signals['has_api_docs'] else 'NO'}")

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
