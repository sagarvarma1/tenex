import json
import anthropic
import openai
from models import ScanReport

SYSTEM_PROMPT = """You are an expert AI strategy consultant. You analyze companies and assess their AI readiness using a structured 5-dimension rubric.

Given scraped website content AND technical signals detected from a company's site, produce a detailed AI readiness assessment.

You MUST respond with valid JSON matching this exact schema:
{
  "company_name": "string",
  "company_summary": "2-3 sentence summary of what the company does",
  "dimensions": [
    {
      "name": "dimension name",
      "score": integer 1-10,
      "justification": "2-3 sentences explaining the score with specific evidence",
      "signals_detected": ["list of specific signals found"]
    }
  ],
  "opportunities": [
    {
      "title": "short title",
      "description": "2-3 sentence description of the AI opportunity",
      "impact": "High" | "Medium" | "Low",
      "timeframe": "Quick Win (<1 month)" | "Strategic (3-6 months)"
    }
  ]
}

You MUST score ALL 5 dimensions listed below, in this exact order.

## Scoring Rubric

### 1. Agent Discoverability
Can an AI agent find, map, and access this site's content?
- 9-10: llms.txt present, robots.txt with explicit AI bot permissions (GPTBot, ClaudeBot allowed), rich structured data (JSON-LD, Schema.org)
- 7-8: llms.txt present, robots.txt exists with no AI bot blocks, some structured data
- 4-6: No llms.txt, robots.txt with no AI bot rules, minimal structured data
- 1-3: No discoverability signals, AI bots blocked, no structured data

### 2. Semantic Digestibility
How clean is the content for LLM consumption (signal vs noise)?
- 9-10: High token density (>40%), valid JSON-LD, >90% alt text, proper heading hierarchy (H1>H2>H3)
- 7-8: Good density (20-40%), valid structured data, decent alt text (>70%), mostly proper headings
- 4-6: Average density (10-20%), some structured data issues, partial alt text, broken heading nesting
- 1-3: Very noisy HTML (<10%), no valid structured data, poor alt text, no heading structure

### 3. Actionability & Friction
Can an agent interact with this site without hitting walls?
- 9-10: MCP support, semantic form fields (>80%), no CAPTCHAs, no auth walls, supports JSON responses
- 7-8: Good form semantics, no CAPTCHAs, no auth wall, standard navigation
- 4-6: Some friction — CAPTCHAs or auth walls present, poor form naming
- 1-3: Heavy friction — CAPTCHAs + auth walls, non-semantic forms, no JSON support

### 4. Data Richness
How much structured, AI-consumable content exists?
- 9-10: Extensive documentation, knowledge base, product catalogs, blog, case studies, whitepapers
- 7-8: Good content depth — blog + detailed product/service pages + some docs
- 4-6: Moderate content — basic product pages and marketing copy
- 1-3: Very sparse, mostly a brochure site with minimal content

### 5. Developer & API Maturity
Can an agent programmatically interact with the company's services?
- 9-10: OpenAPI/Swagger spec discoverable, GraphQL endpoint available, webhook docs present, comprehensive API docs
- 7-8: API docs present, spec or GraphQL available, some webhook mentions
- 4-6: API mentioned but no spec or GraphQL, limited developer resources
- 1-3: No API, GraphQL, or developer presence detected

## Guidelines
- Use the TECHNICAL SIGNALS section as hard evidence — these are measured facts
- Score based on what you can actually observe, not assumptions
- Provide exactly 5-7 opportunities, mix of quick wins and strategic plays
- Be specific to THIS company, not generic AI advice
- Return ONLY the JSON object, no markdown or extra text"""

USER_PROMPT = """Analyze this company's website content and produce an AI readiness assessment across all 5 dimensions.

{technical_signals}

=== WEBSITE CONTENT ===
{content}"""


def _parse_response(text: str) -> ScanReport:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return ScanReport.model_validate(json.loads(cleaned))


async def analyze_with_claude(content: str, technical_signals: str) -> ScanReport:
    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(content=content, technical_signals=technical_signals)}],
    )
    return _parse_response(message.content[0].text)


async def analyze_with_openai(content: str, technical_signals: str) -> ScanReport:
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-5.2",
        max_completion_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(content=content, technical_signals=technical_signals)},
        ],
    )
    return _parse_response(response.choices[0].message.content)


async def analyze(content: str, technical_signals: str, provider: str) -> ScanReport:
    if provider == "claude":
        return await analyze_with_claude(content, technical_signals)
    elif provider == "openai":
        return await analyze_with_openai(content, technical_signals)
    else:
        raise ValueError(f"Unknown provider: {provider}")
