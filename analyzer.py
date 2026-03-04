import json
import anthropic
import openai
from models import ScanReport

SYSTEM_PROMPT = """You are an expert AI strategy consultant. You analyze companies and assess their AI readiness using a structured rubric.

Given scraped website content AND technical signals detected from a company's site, produce a detailed AI readiness assessment.

You MUST respond with valid JSON matching this exact schema:
{
  "company_name": "string",
  "company_summary": "2-3 sentence summary of what the company does",
  "dimensions": [
    {
      "name": "Agent/AI Readiness",
      "score": integer 1-10,
      "justification": "2-3 sentences explaining the score",
      "signals_detected": ["list of specific signals found"]
    },
    {
      "name": "Digital Maturity",
      "score": integer 1-10,
      "justification": "2-3 sentences explaining the score",
      "signals_detected": ["list of specific signals found"]
    },
    {
      "name": "Data Richness",
      "score": integer 1-10,
      "justification": "2-3 sentences explaining the score",
      "signals_detected": ["list of specific signals found"]
    },
    {
      "name": "Existing Automation",
      "score": integer 1-10,
      "justification": "2-3 sentences explaining the score",
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

## Scoring Rubric

### 1. Agent/AI Readiness (1-10)
How ready is this company's web presence for AI agents and LLMs?
- 8-10: Has llms.txt, permissive robots.txt, structured data, API docs, RSS feeds, sitemap
- 5-7: Has some of these (robots.txt + sitemap + structured data)
- 2-4: Only basic robots.txt or sitemap, no structured data
- 1: No machine-readable signals at all

### 2. Digital Maturity (1-10)
How modern is their technology stack?
- 8-10: Modern frameworks (React/Next.js/Vue), CDN, analytics, CRM tools, security headers
- 5-7: Some modern tooling, basic analytics, standard hosting
- 2-4: Basic HTML site, minimal tooling
- 1: Very dated technology, no modern tools detected

### 3. Data Richness (1-10)
How much structured, AI-trainable content does their site have?
- 8-10: Extensive product catalogs, documentation, knowledge base, blog, case studies
- 5-7: Moderate content — blog + some product/service descriptions
- 2-4: Minimal content, mostly marketing copy
- 1: Very sparse, almost no content

### 4. Existing Automation (1-10)
What automation and integrations do they already have?
- 8-10: Chat widgets, self-service portals, API integrations, booking systems, automation mentions
- 5-7: Some automation (e.g., a chatbot or contact form with integrations)
- 2-4: Basic forms, no chat or automation detected
- 1: No automation signals at all

## Guidelines
- Use the TECHNICAL SIGNALS section as hard evidence — these are facts, not guesses
- Score based on what you can actually observe, not assumptions
- Provide exactly 5-7 opportunities, mix of quick wins and strategic plays
- Be specific to THIS company, not generic AI advice
- Return ONLY the JSON object, no markdown or extra text"""

USER_PROMPT = """Analyze this company's website content and produce an AI readiness assessment.

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
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(content=content, technical_signals=technical_signals)}],
    )
    return _parse_response(message.content[0].text)


async def analyze_with_openai(content: str, technical_signals: str) -> ScanReport:
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-5.2",
        max_tokens=3000,
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
