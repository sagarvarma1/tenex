import json
import anthropic
import openai
from models import ScanReport

SYSTEM_PROMPT = """You are an expert AI strategy consultant. You analyze companies and assess their AI readiness.

Given scraped website content from a company, produce a detailed AI readiness assessment.

You MUST respond with valid JSON matching this exact schema:
{
  "company_name": "string",
  "company_summary": "2-3 sentence summary of what the company does",
  "ai_readiness_score": integer 1-10,
  "score_justification": "2-3 sentences explaining the score",
  "current_tech_signals": ["list of any technology/AI/automation signals detected from the site"],
  "opportunities": [
    {
      "title": "short title",
      "description": "2-3 sentence description of the AI opportunity",
      "impact": "High" | "Medium" | "Low",
      "timeframe": "Quick Win (<1 month)" | "Strategic (3-6 months)"
    }
  ]
}

Guidelines:
- Provide exactly 5-7 opportunities, mix of quick wins and strategic plays
- Be specific to THIS company, not generic AI advice
- Score justification should reference concrete observations from the site
- current_tech_signals should note any mentions of technology, data, automation, APIs, etc.
- If the site reveals very little, note that and adjust the score accordingly
- Return ONLY the JSON object, no markdown or extra text"""

USER_PROMPT = """Analyze this company's website content and produce an AI readiness assessment:

{content}"""


def _parse_response(text: str) -> ScanReport:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return ScanReport.model_validate(json.loads(cleaned))


async def analyze_with_claude(content: str) -> ScanReport:
    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(content=content)}],
    )
    return _parse_response(message.content[0].text)


async def analyze_with_openai(content: str) -> ScanReport:
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(content=content)},
        ],
    )
    return _parse_response(response.choices[0].message.content)


async def analyze(content: str, provider: str) -> ScanReport:
    if provider == "claude":
        return await analyze_with_claude(content)
    elif provider == "openai":
        return await analyze_with_openai(content)
    else:
        raise ValueError(f"Unknown provider: {provider}")
