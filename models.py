from pydantic import BaseModel


class Opportunity(BaseModel):
    title: str
    description: str
    impact: str  # "High", "Medium", "Low"
    timeframe: str  # "Quick Win (<1 month)" or "Strategic (3-6 months)"


class ScanReport(BaseModel):
    company_name: str
    company_summary: str
    ai_readiness_score: int  # 1-10
    score_justification: str
    current_tech_signals: list[str]
    opportunities: list[Opportunity]

    @property
    def quick_wins(self) -> list[Opportunity]:
        return [o for o in self.opportunities if "Quick" in o.timeframe]

    @property
    def strategic_plays(self) -> list[Opportunity]:
        return [o for o in self.opportunities if "Strategic" in o.timeframe]


class ScanRequest(BaseModel):
    url: str
    provider: str  # "claude" or "openai"
