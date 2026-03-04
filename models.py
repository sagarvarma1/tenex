from pydantic import BaseModel, computed_field


class Opportunity(BaseModel):
    title: str
    description: str
    impact: str  # "High", "Medium", "Low"
    timeframe: str  # "Quick Win (<1 month)" or "Strategic (3-6 months)"


class DimensionScore(BaseModel):
    name: str  # "Agent/AI Readiness", "Digital Maturity", "Data Richness", "Existing Automation"
    score: int  # 1-10
    justification: str
    signals_detected: list[str]


class ScanReport(BaseModel):
    company_name: str
    company_summary: str
    dimensions: list[DimensionScore]
    opportunities: list[Opportunity]

    @computed_field
    @property
    def ai_readiness_score(self) -> int:
        if not self.dimensions:
            return 0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions))

    @property
    def quick_wins(self) -> list[Opportunity]:
        return [o for o in self.opportunities if "Quick" in o.timeframe]

    @property
    def strategic_plays(self) -> list[Opportunity]:
        return [o for o in self.opportunities if "Strategic" in o.timeframe]


class ScanRequest(BaseModel):
    url: str
    provider: str  # "claude" or "openai"
