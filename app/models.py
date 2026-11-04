from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str
    arxiv_id: str | None = None
    version: int | None = None

class ComparisonRequest(BaseModel):
    arxiv_id: str
    old_version: int | None = None
    new_version: int | None = None

class Change(BaseModel):
    section: str
    category: str = Field(description="substantive or cosmetic")
    change_type: str
    description: str
    before: str
    after: str

class GapReport(BaseModel):
    arxiv_id: str
    old_version: int
    new_version: int
    changes: list[Change]
    summary: str
