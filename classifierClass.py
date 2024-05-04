from pydantic import BaseModel
class ClassifiersReport(BaseModel):
  category: str
  precision : float
  recall : float
  f1_score : float
  support : float