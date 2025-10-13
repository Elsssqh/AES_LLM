# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from model import QwenScorer

app = FastAPI(title="AutoGrade-X: Mandarin Essay Scorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = QwenScorer()

class EssayRequest(BaseModel):
    text: str
    hsk_level: int
@app.post("/score")
async def score_essay(req: EssayRequest):
    if req.hsk_level not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="HSK level must be 1, 2, or 3")

    try:
        # Pass raw essay + hsk_level to model
        raw_output = scorer.generate_json(req.text, req.hsk_level)
        
        # Parse the LLM's raw JSON response
        parsed = json.loads(raw_output)

        # ✅ STANDARDIZE KEYS TO SNAKE_CASE FOR FRONTEND
        standardized = {
            "overall_score": parsed.get("Overall Score") or parsed.get("overall_score", 0),
            "detailed_scores": {
                "grammar": (
                    parsed.get("Detailed Scores", {}).get("Grammar") or
                    parsed.get("detailed_scores", {}).get("grammar", 0)
                ),
                "vocabulary": (
                    parsed.get("Detailed Scores", {}).get("Vocabulary") or
                    parsed.get("detailed_scores", {}).get("vocabulary", 0)
                ),
                "coherence": (
                    parsed.get("Detailed Scores", {}).get("Coherence") or
                    parsed.get("detailed_scores", {}).get("coherence", 0)
                ),
                "cultural_adaptation": (
                    parsed.get("Detailed Scores", {}).get("Cultural Adaptation") or
                    parsed.get("detailed_scores", {}).get("cultural_adaptation", 0)
                ),
            },
            "error_list": (
                parsed.get("Error List") or
                parsed.get("error_list", [])
            ),
            "feedback": parsed.get("Feedback") or parsed.get("feedback", "Tidak ada umpan balik."),
            "text": req.text  # Always include original text for highlighting
        }

        return standardized

    except Exception as e:
        print("Backend error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}"
        )