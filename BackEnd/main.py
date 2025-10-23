# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import logging
import re

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORT MODEL ---
from model import QwenScorer

# --- FASTAPI APP SETUP ---
app = FastAPI(title="AutoGrade-X: Mandarin Essay Scorer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE MODEL ---
scorer = QwenScorer()


# --- Helper: Robust JSON Parser ---
def generate_json_from_llm_output(raw_output: str):
    try:
        data = json.loads(raw_output)
    except Exception:
        logging.warning("Initial JSON parse failed; trying cleanup.")
        try:
            cleaned = re.sub(r'^[^{\[]*', '', raw_output)
            cleaned = re.sub(r'[^}\]]*$', '', cleaned)
            data = json.loads(cleaned)
        except Exception:
            logging.error("Failed to parse JSON after cleanup.")
            # coba parse nested JSON seperti {"raw_response": "..."}
            try:
                outer = json.loads(raw_output)
                if isinstance(outer, dict) and "raw_response" in outer:
                    inner = json.loads(outer["raw_response"])
                    return inner
            except Exception:
                pass
            return {"error": "Failed to parse JSON from LLM output.", "raw_response": raw_output}
    return data


# --- REQUEST MODEL ---
class EssayRequest(BaseModel):
    text: str
    hsk_level: int


# --- API ENDPOINT ---
@app.post("/score")
async def score_essay(req: EssayRequest):
    """
    Endpoint utama untuk menilai esai HSK.
    """
    essay = req.text
    hsk_level = req.hsk_level

    logger.info(f"Received essay for HSK {hsk_level} (length: {len(essay)} chars).")

    if hsk_level not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="HSK level must be 1, 2, or 3")

    try:
        # 1️⃣ Generate response dari model
        raw_json_string_from_model = scorer.generate_json(essay, hsk_level)

        snippet = str(raw_json_string_from_model)[:200]
        logger.info(f"Raw JSON string from model (first 200 chars): {snippet}...")

        if isinstance(raw_json_string_from_model, dict):
            parsed_data_from_model = raw_json_string_from_model
        else:
            parsed_data_from_model = generate_json_from_llm_output(raw_json_string_from_model)


        # 4️⃣ Standarisasi hasil
        standardized_result = {
            "text": essay,
            "overall_score": parsed_data_from_model.get("overall_score")
                or parsed_data_from_model.get("Overall Score")
                or parsed_data_from_model.get("Overall_Score")
                or 0,
            "detailed_scores": {
                "grammar": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("grammar")
                    or (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Grammar")
                    or 0
                ),
                "vocabulary": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("vocabulary")
                    or (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Vocabulary")
                    or 0
                ),
                "coherence": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("coherence")
                    or (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Coherence")
                    or 0
                ),
                "cultural_adaptation": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("cultural_adaptation")
                    or (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Cultural Adaptation")
                    or 0
                ),
            },
            "error_list": parsed_data_from_model.get("error_list")
                or parsed_data_from_model.get("Error List")
                or parsed_data_from_model.get("Error_List")
                or [],
            "feedback": parsed_data_from_model.get("feedback")
                or parsed_data_from_model.get("Feedback")
                or "Tidak ada umpan balik.",
        }

        logger.info(f"Final standardized result to send: {standardized_result}")
        return standardized_result

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}")
        logger.error(f"Raw output was: {raw_json_string_from_model}")
        raise HTTPException(status_code=500, detail=f"Backend error: Invalid JSON from LLM. {str(e)}")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"General Error in /score: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
