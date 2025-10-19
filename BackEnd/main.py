# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import logging

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORT MODEL ---
# 假设 QwenScorer 类在 backend/model.py 中
from model import QwenScorer

# --- FASTAPI APP SETUP ---
app = FastAPI(title="AutoGrade-X: Mandarin Essay Scorer")

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE MODEL ---
scorer = QwenScorer()

# --- REQUEST/RESPONSE MODEL ---
class EssayRequest(BaseModel):
    text: str
    hsk_level: int

# --- API ENDPOINT ---
@app.post("/score")
async def score_essay(req: EssayRequest):
    """
    Endpoint utama untuk menilai esai HSK.
    """
    logger.info(f"Received essay for HSK {req.hsk_level} (length: {len(req.text)} chars).")

    # 1. Validasi input HSK level
    if req.hsk_level not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="HSK level must be 1, 2, or 3")

    try:
        # 2. --- CALL THE LLM SCORING FUNCTION ---
        # Ini memanggil metode `generate_json` di `model.py`
        raw_json_string_from_model = scorer.generate_json(req.text, req.hsk_level)
        logger.info(f"Raw JSON string from model (first 200 chars): {raw_json_string_from_model[:200]}...")

        # 3. --- PARSE THE JSON STRING FROM THE MODEL ---
        # Ubah string respons dari model menjadi objek Python
        parsed_data_from_model = json.loads(raw_json_string_from_model)
        logger.info(f"Parsed data from model: {parsed_data_from_model}")

        # 4. --- CHECK FOR MODEL ERROR OBJECT ---
        # Jika model.py mengembalikan objek error, propagasikan sebagai HTTP 500 error
        if isinstance(parsed_data_from_model, dict) and "error" in parsed_data_from_model:
            logger.warning(f"Model returned an error object: {parsed_data_from_model}")
            raise HTTPException(status_code=500, detail=f"LLM Processing Error: {parsed_data_from_model['error']}")

        # 5. --- FINAL VALIDATION & DEFAULTS ---
        # Pastikan data yang di-parse adalah dictionary
        if not isinstance(parsed_data_from_model, dict):
            raise ValueError("Parsed JSON is not a dictionary/object.")

        # Standarisasi kunci utama
        standardized_result = {
            "text": req.text, # Sertakan teks asli untuk highlighting di frontend
            "overall_score": parsed_data_from_model.get("overall_score") or parsed_data_from_model.get("Overall Score") or parsed_data_from_model.get("Overall_Score") or 0,
            "detailed_scores": {
                "grammar": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("grammar") or
                    (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Grammar") or
                    (parsed_data_from_model.get("Detailed_Scores", {}) or {}).get("Grammar") or
                    0
                ),
                "vocabulary": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("vocabulary") or
                    (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Vocabulary") or
                    (parsed_data_from_model.get("Detailed_Scores", {}) or {}).get("Vocabulary") or
                    0
                ),
                "coherence": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("coherence") or
                    (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Coherence") or
                    (parsed_data_from_model.get("Detailed_Scores", {}) or {}).get("Coherence") or
                    0
                ),
                "cultural_adaptation": (
                    (parsed_data_from_model.get("detailed_scores", {}) or {}).get("cultural_adaptation") or
                    (parsed_data_from_model.get("Detailed Scores", {}) or {}).get("Cultural Adaptation") or
                    (parsed_data_from_model.get("Detailed_Scores", {}) or {}).get("Cultural_Adaptation") or # Handle potential underscore variant
                    0
                ),
            },
            "error_list": parsed_data_from_model.get("error_list") or parsed_data_from_model.get("Error List") or parsed_data_from_model.get("Error_List") or [],
            "feedback": parsed_data_from_model.get("feedback") or parsed_data_from_model.get("Feedback") or "Tidak ada umpan balik.",
        }

        logger.info(f"Final standardized result to send: {standardized_result}")

        # 6. --- RETURN THE STANDARDIZED RESULT ---
        # FastAPI akan secara otomatis mengkonversi dictionary ini ke JSON dan mengirimkannya sebagai respons HTTP.
        return standardized_result

    except json.JSONDecodeError as e:
        # Tangani error jika model.py mengembalikan string yang bukan JSON yang valid
        logger.error(f"JSON Decode Error: {e}")
        logger.error(f"Raw output was: {raw_json_string_from_model}")
        raise HTTPException(status_code=500, detail=f"Backend error: Invalid JSON from LLM. {str(e)}")
    except HTTPException:
        # Re-raise HTTPExceptions (seperti yang dari model error check di atas)
        raise
    except Exception as e:
        # Tangani error tak terduga lainnya selama proses penilaian
        logger.exception(f"General Error in /score: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
