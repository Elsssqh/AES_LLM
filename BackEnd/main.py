# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json # Tambahkan import json
from model import QwenScorer # Pastikan path benar

app = FastAPI(title="AutoGrade-X: Mandarin Essay Scorer")

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
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
        # Panggil model dan dapatkan string JSON
        raw_json_string = scorer.generate_json(req.text, req.hsk_level)
        print(f"Raw JSON string from model: {raw_json_string}") # Log untuk debugging

        # Parse string JSON menjadi objek Python
        parsed_data = json.loads(raw_json_string)

        # --- STANDARISASI KE SNAKE_CASE ---
        # Ambil nilai dari data mentah, prioritaskan format snake_case, fallback ke title_case / format lain
        overall_score = parsed_data.get("overall_score") or parsed_data.get("Overall_Score") or parsed_data.get("Overall Score") or 0

        raw_detailed_scores = parsed_data.get("detailed_scores") or parsed_data.get("Detailed_Scores") or parsed_data.get("Detailed Scores") or {}
        detailed_scores = {
            "grammar": (
                raw_detailed_scores.get("grammar") or
                raw_detailed_scores.get("Grammar") or 0
            ),
            "vocabulary": (
                raw_detailed_scores.get("vocabulary") or
                raw_detailed_scores.get("Vocabulary") or 0
            ),
            "coherence": (
                raw_detailed_scores.get("coherence") or
                raw_detailed_scores.get("Coherence") or 0
            ),
            "cultural_adaptation": (
                raw_detailed_scores.get("cultural_adaptation") or
                raw_detailed_scores.get("Cultural_A Adaptation") or # Coba kunci yang bermasalah
                raw_detailed_scores.get("Cultural Adaptation") or # Coba alternatif
                raw_detailed_scores.get("Cultural_Adaptation") or # Coba alternatif lain
                0
            ),
        }

        # Standarisasi error list
        raw_error_list = parsed_data.get("error_list") or parsed_data.get("Error List") or []
        standardized_error_list = []
        for raw_error in raw_error_list:
             standardized_error = {
                "error_type": raw_error.get("error_type") or raw_error.get("Error_Type") or raw_error.get("Error Type") or "Unknown",
                "error_position": raw_error.get("error_position") or raw_error.get("Error_Position") or raw_error.get("Error Position") or [None, None],
                "incorrect_fragment": raw_error.get("incorrect_fragment") or raw_error.get("Incorrect_Fragment") or raw_error.get("Incorrect Fragment") or "",
                "suggested_correction": raw_error.get("suggested_correction") or raw_error.get("Suggested_Correction") or raw_error.get("Suggested Correction") or "",
                "explanation": raw_error.get("explanation") or raw_error.get("Explanation") or "No explanation provided.",
            }
             standardized_error_list.append(standardized_error)

        feedback = parsed_data.get("feedback") or parsed_data.get("Feedback") or "Tidak ada umpan balik."

        # Buat objek hasil final dalam format snake_case
        final_result = {
            "text": req.text, # Sertakan teks asli
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "error_list": standardized_error_list,
            "feedback": feedback,
        }

        print(f"Final result to send (SNAKE_CASE): {final_result}") # Log final result sebelum dikirim

        return final_result # Kembalikan hasil yang sudah distandarisasi

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw output was: {raw_json_string}")
        raise HTTPException(status_code=500, detail=f"Backend error: Invalid JSON from LLM. {str(e)}")
    except Exception as e:
        print(f"General Error in /score: {e}")
        import traceback
        traceback.print_exc() # Cetak stack trace penuh untuk debugging
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
