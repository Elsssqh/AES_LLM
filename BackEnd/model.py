# backend/model.py
# Description: Integrates Qwen-1.8B-Chat LLM for automated HSK essay scoring.
#              Incorporates Jieba for preprocessing and enhanced prompt tuning.

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import logging

# --- Tambahkan import jieba ---
import jieba.posseg as pseg # Untuk POS tagging
import jieba # Untuk segmentasi dasar jika diperlukan

# Configure logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)

class QwenScorer:
    """
    Wrapper for the Qwen-1.8B-Chat model to score HSK Mandarin essays
    written by Indonesian learners, focusing on culturally specific errors.
    Integrates Jieba for preprocessing.
    """

    def __init__(self):
        """
        Initializes the QwenScorer by loading the model and tokenizer.
        """
        logger.info("Loading Qwen-1.8B-Chat...")
        try:
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-1_8B-Chat", # Use forward slash for path consistency
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded successfully.")

            # Load Model
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-1_8B-Chat",
                device_map="auto",       # Automatically distribute across available devices (GPU/CPU)
                trust_remote_code=True,
                torch_dtype="auto"       # Automatically infer the best dtype
            ).eval() # Set to evaluation mode for inference
            logger.info("Qwen-1.8B-Chat model loaded and set to evaluation mode successfully.")
            
            # --- Inisialisasi Jieba (Opsional: Muat kamus kustom jika ada) ---
            # jieba.load_userdict("path/to/your/custom_dict.txt") # Jika ada kamus kustom
            logger.info("Jieba library initialized.")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-1.8B-Chat model or tokenizer: {e}")
            raise # Re-raise the exception to halt initialization if model fails to load

    def generate_json(self, essay: str, hsk_level: int) -> str:
        """
        Generates a structured JSON assessment of a Mandarin essay by an Indonesian learner,
        focusing on errors specific to their linguistic background.
        Integrates Jieba for preprocessing.
        """
        logger.info(f"Initiating scoring for HSK {hsk_level} essay (length: {len(essay)} chars).")

        # --- PREPROCESSING DENGAN JIEBA ---
        logger.info("Starting Jieba preprocessing (segmentation and POS tagging)...")
        try:
            # Segmentasi dan POS tagging menggunakan jieba
            words_with_pos = list(pseg.cut(essay)) # List of (word, flag)
            segmented_words = [word for word, flag in words_with_pos] # List of words only
            pos_tags = [(word, flag) for word, flag in words_with_pos] # List of (word, pos_tag)
            
            # Buat representasi teks yang diformat untuk prompt
            formatted_segmentation = " ".join(segmented_words)
            formatted_pos_tags = "\n".join([f"  - '{word}': {pos_tag}" for word, pos_tag in pos_tags])
            
            logger.info(f"Jieba preprocessing completed.")
            logger.debug(f"Segmented words: {formatted_segmentation}")
            logger.debug(f"POS tags:\n{formatted_pos_tags}")
            
        except Exception as pe:
            logger.error(f"Jieba preprocessing failed: {pe}")
            # Jika Jieba gagal, gunakan teks asli tanpa preprocessing tambahan
            formatted_segmentation = essay
            formatted_pos_tags = "Jieba preprocessing failed."
            
        # Define HSK rubrics
        rubrics = {
            1: "HSK 1: 10-15 characters, S-P or S-P-O structures only.",
            2: "HSK 2: 20-40 characters, use 在, 和, simple connectors.",
            3: "HSK 3: 50-100 characters, use 的/得/地, complex sentences."
        }
        rubric_description = rubrics.get(hsk_level, rubrics[1])

        # --- EXTREMELY STRICT PROMPT ENGINEERING ---
        # Force the LLM to fill a pre-defined JSON template.
        # This is the core of prompt tuning to get consistent, structured output.
        prompt = f"""
        You are an AUTOMATED HSK MANDARIN ESSAY SCORING SYSTEM for Indonesian learners.
        You MUST respond ONLY with a SINGLE, VALID JSON object.
        DO NOT provide any text, explanations, markdown, or greetings outside the JSON.

        INPUT ESSAY (Analyze this text ONLY):
        {essay}

        TASK:
        Fill the following PRECISE JSON TEMPLATE with your assessment based on HSK {hsk_level} criteria ({rubric_description}).
        Focus ONLY on detecting the THREE specific error types common to Indonesian learners:
        1. S-P-O-K Interference (e.g., '我吃饭在餐厅' -> '我在餐厅吃饭')
        2. False Friends (e.g., '路忙' -> '路拥挤')
        3. Particle Misuse (e.g., '我妹妹是十岁' -> '我妹妹十岁')

        CRITICAL INSTRUCTIONS:
        - Analyze ONLY the provided 'INPUT ESSAY' character by character.
        - For EACH identified error of the three types listed above:
        a. Identify the EXACT SUBSTRING from the 'INPUT ESSAY' that is incorrect. This is `incorrect_fragment`.
        b. Determine the ZERO-BASED starting character index (inclusive) and the ending character index (exclusive) of this substring within the 'INPUT ESSAY'. Python slicing `essay[start:end]` should yield the `incorrect_fragment`. This is `error_position`.
        c. Provide the precise correction for that substring IN CHINESE CHARACTERS (Hanzi). This is `suggested_correction`.
        d. Classify the error type (`word_order`, `false_friend`, `particle_misuse`).
        e. Write a concise explanation IN ENGLISH explaining WHY the original fragment was incorrect and how the correction fixes it. This is `explanation`.
        - CRITICAL: If NO ERRORS of the specified critical types are found, return an empty `error_list`: [].
        - CRITICAL: DO NOT invent or hallucinate errors that are not present in the 'INPUT ESSAY'.
        - CRITICAL: DO NOT make up scores. Assign realistic scores based on the actual quality of the essay.
        - CRITICAL: DO NOT provide feedback in Chinese. ALL feedback and explanations MUST BE IN ENGLISH.
        - CRITICAL: Respond ONLY with the SINGLE JSON object. NO OTHER TEXT, MARKDOWN, OR EXPLANATIONS OUTSIDE THE JSON.
        - CRITICAL: ALL JSON KEYS MUST BE IN `snake_case` (e.g., `overall_score`, `error_type`, `error_position`, `incorrect_fragment`, `suggested_correction`, `explanation`, `detailed_scores`, `cultural_adaptation`).
        - CRITICAL: `error_position` MUST BE an array of TWO INTEGERS: [start_character_index, end_character_index], derived from the 'INPUT ESSAY' text.
        - CRITICAL: `error_list` MUST BE an array, which can be empty [].
        - CRITICAL: `detailed_scores` MUST CONTAIN keys: `grammar`, `vocabulary`, `coherence`, `cultural_adaptation`.
        - CRITICAL: Corrections (`suggested_correction`) MUST BE IN CHINESE CHARACTERS (Hanzi).
        - CRITICAL: Explanations and general feedback MUST BE IN ENGLISH.

        PRECISE JSON TEMPLATE TO FILL (Return ONLY this, with filled values):
        {{
        "overall_score": 0,
        "detailed_scores": {{
            "grammar": 0,
            "vocabulary": 0,
            "coherence": 0,
            "cultural_adaptation": 0
        }},
        "error_list": [
            {{
            "error_type": "word_order",
            "error_position": [0, 0],
            "incorrect_fragment": "",
            "suggested_correction": "",
            "explanation": ""
            }}
        ],
        "feedback": ""
        }}
        """

        try:
            logger.debug("Sending prompt to Qwen-1.8B-Chat...")
            # --- LLM INFERENCE ---
            response, _ = self.model.chat(self.tokenizer, prompt, history=None)
            logger.debug("Received raw LLM response.")

            # --- ULTRA-ROBUST JSON EXTRACTION & PARSING ---
            # 1. Strip leading/trailing whitespace
            stripped_response = response.strip()
            logger.debug(f"Stripped response length: {len(stripped_response)}")

            # 2. Aggressively remove Markdown code blocks if present
            if stripped_response.startswith("```json"):
                stripped_response = stripped_response[7:] # Remove ```json
            elif stripped_response.startswith("```"):
                stripped_response = stripped_response[3:] # Remove ```
            
            if stripped_response.endswith("```"):
                stripped_response = stripped_response[:-3] # Remove ```
            
            stripped_response = stripped_response.strip() # Strip again after removing wrappers

            # 3. Find the first '{{' and the last '}}'
            first_brace = stripped_response.find('{{')
            last_brace = stripped_response.rfind('}}')

            json_str = None
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                # 4. Extract the substring that should be the JSON object
                json_str = stripped_response[first_brace:last_brace+2] # +2 to include both braces
                logger.info(f"Extracted potential JSON block (length: {len(json_str)} chars). First 200 chars: {json_str[:200]}...")
            else:
                # Fallback: Try to find any JSON-like structure
                fallback_match = re.search(r'\{.*\}', stripped_response, re.DOTALL)
                if fallback_match:
                    json_str = fallback_match.group(0)
                    logger.info(f"Fallback regex found JSON block (length: {len(json_str)} chars). First 200 chars: {json_str[:200]}...")
                else:
                    error_msg = "No JSON block found in LLM response."
                    logger.error(error_msg)
                    return json.dumps({"error": error_msg, "raw_response": response})

            # 5. Attempt to parse the extracted string
            if json_str:
                try:
                    # Basic cleaning: Remove common problematic control characters that can break JSON
                    cleaned_json_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '')
                    # Final parsing attempt
                    parsed_data = json.loads(cleaned_json_str)
                    logger.info("Successfully parsed extracted JSON string.")
                    
                    # --- POST-PARSE VALIDATION & DEFAULTS ---
                    # Ensure the parsed data is a dictionary
                    if not isinstance(parsed_data, dict):
                        raise ValueError("Parsed JSON is not a dictionary/object.")

                    # Ensure required top-level keys exist with correct types or provide defaults
                    required_top_keys = ["overall_score", "detailed_scores", "error_list", "feedback"]
                    for key in required_top_keys:
                        if key not in parsed_data:
                            logger.warning(f"Missing key '{key}' in LLM output. Adding default.")
                            if key == "overall_score":
                                parsed_data[key] = 0
                            elif key == "detailed_scores":
                                parsed_data[key] = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0}
                            elif key == "error_list":
                                parsed_data[key] = []
                            elif key == "feedback":
                                parsed_data[key] = "Tidak ada umpan balik."

                    # Validate types of top-level keys
                    if not isinstance(parsed_data.get("overall_score", None), (int, float)):
                        parsed_data["overall_score"] = 0
                    if not isinstance(parsed_data.get("detailed_scores", None), dict):
                        parsed_data["detailed_scores"] = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0}
                    if not isinstance(parsed_data.get("error_list", None), list):
                         parsed_data["error_list"] = []
                    if not isinstance(parsed_data.get("feedback", None), str):
                         parsed_data["feedback"] = "Tidak ada umpan balik."

                    # --- FINAL STRUCTURE CHECK ---
                    # Ensure `detailed_scores` has all required keys
                    ds = parsed_data.get("detailed_scores", {})
                    required_ds_keys = ["grammar", "vocabulary", "coherence", "cultural_adaptation"]
                    for ds_key in required_ds_keys:
                        if ds_key not in ds or not isinstance(ds[ds_key], (int, float)):
                            logger.warning(f"Missing or invalid detailed score '{ds_key}'. Setting to 0.")
                            ds[ds_key] = 0
                    parsed_data["detailed_scores"] = ds

                    # --- ENHANCED ERROR LIST VALIDATION & SANITIZATION ---
                    # Ensure `error_list` items are dictionaries with required keys and types
                    # AND MOST IMPORTANTLY, validate that errors actually exist in the input essay.
                    el = parsed_data.get("error_list", [])
                    validated_el = []
                    sanitized_count = 0 # Counter for removed hallucinations
                    if isinstance(el, list):
                        for i, error_item in enumerate(el):
                            if isinstance(error_item, dict):
                                # --- CORE VALIDATION LOGIC ---
                                # 1. Get fields
                                incorrect_frag = error_item.get("incorrect_fragment", "")
                                error_pos = error_item.get("error_position", [])
                                error_type = error_item.get("error_type", "unknown_error")

                                # 2. Check if fragment and position are valid strings/lists
                                if not isinstance(incorrect_frag, str) or not isinstance(error_pos, list) or len(error_pos) != 2:
                                    logger.warning(f"Invalid error item structure in error_list[{i}]. Removing.")
                                    sanitized_count += 1
                                    continue # Skip invalid item

                                start_pos, end_pos = error_pos
                                # 3. Check if positions are valid integers within essay bounds
                                if not isinstance(start_pos, int) or not isinstance(end_pos, int) or start_pos < 0 or end_pos > len(essay) or start_pos >= end_pos:
                                     logger.warning(f"Invalid error position {error_pos} in error_list[{i}] for essay length {len(essay)}. Removing.")
                                     sanitized_count += 1
                                     continue # Skip item with invalid position

                                # 4. CRITICAL CHECK: Does the fragment at the given position match the claimed incorrect fragment?
                                extracted_from_essay = essay[start_pos:end_pos]
                                if extracted_from_essay != incorrect_frag:
                                    logger.warning(f"MISMATCH! LLM claims error '{incorrect_frag}' at {error_pos}, but essay[{start_pos}:{end_pos}] is '{extracted_from_essay}'. This is a hallucination. Removing error_list[{i}].")
                                    sanitized_count += 1
                                    # Optionally, you could add a placeholder error indicating a hallucination was detected
                                    # validated_el.append({
                                    #     "error_type": "hallucination",
                                    #     "error_position": error_pos,
                                    #     "incorrect_fragment": incorrect_frag,
                                    #     "suggested_correction": "N/A",
                                    #     "explanation": f"System detected a discrepancy: LLM reported '{incorrect_frag}' at position {error_pos}, but the actual text was '{extracted_from_essay}'. This error might be fabricated."
                                    # })
                                    continue # Remove the hallucinated error

                                # --- If validation passes, keep the error item ---
                                # (You can still do the key/type validation as before if needed, but it's less critical now)
                                validated_el.append(error_item)
                                logger.info(f"Validated genuine error at position {error_pos}: '{incorrect_frag}'")

                            else:
                                logger.warning(f"Non-dictionary item found in error_list at index {i}. Skipping.")
                                sanitized_count += 1
                        parsed_data["error_list"] = validated_el
                        if sanitized_count > 0:
                            logger.info(f"Sanitized {sanitized_count} hallucinated or invalid errors from LLM output.")
                    else:
                         logger.warning("`error_list` is not an array. Setting to empty array.")
                         parsed_data["error_list"] = []


                    # --- FINAL DATA STANDARDIZATION & FORMATTING ---
                    # This is the CRUCIAL step to ensure the output ALWAYS matches the frontend's expectations.
                    # Regardless of what the LLM returns, we build a new, clean object with the correct structure.
                    # This guarantees that main.py receives a consistent format.

                    # 1. Extract and standardize the overall score
                    overall_score_raw = parsed_data.get("overall_score") or parsed_data.get("Overall Score") or parsed_data.get("Overall_Score") or 0
                    # Clamp score between 0 and 100
                    standardized_overall_score = max(0, min(100, int(overall_score_raw)))

                    # 2. Extract and standardize detailed scores
                    raw_detailed_scores = parsed_data.get("detailed_scores") or parsed_data.get("Detailed Scores") or parsed_data.get("Detailed_Scores") or {}
                    standardized_detailed_scores = {
                        "grammar": max(0, min(100, int(
                            raw_detailed_scores.get("grammar") or
                            raw_detailed_scores.get("Grammar") or 0
                        ))),
                        "vocabulary": max(0, min(100, int(
                            raw_detailed_scores.get("vocabulary") or
                            raw_detailed_scores.get("Vocabulary") or 0
                        ))),
                        "coherence": max(0, min(100, int(
                            raw_detailed_scores.get("coherence") or
                            raw_detailed_scores.get("Coherence") or 0
                        ))),
                        "cultural_adaptation": max(0, min(100, int(
                            raw_detailed_scores.get("cultural_adaptation") or
                            raw_detailed_scores.get("Cultural Adaptation") or
                            raw_detailed_scores.get("Cultural_Adaptation") or 0
                        ))),
                    }

                    # 3. Use the validated error list
                    standardized_error_list = parsed_data.get("error_list", [])

                    # 4. Extract and standardize feedback
                    standardized_feedback = parsed_data.get("feedback") or parsed_data.get("Feedback") or "Tidak ada umpan balik."

                    # 5. Build the final, standardized result object
                    final_result = {
                        "text": essay, # Always include the original text for highlighting in frontend
                        "overall_score": standardized_overall_score,
                        "detailed_scores": standardized_detailed_scores,
                        "error_list": standardized_error_list,
                        "feedback": standardized_feedback,
                    }

                    # 6. Convert the final result object to a JSON string and return it
                    final_json_str = json.dumps(final_result, ensure_ascii=False) # ensure_ascii=False for Hanzi
                    logger.info("Final standardized JSON string generated successfully.")
                    return final_json_str # Return the clean, valid JSON string

                except json.JSONDecodeError as je:
                    logger.error(f"JSON Decode Error: {je}")
                    logger.error(f"Raw output was: {json_str if 'json_str' in locals() else 'N/A'}")
                    # Last resort: Try parsing the raw response if it looks like JSON
                    try:
                        parsed_data = json.loads(response)
                        logger.info("Parsed raw LLM response as a fallback.")
                        return json.dumps(parsed_data, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass # If raw response also fails, proceed to return error object

            # If all extraction and parsing attempts fail
            error_msg = f"Failed to decode valid JSON from extracted LLM response block."
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "raw_response": response})

        except Exception as e:
            error_msg = f"LLM inference or processing failed unexpectedly: {repr(e)}"
            logger.exception(error_msg) # Logs the full traceback
            # Return a consistent error format for the backend API to handle
            return json.dumps({
                "error": error_msg,
                "raw_response": response if 'response' in locals() else ""
            })

