# backend/model.py
# Author: Elsie
# Date: 2025
# Description: Integrates Qwen-1.8B-Chat LLM for automated HSK essay scoring.

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)

class QwenScorer:
    """
    Wrapper for the Qwen-1.8B-Chat model to score HSK Mandarin essays
    written by Indonesian learners, focusing on culturally specific errors.
    """

    def __init__(self):
        """
        Initializes the QwenScorer by loading the model and tokenizer.
        """
        logger.info("Starting to load Qwen-1.8B-Chat model...")
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
        except Exception as e:
            logger.error(f"Failed to load Qwen-1.8B-Chat model or tokenizer: {e}")
            raise # Re-raise the exception to halt initialization if model fails to load

    def generate_json(self, essay: str, hsk_level: int) -> str:
        """
        Generates a structured JSON assessment of a Mandarin essay by an Indonesian learner,
        focusing on errors specific to their linguistic background.

        Args:
            essay (str): The HSK essay text to be scored.
            hsk_level (int): The HSK level (1, 2, or 3) of the essay.

        Returns:
            str: A JSON-formatted string representing the assessment.
                 On success, it contains the score, details, errors, and feedback.
                 On failure, it contains an error message.
        """
        logger.info(f"Initiating scoring for HSK {hsk_level} essay (length: {len(essay)} chars).")

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
        prompt =(
            "You are an automated assessment tool for HSK Mandarin essays by Indonesian learners. "
            "You MUST respond ONLY with a SINGLE, VALID JSON object. "
            "DO NOT provide any text, explanations, markdown, or greetings outside the JSON.\n\n"

            f"INPUT ESSAY (Analyze this text ONLY):\n{essay}\n\n"

            f"TASK:\n"
            f"Fill the following PRECISE JSON TEMPLATE with your assessment based on HSK {hsk_level} criteria "
            f"({rubric_description}). "
            "Focus on detecting the THREE specific error types common to Indonesian learners:\n"
            "1. S-P-O-K Interference (e.g., '我吃饭在餐厅' -> '我在餐厅吃饭')\n"
            "2. False Friends (e.g., '路忙' -> '路拥挤')\n"
            "3. Particle Misuse (e.g., '我妹妹是十岁' -> '我妹妹十岁')\n\n"

            "Instructions for filling the template:\n"
            "- Analyze ONLY the provided 'INPUT ESSAY' character by character.\n"
            "- For EACH identified error of the three types listed above:\n"
            "   a. Identify the EXACT SUBSTRING from the 'INPUT ESSAY' that is incorrect. This is `incorrect_fragment`.\n"
            "   b. Determine the ZERO-BASED starting character index (inclusive) and the ending character index (exclusive) of this substring within the 'INPUT ESSAY'. Python slicing `essay[start:end]` should yield the `incorrect_fragment`. This is `error_position`.\n"
            "   c. Provide the precise correction for that substring IN CHINESE CHARACTERS (Hanzi). This is `suggested_correction`.\n"
            "   d. Classify the error type (`word_order`, `false_friend`, `particle_misuse`).\n"
            "   e. Write a concise explanation IN ENGLISH explaining WHY the original fragment was incorrect and how the correction fixes it. This is `explanation`.\n"
            "- CRITICAL: If NO ERRORS of the specified critical types are found, return an empty `error_list`: [].\n"
            "- CRITICAL: Output MUST be a SINGLE, VALID JSON OBJECT enclosed by {}. NO OTHER TEXT, MARKDOWN, OR EXPLANATIONS OUTSIDE THE JSON.\n"
            "- CRITICAL: ALL JSON KEYS MUST BE IN `snake_case` (e.g., `overall_score`, `error_type`, `error_position`, `incorrect_fragment`, `suggested_correction`, `explanation`, `detailed_scores`, `cultural_adaptation`).\n"
            "- CRITICAL: `error_position` MUST BE an array of TWO INTEGERS: [start_character_index, end_character_index], derived from the 'INPUT ESSAY' text.\n"
            "- CRITICAL: `error_list` MUST BE an array, which can be empty [].\n"
            "- CRITICAL: `detailed_scores` MUST CONTAIN keys: `grammar`, `vocabulary`, `coherence`, `cultural_adaptation`.\n"
            "- CRITICAL: Explanations and general feedback MUST BE IN ENGLISH.\n"
            "- CRITICAL: Corrections (`suggested_correction`) MUST BE IN CHINESE CHARACTERS (Hanzi).\n\n"

            "PRECISE JSON TEMPLATE TO FILL (Return ONLY this, with filled values):\n"
            "{\n"
            '  "overall_score": 0,\n' # Integer, overall score from 0 to 100
            '  "detailed_scores": {\n'
            '    "grammar": 0,\n'       # Integer, score for grammar (0-100)
            '    "vocabulary": 0,\n'    # Integer, score for vocabulary (0-100)
            '    "coherence": 0,\n'     # Integer, score for coherence (0-100)
            '    "cultural_adaptation": 0\n' # Integer, score for detecting learner-specific errors (0-100)
            '  },\n'
            '  "error_list": [\n' # Array of specific errors found, can be empty []
            '    {\n'
            '      "error_type": "word_order",\n' # String, MUST BE ONE OF: "word_order", "false_friend", "particle_misuse"
            '      "error_position": [0, 0],\n' # Array of two integers, [start_index, end_index] BASED ON THE INPUT ESSAY
            '      "incorrect_fragment": "",\n' # String, the EXACT erroneous text copied from the INPUT ESSAY
            '      "suggested_correction": "",\n' # String, the corrected Chinese text (Hanzi)
            '      "explanation": ""\n' # String, reason for error and correction IN ENGLISH
            '    }\n'
            '  ],\n'
            '  "feedback": ""\n'
            "}\n\n"
            "FILL TEMPLATE NOW. OUTPUT ONLY THE JSON."
        )  # Removed invalid C-style comment that broke the string concatenation

        try:
            logger.debug("Sending prompt to Qwen-1.8B-Chat...")
            # --- LLM INFERENCE ---
            response, _ = self.model.chat(self.tokenizer, prompt, history=None)
            logger.debug("Received raw LLM response.")

            # --- ULTRA-ROBUST JSON EXTRACTION & PARSING ---
            # 1. Strip leading/trailing whitespace
            stripped_response = response.strip()
            logger.debug(f"Stripped response length: {len(stripped_response)}")

            # 2. Find the first '{' and the last '}'
            first_brace = stripped_response.find('{')
            last_brace = stripped_response.rfind('}')

            json_str = None
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                # 3. Extract the substring that should be the JSON object
                json_str = stripped_response[first_brace:last_brace+1]
                logger.info(f"Extracted potential JSON block (length: {len(json_str)} chars). First 200 chars: {json_str[:200]}...")
            else:
                error_msg = "No balanced JSON braces {} found in LLM response."
                logger.warning(error_msg)
                return json.dumps({"error": error_msg, "raw_response": response})

            # 4. Attempt to parse the extracted string
            if json_str:
                try:
                    # Basic cleaning: Remove common problematic control characters that can break JSON
                    # Be very cautious here. Over-cleaning can corrupt legitimate content.
                    # Removing newlines, carriage returns, tabs is generally safe for JSON *values* if LLM escaped them poorly.
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
                                parsed_data[key] = "No feedback generated."

                    # Validate types of top-level keys
                    if not isinstance(parsed_data.get("overall_score", None), (int, float)):
                        parsed_data["overall_score"] = 0
                    if not isinstance(parsed_data.get("detailed_scores", None), dict):
                        parsed_data["detailed_scores"] = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0}
                    if not isinstance(parsed_data.get("error_list", None), list):
                         parsed_data["error_list"] = []
                    if not isinstance(parsed_data.get("feedback", None), str):
                         parsed_data["feedback"] = "No feedback generated."

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


                    # Return the validated and potentially corrected JSON string
                    final_json_str = json.dumps(parsed_data, ensure_ascii=False) # ensure_ascii=False for Hanzi
                    logger.info("Final validated JSON string generated successfully.")
                    return final_json_str

                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing failed after extraction/cleaning: {je}")
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
                "raw_response": response if 'response' in locals() else "(No response captured)"
            })
