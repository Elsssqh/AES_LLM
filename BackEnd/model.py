from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

class QwenScorer:
    def __init__(self):
        print("Loading Qwen-1.8B-Chat...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto"
        ).eval()
        print("Model loaded successfully.")

    def generate_json(self, essay: str, hsk_level: int) -> str:
        # Build prompt dynamically based on HSK level (optional enhancement)
        rubric = {
            1: "HSK 1: 10-15 karakter, hanya pola S-P atau S-P-O",
            2: "HSK 2: 20-40 karakter, gunakan 在, 和, kalimat sederhana",
            3: "HSK 3: 50-100 karakter, gunakan 的/得/地, kalimat kompleks"
        }

        prompt = f"""
        作为中文作文评分专家，请根据以下HSK {hsk_level}评分标准评估这篇作文：
        {rubric[hsk_level]}
        1. 语法准确性（的/得/地、词序、助词）
        2. 词汇水平（HSK{hsk_level}词汇覆盖率）
        3. 连贯性（逻辑连接词、段落衔接）
        4. 文化适应性（印尼学习者常见错误：S-P-O-K干扰、false friends）

        【作文内容】
        {essay}

        请用严格有效的JSON格式输出，包含：
        - 总分（100分制）
        - 分项评分（grammar, vocabulary, coherence, cultural_adaptation）
        - 错误列表（type, position, correction, explanation）
        - 针对性反馈（针对印尼学习者的改进建议）

        输出示例（仅JSON，无其他文字）：
        {{
        "overall_score": 85,
        "detailed_scores": {{
            "grammar": 90,
            "vocabulary": 80,
            "coherence": 85,
            "cultural_adaptation": 90
        }},
        "error_list": [
            {{
            "error_type": "word_order",
            "error_position": [4, 6],
            "incorrect_fragment": "吃饭在",
            "suggested_correction": "在吃饭",
            "explanation": "Struktur S-P-O-K dari Bahasa Indonesia salah. Gunakan S-K-P-O: '我在餐厅吃饭'."
            }}
        ],
        "feedback": "Esai sudah baik, tapi perhatikan urutan kata dan partikel."
        }}
        """

        try:
            response, _ = self.model.chat(self.tokenizer, prompt, history=None)
            print("Raw LLM response:", response[:200] + "..." if len(response) > 200 else response)

            # Extract JSON from response (robust extraction)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Validate JSON
                json.loads(json_str)  # Will raise exception if invalid
                return json_str
            else:
                return '{"error": "No JSON found in LLM response"}'
        except Exception as e:
            return json.dumps({
                "error": f"LLM inference failed: {str(e)}",
                "raw_response": response if 'response' in locals() else ""
            })