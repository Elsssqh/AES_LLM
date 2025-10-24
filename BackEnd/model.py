# -*- coding: utf-8 -*-
# Pastikan encoding UTF-8 di awal

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import logging
import math
from typing import List, Tuple, Dict, Optional, Any
import time # Impor time
import jieba
import jieba.posseg as pseg
import torch

# ---------------- Logger ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# ---------------- Helpers ----------------
# (Helper functions cosine_similarity, etc. tetap sama)
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2): return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0: return 0.0
    return dot / (n1 * n2)

# ---------------- QwenScorer (Nama Kelas Sesuai Import) ----------------

class QwenScorer:
    """
    Implementasi Chain of Prompts (Rantai Prompt)
    dengan instruksi prompt dalam Bahasa Mandarin.
    """

    def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
        logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Menggunakan device: {self.device}")
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Tokenizer berhasil dimuat.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True, torch_dtype="auto"
            ).eval()
            logger.info("Model Qwen-1.8B berhasil dimuat dan diatur ke mode eval.")
        except Exception as e:
            logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
            raise
        try:
            jieba.setLogLevel(logging.INFO)
            jieba.initialize()
            logger.info("Jieba berhasil diinisialisasi.")
        except Exception as e:
            logger.warning(f"Gagal inisialisasi Jieba sepenuhnya: {e}")
            pass

        self.rubric_weights = {
            "grammar": 0.30,
            "vocabulary": 0.30,
            "coherence": 0.20,
            "cultural_adaptation": 0.20
        }
        logger.info(f"Rubric weights set (untuk output JSON): {self.rubric_weights}")

    def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
        # (Fungsi ini tidak berubah)
        try:
            cleaned_essay = re.sub(r'\s+', '', essay).strip()
            if not cleaned_essay:
                 logger.warning("Input esai kosong setelah dibersihkan.")
                 return "", ""
            words_with_pos = list(pseg.cut(cleaned_essay))
            segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
            pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
            logger.debug(f"Jieba Segmented: {segmented}")
            return segmented, pos_lines
        except Exception as e:
            logger.exception("Preprocessing Jieba gagal.")
            return essay, "Jieba preprocessing gagal."

    # --- LANGKAH 1: PROMPT DETEKSI KESALAHAN (Versi Mandarin) ---
    
    def _build_error_detection_prompt(self, essay: str) -> str:
        """Membangun prompt yang HANYA fokus mencari kesalahan (dalam B. Mandarin)."""
        return f"""
        您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。
        您的任务【仅仅】是找出下文中的语法、词汇或语序错误。

        请【严格】遵守以下格式：
        - 如果发现错误，请使用此格式： 错误类型 | 错误原文 | 修正建议 | 简短解释
        - 每个错误占一行。
        - 如果【没有发现任何错误】，请【只】回答 'TIDAK ADA KESALAHAN'。

        --- 示例 ---
        示例 1:
        输入: 我妹妹是十岁。
        输出: 助词误用(是) | 我妹妹是十岁 | 我妹妹十岁 | 表达年龄时通常不需要'是'。

        示例 2:
        输入: 我们住雅加达在。
        输出: 语序干扰(SPOK) | 我们住雅加达在 | 我们住在雅加达 | 地点状语(在雅加达)应放在动词(住)之前。

        示例 3:
        输入: 路很忙。
        输出: 词语误用(False Friend) | 路很忙 | 路很拥挤 | '忙'(máng)通常用于人，而非道路。

        示例 4:
        输入: 我喜欢学中文。
        输出: TIDAK ADA KESALAHAN
        --- 示例结束 ---

        --- 主要任务 ---
        请分析以下作文，找出所有错误。请严格遵守格式。

        作文：
        "{essay}"
        """

    def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
        """
        Mem-parsing output teks dari _build_error_detection_prompt.
        (Fungsi ini tidak berubah, karena 'TIDAK ADA KESALAHAN' dan '|' bersifat universal)
        """
        validated_error_list = []
        # Keyword 'TIDAK ADA KESALAHAN' sengaja tidak diterjemahkan agar unik
        if "TIDAK ADA KESALAHAN" in error_response or error_response.strip() == "":
            return []
        
        # Pola regex untuk menangkap 4 bagian yang dipisahkan oleh '|'
        pattern = re.compile(r"(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)")
        
        for line in error_response.split('\n'):
            line = line.strip()
            match = pattern.search(line)
            
            if match:
                try:
                    err_type = match.group(1).strip()
                    incorrect_frag = match.group(2).strip()
                    correction = match.group(3).strip()
                    explanation = match.group(4).strip()
                    
                    start_index = essay_text.find(incorrect_frag)
                    if start_index != -1:
                        end_index = start_index + len(incorrect_frag)
                        pos = [start_index, end_index]
                    else:
                        logger.warning(f"Tidak dapat menemukan posisi untuk fragmen: '{incorrect_frag}'. Menggunakan posisi default [0, 0].")
                        pos = [0, 0]

                    validated_error_list.append({
                        "error_type": err_type,
                        "error_position": pos,
                        "incorrect_fragment": incorrect_frag,
                        "suggested_correction": correction,
                        "explanation": explanation
                    })
                except Exception as e:
                    logger.warning(f"Gagal mem-parsing baris error: '{line}'. Error: {e}")
                    
        return validated_error_list

    # --- LANGKAH 2: PROMPT PENILAIAN (Versi Mandarin) ---

    def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
        """
        Membangun prompt yang HANYA fokus memberi skor (dalam B. Mandarin).
        VERSI DIPERKETAT v3: Menghapus SEMUA distraksi (termasuk error_summary).
        """
        
        # KITA HAPUS error_summary. Ternyata ini mengganggu model kecil.
        # error_summary = "未检测到语法错误。"
        # if detected_errors:
        #     error_summary = f"检测到 {len(detected_errors)} 个错误。" 
        
        return f"""
        您是HSK作文评分员。
        您的任务【仅仅】是提供分数（0-100）。
        【不要】写任何评语或解释。

        HSK等级: {hsk_level}
        作文: "{essay}"

        请【必须】按照以下纯文本格式提供所有5个分数。
        【不要】写任何其他文字。

        语法准确性: [分数]
        词汇水平: [分数]
        篇章连贯: [分数]
        任务完成度: [分数]
        总体得分: [分数]
        """

    def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Mengekstrak skor dari output teks.
        (Fungsi ini tidak berubah, sudah mendukung keyword Mandarin)
        """
        try:
            extracted_data = {"score": {}}
            found_any_score = False
            patterns = {
                # Regex sudah mencakup keyword Mandarin (语法准确性) dan Inggris (grammar)
                "grammar": r"(?:语法准确性|grammar)\s*[:：分]?\s*(\d{1,3})",
                "vocabulary": r"(?:词汇水平|vocabulary)\s*[:：分]?\s*(\d{1,3})",
                "coherence": r"(?:篇章连贯|连贯性|coherence)\s*[:：分]?\s*(\d{1,3})",
                "task_fulfillment": r"(?:任务完成度|task_fulfillment|cultural_adaptation)\s*[:：分]?\s*(\d{1,3})",
                "overall": r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    score_val = int(match.group(1))
                    score_clamped = max(0, min(100, score_val))
                    extracted_data["score"][key] = score_clamped
                    found_any_score = True
                    logger.debug(f"Parser Skor: Ditemukan skor {key}={score_clamped}")

            if not found_any_score:
                 logger.warning("Parser Skor: Tidak ada skor yang dapat diekstrak dari teks.")
                 return None
            
            extracted_data["feedback"] = ""
            extracted_data["errors"] = []
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Parser Skor: Ekstraksi skor dari teks gagal total: {e}")
            return None

    # --- LANGKAH 3: PROMPT UMPAN BALIK (Versi Mandarin) ---
    
    def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
        """Membangun prompt yang HANYA menghasilkan feedback kualitatif (dalam B. Mandarin)."""
        
        # Gunakan key bahasa Inggris (grammar, vocab, dll) karena itu yang disimpan
        # oleh _extract_scores_from_text
        score_summary = (
            f"总体得分: {scores.get('overall', 'N/A')}, "
            f"语法: {scores.get('grammar', 'N/A')}, "
            f"词汇: {scores.get('vocabulary', 'N/A')}"
        )
        
        error_summary = "未发现主要错误。"
        if errors:
            error_summary = "发现的主要错误:\n"
            for err in errors[:2]: # Tampilkan maks 2 kesalahan
                error_summary += f"- {err.get('explanation', 'N/A')}\n"

        return f"""
您是一位友好且善于鼓励的中文老师。
您的任务是根据学生的作文、得分和错误，写一段简短的评语（2-3句话）。
请用中文书写评语，并在括号()中附上简短的印尼语翻译。

学生作文:
"{essay}"

所得分数:
{score_summary}

错误备注:
{error_summary}

请现在撰写您的评语：
"""

    # --- FUNGSI UTAMA: GENERATE_JSON ---

    def generate_json(self, essay: str, hsk_level: int = 3) -> str:
        """
        Fungsi utama untuk menilai esai menggunakan arsitektur Chain of Prompts.
        (Logika fungsi ini tidak berubah, hanya memanggil prompt baru)
        """
        start_time = time.time()
        logger.info(f"Menerima permintaan penilaian (generate_json) untuk esai HSK {hsk_level} (panjang: {len(essay)} karakter).")

        if not essay or not essay.strip():
            logger.warning("Input esai kosong atau hanya berisi spasi.")
            error_result = {"error": "Input esai kosong.", "essay": essay}
            duration = time.time() - start_time
            error_result["processing_time"] = f"{duration:.2f} detik"
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        # --- LANGKAH 1: DETEKSI KESALAHAN ---
        logger.info("Memulai Langkah 1: Mendeteksi Kesalahan...")
        validated_error_list = []
        try:
            error_prompt = self._build_error_detection_prompt(essay)
            error_response, _ = self.model.chat(self.tokenizer, error_prompt, history=None)
            logger.debug(f"Langkah 1 (Raw Response): {error_response}")
            validated_error_list = self._parse_errors_from_text(error_response, essay)
            logger.info(f"Langkah 1 Selesai. Ditemukan {len(validated_error_list)} kesalahan.")
        except Exception as e:
            logger.exception("Langkah 1 (Deteksi Kesalahan) Gagal.")
            validated_error_list = []

        # --- LANGKAH 2: PENILAIAN ---
        logger.info("Memulai Langkah 2: Memberikan Skor...")
        parsed_scores = {}
        grammar_s, vocab_s, coherence_s, cultural_s, overall_s = 0, 0, 0, 0, 0
        try:
            scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
            scoring_response, _ = self.model.chat(self.tokenizer, scoring_prompt, history=None)
            logger.debug(f"Langkah 2 (Raw Response): {scoring_response}")
            
            parsed_scores_data = self._extract_scores_from_text(scoring_response)
            if not parsed_scores_data or "score" not in parsed_scores_data:
                logger.error("Langkah 2 Gagal: Tidak dapat mem-parsing skor dari model.")
                raise ValueError("Gagal mem-parsing skor.")
                
            parsed_scores = parsed_scores_data.get("score", {})
            
            grammar_s = parsed_scores.get("grammar", 0)
            vocab_s = parsed_scores.get("vocabulary", 0)
            coherence_s = parsed_scores.get("coherence", 0)
            cultural_s = parsed_scores.get("task_fulfillment", 0)
            overall_s = parsed_scores.get("overall", 0)

            if overall_s == 0 and (grammar_s > 0 or vocab_s > 0):
                logger.info("Skor 'overall' tidak ada/0. Menghitung berdasarkan bobot rubrik...")
                calc_score = (grammar_s * self.rubric_weights["grammar"]) + \
                             (vocab_s * self.rubric_weights["vocabulary"]) + \
                             (coherence_s * self.rubric_weights["coherence"]) + \
                             (cultural_s * self.rubric_weights["cultural_adaptation"])
                overall_s = max(0, min(100, int(round(calc_score))))

            logger.info(f"Langkah 2 Selesai. Skor diterima (Overall: {overall_s}).")

        except Exception as e:
            logger.exception("Langkah 2 (Penilaian) Gagal Total.")
            parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "task_fulfillment": 0, "overall": 0}


        # --- LANGKAH 3: UMPAN BALIK ---
        logger.info("Memulai Langkah 3: Menghasilkan Feedback...")
        feedback = "Gagal menghasilkan feedback."
        try:
            # `parsed_scores` menggunakan key B. Inggris (grammar, vocab), ini sesuai
            feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
            feedback_response, _ = self.model.chat(self.tokenizer, feedback_prompt, history=None)
            feedback = feedback_response.strip()
            
            if not feedback:
                if not validated_error_list and overall_s > 80:
                    feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik, tidak ditemukan kesalahan signifikan. Teruslah berusaha!)"
                elif validated_error_list:
                     feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan dalam esai, silakan periksa daftar kesalahan untuk detailnya.)"
                else:
                    feedback = "请重新检查你的作文。(Harap periksa kembali esai Anda.)"
            
            logger.info("Langkah 3 Selesai.")
        except Exception as e:
            logger.exception("Langkah 3 (Feedback) Gagal.")
            if validated_error_list:
                feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan dalam esai, silakan periksa daftar kesalahan untuk detailnya.)"
            elif overall_s > 80:
                feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik, tidak ditemukan kesalahan signifikan. Teruslah berusaha!)"


        # --- FINAL: PERAKITAN JSON ---
        final_result = {
            "text": essay,
            "overall_score": overall_s,
            "detailed_scores": {
                "grammar": grammar_s,
                "vocabulary": vocab_s,
                "coherence": coherence_s,
                "cultural_adaptation": cultural_s 
            },
            "error_list": validated_error_list,
            "feedback": feedback
        }

        end_time = time.time()
        duration = end_time - start_time
        final_result["processing_time"] = f"{duration:.2f} detik"
        logger.info(f"Semua langkah selesai. Waktu pemrosesan: {duration:.2f} detik")

        return json.dumps(final_result, ensure_ascii=False, indent=2)


# ---------------- Simulasi (Main execution) ----------------
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    try:
        scorer = QwenScorer()

        logger.info("="*50 + "\nMENJALANKAN SIMULASI 1: Esai Lampiran 2 (Esai Bagus)\n" + "="*50)
        essay_lampiran_2 = "上个星期六，我和朋友去公园玩。我们早上九点起床。我吃早饭，然后穿衣服。朋友开车带我们去公园。公园里有很多人。我们放风筝，吃午饭，然后回家。我玩得很开心。"
        result_json_1 = scorer.generate_json(essay_lampiran_2, hsk_level=2)
        print("\n--- HASIL SIMULASI 1 (JSON) ---")
        print(result_json_1)
        print("---------------------------------\n")

        logger.info("="*50 + "\nMENJALANKAN SIMULASI 2: Esai dengan Kesalahan Khas\n" + "="*50)
        essay_errors = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
        result_json_2 = scorer.generate_json(essay_errors, hsk_level=3)
        print("\n--- HASIL SIMULASI 2 (JSON) ---")
        print(result_json_2)
        print("---------------------------------\n")

        logger.info("="*50 + "\nMENJALANKAN SIMULASI 3: Esai Pendek Bagus\n" + "="*50)
        essay_short_good = "我喜欢学中文。"
        result_json_3 = scorer.generate_json(essay_short_good, hsk_level=1)
        print("\n--- HASIL SIMULASI 3 (JSON) ---")
        print(result_json_3)
        print("---------------------------------\n")

        logger.info("Semua simulasi selesai.")
    except Exception as e:
        logger.critical(f"Gagal menjalankan program utama: {e}", exc_info=True)
        logger.critical("Pastikan koneksi internet, 'transformers', 'torch', 'jieba' terinstal.")