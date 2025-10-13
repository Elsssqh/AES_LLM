import jieba

def normalize_text(text: str) -> str:
    return text.strip().replace(" ", "").replace("　", "")

def segment_hanzi(text: str):
    return list(jieba.cut(text))

def validate_hsk_vocab(tokens, hsk_level=3):
    # Simplified: in real system, load official HSK wordlists
    hsk1_words = {"我", "是", "的", "在", "了", "有", "和", "你", "们", "好"}
    hsk2_words = {"学习", "喜欢", "吃饭", "朋友", "家", "去", "来"}
    hsk3_words = {"餐厅", "医生", "老师", "妹妹", "雅加达", "住"}

    hsk_vocab = hsk1_words
    if hsk_level >= 2:
        hsk_vocab |= hsk2_words
    if hsk_level >= 3:
        hsk_vocab |= hsk3_words

    unknown = [w for w in tokens if w not in hsk_vocab and len(w) > 1]
    return unknown