import streamlit as st
from datetime import datetime
from underthesea import ner
import time
import re

def typewriter_effect(placeholder, text, delay=0.02):
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.text(typed_text)
        time.sleep(delay)
    placeholder.text("")
    return placeholder

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_keyword_name(sentence):
    try:
        entities = ner(sentence)
    except Exception:
        entities = []

    important_words = []
    for entity in entities:
        if len(entity) >= 4:
            word = entity[0]
            pos_tag = entity[1]
            chunk_tag = entity[2]
            ner_tag = entity[3]

            if ner_tag in ["B-ORG", "I-LOC", "B-MISC", "B-PER", "B-DATE"]:
                important_words.append(word)
            elif chunk_tag in ["B-NP", "B-AP", "B-VP"] and pos_tag in ["N", "A", "V"]:
                important_words.append(word)

    chat_name = " ".join(important_words).strip()
    chat_name = re.sub(r"\s+", " ", chat_name)

    if not chat_name:
        chat_name = "Cuộc trò chuyện mới"

    return chat_name[:50]