import re
import wordninja
from langdetect import detect, LangDetectException
from concurrent.futures import ProcessPoolExecutor

def preprocess_text(text):
  text = re.sub(r'[\n\t\003]', ' ', text)

    if len(text.strip()) < 50:
        return None

    try:
        if detect(text) != 'en':
            return None
    except LangDetectException:
        return None 

    ligatures = {"ﬁ": "fi", "ﬂ": "fl"}
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)

    text = ' '.join(wordninja.split(text))

    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 30:
        return None

    return text

def preprocess_sentences_parallel(sentences):
    with ProcessPoolExecutor() as executor:
        preprocessed_sentences = list(executor.map(preprocess_text, sentences))
    return [sent for sent in preprocessed_sentences if sent is not None]
