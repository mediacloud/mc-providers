import logging
import os
import re
from collections import Counter
from typing import List

# PyPI
import fasttext
import spacy

logger = logging.getLogger(__name__)

# this install.sh script will have downloaded and moved this to the proper place
MODEL_NAME = 'lid.176.bin'

this_dir = os.path.dirname(os.path.realpath(__file__))

# manage this like a lazy-loaded singleton so it is fast after the first time
_stopwords_by_language = {}


fasttext_model: fasttext.FastText._FastText | None = None

# XXX possibly tell spacy what NOT to load?
# https://spacy.io/usage/processing-pipelines#disabling

def _get_model() -> fasttext.FastText._FastText:
    try:
        global fasttext_model
        if fasttext_model is None:
            fasttext_model = fasttext.load_model(os.path.join(this_dir, MODEL_NAME))
        return fasttext_model
    except ValueError:
        raise ValueError("Couldn't load fasttext lang detection model - make sure install.sh ran and saved to {}".format(
            os.path.join(this_dir, MODEL_NAME)))


def detect(text: str) -> list[list[list[str]]]:
    cleaned_text = text.replace('\n', '')
    return _get_model().predict([cleaned_text])  # [['__label__en']], [array([0.9331119], dtype=float32)]


def top_detected(text: str) -> str:
    guesses = detect(text)
    return guesses[0][0][0].replace('__label__', '')


def stopwords_for_language(lang_code: str) -> set:
    # manage the _stopwords_by_language dict, from alpha2 to list
    if len(lang_code) != 2:
        raise RuntimeError('Invalid language "{}" - use 2 letter alpha code'.format(lang_code))
    if lang_code not in _stopwords_by_language:
        file_path = os.path.join(this_dir, '{}_stop_words.txt'.format(lang_code))
        if not os.path.exists(file_path):
            logger.info('Language "{}" has no stopwords list, accepting all terms'.format(lang_code))
            return set()
        with open(file_path) as f:
            lines = f.read().splitlines()
            _stopwords_by_language[lang_code] = set(line.strip() for line in lines
                                                 if not line.startswith('#') and len(line) > 0)
    return _stopwords_by_language[lang_code]


_IGNORE_BY_LANG = {
    "en": set(["'s"]),
}
_DEFAULT_IGNORE: set[str] = set()

_MIN_WORD_LENGTH = 2            # zero to disable

def terms_without_stopwords(lang_code: str, text: str) -> List[str]:
    """
    backwards compatibility: take text for single document, return terms
    """
    return terms_without_stopwords_list(lang_code, [text])[0]

def terms_without_stopwords_list(lang_code: str, texts: list[str],
                                 min_word_length: int = _MIN_WORD_LENGTH) -> List[List[str]]:
    """
    take list of documents, return list of token lists
    (for both total term counts and per-document counts)
    """
    try:
        lang_stopwords = stopwords_for_language(lang_code)
    except RuntimeError:
        # no stopwords for this language, so just let them all through
        logger.info(f"No stopwords for {lang_code}")
        lang_stopwords = set()
    ignore = _IGNORE_BY_LANG.get(lang_code, _DEFAULT_IGNORE)

    # See https://github.com/mediacloud/mc-providers/issues/54
    # "xx" means the multilingual, language-agnostic tokenization
    nlp = spacy.blank("xx")

    results: list[list[str]] = []   # list of list of terms
    for text in texts:
        terms: list[str] = []

        # from https://github.com/mediacloud/backend/blob/master/apps/common/src/python/mediawords/languages/__init__.py#L128
        # Normalize apostrophe so that "it’s" and "it's" get counted together
        # NOTE: https://stackoverflow.com/questions/67229023/correctly-tokenize-english-contractions-with-unicode-apostrophes
        # has a longer list: '\u02B9', '\u02BB', '\u02BC', '\u02BD', '\u02C8', '\u02CA', '\u02CB'
        text = text.replace("’", "'")

        for t in nlp(text):
            if t.is_punct or t.is_space or t.is_currency or len(t.text) < min_word_length:
                continue
            term = t.lower_
            if term in lang_stopwords or term in ignore:
                continue
            terms.append(term)
        results.append(terms)
    return results
