import re

import nltk
from cleantext import clean
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def convert_text_into_sentences(text):
    assert isinstance(text, str)

    # Normalize a text.
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
    )

    # Remove tags.
    text = re.sub("<[^<]+?>", "", text)
    # Remove invalid charactors.
    text = re.sub("[#%'\(\)\*\+\-\\\/:;<=>@^_`|~\[\]]+", "", text)
    # Convert a text into sentences.
    sentences = sent_tokenize(text)

    return sentences
