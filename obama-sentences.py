"Save 1M of text as sentences, one per line"
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer

import codecs
def get_text(filename:str):
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    with codecs.open(filename, mode='r') as f:
        s = f.read()
    return s

text = get_text("notebooks/data/obama-speeches.txt").lower()
text = text[:1_000_000]

nlp = spacy.load("en_core_web_sm")
nlp.max_length = len(text)
doc = nlp(text)

sentences = [str(s).strip() for s in doc.sents]
sentences = [s for s in sentences if len(s)>0]

for s in sentences:
	print(s)
