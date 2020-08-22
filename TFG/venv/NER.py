# ------------------ NER WITH NLTK AND SPACY ----------------------
# https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

#INFORMATION EXTRACTION
def preprocess(sent):
    sent = nltk.word_tokenize(sent) #Chunks the sentence
    sent = nltk.pos_tag(sent) #Does part-of-speach tagging -> tuples (word, part-of-speech)
    # nltk.help.upenn_tagset('RB') --> Check Tag
    return sent

sent = preprocess(ex)
#print(nltk.help.upenn_tagset('NN'))
print(sent)

#CHUNK PATTERN
"""
Rule: A noun phrase (NP) should be formed whenever the chunker finds and optional determiner (DT)
followed by any number of adjectives (JJ) and then a noun (NN)
"""
pattern = 'NP: {<DT>?<JJ>*<NN>}'

#CHUNKING

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent) #Returns a chunk tree
print("\n", cs)

#CONVERTING IT TO IOB (Inside-Outside-Beginning) TAGS -> Standard way to represent chunk structure
"""
I: Tag is inside a chunk
O: Tag is outside a chunk
B: Tag is the beginning of a chunk (only when followed by a tag of the same type without O tags between them)
"""
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

iob_tagged = tree2conlltags(cs)
print("\n")
pprint(iob_tagged)

#CLASSIFYING THE TAGS WITH A TRAINED CLASSIFIER (ne_chunk) -> Classifies as (Person, Location(GPE), Organization)

ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
print("\n")
print(ne_tree)

# (!) Google has been recognized as a person

############# USING SpaCy ##########################
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print(doc)
print("\n")
pprint([(X.text, X.label_) for X in doc.ents])
print("\n")
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

# -------------------------------- EXTRACTING NAMED ENTITY FROM AN ARTICLE ------------------------------
print ("-------------------------------- EXTRACTING NAMED ENTITY FROM AN ARTICLE ------------------------------")

from bs4 import BeautifulSoup
import requests
import re

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)

print(len(article.ents))

sentences = [x for x in article.sents]
#displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')
