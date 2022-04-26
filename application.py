import pandas as pd
import spacy
from spacy import displacy
import pickle
from spacy import displacy

nlp = spacy.load("ner")

doc=nlp("They wanna check the northeast windows of the apartment")
displacy.render(doc, style="ent", jupyter=True)
print(doc.ents)

doc=nlp("show us the east side of the park")
displacy.render(doc, style="ent", jupyter=True)
print(doc.ents)

doc=nlp("I'm curious if the southern façade can be changed a bit more")
displacy.render(doc, style="ent", jupyter=True)
print(doc.ents)