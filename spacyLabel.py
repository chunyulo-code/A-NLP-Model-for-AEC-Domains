import pandas as pd
import spacy
from spacy import displacy
#SpaCy 2.x brough significant speed and accuracy improvements
#spacy.__version__

text = "Apple acquired Zoom in China on Wednesday 6th May 2020.\
This news has made Apple and Google stock jump by 5% on Dow Jones Index in the \
United States of America"

print("text is done")

nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

entities = []
labels = []
position_start = []
position_end = []

for ent in doc.ents:
    entities.append(ent)
    labels.append(ent.label_)
    position_start.append(ent.start_char)
    position_end.append(ent.end_char)

df = pd.DataFrame(
    {'Entities': entities, 'Labels': labels, 'Position_Start': position_start, 'Position_End': position_end})

print(df)

res1 = spacy.explain("ORG")
res2 = spacy.explain("GPE")

print(res1, res2)
print(nlp.pipe_names)
