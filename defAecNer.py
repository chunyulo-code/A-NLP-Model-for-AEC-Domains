# import libs --------------------

from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training import Example
from spacy import displacy
import pickle



# prepare training data --------------------

TRAIN_DATA = [
    ("Check the south facade of the building", {"entities": [(10, 15, "DIRECTION"), (30, 38, "BUILDING")]}),
    ("I wanna see how is the north side of the stadium look like", {"entities": [(23, 28, "DIRECTION"), (41, 48, "BUILDING")]}),
    ("Boss wanna see how is the west side of the hospital look like", {"entities": [(26, 30, "DIRECTION"), (43, 51, "BUILDING")]}),
    ("Go east side of that tower", {"entities": [(3, 7, "DIRECTION"), (21, 26, "BUILDING")]}),
    ("Fly above the playground", {"entities": [(4, 9, "DIRECTION"), (14, 24, "BUILDING")]}),
    ("Fly above the soccer field", {"entities": [(4, 9, "DIRECTION"), (14, 26, "BUILDING")]}),
    ("Fly to the west of the basketball court", {"entities": [(11, 15, "DIRECTION"), (23, 39, "BUILDING")]}),
    ("Go to the southwest side of the bridge", {"entities": [(10, 19, "DIRECTION"), (32, 38, "BUILDING")]}),
]



# create blank model --------------------

model = None
output_dir=Path("ner/")
n_iter=100

   #load the model

if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  
    print("Created blank 'en' model")


# create pipeline --------------------

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')


# start training --------------------

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
example = []
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                [example], 
                drop=0.5,  
                sgd=optimizer,
                losses=losses)
        print(losses)




if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
pickle.dump(nlp, open( "aecNlp.pkl", "wb" ))



doc=nlp("They wanna check the northeast windows of the apartment")
displacy.render(doc, style="ent", jupyter=True)
print(doc.ents)

doc=nlp("show us the east side of the park")
displacy.render(doc, style="ent", jupyter=True)
print(doc.ents)