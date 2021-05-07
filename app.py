from flask import Flask, render_template, url_for, request
import re
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_sm
from processing import build_model, predict, load_data

nlp = spacy.load('en_core_web_md')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=["POST"])
def process():
    model = build_model(17)
    model.load_weights("model.hdf5")
    if request.method == 'POST':
        choice = request.form['taskoption']
        rawtext = request.form['rawtext']
        df = load_data()
        doc = predict(df,rawtext, model)




        ORG_named_entity = doc.loc[(doc['named entity'] == 'B-org') | (doc['named entity'] == 'I-org')]['output']
        PER_named_entity = doc.loc[(doc['named entity'] == 'B-per') | (doc['named entity'] == 'I-per')]['output']
        GEO_named_entity = doc.loc[(doc['named entity'] == 'B-geo') | (doc['named entity'] == 'I-geo')]['output']
        GPE_named_entity = doc.loc[(doc['named entity'] == 'B-gpe') | (doc['named entity'] == 'I-gpe')]['output']
        TIM_named_entity = doc.loc[(doc['named entity'] == 'B-tim') | (doc['named entity'] == 'I-tim')]['output']
        ART_named_entity = doc.loc[(doc['named entity'] == 'B-art') | (doc['named entity'] == 'I-art')]['output']
        EVE_named_entity = doc.loc[(doc['named entity'] == 'B-eve') | (doc['named entity'] == 'I-eve')]['output']
        NAT_named_entity = doc.loc[(doc['named entity'] == 'B-nat') | (doc['named entity'] == 'I-nat')]['output']
        if choice == 'organization':
            results = ORG_named_entity
            num_of_results = len(results)
        elif choice == 'person':
            results = PER_named_entity
            num_of_results = len(results)
        elif choice == 'geographical':
            results = GEO_named_entity
            num_of_results = len(results)
        elif choice == 'geopolitical':
            results = GPE_named_entity
            num_of_results = len(results)
        elif choice == 'time':
            results = TIM_named_entity
            num_of_results = len(results)
        elif choice == 'event':
            results = EVE_named_entity
            num_of_results = len(results)
        elif choice == 'art':
            results = ART_named_entity
            num_of_results = len(results)
        elif choice == 'natural phenomenon':
            results = NAT_named_entity
            num_of_results = len(results)

    return render_template("index.html", results=results, num_of_results=num_of_results)


if __name__ == '__main__':
    app.run()