# Necessary imports
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# Headings for Web Application
st.title("Analyze your Text")
st.subheader("What type of NLP service would you like to use?")

# Picking what NLP task you want to do
# option is stored in this variable
option = st.selectbox(
    'NLP Service', ('Sentiment Analysis', 'Entity Extraction'))

# Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text')  # text is stored in this variable

# Display results of the NLP task
st.header("Results")

# Function to take in dictionary of entities, type of entity, and returns specific entities of specific type


def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList


# Sentiment Analysis
if option == 'Sentiment Analysis':

    # Creating graph for sentiment across each sentence in the text inputted
    sents = sent_tokenize(text)
    entireText = TextBlob(text)
    sentScores = []
    for sent in sents:
        text = TextBlob(sent)
        score = text.sentiment[0]
        sentScores.append(score)

    # Plotting sentiment scores per sentencein line graph
    st.line_chart(sentScores)
    st.write("Line Graph shows the sentiments of each sentence from a given text.")
    st.write(" ")

    # Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    st.write("The sentiment of the overall text below.")
    st.write(sentimentTotal)
    st.write(
        "Polarity is in between [-1,1]. It indicated the sentiment like whether the text expresses negative(-1), positive(1) or neutral sentiment.")
    st.write(
        "Subjectivity is in between [0,1], where 0 signifies objective and 1 signifies subjective.")

# Named Entity Recognition
else:

    # Getting Entity and type of Entity
    entities = []
    entityLabels = []
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    # Creating dictionary with entity and entity types
    entDict = dict(zip(entities, entityLabels))

    # Using function to create lists of entities of each type
    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    # Displaying entities of each type
    st.write("Organization Entities: " + str(entOrg))
    st.write("Cardinal Entities: " + str(entCardinal))
    st.write("Personal Entities: " + str(entPerson))
    st.write("Date Entities: " + str(entDate))
    st.write("GPE Entities: " + str(entGPE))
