
# 🌙 DREAMsense

> **To dream is human. To understand those dreams — that's where AI comes in.**

DREAMsense is an NLP and Machine Learning-based application that analyzes dream narratives and classifies their emotional tone. The project explores how Natural Language Processing can transform unstructured dream descriptions into meaningful, data-driven emotional insights.

The system processes dream text using NLP techniques such as **TF-IDF, VADER sentiment analysis, LDA topic modeling, and Named Entity Recognition (NER)**, followed by Machine Learning-based classification.

---

## ✨ Overview

Dreams are highly subjective and often contain complex emotions, symbolic elements, and recurring themes. Traditional dream interpretation relies heavily on psychological theories and subjective analysis.

DREAMsense explores an alternative approach by combining **Natural Language Processing and Machine Learning** to analyze textual dream narratives.

The system focuses primarily on:

- 🧠 Dream narrative analysis
- ❤️ Emotion and sentiment classification
- 🔍 Identification of important textual and symbolic features
- 📊 Data-driven analysis of dream patterns
- 🤖 Machine Learning-based classification
- 🌐 Interactive dream analysis through a Streamlit interface

---

## 🎯 Objectives

The main objectives of DREAMsense are:

- Build a Machine Learning model capable of analyzing and classifying dream narratives.
- Extract emotional and symbolic features from raw dream text using NLP.
- Explore the application of AI in psychological and cognitive analysis.
- Provide a user-friendly interface for entering and analyzing dreams.

---

## 🔄 Project Workflow

```text
                 Dream Narrative
                       │
                       ▼
              Text Preprocessing
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Tokenization  Lemmatization  POS Tagging
          │
          ▼
      Feature Extraction
          │
     ┌────┼────┬─────────────┐
     ▼    ▼    ▼             ▼
   TF-IDF VADER  LDA          NER
     │    │    │              │
     └────┴────┴──────────────┘
                  │
                  ▼
          Machine Learning Model
                  │
                  ▼
       Random Forest Classifier
                  │
                  ▼
       Emotion / Sentiment Output
