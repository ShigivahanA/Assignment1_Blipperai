#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the VADER sentiment analyzer
nltk.download("vader_lexicon")  # Download the VADER lexicon
sia = SentimentIntensityAnalyzer()

# Create a recognizer
r = sr.Recognizer()

# File path to the single audio file (change this to your specific audio file)
audio_file_path = "E:\gggg\youtube_OfcVGnlR2TA_audio.wav"

# List to store transcriptions and sentiment analysis results
results = []

# Custom function to map compound scores to "happy," "sad," or "neutral"
def map_sentiment(compound):
    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    else:
        return "neutral"

# Process the single audio file
with sr.AudioFile(audio_file_path) as source:
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        sentiment = sia.polarity_scores(text)
        sentiment_label = map_sentiment(sentiment["compound"])  # Map to "happy," "sad," or "neutral"

        results.append(("your_audio_file.wav", text, sentiment_label))
        print(f'Transcription for your_audio_file.wav:')
        print(text)
        print(f'Sentiment Analysis for your_audio_file.wav: {sentiment_label}')
    except Exception as e:
        print(f"Error transcribing your_audio_file.wav: {str(e)}")

# Now you have the transcription and sentiment analysis results for the single audio file
print("Transcription and Sentiment Analysis for your_audio_file.wav:")
if results:
    audio_file, text, sentiment_label = results[0]
    print(f"File: {audio_file}\nTranscription: {text}\nSentiment: {sentiment_label}\n")
else:
    print("No results to display.")

