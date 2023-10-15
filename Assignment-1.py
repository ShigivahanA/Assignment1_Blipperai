#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Directory containing audio files
audio_dir = input("Enter the path")

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

# Iterate through audio files in the directory
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".wav"):  # Assuming all files are in WAV format
        audio_path = os.path.join(audio_dir, audio_file)

        with sr.AudioFile(audio_path) as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                sentiment = sia.polarity_scores(text)
                sentiment_label = map_sentiment(sentiment["compound"])  # Map to "happy," "sad," or "neutral"
                results.append((audio_file, text, sentiment_label))
                print(f'Transcription for {audio_file}:')
                print(text)
                print(f'Sentiment Analysis for {audio_file}: {sentiment_label}')
            except Exception as e:
                print(f"Error transcribing {audio_file}: {str(e)}")

# Now you have a list of transcriptions and sentiment analysis labels for each audio file
print("Transcriptions and Sentiment Analysis:")
for file, text, sentiment_label in results:
    print(f"File: {file}\nTranscription: {text}\nSentiment: {sentiment_label}\n")

# Create a Pandas DataFrame from the results
df = pd.DataFrame(results, columns=["File", "Transcription", "Sentiment"])

# Generate tables
print("Results Table:")
print(df)

# Generate a line chart for sentiment
plt.figure(figsize=(10, 6))
plt.plot(df["File"], df["Sentiment"], marker='o')
plt.xlabel("File")
plt.ylabel("Sentiment")
plt.title("Sentiment Analysis Results")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:




