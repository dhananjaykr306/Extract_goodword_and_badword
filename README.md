# Extract_goodword_and_badword
# using nlp extract good and bad word
# the speech should be in text file
# it should read from the .txt file
#To extract good and bad words from a text file using Natural Language Processing (NLP), you can follow these general steps:

1. **Text Preprocessing**: First, you'll need to preprocess the text to tokenize it into words, remove punctuation, convert to lowercase, and handle any other specific cleaning steps necessary for your data.

2. **Word Sentiment Analysis**: You can use a pre-trained sentiment analysis model or lexicon-based sentiment analysis to determine whether each word in the text has a positive or negative sentiment. Lexicon-based approaches use lists of positive and negative words to classify sentiment.

3. **Filtering Good and Bad Words**: After analyzing the sentiment of each word, you can filter out the words with positive sentiment as "good" words and words with negative sentiment as "bad" words.

Here's a Python example using the NLTK library for sentiment analysis:

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER sentiment analysis lexicon (one-time setup)
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Read the text from a file
with open('your_text_file.txt', 'r') as file:
    text = file.read()

# Tokenize the text
words = nltk.word_tokenize(text)

# Initialize lists to store good and bad words
good_words = []
bad_words = []

# Analyze sentiment for each word
for word in words:
    sentiment = sid.polarity_scores(word)
    
    # Classify as good or bad based on sentiment score
    if sentiment['compound'] >= 0.05:
        good_words.append(word)
    elif sentiment['compound'] <= -0.05:
        bad_words.append(word)

# Print the extracted good and bad words
print("Good Words:", good_words)
print("Bad Words:", bad_words)
```

In this example, we're using the VADER sentiment analyzer from NLTK to classify words as good or bad based on their sentiment scores. You can adjust the threshold values (0.05 and -0.05) to customize the classification criteria according to your specific needs.

Keep in mind that sentiment analysis is not perfect and may not always accurately classify words as good or bad, as it depends on the training data and the context of the text. You may need to fine-tune or use more advanced models for more accurate results, especially if the text contains complex language or sarcasm.
