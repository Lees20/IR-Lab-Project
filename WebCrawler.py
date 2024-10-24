import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import json
import math

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Wikipedia page URL
url = 'https://en.wikipedia.org/wiki/George_II_of_Great_Britain'

try:
    # Request the page
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Parse the page content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get the page title
    title = soup.title.string if soup.title else 'No Title'
    print(f'Title of the page: {title}')

    # Get all paragraphs
    paragraphs = soup.find_all('p')

    # Extract text from paragraphs
    content = ' '.join([para.get_text() for para in paragraphs])

    # Create a DataFrame with the data
    data = pd.DataFrame({
        'Title': [title],
        'Content': [content]
    })

    # Save to JSON
    data.to_json('wikipedia_data.json', orient='records', lines=True)
    print('Data saved to JSON file.')

except requests.exceptions.RequestException as e:
    print(f'Failed to retrieve the page: {e}')

# Load the JSON data
try:
    data = pd.read_json('wikipedia_data.json', lines=True)
    text = data['Content'][0]

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove special characters
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # 3. Tokenize text
    tokens = word_tokenize(text, language='english')

    # 4. Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Create cleaned text
    cleaned_text = ' '.join(lemmatized_tokens)

    # Save cleaned data to JSON
    cleaned_data = pd.DataFrame({'Title': [data['Title'][0]], 'Cleaned_Content': [cleaned_text]})
    cleaned_data.to_json('cleaned_wikipedia_data.json', orient='records', lines=True)
    print("Cleaned data saved to 'cleaned_wikipedia_data.json'")

except FileNotFoundError:
    print("The file 'wikipedia_data.json' was not found.")
except Exception as e:
    print(f'An error occurred: {e}')

# Step 3: Create an Inverted Index
try:
    # Load the cleaned JSON data
    cleaned_data = pd.read_json('cleaned_wikipedia_data.json', lines=True)
    cleaned_text = cleaned_data['Cleaned_Content'][0]

    # Tokenize the cleaned text
    tokens = word_tokenize(cleaned_text, language='english')

    # Create an inverted index
    inverted_index = defaultdict(list)
    for idx, token in enumerate(tokens):
        inverted_index[token].append(idx)

    # Save the inverted index to a JSON file
    with open('inverted_index.json', 'w') as f:
        json.dump(inverted_index, f)

    print("Inverted index saved to 'inverted_index.json'")

except FileNotFoundError:
    print("The file 'cleaned_wikipedia_data.json' was not found.")
except Exception as e:
    print(f'An error occurred: {e}')

# Step 4: Search Engine Implementation (Command Line Interface)
# Load the inverted index
with open('inverted_index.json', 'r') as f:
    inverted_index = json.load(f)

# Command-line search
def search(query):
    query = query.lower()
    tokens = word_tokenize(query, language='english')
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Boolean retrieval
    results = set()
    for token in lemmatized_tokens:
        if token in inverted_index:
            if not results:
                results = set(inverted_index[token])
            else:
                results &= set(inverted_index[token])

    return list(results)

if __name__ == '__main__':
    while True:
        user_query = input("Enter your search query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        search_results = search(user_query)
        if search_results:
            print(f'Results found at positions: {search_results}')
        else:
            print('No results found.')