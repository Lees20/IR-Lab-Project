import requests
from bs4 import BeautifulSoup
import re
import json
import nltk
nltk.data.path.append('/Users/panteliskarabetsos/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from urllib.parse import urljoin
import time
import math
import numpy as np

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Wikipedia base URL
base_url = 'https://en.wikipedia.org'
start_url = f'{base_url}/wiki/Artificial_intelligence'

# Set of visited URLs to avoid duplicates
visited_urls = set()
articles = []
inverted_index = {}

def preprocess_text(text):
    """Preprocesses text by normalizing, removing stop words, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

def build_inverted_index():
    """Builds an inverted index from the articles collected."""
    global inverted_index
    for idx, article in enumerate(articles):
        tokens = preprocess_text(article['Cleaned_Content'])
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            if stemmed_token not in inverted_index:
                inverted_index[stemmed_token] = set()
            inverted_index[stemmed_token].add(idx)

def crawl_wikipedia(url, max_articles=10):
    """Crawls Wikipedia starting from a given URL and collects up to max_articles."""
    queue = [url]
    
    while queue and len(articles) < max_articles:
        current_url = queue.pop(0)
        
        if current_url in visited_urls:
            continue
        
        try:
            response = requests.get(current_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title and content
            title = soup.title.string if soup.title else 'No Title'
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            cleaned_content = ' '.join(preprocess_text(content))
            
            # Save article data
            articles.append({
                'Title': title,
                'URL': current_url,
                'Content': content,
                'Cleaned_Content': cleaned_content
            })
            print(f'Collected and processed article: {title} from {current_url}')
            
            # Mark the URL as visited
            visited_urls.add(current_url)
            
            # Find links to other Wikipedia articles
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(base_url, href)
                    if full_url not in visited_urls:
                        queue.append(full_url)
            
            
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {current_url}: {e}")
            continue

# Start crawling with a limit of 30 articles
crawl_wikipedia(start_url, max_articles=10)

# Build inverted index after collecting articles
build_inverted_index()

# Save collected articles to JSON file
with open('wikipedia_data_crawled_30.json', 'w') as f:
    json.dump(articles, f, indent=4)
print("Data saved to 'wikipedia_data_crawled_30.json'")

def infix_to_postfix(query):
    """Convert an infix Boolean query to postfix notation."""
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operators = []

    tokens = query.split()
    for token in tokens:
        token = token.upper()
        if token in precedence:
            while (operators and operators[-1] != '(' and
                   precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # Remove '('
        else:
            output.append(token)  # Operand (term)
    
    while operators:
        output.append(operators.pop())
    
    return output

def infix_to_postfix(query):
    """Convert an infix Boolean query to postfix notation, handling implicit AND with NOT."""
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operators = []

    tokens = query.split()
    i = 0
    while i < len(tokens):
        token = tokens[i].upper()
        
        if token == 'NOT' and i > 0 and tokens[i - 1].isalnum():
            # Insert implicit AND between terms when `NOT` follows a term without AND
            operators.append('AND')
        
        if token in precedence:
            while (operators and operators[-1] != '(' and
                   precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # Remove '('
        else:
            output.append(token)  # Operand (term)
        
        i += 1
    
    while operators:
        output.append(operators.pop())
    
    return output

def evaluate_postfix(postfix):
    """Evaluates a postfix Boolean query with strict handling for multiple NOT operations."""
    stack = []
    all_docs = set(range(len(articles)))  # Set with all document indices
    
    for token in postfix:
        if token == "AND":
            # Πάρε τα δύο τελευταία σύνολα και κάνε την τομή
            set2 = stack.pop()
            set1 = stack.pop()
            stack.append(set1.intersection(set2))
        elif token == "OR":
            # Πάρε τα δύο τελευταία σύνολα και κάνε την ένωση
            set2 = stack.pop()
            set1 = stack.pop()
            stack.append(set1.union(set2))
        elif token == "NOT":
            # Εφαρμογή του NOT στο αμέσως προηγούμενο σύνολο
            set1 = stack.pop()
            stack.append(all_docs.difference(set1))
        else:
            # Αν το token είναι ένας όρος, πρόσθεσέ το στο stack
            stemmed_token = stemmer.stem(token.lower())
            stack.append(inverted_index.get(stemmed_token, set()))
    
    return stack[0] if stack else set()

def infix_to_postfix(query):
    """Convert an infix Boolean query to postfix notation with implicit AND for NOT handling."""
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operators = []

    tokens = query.split()
    i = 0
    while i < len(tokens):
        token = tokens[i].upper()
        
        if token == 'NOT' and i > 0 and tokens[i - 1].isalnum():
            operators.append('AND')
        
        if token in precedence:
            while (operators and operators[-1] != '(' and
                   precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # Remove '('
        else:
            output.append(token)  # Operand (term)
        
        i += 1
    
    while operators:
        output.append(operators.pop())
    
    return output


def search_interface():
    """Provides a command-line interface for the search engine."""
    print("Welcome to the Boolean Search Engine! Enter your query (type 'exit' to quit):")
    
    while True:
        query = input("Query: ")
        if query.lower() == 'exit':
            break
        
        
        postfix_query = infix_to_postfix(query)
        result_indices = evaluate_postfix(postfix_query)
        
        if result_indices:
            print(f"Found {len(result_indices)} results:")
            for idx in result_indices:
                print(f"- {articles[idx]['Title']} (URL: {articles[idx]['URL']})")
        else:
            print("No results found.")



# Βοηθητική συνάρτηση για υπολογισμό TF
def compute_tf(term, document):
    """Υπολογίζει τη συχνότητα εμφάνισης του όρου στο έγγραφο."""
    term_count = document.count(term)
    return term_count / len(document) if len(document) > 0 else 0

def compute_idf(term, corpus):
    """Υπολογίζει τον αντιστρόφως ανάλογο της συχνότητας εμφάνισης του όρου σε όλα τα έγγραφα."""
    num_docs_with_term = sum(1 for doc in corpus if term in doc)
    if num_docs_with_term == 0:
        return 0
    return math.log(len(corpus) / num_docs_with_term)

# Υπολογισμός TF-IDF με χρήση των βελτιωμένων `compute_tf` και `compute_idf`
def compute_tfidf(query, corpus):
    """Υπολογίζει το TF-IDF σκορ κάθε εγγράφου για τους όρους του ερωτήματος."""
    tfidf_scores = []
    for doc in corpus:
        score = 0
        for term in query:
            tf = compute_tf(term, doc)
            idf = compute_idf(term, corpus)
            score += tf * idf
        tfidf_scores.append(score)
    ranked_indices = np.argsort(tfidf_scores)[::-1]
    return ranked_indices, tfidf_scores


# Υπολογισμός συνημιτονοειδούς ομοιότητας για VSM
from collections import defaultdict

def compute_cosine_similarity(query_terms, corpus):
    """Υπολογίζει τη συνημιτονοειδή ομοιότητα (cosine similarity) μεταξύ του ερωτήματος και των εγγράφων."""
    # Δημιουργία λεξικού με IDF για όλους τους όρους
    term_idf = {}
    num_docs = len(corpus)
    for doc in corpus:
        for term in set(doc):  # Μόνο μοναδικοί όροι
            if term not in term_idf:
                term_idf[term] = math.log((num_docs + 1) / (1 + sum(1 for d in corpus if term in d)))  # Smoothing

    # Υπολογισμός διανύσματος ερωτήματος
    query_vector = defaultdict(float)
    for term in query_terms:
        if term in term_idf:
            query_vector[term] = query_terms.count(term) * term_idf[term]

    # Υπολογισμός διανυσμάτων εγγράφων και cosine similarity
    scores = []
    for doc in corpus:
        doc_vector = defaultdict(float)
        for term in doc:
            if term in term_idf:
                doc_vector[term] = doc.count(term) * term_idf[term]
        
        # Υπολογισμός cosine similarity
        numerator = sum(query_vector[term] * doc_vector[term] for term in query_vector)
        query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))
        doc_norm = math.sqrt(sum(v**2 for v in doc_vector.values()))
        denominator = query_norm * doc_norm
        cosine_similarity = numerator / denominator if denominator != 0 else 0
        scores.append(cosine_similarity)

    # Ταξινόμηση εγγράφων με βάση τα scores
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores


# Υπολογισμός BM25
def compute_bm25(query, corpus, k=1.5, b=0.75):
    avg_doc_len = np.mean([len(doc) for doc in corpus])
    bm25_scores = []
    
    for doc in corpus:
        doc_len = len(doc)
        score = 0
        for term in query:
            term_freq = doc.count(term)
            idf = compute_idf(term, corpus)
            score += idf * ((term_freq * (k + 1)) / (term_freq + k * (1 - b + b * (doc_len / avg_doc_len))))
        bm25_scores.append(score)
    
    ranked_indices = np.argsort(bm25_scores)[::-1]
    return ranked_indices, bm25_scores

# Προσαρμοσμένη συνάρτηση αναζήτησης με κατάταξη
def search_with_ranking(query, corpus, ranking_method="tfidf"):
    """Αναζήτηση και κατάταξη με βάση τον επιλεγμένο αλγόριθμο"""
    query_terms = query.lower().split()
    
    if ranking_method == "tfidf":
        ranked_indices, scores = compute_tfidf(query_terms, corpus)
    elif ranking_method == "vsm":
        ranked_indices, scores = compute_cosine_similarity(query_terms, corpus)
    elif ranking_method == "bm25":
        ranked_indices, scores = compute_bm25(query_terms, corpus)
    else:
        print("Unknown ranking method. Please choose 'tfidf', 'vsm', or 'bm25'.")
        return
    
    print(f"Results ranked by {ranking_method.upper()}:")
    for idx in ranked_indices[:10]:  # Εμφάνιση κορυφαίων 10 αποτελεσμάτων με θετική βαθμολογία
        if scores[idx] > 0:  # Φιλτράρει αποτελέσματα με σκορ 0
            print(f"- {articles[idx]['Title']} (Score: {scores[idx]:.4f}) - URL: {articles[idx]['URL']}")

def search_interface():
    """Provides a command-line interface for the search engine with ranking choice."""
    print("Welcome to the Search Engine with Ranking Options! Enter your query (type 'exit' to quit):")
    
    while True:
        query = input("Query: ")
        if query.lower() == 'exit':
            break
        
        print("Choose ranking method: (1) TF-IDF, (2) VSM, (3) BM25")
        choice = input("Enter 1, 2, or 3: ")
        
        if choice == "1":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="tfidf")
        elif choice == "2":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="vsm")
        elif choice == "3":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="bm25")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Καλούμε τη διεπαφή αναζήτησης
search_interface()





search_interface()