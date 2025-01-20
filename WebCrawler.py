import requests
from bs4 import BeautifulSoup
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from urllib.parse import urljoin
import time
import math
import numpy as np
from collections import defaultdict

# Διαδρομή NLTK
nltk.data.path.append('/Users/panteliskarabetsos/nltk_data')

# Κατέβασμα απαραίτητων δεδομένων NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Αρχικοποίηση εργαλείων
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# Μεταβλητές
visited_urls = set()
articles = []
inverted_index = {}

#Προεπεξεργασία Κειμένου
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

#Crawling
def crawl_wikipedia(url, max_articles=10):
    queue = [url]
    while queue and len(articles) < max_articles:
        current_url = queue.pop(0)
        if current_url in visited_urls:
            continue
        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Απόκτηση τίτλου και περιεχομένου
            title = soup.title.string if soup.title else 'No Title'
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            cleaned_content = ' '.join(preprocess_text(content))

            # Αποθήκευση δεδομένων άρθρου
            articles.append({
                'Title': title,
                'URL': current_url,
                'Content': content,
                'Cleaned_Content': cleaned_content
            })
            print(f'Collected and processed article: {title} from {current_url}')

            visited_urls.add(current_url)

            # Εύρεση συνδέσμων
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(base_url, href)
                    if full_url not in visited_urls:
                        queue.append(full_url)

            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {current_url}: {e}")

#Δημιουργία Ευρετηρίου
def build_inverted_index():
    global inverted_index
    for idx, article in enumerate(articles):
        tokens = preprocess_text(article['Cleaned_Content'])
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            if stemmed_token not in inverted_index:
                inverted_index[stemmed_token] = set()
            inverted_index[stemmed_token].add(idx)
    #print("Inverted Index Built:", inverted_index)


#Boolean Query Processing
def infix_to_postfix(query):
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operators = []
    tokens = re.findall(r'\(|\)|\w+|AND|OR|NOT', query)

    for token in tokens:
        token = token.upper()
        if token in precedence:
            while operators and operators[-1] != '(' and precedence[operators[-1]] >= precedence[token]:
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output

def evaluate_postfix(postfix):
    """Αξιολογεί ένα Boolean query σε postfix μορφή."""
    stack = []
    all_docs = set(range(len(articles))) 

    for token in postfix:
        print(f"Processing token: {token}") 
        if token in {"AND", "OR", "NOT"}: 
            if token == "AND":
                if len(stack) < 2:
                    print("Error: Not enough operands for AND.")
                    return set()
                set2 = stack.pop()
                set1 = stack.pop()
                stack.append(set1.intersection(set2))
            elif token == "OR":
                if len(stack) < 2:
                    print("Error: Not enough operands for OR.")
                    return set()
                set2 = stack.pop()
                set1 = stack.pop()
                stack.append(set1.union(set2))
            elif token == "NOT":
                if not stack:
                    print("Error: Not enough operands for NOT.")
                    return set()
                set1 = stack.pop()
                # Αφαιρούμε τα έγγραφα του set1 από το σύνολο όλων των εγγράφων
                negated_set = all_docs.difference(set1)
                print(f"NOT operation negates documents: {set1}")
                stack.append(negated_set)
        else:  # Αν είναι όρος
            stemmed_token = stemmer.stem(token.lower())
            result_set = inverted_index.get(stemmed_token, set())
            print(f"Token '{token}' maps to documents: {result_set}") 
            stack.append(result_set)

    result = stack.pop() if stack else set()
    print(f"Final result: {result}")  
    return result


#Κατάταξη 
def compute_tfidf(query_terms, corpus):
    """Υπολογίζει το TF-IDF σκορ κάθε εγγράφου για τους όρους του ερωτήματος."""
    tfidf_scores = []
    num_docs = len(corpus)

    # Υπολογισμός IDF με ελάχιστο όριο
    idf = {}
    for term in query_terms:
        doc_count = sum(1 for doc in corpus if term in doc)
        idf[term] = max(math.log((num_docs + 1) / (doc_count + 1)), 0.1)  

    # Υπολογισμός TF-IDF για κάθε έγγραφο
    for doc in corpus:
        score = 0
        for term in query_terms:
            tf = doc.count(term) / len(doc) if len(doc) > 0 else 0
            score += tf * idf.get(term, 0)
        tfidf_scores.append(score)

    ranked_indices = np.argsort(tfidf_scores)[::-1]
    return ranked_indices, tfidf_scores


def compute_bm25(query_terms, corpus, k1=1.5, b=0.75):
    """Υπολογίζει το BM25 σκορ κάθε εγγράφου για τους όρους του ερωτήματος."""
    bm25_scores = []
    num_docs = len(corpus)
    avg_doc_len = np.mean([len(doc) for doc in corpus]) 

    # Υπολογισμός IDF για κάθε όρο
    idf = {}
    for term in query_terms:
        doc_count = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)  

    # Υπολογισμός BM25 για κάθε έγγραφο
    for doc in corpus:
        doc_len = len(doc)
        score = 0
        for term in query_terms:
            freq = doc.count(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf.get(term, 0) * (numerator / denominator)
        bm25_scores.append(score)

    ranked_indices = np.argsort(bm25_scores)[::-1]
    return ranked_indices, bm25_scores

def compute_vsm(query_terms, corpus, debug=False):
    """Υπολογίζει τη συνημιτονοειδή ομοιότητα (cosine similarity) για κάθε έγγραφο."""
    from collections import defaultdict
    from math import log, sqrt

    # Υπολογισμός IDF
    term_idf = {}
    num_docs = len(corpus)
    for doc in corpus:
        for term in set(doc):
            if term not in term_idf:
                doc_count = sum(1 for d in corpus if term in d)
                term_idf[term] = max(log((num_docs + 1) / (1 + doc_count)), 0.1)  # Κατώτατο όριο IDF

    if debug:
        print(f"IDF Values: {term_idf}")

    # Δημιουργία διανύσματος ερωτήματος
    query_vector = defaultdict(float)
    for term in query_terms:
        term = term.lower()
        if term in term_idf:
            query_vector[term] = query_terms.count(term) * term_idf[term]

    if debug:
        print(f"Query Vector: {dict(query_vector)}")

    # Υπολογισμός διανυσμάτων εγγράφων και cosine similarity
    scores = []
    for idx, doc in enumerate(corpus):
        doc_vector = defaultdict(float)
        for term in doc:
            if term in term_idf:
                doc_vector[term] = doc.count(term) * term_idf[term]

        # Υπολογισμός cosine similarity
        numerator = sum(query_vector[term] * doc_vector[term] for term in query_vector)
        query_norm = sqrt(sum(v**2 for v in query_vector.values()))
        doc_norm = sqrt(sum(v**2 for v in doc_vector.values()))
        denominator = query_norm * doc_norm

        if denominator == 0:
            scores.append(0)
        else:
            scores.append(numerator / denominator)

        if debug:
            print(f"Doc {idx} Vector: {dict(doc_vector)}, Score: {scores[-1]}")

    
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores


def search_with_ranking(query, corpus, ranking_method="tfidf"):
    """Αναζήτηση και κατάταξη με βάση τον επιλεγμένο αλγόριθμο."""
    postfix_query = infix_to_postfix(query)
    filtered_indices = evaluate_postfix(postfix_query)

    if not filtered_indices:
        print("No results found.")
        return None, None  

    filtered_indices = list(filtered_indices)
    filtered_corpus = [corpus[idx] for idx in filtered_indices]
    query_terms = [term for term in query.lower().split() if term not in {"AND", "OR", "NOT"}]

    if ranking_method == "tfidf":
        ranked_indices, scores = compute_tfidf(query_terms, filtered_corpus)
    elif ranking_method == "bm25":
        ranked_indices, scores = compute_bm25(query_terms, filtered_corpus)
    elif ranking_method == "vsm":
        ranked_indices, scores = compute_vsm(query_terms, filtered_corpus)
    else:
        print("Invalid ranking method.")
        return None, None  

    # Φιλτράρισμα έγγραφων με σκορ > 0
    valid_results = [
        (filtered_indices[idx], scores[idx])
        for idx in ranked_indices if scores[idx] > 0
    ]

    if not valid_results:
        print("No valid results found.")
        return None, None 

    print(f"Results ranked by {ranking_method.upper()}:")
    for rank, (doc_idx, score) in enumerate(valid_results):
        print(f"- {articles[doc_idx]['Title']} (Score: {score:.4f}) - URL: {articles[doc_idx]['URL']}")

    return [doc_idx for doc_idx, _ in valid_results], [score for _, score in valid_results]


#Διεπαφή Αναζήτησης
def search_interface():
    """Διεπαφή αναζήτησης για Boolean queries."""
    print("Welcome to the Search Engine with Ranking Options! Enter your query (type 'exit' to quit):")
    
    while True:
        query = input("Query: ")
        if query.lower() == 'exit':
            break
        
        print("Choose ranking method: (1) TF-IDF, (2) BM25, (3) VSM")
        choice = input("Enter 1, 2, or 3: ")
        
        if choice == "1":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="tfidf")
        elif choice == "2":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="bm25")
        elif choice == "3":
            search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in articles], ranking_method="vsm")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def evaluate_search_engine(test_queries, corpus):
    while True:
        print("\nChoose ranking method for evaluation: (1) TF-IDF, (2) BM25, (3) VSM, (4) Exit")
        choice = input("Enter 1, 2, 3, or 4: ")

        if choice == "1":
            ranking_method = "tfidf"
        elif choice == "2":
            ranking_method = "bm25"
        elif choice == "3":
            ranking_method = "vsm"
        elif choice == "4":
            print("Exiting evaluation.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            continue

        results = []
        all_precisions = []

        for query, relevant_docs in test_queries.items():
            print(f"\nEvaluating query: {query}")
            ranked_indices, scores = search_with_ranking(query, [doc["Cleaned_Content"].split() for doc in corpus], ranking_method)

            if ranked_indices is None or scores is None:  # Handle case where no results are found
                print(f"No results for query: {query}")
                continue

            # Υπολογισμός Precision, Recall, F1
            retrieved_docs = ranked_indices[:len(relevant_docs)]  
            true_positive = len(set(retrieved_docs) & set(relevant_docs))
            precision = true_positive / len(retrieved_docs) if retrieved_docs else 0
            recall = true_positive / len(relevant_docs) if relevant_docs else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

            # Υπολογισμός Average Precision 
            precisions = []
            retrieved_set = set()
            for rank, idx in enumerate(ranked_indices):
                if idx in relevant_docs:
                    retrieved_set.add(idx)
                    precision_at_k = len(retrieved_set) / (rank + 1)
                    precisions.append(precision_at_k)
            average_precision = sum(precisions) / len(relevant_docs) if precisions else 0
            all_precisions.append(average_precision)

            results.append({
                "Query": query,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Average Precision": average_precision,
            })

        # Υπολογισμός Συνολικών Μετρικών
        mean_average_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
        avg_precision = sum(res["Precision"] for res in results) / len(results) if results else 0
        avg_recall = sum(res["Recall"] for res in results) / len(results) if results else 0
        avg_f1 = sum(res["F1-Score"] for res in results) / len(results) if results else 0

        print("\nΑποτελέσματα ανά Ερώτημα:")
        for res in results:
            print(f"- Query: {res['Query']}, Precision: {res['Precision']:.2f}, Recall: {res['Recall']:.2f}, "
                  f"F1: {res['F1-Score']:.2f}, Average Precision: {res['Average Precision']:.2f}")

        print("\nΣυνολικά Αποτελέσματα:")
        print(f"Μέση Ακρίβεια (MAP): {mean_average_precision:.2f}")
        print(f"Μέση Precision: {avg_precision:.2f}")
        print(f"Μέση Recall: {avg_recall:.2f}")
        print(f"Μέσο F1-Score: {avg_f1:.2f}")




# ΕΚΤΕΛΕΣΗ
if __name__ == "__main__":
    base_url = "https://en.wikipedia.org"
    start_url = f"{base_url}/wiki/Artificial_intelligence"

    crawl_wikipedia(start_url, max_articles=10)
    build_inverted_index()
    search_interface()

    #Eρωτήματα δοκιμής
    test_queries = {
        "ai": [0, 2, 4],
        "machine learning": [1, 3],
        "deep learning": [5, 6, 7],
    }
    evaluate_search_engine(test_queries, articles)
