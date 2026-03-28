import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import movie_reviews
import pandas as pd
from collections import Counter

"""
Movie Review Dataset POS Tagging Script

This script performs Part-of-Speech (POS) tagging on movie review datasets.
It uses NLTK for tokenization and POS tagging, and supports both English and Chinese text.

Usage:
    python movie_review.py
"""


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    nltk.download('movie_reviews')


def pos_tag_review(text):
    """
    Perform POS tagging on a movie review text.
    
    Args:
        text (str): Review text to tag
        
    Returns:
        list: List of (word, pos_tag) tuples
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags


def analyze_pos_distribution(reviews):
    """
    Analyze the POS tag distribution in reviews.
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        dict: Counter of POS tags
    """
    all_pos_tags = []
    for review in reviews:
        tagged = pos_tag_review(review)
        pos_list = [tag for word, tag in tagged]
        all_pos_tags.extend(pos_list)
    
    return Counter(all_pos_tags)


def process_movie_reviews_corpus():
    """
    Process NLTK movie reviews corpus with POS tagging.
    """
    reviews_data = []
    
    # Process positive reviews
    for fileid in movie_reviews.fileids('pos')[:10]:  # Sample first 10
        text = movie_reviews.raw(fileid)
        pos_tags = pos_tag_review(text)
        reviews_data.append({
            'review_id': fileid,
            'sentiment': 'positive',
            'text_length': len(text),
            'token_count': len(pos_tags),
            'pos_tags': pos_tags
        })
    
    # Process negative reviews
    for fileid in movie_reviews.fileids('neg')[:10]:  # Sample first 10
        text = movie_reviews.raw(fileid)
        pos_tags = pos_tag_review(text)
        reviews_data.append({
            'review_id': fileid,
            'sentiment': 'negative',
            'text_length': len(text),
            'token_count': len(pos_tags),
            'pos_tags': pos_tags
        })
    
    return pd.DataFrame(reviews_data)


def get_pos_statistics(df):
    """
    Get POS tagging statistics by sentiment.
    
    Args:
        df (DataFrame): Reviews dataframe
        
    Returns:
        dict: Statistics grouped by sentiment
    """
    stats = {}
    for sentiment in ['positive', 'negative']:
        reviews = df[df['sentiment'] == sentiment]['pos_tags'].tolist()
        all_tags = []
        for tagged_review in reviews:
            all_tags.extend([tag for word, tag in tagged_review])
        stats[sentiment] = Counter(all_tags)
    
    return stats


if __name__ == '__main__':
    print("Processing movie reviews...")
    df = process_movie_reviews_corpus()
    
    print("\n=== Reviews Summary ===")
    print(df[['review_id', 'sentiment', 'token_count']].head(10))
    
    print("\n=== POS Tag Statistics ===")
    stats = get_pos_statistics(df)
    for sentiment, counter in stats.items():
        print(f"\n{sentiment.upper()} Reviews - Top 10 POS Tags:")
        for tag, count in counter.most_common(10):
            print(f"  {tag}: {count}")