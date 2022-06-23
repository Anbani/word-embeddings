from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict
import fitz

def wordcount(list_of_words):
    # Create wordcount map
    wordfreq = defaultdict(int)
    stats = {}
    stats['max_group_length'] = 0
    stats['max_word_length'] = 0
    stats['min_word_length'] = 1e10
    stats['total_words'] = 0
    stats['total_groups'] = len(list_of_words)


    for group in tqdm(list_of_words):
        stats['max_group_length'] = max(stats['max_group_length'], len(group))
        for word in group:
            wordfreq[word] += 1
            stats['total_words'] += 1
            stats['max_word_length'] = max(stats['max_word_length'], len(word))
            stats['min_word_length'] = min(stats['min_word_length'], len(word))

    stats['total_unique_words'] = len(wordfreq)
    
    # Create convinience Pandas dataframe
    df = pd.DataFrame.from_dict(
        wordfreq, 
        orient='index'
    ).reset_index().rename(
        columns={
            'index' : 'word', 
            0 : 'count'
        }
    ).sort_values('count', ascending=False).reset_index(drop=True)

    df['len'] = df.word.str.len()

    return df, stats, wordfreq

