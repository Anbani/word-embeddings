import numpy as np
def related_evaluation(queries, model):
    """ Ranked evaluation metric for related terms

    Args:
        model (KeyedVector): Any word embedding model supported by Gensim v4.0+

    Returns:
        float   - evaluation score
        Dict    - details of evaluation
    """    
    results = []
    for i, subject in enumerate(queries):
        suggestions, values = zip(
            *model.wv.most_similar(positive=[subject], topn=model.max_final_vocab))
        
        result = {
            'subject' : subject,
            'related' : {}
        }

        for r in queries[subject]:
            ix = suggestions.index(r)
            result['related'][r] = ix
            # result['related'][r] = 1-ix/len(suggestions)

        result['score'] = np.round(1 - (np.mean(list(result['related'].values())) / len(model.wv.key_to_index)), 4)
        results.append(result)

    final_score = np.mean(list(map(lambda x: x['score'], results)))

    return final_score, results


def analogy(text, model, **kwargs):
    """ Find analogy vectors

    Args:
        text (string): First three words of the analogy sentence. E.g.: "Man King Woman" -> "Queen" to guess
        model (KeyedVector): Any word embedding model supported by Gensim v4.0+

    Returns:
        string: Fourth word of the analogy. E.g: "Queen" in the previous example. 
    """
    a,b,c = text.split(' ')
    return model.most_similar(positive=[b,c], negative=[a], **kwargs)

def analogy_result_rank(text, model):
    """ Wrapper around analogy utility finding position of the correct fourth word. 

    Args:
        text (string): All four words of the analogy to query. 
        model (KeyedVector): Any word embedding model supported by Gensim v4.0+

    Returns:
        string:     Fourth word to seek
        int:        Position of the word
        int:        Total length of the dict
    """
    a,b,c,d = text.split(' ')
    words = model.wv.most_similar(positive=[b,c], negative=[a], topn=model.max_final_vocab)
    for ix, word in enumerate(words):
        if d in word:
            return *word, ix, len(words)

    return d, -1, len(words)

def analogy_evaluation(queries, model):
    """ Wrapper around analogy utility to query multiple relevant sentences for the model to be evaluated with. 
    Calculates evaluation score with (1 - rank/total_words) formula for language model to be maximized for. 

    Args:
        model (KeyedVector): Any word embedding model supported by Gensim v4.0+

    Returns:
        float:  Average of the evaluation scores
        Dict:   Details of the evaluation
    """

    results = {}

    for q in queries:
        a,b,c,d = q.split(' ')

        query = f"{a} {b} {c} {d}"
        _, _, ix, total = analogy_result_rank(query, model)
        results[query] = ix

        query = f"{c} {d} {a} {b}"
        _, _, ix, total = analogy_result_rank(query, model)
        results[query] = ix

    return np.round(1 - np.mean(list(results.values())) / len(model.wv.key_to_index), 4), results


def evaluate(analogy_queries, related_queries, model):
    """ Wrapper around language model evaluation metrics. 
    Calculates final score based on maximum of either of the evaluation metrics to maximize model expertise in either domain. 

    Args:
        model (KeyedVector): Any word embedding model supported by Gensim v4.0+

    Returns:
        float:  Maximum score of the evaluation metrics
        Dict:   Details of the evaluation.
    """
    eval_analogy, details_analogy = analogy_evaluation(analogy_queries, model)
    eval_related, details_related = related_evaluation(related_queries, model)
    return np.mean([eval_analogy, eval_related]), {
        "analogy" : eval_analogy,
        "related" : eval_related,
        "details" : {
            "details_analogy" : details_analogy,
            "details_related" :  details_related
        }
    }
