from anbani.nlp.preprocessing import sentence_tokenize
from tqdm import tqdm

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self, tokenizer=sentence_tokenize):
        with open(self.dataset_path, "r") as file:
            text = file.read()
            text = tokenizer(text)
            return text

    def load_streaming(self):
        # TODO Gensim LineSentences for example
        raise Exception('Streaming dataset not implemented')
