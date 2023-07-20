import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceSimilarity:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def mean_pooling(self, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging

        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_similarity_two_sentences(self, a, b):
        sentences = [a, b]

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # compute cosine similarity between embeddings
        cosine_scores = sentence_embeddings[0] @ sentence_embeddings[1].T
        return cosine_scores

    def get_most_similar_in_list(self, query_word, list):
        sentences = [query_word] + [word.replace("_", " ") for word in list]
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # compute cosine similarity between embeddings
        cosine_scores = sentence_embeddings[0] @ sentence_embeddings[1:].T
        print(
            f"word queried : {query_word} | word list : {list} | cosine scores : {cosine_scores}"
        )

        return list[torch.argmax(cosine_scores).item()]
