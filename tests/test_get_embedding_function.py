import unittest
from langchain_ollama import OllamaEmbeddings
from get_embedding_function import get_embedding_function

class TestGetEmbeddingFunction(unittest.TestCase):
    def test_returns_ollama_embeddings(self):
        """
        Tests if get_embedding_function returns an instance of OllamaEmbeddings.
        """
        embeddings = get_embedding_function()
        self.assertIsInstance(embeddings, OllamaEmbeddings)

if __name__ == '__main__':
    unittest.main()
