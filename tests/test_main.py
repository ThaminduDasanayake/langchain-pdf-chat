import unittest
from unittest.mock import patch, MagicMock
from main import query_rag
import config # For accessing PROMPT_TEMPLATE

# Mocking Document class for similarity_search_with_score results
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class TestMain(unittest.TestCase):

    @patch('main.get_embedding_function')
    @patch('main.OllamaLLM')
    @patch('main.ChatPromptTemplate')
    @patch('main.Chroma')
    def test_query_rag(self, MockChroma, MockChatPromptTemplate, MockOllamaLLM, MockGetEmbeddingFunction):
        """
        Tests the query_rag function with mocked dependencies.
        """
        # Setup mock instances and their return values
        mock_db = MockChroma.return_value
        mock_similarity_results = [
            (MockDocument(page_content="Result 1", metadata={"id": "source1"}), 0.8),
            (MockDocument(page_content="Result 2", metadata={"id": "source2"}), 0.7)
        ]
        mock_db.similarity_search_with_score.return_value = mock_similarity_results

        mock_prompt_template_instance = MockChatPromptTemplate.from_template.return_value
        mock_prompt_template_instance.format.return_value = "Formatted prompt"

        mock_llm_instance = MockOllamaLLM.return_value
        mock_llm_instance.invoke.return_value = "LLM response"

        # Mock get_embedding_function as it's called within query_rag
        MockGetEmbeddingFunction.return_value = MagicMock()


        # Call the function
        query_text = "Test query"
        response, sources = query_rag(query_text)

        # Assertions
        MockChroma.assert_called_once_with(
            persist_directory=config.CHROMA_PATH,
            embedding_function=MockGetEmbeddingFunction.return_value
        )
        mock_db.similarity_search_with_score.assert_called_once_with(query_text, k=8)

        MockChatPromptTemplate.from_template.assert_called_once_with(config.PROMPT_TEMPLATE)
        mock_prompt_template_instance.format.assert_called_once_with(
            context="Result 1\n\n---\n\nResult 2",
            question=query_text
        )

        MockOllamaLLM.assert_called_once_with(
            model=config.LLM_MODEL,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            num_predict=500
        )
        mock_llm_instance.invoke.assert_called_once_with("Formatted prompt")

        self.assertEqual(response, "LLM response")
        self.assertEqual(sources, ["source1", "source2"])

if __name__ == '__main__':
    unittest.main()
