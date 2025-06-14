import unittest
from unittest.mock import MagicMock
# Assuming Document is imported from langchain_core.documents
from langchain_core.documents import Document
from populate_database import calculate_chunk_ids

class TestPopulateDatabase(unittest.TestCase):
    def test_calculate_chunk_ids(self):
        """
        Tests the calculate_chunk_ids function with mocked Document objects.
        """
        # Create mock Document objects
        doc1_page1_chunk1 = Document(page_content="Content 1", metadata={"source": "doc1.pdf", "page": 1})
        doc1_page1_chunk2 = Document(page_content="Content 2", metadata={"source": "doc1.pdf", "page": 1})
        doc1_page2_chunk1 = Document(page_content="Content 3", metadata={"source": "doc1.pdf", "page": 2})
        doc2_page1_chunk1 = Document(page_content="Content 4", metadata={"source": "doc2.pdf", "page": 1})

        chunks = [
            doc1_page1_chunk1,
            doc1_page1_chunk2,
            doc1_page2_chunk1,
            doc2_page1_chunk1,
        ]

        # Apply the function
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Verify the generated IDs
        self.assertEqual(chunks_with_ids[0].metadata["id"], "doc1.pdf:1:0")
        self.assertEqual(chunks_with_ids[1].metadata["id"], "doc1.pdf:1:1")
        self.assertEqual(chunks_with_ids[2].metadata["id"], "doc1.pdf:2:0")
        self.assertEqual(chunks_with_ids[3].metadata["id"], "doc2.pdf:1:0")

if __name__ == '__main__':
    unittest.main()
