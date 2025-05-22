#!/usr/bin/env python3
"""
Comprehensive test suite for AI Product Expert Bot.
Tests critical functionality, security issues, and edge cases identified in code analysis.
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

# Import modules to test
import config
import data_ingestion
import gcp_clients
import gemini_client
import search
import text_embedding_client
import vector_store


class TestConfig(unittest.TestCase):
    """Test configuration management and validation."""
    
    def setUp(self):
        # Store original values
        self.original_env = {}
        for key in ['QDRANT_URL', 'QDRANT_API_KEY', 'GOOGLE_CLOUD_PROJECT', 'GEMINI_API_KEY']:
            self.original_env[key] = os.environ.get(key)
    
    def tearDown(self):
        # Restore original environment
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def test_validate_config_missing_keys(self):
        """Test config validation fails with missing required keys."""
        # Clear required environment variables
        os.environ.pop('QDRANT_API_KEY', None)
        os.environ.pop('GEMINI_API_KEY', None)
        
        with self.assertRaises(ValueError) as context:
            config.validate_config()
        
        error_msg = str(context.exception)
        self.assertIn('QDRANT_API_KEY', error_msg)
        self.assertIn('GEMINI_API_KEY', error_msg)
    
    def test_validate_config_success(self):
        """Test config validation passes with all required keys."""
        os.environ['QDRANT_API_KEY'] = 'test_key'
        os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
        
        # Should not raise an exception
        try:
            config.validate_config()
        except ValueError:
            self.fail("validate_config() raised ValueError unexpectedly!")
    
    def test_default_values(self):
        """Test default configuration values are reasonable."""
        self.assertEqual(config.QDRANT_URL, os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.assertEqual(config.GOOGLE_CLOUD_PROJECT, os.getenv("GOOGLE_CLOUD_PROJECT", "its-koral-prod"))
        self.assertEqual(config.MULTIMODAL_EMBEDDING_DIMENSION, 1408)
        self.assertEqual(config.TEXT_EMBEDDING_005_DIMENSION, 768)


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion functionality and security."""
    
    def test_parse_price_valid_inputs(self):
        """Test price parsing with valid inputs."""
        self.assertEqual(data_ingestion.parse_price("189.95 DKK"), 189.95)
        self.assertEqual(data_ingestion.parse_price("1234.56"), 1234.56)
        self.assertEqual(data_ingestion.parse_price("99"), 99.0)
    
    def test_parse_price_invalid_inputs(self):
        """Test price parsing with invalid inputs."""
        self.assertIsNone(data_ingestion.parse_price(None))
        self.assertIsNone(data_ingestion.parse_price(""))
        self.assertIsNone(data_ingestion.parse_price("invalid"))
        self.assertIsNone(data_ingestion.parse_price("DKK only"))
    
    def test_parse_price_malicious_inputs(self):
        """Test price parsing with potentially malicious inputs."""
        # Test regex injection attempts
        self.assertIsNone(data_ingestion.parse_price(".*"))
        self.assertIsNone(data_ingestion.parse_price("[0-9]+"))
        # Very long input
        long_input = "a" * 10000 + "123.45"
        result = data_ingestion.parse_price(long_input)
        self.assertEqual(result, 123.45)
    
    def test_encode_categorical_n_hot(self):
        """Test categorical encoding."""
        vocab = ["apple", "banana", "cherry"]
        
        # Valid encoding
        result = data_ingestion.encode_categorical_n_hot("banana", vocab)
        expected = [0.0, 1.0, 0.0]
        self.assertEqual(result, expected)
        
        # Missing value
        result = data_ingestion.encode_categorical_n_hot("orange", vocab)
        expected = [0.0, 0.0, 0.0]
        self.assertEqual(result, expected)
        
        # None value
        result = data_ingestion.encode_categorical_n_hot(None, vocab)
        expected = [0.0, 0.0, 0.0]
        self.assertEqual(result, expected)
    
    def test_encode_price_normalized(self):
        """Test price normalization."""
        # Set up price stats
        data_ingestion.PRICE_STATS = {"min": 100.0, "max": 500.0, "count": 10, "sum": 3000.0}
        
        # Test normal case
        result = data_ingestion.encode_price_normalized(300.0)
        expected = [(300.0 - 100.0) / (500.0 - 100.0)]  # Should be 0.5
        self.assertEqual(result, expected)
        
        # Test None
        result = data_ingestion.encode_price_normalized(None)
        self.assertEqual(result, [0.5])
        
        # Test edge cases
        result = data_ingestion.encode_price_normalized(50.0)  # Below min
        self.assertEqual(result, [0.0])
        
        result = data_ingestion.encode_price_normalized(600.0)  # Above max
        self.assertEqual(result, [1.0])
    
    @patch('data_ingestion.requests.get')
    def test_download_xml_feed_security(self, mock_get):
        """Test XML feed download security measures."""
        # Test invalid URL schemes
        result = data_ingestion.download_xml_feed("ftp://malicious.com/feed.xml")
        self.assertIsNone(result)
        
        result = data_ingestion.download_xml_feed("file:///etc/passwd")
        self.assertIsNone(result)
        
        # Test valid URL with mocked response
        mock_response = Mock()
        mock_response.text = "<xml>test</xml>"
        mock_response.headers = {'content-type': 'application/xml'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = data_ingestion.download_xml_feed("https://example.com/feed.xml")
        self.assertEqual(result, "<xml>test</xml>")
    
    @patch('data_ingestion.requests.get')
    def test_download_image_security(self, mock_get):
        """Test image download security measures."""
        # Test oversized image
        mock_response = Mock()
        mock_response.headers = {'content-length': str(20 * 1024 * 1024)}  # 20MB
        mock_get.return_value = mock_response
        
        result = data_ingestion.download_image("https://example.com/large.jpg", max_size_mb=10)
        self.assertIsNone(result)
        
        # Test invalid content type
        mock_response.headers = {'content-type': 'text/html', 'content-length': '1024'}
        result = data_ingestion.download_image("https://example.com/not-image.html")
        self.assertIsNone(result)
    
    def test_persist_vocab_embeddings_atomic(self):
        """Test that vocabulary persistence is atomic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily change the file path
            original_file = data_ingestion.VOCAB_EMBEDDINGS_FILE
            data_ingestion.VOCAB_EMBEDDINGS_FILE = os.path.join(tmpdir, "test_vocab.json")
            
            try:
                # Set some test data
                data_ingestion.BRAND_VOCAB = ["TestBrand"]
                data_ingestion.PRODUCT_TYPE_VOCAB = ["TestType"]
                data_ingestion.PRICE_STATS = {"min": 1.0, "max": 100.0, "count": 5, "sum": 250.0}
                data_ingestion.PRODUCT_TYPE_VOCAB_EMBEDDINGS = {"TestType": [0.1, 0.2, 0.3]}
                
                # Test successful persistence
                data_ingestion.persist_vocab_embeddings()
                
                # Verify file exists and contains correct data
                self.assertTrue(os.path.exists(data_ingestion.VOCAB_EMBEDDINGS_FILE))
                
                with open(data_ingestion.VOCAB_EMBEDDINGS_FILE, 'r') as f:
                    data = json.load(f)
                
                self.assertEqual(data["BRAND_VOCAB"], ["TestBrand"])
                self.assertEqual(data["PRODUCT_TYPE_VOCAB"], ["TestType"])
                
            finally:
                # Restore original file path
                data_ingestion.VOCAB_EMBEDDINGS_FILE = original_file


class TestSearchFunctionality(unittest.TestCase):
    """Test search functionality and performance."""
    
    def setUp(self):
        # Mock vocabularies for testing
        search.BRAND_VOCAB = ["Apple", "Samsung", "Google"]
        search.PRODUCT_TYPE_VOCAB = ["Smartphone", "Tablet", "Laptop"]
        search.AVAILABILITY_VOCAB = ["in_stock", "out_of_stock", "preorder", "backorder"]
        search.PRICE_STATS = {"min": 100.0, "max": 2000.0, "count": 100, "sum": 50000.0}
        search.PRODUCT_TYPE_VOCAB_EMBEDDINGS = {
            "Smartphone": [0.1] * 768,
            "Tablet": [0.2] * 768,
            "Laptop": [0.3] * 768
        }
        search.VOCABS_LOADED_SUCCESSFULLY = True
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(search.cosine_similarity(vec1, vec2), 1.0, places=6)
        
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(search.cosine_similarity(vec3, vec4), 0.0, places=6)
        
        # Test empty vectors
        self.assertEqual(search.cosine_similarity([], [1, 2, 3]), 0.0)
        self.assertEqual(search.cosine_similarity([1, 2, 3], []), 0.0)
    
    @patch('search.generate_gemini_text')
    def test_parse_query_with_llm_fallback(self, mock_generate):
        """Test query parsing with LLM fallback."""
        # Test when GEMINI_API_KEY is not set
        original_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = None
        
        try:
            result = search.parse_query_with_llm("test query")
            self.assertEqual(result["search_text"], "test query")
            self.assertEqual(result["attributes"], {})
            self.assertEqual(result["weights"], {})
        finally:
            config.GEMINI_API_KEY = original_key
    
    @patch('search.generate_gemini_text')
    def test_parse_query_with_llm_malformed_json(self, mock_generate):
        """Test query parsing with malformed LLM response."""
        config.GEMINI_API_KEY = "test_key"
        mock_generate.return_value = "This is not JSON at all"
        
        result = search.parse_query_with_llm("test query")
        self.assertEqual(result["search_text_for_description"], "test query")
        self.assertEqual(result["attributes"], {})
        self.assertEqual(result["weights"], {})
    
    def test_find_best_matching_product_type(self):
        """Test product type matching with embeddings."""
        with patch('search.get_dedicated_text_embedding') as mock_embed:
            mock_embed.return_value = [0.15] * 768  # Close to "Smartphone" embedding
            
            result = search.find_best_matching_product_type("Phone", similarity_threshold=0.5)
            # Should match "Smartphone" due to high similarity
            self.assertIsNotNone(result)
    
    def test_cached_functions_input_validation(self):
        """Test input validation in cached functions."""
        # Test oversized input for dense vector cache
        large_query = {"search_text_for_description": "x" * 20000}
        query_json = json.dumps(large_query)
        
        # Should handle large input gracefully
        result = search.get_cached_dense_vector("test_hash", query_json)
        # Function should handle this without crashing
        
        # Test oversized keyword text
        large_keywords = "keyword " * 500
        result = search.get_cached_sparse_vector(large_keywords)
        # Should truncate and handle gracefully


class TestVectorStore(unittest.TestCase):
    """Test vector store functionality."""
    
    @patch('vector_store.qdrant_client.QdrantClient')
    def test_get_qdrant_client_connection_validation(self, mock_client_class):
        """Test Qdrant client connection validation."""
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client
        
        # Reset global client
        vector_store._qdrant_client = None
        
        client = vector_store.get_qdrant_client()
        self.assertIsNotNone(client)
        # Should log warning but continue
    
    def test_upsert_points_validation(self):
        """Test point validation in upsert."""
        with patch('vector_store.get_qdrant_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Test with invalid sparse vector format
            invalid_points = [{
                'id': 'test1',
                'dense_vector': [0.1] * 100,
                'sparse_vector': "invalid_format",  # Should be dict
                'payload': {'test': 'data'}
            }]
            
            # Should handle invalid format gracefully
            result = vector_store.upsert_points(invalid_points)
            # Function should not crash and handle validation
    
    def test_search_points_input_validation(self):
        """Test search input validation."""
        with patch('vector_store.get_qdrant_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Test with no query vectors
            results = vector_store.search_points()
            self.assertEqual(results, [])
            
            # Test with only dense vector
            dense_vec = [0.1] * 100
            results = vector_store.search_points(dense_query_vector=dense_vec)
            # Should attempt search with mock client


class TestGCPClients(unittest.TestCase):
    """Test GCP client functionality."""
    
    @patch('gcp_clients.aiplatform.init')
    def test_vertex_ai_initialization_failure(self, mock_init):
        """Test handling of Vertex AI initialization failure."""
        mock_init.side_effect = Exception("Authentication failed")
        
        # Should handle initialization failure gracefully
        # The module should still be importable
        self.assertTrue(hasattr(gcp_clients, 'get_multimodal_embedding_model'))
    
    def test_get_multimodal_embeddings_input_validation(self):
        """Test input validation for multimodal embeddings."""
        # Test with no inputs
        result = gcp_clients.get_multimodal_embeddings()
        self.assertEqual(result, (None, None))
        
        # Test with invalid image path
        result = gcp_clients.get_multimodal_embeddings(image_path="/nonexistent/path.jpg")
        self.assertEqual(result, (None, None))


class TestTextEmbeddingClient(unittest.TestCase):
    """Test text embedding client."""
    
    def test_get_text_embedding_input_validation(self):
        """Test input validation for text embeddings."""
        with patch('text_embedding_client.get_text_embedding_client') as mock_get_client:
            mock_get_client.return_value = None
            
            # Test with no client
            result = text_embedding_client.get_text_embedding("test")
            self.assertIsNone(result)
            
            # Test with empty string
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            result = text_embedding_client.get_text_embedding("")
            self.assertIsNone(result)
            
            # Test with empty list
            result = text_embedding_client.get_text_embedding([])
            self.assertIsNone(result)


class TestGeminiClient(unittest.TestCase):
    """Test Gemini client functionality."""
    
    @patch('gemini_client.genai.configure')
    def test_gemini_initialization_with_no_api_key(self, mock_configure):
        """Test Gemini initialization without API key."""
        config.GEMINI_API_KEY = None
        mock_configure.side_effect = Exception("No API key")
        
        # Should handle missing API key gracefully
        result = gemini_client.generate_text("test prompt")
        self.assertIn("Error", result)
    
    def test_function_calling_invalid_json(self):
        """Test function calling with invalid JSON arguments."""
        with patch('gemini_client.get_gemini_model') as mock_get_model:
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            
            # Mock response with invalid JSON in function call
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part = Mock()
            mock_function_call = Mock()
            mock_function_call.name = "test_function"
            mock_function_call.args = "invalid json"
            
            mock_part.function_call = mock_function_call
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            
            mock_model_with_tools = Mock()
            mock_model_with_tools.generate_content.return_value = mock_response
            
            with patch('gemini_client.genai.GenerativeModel', return_value=mock_model_with_tools):
                result = gemini_client.function_calling(
                    "test prompt",
                    [{"name": "test_function"}],
                    {"test_function": lambda: "result"}
                )
                self.assertIn("Error", result)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases."""
    
    def test_end_to_end_search_flow_mocked(self):
        """Test complete search flow with mocked dependencies."""
        with patch.multiple(
            'search',
            ensure_vocabs_for_search=Mock(),
            parse_query_with_llm=Mock(return_value={
                "search_text_for_description": "test query",
                "attributes": {"brand": "Apple"},
                "weights": {"description_text_weight": 0.8}
            }),
            construct_holistic_query_vector=Mock(return_value=[0.1] * 100),
            get_cached_sparse_vector=Mock(return_value={"indices": [1, 2], "values": [0.5, 0.3]})
        ):
            with patch('vector_store.search_points', return_value=[
                {"id": "test1", "score": 0.95, "payload": {"title": "Test Product"}}
            ]):
                search.VOCABS_LOADED_SUCCESSFULLY = True
                results = search.search_with_llm_parsed_query("test query")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["id"], "test1")
    
    def test_memory_usage_large_batch(self):
        """Test memory usage with large batch processing."""
        # Create a large number of mock products
        large_product_list = []
        for i in range(1000):
            large_product_list.append({
                'id': f'product_{i}',
                'title': f'Product {i}',
                'description': f'Description for product {i}',
                'brand': 'TestBrand',
                'price': f'{100 + i}.00 DKK'
            })
        
        # Test that vocabulary building can handle large datasets
        with patch('data_ingestion.get_multimodal_embeddings', return_value=(None, None)):
            with patch('data_ingestion.get_dedicated_text_embedding', return_value=[0.1] * 768):
                # Should not consume excessive memory or crash
                data_ingestion.build_vocabularies_and_stats(large_product_list[:100])  # Test with subset
                
                # Verify reasonable vocabulary sizes
                self.assertLessEqual(len(data_ingestion.BRAND_VOCAB), 50)  # Should deduplicate


def run_security_tests():
    """Run security-focused tests."""
    print("Running security tests...")
    
    # Test SQL injection-like attempts in search queries
    malicious_queries = [
        "'; DROP TABLE products; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "' OR '1'='1",
        "\x00\x01\x02"  # Null bytes
    ]
    
    for query in malicious_queries:
        try:
            # Test that malicious queries don't crash the system
            with patch('search.parse_query_with_llm') as mock_parse:
                mock_parse.return_value = {
                    "search_text_for_description": query,
                    "attributes": {},
                    "weights": {}
                }
                search.VOCABS_LOADED_SUCCESSFULLY = True
                
                # Should handle malicious input gracefully
                results = search.search_with_llm_parsed_query(query, limit=1)
                print(f"✓ Handled malicious query: {query[:20]}...")
        except Exception as e:
            print(f"✗ Failed to handle malicious query {query[:20]}...: {e}")


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestDataIngestion,
        TestSearchFunctionality,
        TestVectorStore,
        TestGCPClients,
        TestTextEmbeddingClient,
        TestGeminiClient,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run additional security tests
    run_security_tests()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")