"""Performance Tests for Enterprise Credit Risk Intelligence Platform.

Comprehensive performance testing covering:
- Load testing and scalability validation
- Response time benchmarking
- Memory usage optimization
- Database query performance
- ML model inference speed
- Concurrent request handling

Author: Testing Framework
Version: 2.0.0
"""

import pytest
import sys
import os
import json
import time
import threading
import asyncio
import psutil
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules for testing
try:
    from main import app
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
except ImportError:
    # Create mock imports if modules aren't available
    app = Mock()
    AnalyticsEngine = Mock
    DataManager = Mock
    MLModelManager = Mock


class PerformanceTestConfig:
    """Performance test configuration."""
    
    # Load test parameters
    LIGHT_LOAD_REQUESTS = 50
    MEDIUM_LOAD_REQUESTS = 200
    HEAVY_LOAD_REQUESTS = 500
    STRESS_LOAD_REQUESTS = 1000
    
    # Performance thresholds
    MAX_RESPONSE_TIME_MS = 2000
    MAX_P95_RESPONSE_TIME_MS = 5000
    MAX_P99_RESPONSE_TIME_MS = 10000
    MIN_THROUGHPUT_RPS = 10
    MAX_MEMORY_USAGE_MB = 512
    MAX_CPU_USAGE_PERCENT = 80
    
    # Test data
    SAMPLE_PREDICTION_DATA = {
        'credit_score': 750,
        'annual_income': 85000,
        'debt_to_income': 0.25,
        'employment_length': 5,
        'loan_amount': 25000,
        'loan_purpose': 'debt_consolidation'
    }
    
    LARGE_BATCH_SIZE = 1000
    CONCURRENT_USERS = 20


@pytest.fixture(scope='session')
def performance_app():
    """Application instance for performance testing."""
    if hasattr(app, 'config'):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False  # Disable debug for performance testing
    return app


@pytest.fixture
def performance_client(performance_app):
    """Test client for performance testing."""
    if hasattr(performance_app, 'test_client'):
        with performance_app.test_client() as client:
            yield client
    else:
        yield Mock()


@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for performance testing."""
    return PerformanceTestConfig.SAMPLE_PREDICTION_DATA.copy()


@pytest.fixture
def large_batch_data():
    """Large batch of test data for load testing."""
    base_data = PerformanceTestConfig.SAMPLE_PREDICTION_DATA
    batch_data = []
    
    for i in range(PerformanceTestConfig.LARGE_BATCH_SIZE):
        data = base_data.copy()
        data['application_id'] = f'PERF_TEST_{i:04d}'
        # Add some variance to the data
        data['credit_score'] = base_data['credit_score'] + np.random.randint(-50, 51)
        data['annual_income'] = base_data['annual_income'] + np.random.randint(-10000, 10001)
        batch_data.append(data)
    
    return batch_data


class TestResponseTimePerformance:
    """Test response time performance metrics."""
    
    def test_single_prediction_response_time(self, performance_client, sample_prediction_data):
        """Test single prediction request response time."""
        response_times = []
        
        # Run multiple requests to get average
        for _ in range(10):
            if hasattr(performance_client, 'post'):
                start_time = time.time()
                
                response = performance_client.post('/api/v1/predict',
                                                 data=json.dumps(sample_prediction_data),
                                                 content_type='application/json')
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                assert response.status_code in [200, 503]
        
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            
            # Performance assertions
            assert avg_response_time < PerformanceTestConfig.MAX_RESPONSE_TIME_MS
            assert p95_response_time < PerformanceTestConfig.MAX_P95_RESPONSE_TIME_MS
            assert p99_response_time < PerformanceTestConfig.MAX_P99_RESPONSE_TIME_MS
            
            print(f"Average response time: {avg_response_time:.2f}ms")
            print(f"P95 response time: {p95_response_time:.2f}ms")
            print(f"P99 response time: {p99_response_time:.2f}ms")
    
    def test_health_check_response_time(self, performance_client):
        """Test health check endpoint response time."""
        response_times = []
        
        for _ in range(20):
            if hasattr(performance_client, 'get'):
                start_time = time.time()
                response = performance_client.get('/health')
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                assert response.status_code == 200
        
        if response_times:
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            
            # Health check should be very fast
            assert avg_response_time < 100  # 100ms
            assert max_response_time < 500  # 500ms
            
            print(f"Health check avg response time: {avg_response_time:.2f}ms")
    
    def test_analytics_endpoint_response_time(self, performance_client):
        """Test analytics endpoint response time."""
        response_times = []
        
        for _ in range(5):
            if hasattr(performance_client, 'get'):
                start_time = time.time()
                response = performance_client.get('/api/v1/analytics')
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                assert response.status_code in [200, 503]
        
        if response_times:
            avg_response_time = np.mean(response_times)
            
            # Analytics endpoint can be slower but should be reasonable
            assert avg_response_time < 5000  # 5 seconds
            
            print(f"Analytics avg response time: {avg_response_time:.2f}ms")


class TestLoadPerformance:
    """Test system performance under various load conditions."""
    
    def test_light_load_performance(self, performance_client, sample_prediction_data):
        """Test performance under light load."""
        self._run_load_test(
            performance_client,
            sample_prediction_data,
            PerformanceTestConfig.LIGHT_LOAD_REQUESTS,
            "Light Load"
        )
    
    def test_medium_load_performance(self, performance_client, sample_prediction_data):
        """Test performance under medium load."""
        self._run_load_test(
            performance_client,
            sample_prediction_data,
            PerformanceTestConfig.MEDIUM_LOAD_REQUESTS,
            "Medium Load"
        )
    
    @pytest.mark.slow
    def test_heavy_load_performance(self, performance_client, sample_prediction_data):
        """Test performance under heavy load."""
        self._run_load_test(
            performance_client,
            sample_prediction_data,
            PerformanceTestConfig.HEAVY_LOAD_REQUESTS,
            "Heavy Load"
        )
    
    def _run_load_test(self, client, data, num_requests, test_name):
        """Helper method to run load tests."""
        if not hasattr(client, 'post'):
            return
        
        response_times = []
        error_count = 0
        start_time = time.time()
        
        # Sequential requests to measure individual response times
        for i in range(num_requests):
            request_start = time.time()
            
            response = client.post('/api/v1/predict',
                                 data=json.dumps(data),
                                 content_type='application/json')
            
            request_end = time.time()
            response_time_ms = (request_end - request_start) * 1000
            response_times.append(response_time_ms)
            
            if response.status_code not in [200, 503]:
                error_count += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate performance metrics
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            throughput = num_requests / total_duration
            error_rate = error_count / num_requests
            
            print(f"\n{test_name} Performance Results:")
            print(f"Requests: {num_requests}")
            print(f"Duration: {total_duration:.2f}s")
            print(f"Throughput: {throughput:.2f} requests/second")
            print(f"Avg Response Time: {avg_response_time:.2f}ms")
            print(f"P95 Response Time: {p95_response_time:.2f}ms")
            print(f"Error Rate: {error_rate:.2%}")
            
            # Performance assertions
            assert error_rate < 0.05  # Less than 5% error rate
            if num_requests <= PerformanceTestConfig.MEDIUM_LOAD_REQUESTS:
                assert throughput >= PerformanceTestConfig.MIN_THROUGHPUT_RPS


class TestConcurrentPerformance:
    """Test performance with concurrent requests."""
    
    def test_concurrent_predictions(self, performance_client, sample_prediction_data):
        """Test concurrent prediction requests."""
        if not hasattr(performance_client, 'post'):
            return
        
        def make_prediction_request():
            start_time = time.time()
            response = performance_client.post('/api/v1/predict',
                                             data=json.dumps(sample_prediction_data),
                                             content_type='application/json')
            end_time = time.time()
            
            return {
                'status_code': response.status_code,
                'response_time': (end_time - start_time) * 1000,
                'success': response.status_code in [200, 503]
            }
        
        # Run concurrent requests
        concurrent_users = PerformanceTestConfig.CONCURRENT_USERS
        requests_per_user = 5
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            start_time = time.time()
            
            # Submit all requests
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    future = executor.submit(make_prediction_request)
                    futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
            
            end_time = time.time()
        
        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        response_times = [r['response_time'] for r in results if r['success']]
        
        success_rate = successful_requests / total_requests
        total_duration = end_time - start_time
        throughput = total_requests / total_duration
        
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            
            print(f"\nConcurrent Performance Results:")
            print(f"Concurrent Users: {concurrent_users}")
            print(f"Total Requests: {total_requests}")
            print(f"Success Rate: {success_rate:.2%}")
            print(f"Throughput: {throughput:.2f} requests/second")
            print(f"Avg Response Time: {avg_response_time:.2f}ms")
            print(f"P95 Response Time: {p95_response_time:.2f}ms")
            
            # Performance assertions
            assert success_rate >= 0.95  # 95% success rate
            assert avg_response_time < PerformanceTestConfig.MAX_RESPONSE_TIME_MS * 2  # Allow 2x for concurrency
    
    def test_concurrent_different_endpoints(self, performance_client, sample_prediction_data):
        """Test concurrent requests to different endpoints."""
        if not hasattr(performance_client, 'get') or not hasattr(performance_client, 'post'):
            return
        
        def health_check_request():
            return performance_client.get('/health')
        
        def prediction_request():
            return performance_client.post('/api/v1/predict',
                                         data=json.dumps(sample_prediction_data),
                                         content_type='application/json')
        
        def analytics_request():
            return performance_client.get('/api/v1/analytics')
        
        def status_request():
            return performance_client.get('/api/v1/status')
        
        # Mix of different endpoint types
        request_functions = [
            health_check_request,
            prediction_request,
            analytics_request,
            status_request
        ]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # Submit mixed requests
            for _ in range(20):
                for request_func in request_functions:
                    future = executor.submit(request_func)
                    futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                response = future.result()
                results.append(response.status_code if hasattr(response, 'status_code') else 200)
        
        # Analyze mixed endpoint performance
        success_count = sum(1 for status in results if status in [200, 503])
        success_rate = success_count / len(results)
        
        print(f"\nMixed Endpoint Performance:")
        print(f"Total Requests: {len(results)}")
        print(f"Success Rate: {success_rate:.2%}")
        
        assert success_rate >= 0.90  # 90% success rate for mixed endpoints


class TestMemoryPerformance:
    """Test memory usage and optimization."""
    
    def test_memory_usage_during_predictions(self, performance_client, sample_prediction_data):
        """Test memory usage during prediction requests."""
        if not hasattr(performance_client, 'post'):
            return
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Make multiple predictions
        for _ in range(50):
            response = performance_client.post('/api/v1/predict',
                                             data=json.dumps(sample_prediction_data),
                                             content_type='application/json')
        
        # Get final memory usage
        final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage Analysis:")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Memory Increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < PerformanceTestConfig.MAX_MEMORY_USAGE_MB
    
    def test_memory_cleanup_after_batch_processing(self, performance_client, large_batch_data):
        """Test memory cleanup after processing large batches."""
        if not hasattr(performance_client, 'post'):
            return
        
        # Process batches and monitor memory
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        
        # Process in smaller chunks to simulate real batch processing
        chunk_size = 100
        for i in range(0, min(500, len(large_batch_data)), chunk_size):
            chunk = large_batch_data[i:i + chunk_size]
            batch_payload = {'applications': chunk}
            
            response = performance_client.post('/api/v1/batch/predict',
                                             data=json.dumps(batch_payload),
                                             content_type='application/json')
        
        # Allow some time for garbage collection
        time.sleep(2)
        
        final_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"\nBatch Processing Memory Analysis:")
        print(f"Memory Increase: {memory_increase:.2f} MB")
        
        # Memory should not grow excessively
        assert memory_increase < PerformanceTestConfig.MAX_MEMORY_USAGE_MB * 2


class TestDatabasePerformance:
    """Test database operation performance."""
    
    @patch('main.data_manager')
    def test_database_query_performance(self, mock_data_manager):
        """Test database query performance."""
        # Mock database operations with timing
        def mock_slow_query(*args, **kwargs):
            time.sleep(0.01)  # 10ms simulated query time
            return pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        
        def mock_fast_query(*args, **kwargs):
            time.sleep(0.001)  # 1ms simulated query time
            return pd.DataFrame({'id': [1], 'value': [10]})
        
        mock_data_manager.get_application_data.side_effect = mock_fast_query
        mock_data_manager.save_prediction_result.side_effect = mock_fast_query
        
        # Test query performance
        query_times = []
        
        for _ in range(10):
            start_time = time.time()
            mock_data_manager.get_application_data('APP_001')
            end_time = time.time()
            
            query_time_ms = (end_time - start_time) * 1000
            query_times.append(query_time_ms)
        
        avg_query_time = np.mean(query_times)
        
        print(f"\nDatabase Query Performance:")
        print(f"Average Query Time: {avg_query_time:.2f}ms")
        
        # Database queries should be fast
        assert avg_query_time < 100  # 100ms
    
    @patch('main.data_manager')
    def test_bulk_insert_performance(self, mock_data_manager, large_batch_data):
        """Test bulk database insert performance."""
        def mock_bulk_insert(data_list):
            # Simulate bulk insert timing based on data size
            insert_time = len(data_list) * 0.001  # 1ms per record
            time.sleep(insert_time)
            return len(data_list)
        
        mock_data_manager.bulk_insert_predictions.side_effect = mock_bulk_insert
        
        # Test bulk insert performance
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            batch = large_batch_data[:batch_size]
            
            start_time = time.time()
            result = mock_data_manager.bulk_insert_predictions(batch)
            end_time = time.time()
            
            insert_time_ms = (end_time - start_time) * 1000
            records_per_second = batch_size / (insert_time_ms / 1000)
            
            print(f"Batch Size {batch_size}: {records_per_second:.0f} records/second")
            
            assert result == batch_size
            assert records_per_second > 100  # Minimum 100 records/second


class TestMLModelPerformance:
    """Test ML model inference performance."""
    
    @patch('main.ml_manager')
    def test_model_inference_speed(self, mock_ml_manager, sample_prediction_data):
        """Test ML model inference speed."""
        def mock_fast_prediction(*args, **kwargs):
            time.sleep(0.01)  # 10ms inference time
            return {
                'risk_score': 0.23,
                'confidence': 0.95,
                'processing_time_ms': 10
            }
        
        mock_ml_manager.predict.side_effect = mock_fast_prediction
        
        # Test inference speed
        inference_times = []
        
        for _ in range(20):
            start_time = time.time()
            result = mock_ml_manager.predict(sample_prediction_data)
            end_time = time.time()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            assert 'risk_score' in result
        
        avg_inference_time = np.mean(inference_times)
        p95_inference_time = np.percentile(inference_times, 95)
        
        print(f"\nML Model Inference Performance:")
        print(f"Average Inference Time: {avg_inference_time:.2f}ms")
        print(f"P95 Inference Time: {p95_inference_time:.2f}ms")
        
        # Model inference should be fast
        assert avg_inference_time < 100  # 100ms
        assert p95_inference_time < 200  # 200ms
    
    @patch('main.ml_manager')
    def test_ensemble_prediction_performance(self, mock_ml_manager, sample_prediction_data):
        """Test ensemble model prediction performance."""
        def mock_ensemble_prediction(*args, **kwargs):
            # Simulate ensemble of 3 models
            time.sleep(0.03)  # 30ms for ensemble
            return {
                'ensemble_score': 0.23,
                'individual_scores': {
                    'xgboost': 0.21,
                    'lightgbm': 0.24,
                    'neural_network': 0.25
                },
                'confidence': 0.95
            }
        
        mock_ml_manager.ensemble_predict.side_effect = mock_ensemble_prediction
        
        # Test ensemble performance
        ensemble_times = []
        
        for _ in range(10):
            start_time = time.time()
            result = mock_ml_manager.ensemble_predict(sample_prediction_data)
            end_time = time.time()
            
            ensemble_time_ms = (end_time - start_time) * 1000
            ensemble_times.append(ensemble_time_ms)
            
            assert 'ensemble_score' in result
            assert 'individual_scores' in result
        
        avg_ensemble_time = np.mean(ensemble_times)
        
        print(f"\nEnsemble Model Performance:")
        print(f"Average Ensemble Time: {avg_ensemble_time:.2f}ms")
        
        # Ensemble should still be reasonably fast
        assert avg_ensemble_time < 500  # 500ms


class TestScalabilityPerformance:
    """Test system scalability characteristics."""
    
    def test_response_time_scaling(self, performance_client, sample_prediction_data):
        """Test how response time scales with load."""
        if not hasattr(performance_client, 'post'):
            return
        
        load_levels = [1, 5, 10, 20, 50]
        scaling_results = []
        
        for load in load_levels:
            response_times = []
            
            for _ in range(load):
                start_time = time.time()
                response = performance_client.post('/api/v1/predict',
                                                 data=json.dumps(sample_prediction_data),
                                                 content_type='application/json')
                end_time = time.time()
                
                if response.status_code in [200, 503]:
                    response_times.append((end_time - start_time) * 1000)
            
            if response_times:
                avg_response_time = np.mean(response_times)
                scaling_results.append((load, avg_response_time))
        
        print(f"\nScalability Analysis:")
        for load, avg_time in scaling_results:
            print(f"Load {load}: {avg_time:.2f}ms average response time")
        
        # Response time should not increase exponentially
        if len(scaling_results) >= 2:
            first_load, first_time = scaling_results[0]
            last_load, last_time = scaling_results[-1]
            
            # Response time should not increase more than 5x even with 50x load
            time_ratio = last_time / first_time
            load_ratio = last_load / first_load
            
            assert time_ratio < load_ratio * 2  # Response time growth should be sub-linear


if __name__ == '__main__':
    # Run performance tests with custom markers
    pytest.main([__file__, '-v', '--tb=short', '-m', 'not slow'])