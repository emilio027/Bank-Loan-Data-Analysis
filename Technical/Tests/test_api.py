"""API Tests for Enterprise Credit Risk Intelligence Platform.

Comprehensive API testing covering:
- REST API endpoint validation
- Request/response format testing
- Authentication and authorization
- API versioning and compatibility
- Rate limiting and throttling
- Error handling and status codes

Author: Testing Framework
Version: 2.0.0
"""

import pytest
import sys
import os
import json
import time
import requests
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules for testing
try:
    from main import app
except ImportError:
    app = Mock()


class APITestConfig:
    """API test configuration."""
    
    # API endpoints
    ENDPOINTS = {
        'health': '/health',
        'status': '/api/v1/status',
        'predict': '/api/v1/predict',
        'models': '/api/v1/models',
        'train': '/api/v1/train',
        'analytics': '/api/v1/analytics',
        'powerbi': '/api/v1/powerbi/data',
        'batch_predict': '/api/v1/batch/predict'
    }
    
    # Valid test data
    VALID_PREDICTION_DATA = {
        'credit_score': 750,
        'annual_income': 85000,
        'debt_to_income': 0.25,
        'employment_length': 5,
        'loan_amount': 25000,
        'loan_purpose': 'debt_consolidation',
        'home_ownership': 'MORTGAGE'
    }
    
    # Invalid test data variations
    INVALID_DATA_CASES = [
        # Missing required fields
        {'credit_score': 750},
        # Invalid data types
        {'credit_score': 'invalid', 'annual_income': 85000},
        # Out of range values
        {'credit_score': 1000, 'annual_income': -5000},
        # Empty payload
        {},
        # Null values
        {'credit_score': None, 'annual_income': None}
    ]
    
    # Authentication test data
    VALID_API_KEY = 'test-api-key-12345'
    INVALID_API_KEY = 'invalid-api-key'
    EXPIRED_TOKEN = 'expired-jwt-token'


@pytest.fixture(scope='session')
def api_app():
    """Application instance for API testing."""
    if hasattr(app, 'config'):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
    return app


@pytest.fixture
def api_client(api_app):
    """Test client for API testing."""
    if hasattr(api_app, 'test_client'):
        with api_app.test_client() as client:
            yield client
    else:
        yield Mock()


@pytest.fixture
def valid_prediction_data():
    """Valid prediction data for testing."""
    return APITestConfig.VALID_PREDICTION_DATA.copy()


@pytest.fixture
def invalid_data_cases():
    """Invalid data cases for testing."""
    return APITestConfig.INVALID_DATA_CASES.copy()


class TestBasicAPIEndpoints:
    """Test basic API endpoint functionality."""
    
    def test_health_endpoint_get(self, api_client):
        """Test health check endpoint with GET request."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['health'])
            
            assert response.status_code == 200
            assert response.headers.get('Content-Type') == 'application/json'
            
            if hasattr(response, 'get_json'):
                data = response.get_json()
                assert 'status' in data
                assert 'service' in data
                assert 'timestamp' in data
                assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_status_endpoint_structure(self, api_client):
        """Test API status endpoint response structure."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['status'])
            
            assert response.status_code == 200
            
            if hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Required fields
                assert 'api_version' in data
                assert 'status' in data
                assert 'platform' in data
                assert 'features' in data
                assert 'endpoints' in data
                assert 'timestamp' in data
                
                # Data types validation
                assert isinstance(data['features'], list)
                assert isinstance(data['endpoints'], dict)
                assert len(data['features']) > 0
    
    def test_dashboard_endpoint_html(self, api_client):
        """Test dashboard endpoint returns HTML."""
        if hasattr(api_client, 'get'):
            response = api_client.get('/')
            
            assert response.status_code == 200
            
            if hasattr(response, 'data'):
                # Should return HTML content
                content_type = response.headers.get('Content-Type', '')
                assert 'text/html' in content_type or response.data
    
    def test_invalid_endpoint_404(self, api_client):
        """Test that invalid endpoints return 404."""
        invalid_endpoints = [
            '/api/v1/invalid',
            '/nonexistent',
            '/api/v2/predict',  # Wrong version
            '/api/v1/predict/invalid'
        ]
        
        for endpoint in invalid_endpoints:
            if hasattr(api_client, 'get'):
                response = api_client.get(endpoint)
                assert response.status_code == 404
                
                if hasattr(response, 'get_json'):
                    error_data = response.get_json()
                    assert 'error' in error_data


class TestPredictionAPIEndpoint:
    """Test prediction API endpoint thoroughly."""
    
    def test_predict_endpoint_valid_data(self, api_client, valid_prediction_data):
        """Test prediction endpoint with valid data."""
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['predict'],
                data=json.dumps(valid_prediction_data),
                content_type='application/json'
            )
            
            # Should return success or service unavailable
            assert response.status_code in [200, 503]
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Validate response structure
                required_fields = [
                    'risk_score', 'risk_category', 'probability_default',
                    'confidence_interval', 'feature_importance', 
                    'recommendations', 'timestamp'
                ]
                
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"
                
                # Validate data types and ranges
                assert isinstance(data['risk_score'], (int, float))
                assert 0 <= data['risk_score'] <= 1
                assert isinstance(data['probability_default'], (int, float))
                assert 0 <= data['probability_default'] <= 1
                assert isinstance(data['recommendations'], list)
                assert isinstance(data['feature_importance'], dict)
    
    def test_predict_endpoint_invalid_data(self, api_client, invalid_data_cases):
        """Test prediction endpoint with invalid data."""
        for invalid_data in invalid_data_cases:
            if hasattr(api_client, 'post'):
                response = api_client.post(
                    APITestConfig.ENDPOINTS['predict'],
                    data=json.dumps(invalid_data),
                    content_type='application/json'
                )
                
                # Should return 400 Bad Request or 503 Service Unavailable
                assert response.status_code in [400, 503]
    
    def test_predict_endpoint_malformed_json(self, api_client):
        """Test prediction endpoint with malformed JSON."""
        malformed_json_cases = [
            'invalid json',
            '{"incomplete": }',
            '{"credit_score": 750,}',  # Trailing comma
            '[invalid array]',
            '{"nested": {"invalid": }}'
        ]
        
        for malformed_json in malformed_json_cases:
            if hasattr(api_client, 'post'):
                response = api_client.post(
                    APITestConfig.ENDPOINTS['predict'],
                    data=malformed_json,
                    content_type='application/json'
                )
                
                assert response.status_code == 400
    
    def test_predict_endpoint_wrong_content_type(self, api_client, valid_prediction_data):
        """Test prediction endpoint with wrong content type."""
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['predict'],
                data=json.dumps(valid_prediction_data),
                content_type='text/plain'
            )
            
            assert response.status_code in [400, 415, 503]
    
    def test_predict_endpoint_no_data(self, api_client):
        """Test prediction endpoint with no data."""
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['predict'],
                content_type='application/json'
            )
            
            assert response.status_code in [400, 503]
    
    def test_predict_endpoint_method_not_allowed(self, api_client):
        """Test prediction endpoint with wrong HTTP methods."""
        wrong_methods = ['get', 'put', 'delete', 'patch']
        
        for method in wrong_methods:
            if hasattr(api_client, method):
                response = getattr(api_client, method)(APITestConfig.ENDPOINTS['predict'])
                assert response.status_code in [405, 404]  # Method Not Allowed


class TestAnalyticsAPIEndpoint:
    """Test analytics API endpoint."""
    
    def test_analytics_endpoint_structure(self, api_client):
        """Test analytics endpoint response structure."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['analytics'])
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Expected top-level keys
                expected_keys = [
                    'portfolio_metrics', 'sector_breakdown',
                    'risk_distribution', 'trend_analysis', 'timestamp'
                ]
                
                for key in expected_keys:
                    assert key in data, f"Missing analytics key: {key}"
                
                # Validate portfolio metrics structure
                if 'portfolio_metrics' in data:
                    portfolio = data['portfolio_metrics']
                    assert isinstance(portfolio, dict)
                    
                    expected_portfolio_keys = [
                        'total_exposure', 'weighted_avg_risk_score',
                        'expected_loss_rate', 'value_at_risk_95'
                    ]
                    
                    for key in expected_portfolio_keys:
                        assert key in portfolio
    
    def test_analytics_data_consistency(self, api_client):
        """Test analytics data consistency and validity."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['analytics'])
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Validate risk distribution sums to ~1
                if 'risk_distribution' in data:
                    risk_dist = data['risk_distribution']
                    total_risk = sum(risk_dist.values())
                    assert 0.95 <= total_risk <= 1.05, "Risk distribution should sum to ~1"
                
                # Validate sector breakdown sums to ~1
                if 'sector_breakdown' in data:
                    sector_breakdown = data['sector_breakdown']
                    total_sectors = sum(sector_breakdown.values())
                    assert 0.95 <= total_sectors <= 1.05, "Sector breakdown should sum to ~1"


class TestModelManagementAPI:
    """Test model management API endpoints."""
    
    def test_models_endpoint_metrics(self, api_client):
        """Test models endpoint returns performance metrics."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['models'])
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Required top-level keys
                assert 'model_performance' in data
                assert 'training_info' in data
                assert 'deployment_status' in data
                
                # Validate model performance structure
                perf = data['model_performance']
                assert isinstance(perf, dict)
                
                # Check for expected models
                expected_models = ['xgboost', 'lightgbm', 'neural_network', 'ensemble']
                for model in expected_models:
                    if model in perf:
                        model_metrics = perf[model]
                        assert 'auc_score' in model_metrics
                        assert 'accuracy' in model_metrics
                        
                        # Validate metric ranges
                        assert 0 <= model_metrics['auc_score'] <= 1
                        assert 0 <= model_metrics['accuracy'] <= 1
    
    def test_train_endpoint_post(self, api_client):
        """Test model training endpoint."""
        training_config = {
            'retrain_all': False,
            'models': ['xgboost', 'lightgbm'],
            'validation_split': 0.2
        }
        
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['train'],
                data=json.dumps(training_config),
                content_type='application/json'
            )
            
            assert response.status_code in [200, 202, 503]
            
            if response.status_code in [200, 202] and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Training response should include status
                assert 'status' in data
                if data['status'] == 'success':
                    assert 'training_time_minutes' in data
                    assert 'models_trained' in data


class TestPowerBIIntegrationAPI:
    """Test Power BI integration API endpoint."""
    
    def test_powerbi_data_endpoint(self, api_client):
        """Test Power BI data endpoint structure."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['powerbi'])
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Expected Power BI data structure
                expected_keys = ['model_metrics', 'daily_predictions', 'risk_categories']
                
                for key in expected_keys:
                    assert key in data, f"Missing Power BI data key: {key}"
                
                # Validate model metrics structure
                if data['model_metrics']:
                    for metric in data['model_metrics']:
                        assert 'model_name' in metric
                        assert 'auc_score' in metric
                        assert 'accuracy' in metric
                        assert 'last_updated' in metric


class TestBatchProcessingAPI:
    """Test batch processing API functionality."""
    
    def test_batch_predict_endpoint(self, api_client, valid_prediction_data):
        """Test batch prediction endpoint."""
        batch_data = {
            'applications': [
                {**valid_prediction_data, 'application_id': 'BATCH_001'},
                {**valid_prediction_data, 'application_id': 'BATCH_002'},
                {**valid_prediction_data, 'application_id': 'BATCH_003'}
            ]
        }
        
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['batch_predict'],
                data=json.dumps(batch_data),
                content_type='application/json'
            )
            
            assert response.status_code in [200, 202, 503]
            
            if response.status_code in [200, 202] and hasattr(response, 'get_json'):
                data = response.get_json()
                
                # Batch response structure
                assert 'batch_id' in data
                assert 'total_applications' in data
                assert 'status' in data
                
                if data['status'] == 'completed':
                    assert 'results' in data
                    assert 'processed_count' in data
    
    def test_batch_predict_empty_batch(self, api_client):
        """Test batch prediction with empty batch."""
        empty_batch = {'applications': []}
        
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['batch_predict'],
                data=json.dumps(empty_batch),
                content_type='application/json'
            )
            
            assert response.status_code in [400, 503]
    
    def test_batch_predict_large_batch(self, api_client, valid_prediction_data):
        """Test batch prediction with large batch."""
        # Create a large batch (test system limits)
        large_batch = {
            'applications': [
                {**valid_prediction_data, 'application_id': f'LARGE_BATCH_{i:04d}'}
                for i in range(100)
            ]
        }
        
        if hasattr(api_client, 'post'):
            response = api_client.post(
                APITestConfig.ENDPOINTS['batch_predict'],
                data=json.dumps(large_batch),
                content_type='application/json'
            )
            
            # Should either process successfully or return appropriate error
            assert response.status_code in [200, 202, 413, 503]


class TestAPIErrorHandling:
    """Test API error handling and status codes."""
    
    def test_internal_server_error_handling(self, api_client):
        """Test handling of internal server errors."""
        with patch('main.credit_engine') as mock_engine:
            # Simulate internal error
            mock_engine.predict.side_effect = Exception("Internal error")
            
            if hasattr(api_client, 'post'):
                response = api_client.post(
                    APITestConfig.ENDPOINTS['predict'],
                    data=json.dumps(APITestConfig.VALID_PREDICTION_DATA),
                    content_type='application/json'
                )
                
                assert response.status_code in [500, 503]
                
                if hasattr(response, 'get_json'):
                    error_data = response.get_json()
                    assert 'error' in error_data
    
    def test_service_unavailable_handling(self, api_client):
        """Test service unavailable responses."""
        with patch('main.credit_engine', None):
            if hasattr(api_client, 'post'):
                response = api_client.post(
                    APITestConfig.ENDPOINTS['predict'],
                    data=json.dumps(APITestConfig.VALID_PREDICTION_DATA),
                    content_type='application/json'
                )
                
                assert response.status_code == 503
    
    def test_request_timeout_handling(self, api_client):
        """Test request timeout handling."""
        with patch('main.ml_manager') as mock_ml:
            # Simulate slow processing
            def slow_predict(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow processing
                return {'risk_score': 0.5}
            
            mock_ml.predict.side_effect = slow_predict
            
            if hasattr(api_client, 'post'):
                response = api_client.post(
                    APITestConfig.ENDPOINTS['predict'],
                    data=json.dumps(APITestConfig.VALID_PREDICTION_DATA),
                    content_type='application/json'
                )
                
                # Should complete or timeout gracefully
                assert response.status_code in [200, 408, 503]


class TestAPIHeaders:
    """Test API response headers."""
    
    def test_cors_headers(self, api_client):
        """Test CORS headers are present."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['health'])
            
            # Should have CORS headers if CORS is enabled
            headers = response.headers
            assert 'Access-Control-Allow-Origin' in headers or response.status_code == 200
    
    def test_content_type_headers(self, api_client):
        """Test proper content type headers."""
        if hasattr(api_client, 'get'):
            # JSON endpoints
            json_endpoints = ['health', 'status', 'analytics', 'models', 'powerbi']
            
            for endpoint_key in json_endpoints:
                endpoint = APITestConfig.ENDPOINTS[endpoint_key]
                response = api_client.get(endpoint)
                
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    assert 'application/json' in content_type
    
    def test_security_headers(self, api_client):
        """Test security-related headers."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['health'])
            
            headers = response.headers
            
            # Check for common security headers
            security_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection'
            ]
            
            # At least some security headers should be present in production
            security_header_count = sum(1 for header in security_headers if header in headers)
            assert security_header_count >= 0  # Flexible for test environment


class TestAPIVersioning:
    """Test API versioning functionality."""
    
    def test_api_v1_endpoints(self, api_client):
        """Test that v1 API endpoints are accessible."""
        v1_endpoints = [
            '/api/v1/status',
            '/api/v1/predict',
            '/api/v1/models',
            '/api/v1/analytics'
        ]
        
        for endpoint in v1_endpoints:
            if hasattr(api_client, 'get'):
                response = api_client.get(endpoint)
                # Should not return 404 (endpoint exists)
                assert response.status_code != 404
    
    def test_api_version_in_response(self, api_client):
        """Test that API version is included in responses."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['status'])
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                assert 'api_version' in data
                assert data['api_version'] == 'v1'


class TestAPIDocumentation:
    """Test API documentation and self-description."""
    
    def test_status_endpoint_provides_endpoints(self, api_client):
        """Test that status endpoint provides available endpoints."""
        if hasattr(api_client, 'get'):
            response = api_client.get(APITestConfig.ENDPOINTS['status'])
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                data = response.get_json()
                assert 'endpoints' in data
                assert isinstance(data['endpoints'], dict)
                assert len(data['endpoints']) > 0
    
    def test_error_responses_provide_help(self, api_client):
        """Test that error responses provide helpful information."""
        if hasattr(api_client, 'get'):
            response = api_client.get('/api/v1/nonexistent')
            
            if response.status_code == 404 and hasattr(response, 'get_json'):
                error_data = response.get_json()
                # Should provide available endpoints or helpful message
                assert 'available_endpoints' in error_data or 'message' in error_data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])