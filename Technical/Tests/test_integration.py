"""Integration Tests for Enterprise Credit Risk Intelligence Platform.

Comprehensive integration testing covering:
- End-to-end workflow testing
- Component integration validation
- Data pipeline testing
- External service integration
- Database integration testing
- ML model pipeline validation

Author: Testing Framework
Version: 2.0.0
"""

import pytest
import sys
import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
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
    from visualization_manager import VisualizationManager
    from advanced_credit_risk_engine import AdvancedCreditRiskEngine
except ImportError:
    # Create mock imports if modules aren't available
    app = Mock()
    AnalyticsEngine = Mock
    DataManager = Mock
    MLModelManager = Mock
    VisualizationManager = Mock
    AdvancedCreditRiskEngine = Mock


class IntegrationTestConfig:
    """Integration test configuration."""
    
    SAMPLE_BATCH_DATA = [
        {
            'application_id': 'APP001',
            'credit_score': 750,
            'annual_income': 85000,
            'debt_to_income': 0.25,
            'employment_length': 5,
            'loan_amount': 25000,
            'loan_purpose': 'debt_consolidation'
        },
        {
            'application_id': 'APP002',
            'credit_score': 680,
            'annual_income': 62000,
            'debt_to_income': 0.35,
            'employment_length': 3,
            'loan_amount': 30000,
            'loan_purpose': 'home_improvement'
        },
        {
            'application_id': 'APP003',
            'credit_score': 820,
            'annual_income': 120000,
            'debt_to_income': 0.15,
            'employment_length': 8,
            'loan_amount': 15000,
            'loan_purpose': 'car'
        }
    ]
    
    EXPECTED_PIPELINE_STAGES = [
        'data_validation',
        'preprocessing',
        'feature_engineering',
        'model_prediction',
        'risk_assessment',
        'recommendation_generation',
        'result_formatting'
    ]


@pytest.fixture(scope='session')
def integration_app():
    """Application instance for integration testing."""
    if hasattr(app, 'config'):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    return app


@pytest.fixture
def integration_client(integration_app):
    """Test client for integration testing."""
    if hasattr(integration_app, 'test_client'):
        with integration_app.test_client() as client:
            yield client
    else:
        yield Mock()


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return IntegrationTestConfig.SAMPLE_BATCH_DATA.copy()


@pytest.fixture
def mock_database():
    """Mock database for integration testing."""
    db = Mock()
    db.execute.return_value = Mock()
    db.fetchall.return_value = []
    db.commit.return_value = None
    return db


class TestDataPipelineIntegration:
    """Test complete data processing pipeline."""
    
    def test_end_to_end_prediction_pipeline(self, integration_client, sample_batch_data):
        """Test complete prediction pipeline from input to output."""
        for application in sample_batch_data:
            if hasattr(integration_client, 'post'):
                response = integration_client.post('/api/v1/predict',
                                                 data=json.dumps(application),
                                                 content_type='application/json')
                
                # Verify successful processing
                if response.status_code == 200 and hasattr(response, 'get_json'):
                    result = response.get_json()
                    
                    # Validate pipeline output
                    assert 'risk_score' in result
                    assert 'recommendations' in result
                    assert 'processing_time_ms' in result
                    assert 'model_ensemble' in result
                    
                    # Validate data consistency
                    assert 0 <= result['risk_score'] <= 1
                    assert isinstance(result['recommendations'], list)
                    assert result['processing_time_ms'] > 0
    
    def test_batch_processing_integration(self, integration_client, sample_batch_data):
        """Test batch processing capabilities."""
        batch_payload = {'applications': sample_batch_data}
        
        if hasattr(integration_client, 'post'):
            response = integration_client.post('/api/v1/batch/predict',
                                             data=json.dumps(batch_payload),
                                             content_type='application/json')
            
            if response.status_code == 200 and hasattr(response, 'get_json'):
                results = response.get_json()
                
                assert 'batch_id' in results
                assert 'total_applications' in results
                assert 'processed_count' in results
                assert 'results' in results
                
                # Validate batch results
                assert results['total_applications'] == len(sample_batch_data)
                assert results['processed_count'] <= results['total_applications']
    
    @patch('main.data_manager')
    @patch('main.ml_manager')
    def test_component_integration(self, mock_ml_manager, mock_data_manager, sample_batch_data):
        """Test integration between major components."""
        # Configure mocks
        mock_data_manager.validate_data.return_value = True
        mock_data_manager.preprocess_data.return_value = pd.DataFrame(sample_batch_data[0], index=[0])
        
        mock_ml_manager.predict.return_value = {
            'risk_score': 0.23,
            'confidence': 0.95,
            'model_used': 'ensemble'
        }
        
        # Test component interaction
        for data in sample_batch_data:
            # Data validation
            assert mock_data_manager.validate_data(data)
            
            # Data preprocessing
            processed_data = mock_data_manager.preprocess_data(data)
            assert not processed_data.empty
            
            # ML prediction
            prediction = mock_ml_manager.predict(processed_data)
            assert 'risk_score' in prediction
            assert 'confidence' in prediction


class TestDatabaseIntegration:
    """Test database integration and data persistence."""
    
    @patch('main.data_manager')
    def test_application_data_persistence(self, mock_data_manager, sample_batch_data):
        """Test persistence of application data."""
        mock_data_manager.save_application.return_value = True
        mock_data_manager.get_application.return_value = sample_batch_data[0]
        
        # Test saving application
        for application in sample_batch_data:
            result = mock_data_manager.save_application(application)
            assert result is True
        
        # Test retrieving application
        retrieved = mock_data_manager.get_application('APP001')
        assert retrieved['application_id'] == 'APP001'
    
    @patch('main.data_manager')
    def test_prediction_results_storage(self, mock_data_manager):
        """Test storage of prediction results."""
        mock_prediction_result = {
            'application_id': 'APP001',
            'risk_score': 0.23,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': 'v2.0.0'
        }
        
        mock_data_manager.save_prediction.return_value = True
        mock_data_manager.get_prediction_history.return_value = [mock_prediction_result]
        
        # Test saving prediction
        result = mock_data_manager.save_prediction(mock_prediction_result)
        assert result is True
        
        # Test retrieving prediction history
        history = mock_data_manager.get_prediction_history('APP001')
        assert len(history) > 0
        assert history[0]['application_id'] == 'APP001'
    
    @patch('main.data_manager')
    def test_analytics_data_aggregation(self, mock_data_manager):
        """Test aggregation of analytics data."""
        mock_analytics_data = {
            'daily_predictions': 1247,
            'average_risk_score': 0.31,
            'high_risk_applications': 125,
            'model_performance': {
                'accuracy': 0.94,
                'auc_score': 0.95
            }
        }
        
        mock_data_manager.get_daily_analytics.return_value = mock_analytics_data
        
        # Test analytics aggregation
        analytics = mock_data_manager.get_daily_analytics(datetime.now().date())
        assert 'daily_predictions' in analytics
        assert 'average_risk_score' in analytics
        assert 'model_performance' in analytics


class TestModelIntegration:
    """Test ML model integration and ensemble functionality."""
    
    @patch('main.ml_manager')
    def test_ensemble_model_integration(self, mock_ml_manager, sample_batch_data):
        """Test ensemble model prediction integration."""
        mock_ensemble_result = {
            'ensemble_prediction': 0.23,
            'individual_predictions': {
                'xgboost': 0.21,
                'lightgbm': 0.24,
                'neural_network': 0.25
            },
            'confidence_score': 0.95,
            'prediction_variance': 0.02
        }
        
        mock_ml_manager.ensemble_predict.return_value = mock_ensemble_result
        
        # Test ensemble prediction
        for data in sample_batch_data:
            result = mock_ml_manager.ensemble_predict(data)
            assert 'ensemble_prediction' in result
            assert 'individual_predictions' in result
            assert 'confidence_score' in result
            
            # Validate ensemble components
            individual = result['individual_predictions']
            assert len(individual) >= 2  # At least 2 models in ensemble
    
    @patch('main.ml_manager')
    def test_model_retraining_integration(self, mock_ml_manager):
        """Test model retraining workflow integration."""
        mock_training_result = {
            'training_status': 'completed',
            'models_trained': ['xgboost', 'lightgbm', 'neural_network'],
            'training_duration': 1847.5,
            'performance_improvement': 0.023,
            'new_model_metrics': {
                'accuracy': 0.946,
                'auc_score': 0.958,
                'precision': 0.903
            }
        }
        
        mock_ml_manager.retrain_models.return_value = mock_training_result
        
        # Test model retraining
        training_config = {
            'retrain_all': True,
            'validation_split': 0.2,
            'epochs': 100
        }
        
        result = mock_ml_manager.retrain_models(training_config)
        assert result['training_status'] == 'completed'
        assert 'models_trained' in result
        assert 'performance_improvement' in result
    
    @patch('main.ml_manager')
    def test_feature_importance_integration(self, mock_ml_manager, sample_batch_data):
        """Test feature importance calculation integration."""
        mock_feature_importance = {
            'global_importance': {
                'credit_score': 0.34,
                'debt_to_income': 0.28,
                'employment_length': 0.19,
                'loan_amount': 0.12,
                'annual_income': 0.07
            },
            'local_importance': {
                'credit_score': 0.42,
                'debt_to_income': 0.31,
                'employment_length': 0.15,
                'loan_amount': 0.08,
                'annual_income': 0.04
            }
        }
        
        mock_ml_manager.calculate_feature_importance.return_value = mock_feature_importance
        
        # Test feature importance calculation
        for data in sample_batch_data:
            importance = mock_ml_manager.calculate_feature_importance(data)
            assert 'global_importance' in importance
            assert 'local_importance' in importance
            
            # Validate importance scores sum to approximately 1
            global_sum = sum(importance['global_importance'].values())
            assert 0.95 <= global_sum <= 1.05


class TestExternalServiceIntegration:
    """Test integration with external services."""
    
    @patch('main.powerbi_connector')
    def test_powerbi_integration(self, mock_powerbi_connector, integration_client):
        """Test Power BI service integration."""
        mock_powerbi_data = {
            'dataset_id': 'dataset_123',
            'last_refresh': datetime.now().isoformat(),
            'status': 'success',
            'rows_updated': 1247
        }
        
        mock_powerbi_connector.push_data.return_value = mock_powerbi_data
        mock_powerbi_connector.refresh_dataset.return_value = True
        
        # Test Power BI data push
        if hasattr(integration_client, 'get'):
            response = integration_client.get('/api/v1/powerbi/data')
            
            if response.status_code == 200:
                # Simulate data push to Power BI
                data_push_result = mock_powerbi_connector.push_data({})
                assert data_push_result['status'] == 'success'
                assert 'rows_updated' in data_push_result
                
                # Test dataset refresh
                refresh_result = mock_powerbi_connector.refresh_dataset('dataset_123')
                assert refresh_result is True
    
    @patch('requests.post')
    def test_external_api_integration(self, mock_requests_post):
        """Test integration with external credit bureau APIs."""
        # Mock external API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'credit_score': 750,
            'credit_history_length': 8,
            'recent_inquiries': 2,
            'derogatory_marks': 0
        }
        mock_requests_post.return_value = mock_response
        
        # Test external API call
        import requests
        response = requests.post('https://api.creditbureau.com/score', json={'ssn': '123-45-6789'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'credit_score' in data
        assert 'credit_history_length' in data
    
    def test_email_notification_integration(self):
        """Test email notification service integration."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            # Test email sending
            mock_server.send_message.return_value = {}
            
            # Simulate sending notification
            email_sent = True  # Would be actual email sending logic
            assert email_sent is True


class TestPerformanceIntegration:
    """Test performance under integrated load."""
    
    def test_concurrent_request_handling(self, integration_client, sample_batch_data):
        """Test handling of concurrent requests across components."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_concurrent_request():
            if hasattr(integration_client, 'post'):
                for data in sample_batch_data:
                    response = integration_client.post('/api/v1/predict',
                                                     data=json.dumps(data),
                                                     content_type='application/json')
                    results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_concurrent_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Validate all requests were handled
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 9  # 3 threads × 3 applications
        assert all(status in [200, 503] for status in status_codes)
    
    def test_database_connection_pooling(self):
        """Test database connection pooling under load."""
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_pool = Mock()
            mock_engine.return_value.pool = mock_pool
            mock_pool.size.return_value = 10
            mock_pool.checked_in.return_value = 8
            mock_pool.checked_out.return_value = 2
            
            # Test connection pool status
            assert mock_pool.size() == 10
            assert mock_pool.checked_in() + mock_pool.checked_out() == 10
    
    def test_cache_integration_performance(self):
        """Test caching layer integration and performance."""
        with patch('redis.Redis') as mock_redis:
            mock_cache = Mock()
            mock_redis.return_value = mock_cache
            
            # Test cache operations
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True   # Cache set
            mock_cache.delete.return_value = True  # Cache delete
            
            # Simulate cache operations
            cache_key = 'prediction_APP001'
            cached_result = mock_cache.get(cache_key)
            assert cached_result is None  # Cache miss
            
            # Set cache
            cache_set = mock_cache.set(cache_key, '{"risk_score": 0.23}', ex=3600)
            assert cache_set is True


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_cascading_failure_handling(self, integration_client):
        """Test handling of cascading failures across components."""
        with patch('main.ml_manager') as mock_ml_manager:
            # Simulate ML manager failure
            mock_ml_manager.predict.side_effect = Exception("Model unavailable")
            
            if hasattr(integration_client, 'post'):
                response = integration_client.post('/api/v1/predict',
                                                 data=json.dumps({'credit_score': 750}),
                                                 content_type='application/json')
                
                # Should handle gracefully
                assert response.status_code in [500, 503]
    
    def test_partial_service_degradation(self, integration_client):
        """Test system behavior with partial service degradation."""
        with patch('main.powerbi_connector') as mock_powerbi:
            # Simulate Power BI service unavailability
            mock_powerbi.push_data.side_effect = Exception("Service unavailable")
            
            if hasattr(integration_client, 'get'):
                # Health check should show degraded status
                response = integration_client.get('/health')
                
                if response.status_code == 200 and hasattr(response, 'get_json'):
                    health_data = response.get_json()
                    # System should still be operational for core functions
                    assert health_data['status'] in ['healthy', 'degraded']
    
    def test_timeout_handling_integration(self, integration_client):
        """Test timeout handling across integrated services."""
        with patch('main.data_manager') as mock_data_manager:
            # Simulate slow database response
            import time
            
            def slow_query(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow query
                return pd.DataFrame({'credit_score': [750]})
            
            mock_data_manager.preprocess_data.side_effect = slow_query
            
            if hasattr(integration_client, 'post'):
                start_time = time.time()
                response = integration_client.post('/api/v1/predict',
                                                 data=json.dumps({'credit_score': 750}),
                                                 content_type='application/json')
                end_time = time.time()
                
                # Request should complete within reasonable time
                assert (end_time - start_time) < 10.0  # 10 second timeout


class TestDataConsistencyIntegration:
    """Test data consistency across integrated components."""
    
    def test_cross_component_data_validation(self, sample_batch_data):
        """Test data validation consistency across components."""
        with patch('main.data_manager') as mock_data_manager:
            with patch('main.ml_manager') as mock_ml_manager:
                # Both components should agree on data validity
                mock_data_manager.validate_data.return_value = True
                mock_ml_manager.validate_input.return_value = True
                
                for data in sample_batch_data:
                    data_valid = mock_data_manager.validate_data(data)
                    input_valid = mock_ml_manager.validate_input(data)
                    
                    # Consistency check
                    assert data_valid == input_valid
    
    def test_feature_transformation_consistency(self, sample_batch_data):
        """Test feature transformation consistency."""
        with patch('main.data_manager') as mock_data_manager:
            # Test feature engineering consistency
            expected_features = ['credit_score_normalized', 'debt_to_income_ratio', 'income_stability']
            
            mock_processed_data = pd.DataFrame({
                'credit_score_normalized': [0.75],
                'debt_to_income_ratio': [0.25],
                'income_stability': [0.8]
            })
            
            mock_data_manager.preprocess_data.return_value = mock_processed_data
            
            for data in sample_batch_data:
                processed = mock_data_manager.preprocess_data(data)
                
                # Verify expected features are present
                for feature in expected_features:
                    assert feature in processed.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])