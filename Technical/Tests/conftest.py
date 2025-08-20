"""Pytest Configuration and Shared Fixtures for Enterprise Credit Risk Intelligence Platform.

This module provides shared fixtures, configuration, and utilities for the test suite.
It includes fixtures for database setup, mock services, test data generation,
and performance monitoring.

Author: Testing Framework
Version: 2.0.0
"""

import pytest
import sys
import os
import json
import tempfile
import shutil
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
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
except ImportError:
    # Create mock imports if modules aren't available
    app = Mock()
    AnalyticsEngine = Mock
    DataManager = Mock
    MLModelManager = Mock
    VisualizationManager = Mock


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "test_main" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that take longer
        if hasattr(item, 'function'):
            if any(keyword in item.function.__name__ for keyword in 
                   ['stress', 'load', 'batch', 'concurrent', 'heavy']):
                item.add_marker(pytest.mark.slow)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def test_data_config():
    """Test data configuration constants."""
    return {
        'VALID_CREDIT_SCORES': [300, 500, 650, 750, 850],
        'VALID_INCOME_RANGES': [25000, 50000, 75000, 100000, 150000],
        'VALID_DTI_RATIOS': [0.1, 0.2, 0.3, 0.4, 0.5],
        'VALID_EMPLOYMENT_LENGTHS': [0, 1, 3, 5, 10],
        'VALID_LOAN_PURPOSES': [
            'debt_consolidation', 'home_improvement', 'car', 
            'major_purchase', 'medical', 'vacation'
        ],
        'VALID_HOME_OWNERSHIP': ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    }


@pytest.fixture
def sample_application_data(test_data_config):
    """Generate sample application data for testing."""
    config = test_data_config
    
    return {
        'application_id': 'TEST_APP_001',
        'credit_score': np.random.choice(config['VALID_CREDIT_SCORES']),
        'annual_income': np.random.choice(config['VALID_INCOME_RANGES']),
        'debt_to_income': np.random.choice(config['VALID_DTI_RATIOS']),
        'employment_length': np.random.choice(config['VALID_EMPLOYMENT_LENGTHS']),
        'loan_amount': np.random.randint(5000, 50000),
        'loan_purpose': np.random.choice(config['VALID_LOAN_PURPOSES']),
        'home_ownership': np.random.choice(config['VALID_HOME_OWNERSHIP']),
        'application_date': datetime.now().isoformat(),
        'state': 'CA',
        'verification_status': 'verified'
    }


@pytest.fixture
def batch_application_data(test_data_config):
    """Generate batch of application data for testing."""
    config = test_data_config
    batch_size = 10
    
    batch_data = []
    for i in range(batch_size):
        data = {
            'application_id': f'BATCH_TEST_{i:03d}',
            'credit_score': np.random.choice(config['VALID_CREDIT_SCORES']),
            'annual_income': np.random.choice(config['VALID_INCOME_RANGES']),
            'debt_to_income': np.random.choice(config['VALID_DTI_RATIOS']),
            'employment_length': np.random.choice(config['VALID_EMPLOYMENT_LENGTHS']),
            'loan_amount': np.random.randint(5000, 50000),
            'loan_purpose': np.random.choice(config['VALID_LOAN_PURPOSES']),
            'home_ownership': np.random.choice(config['VALID_HOME_OWNERSHIP'])
        }
        batch_data.append(data)
    
    return batch_data


@pytest.fixture
def invalid_data_samples():
    """Generate various invalid data samples for testing."""
    return [
        # Missing required fields
        {'credit_score': 750},
        {'annual_income': 50000},
        {},
        
        # Invalid data types
        {'credit_score': 'seven fifty', 'annual_income': 50000},
        {'credit_score': 750, 'annual_income': 'fifty thousand'},
        {'credit_score': [750], 'annual_income': 50000},
        
        # Out of range values
        {'credit_score': 1000, 'annual_income': 50000},  # Credit score too high
        {'credit_score': 200, 'annual_income': 50000},   # Credit score too low
        {'credit_score': 750, 'annual_income': -10000},  # Negative income
        {'credit_score': 750, 'debt_to_income': 2.0},    # DTI > 1
        
        # Null/None values
        {'credit_score': None, 'annual_income': 50000},
        {'credit_score': 750, 'annual_income': None},
        
        # Invalid enum values
        {'credit_score': 750, 'loan_purpose': 'invalid_purpose'},
        {'credit_score': 750, 'home_ownership': 'INVALID'}
    ]


@pytest.fixture
def mock_model_predictions():
    """Mock model prediction results for testing."""
    return {
        'xgboost': {
            'risk_score': 0.23,
            'probability_default': 0.047,
            'confidence': 0.95,
            'model_version': '1.2.0'
        },
        'lightgbm': {
            'risk_score': 0.25,
            'probability_default': 0.051,
            'confidence': 0.93,
            'model_version': '1.1.0'
        },
        'neural_network': {
            'risk_score': 0.22,
            'probability_default': 0.044,
            'confidence': 0.97,
            'model_version': '2.0.0'
        },
        'ensemble': {
            'risk_score': 0.233,
            'probability_default': 0.047,
            'confidence': 0.95,
            'individual_scores': [0.23, 0.25, 0.22],
            'model_version': 'ensemble_1.0'
        }
    }


# ============================================================================
# Application and Client Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def test_app():
    """Create a test application instance."""
    if hasattr(app, 'config'):
        app.config.update({
            'TESTING': True,
            'DEBUG': False,
            'WTF_CSRF_ENABLED': False,
            'SECRET_KEY': 'test-secret-key-for-testing',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
            'JSON_SORT_KEYS': False
        })
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    if hasattr(test_app, 'test_client'):
        with test_app.test_client() as client:
            yield client
    else:
        yield Mock()


@pytest.fixture
def app_context(test_app):
    """Create an application context."""
    if hasattr(test_app, 'app_context'):
        with test_app.app_context():
            yield test_app
    else:
        yield test_app


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def temp_db_path():
    """Create a temporary database file path."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_database.db')
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    db = Mock()
    
    # Mock database methods
    db.execute = Mock(return_value=Mock())
    db.fetchall = Mock(return_value=[])
    db.fetchone = Mock(return_value=None)
    db.commit = Mock()
    db.rollback = Mock()
    db.close = Mock()
    
    # Mock cursor
    cursor = Mock()
    cursor.execute = Mock()
    cursor.fetchall = Mock(return_value=[])
    cursor.fetchone = Mock(return_value=None)
    db.cursor = Mock(return_value=cursor)
    
    return db


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(':memory:')
    
    # Create test tables
    conn.execute('''
        CREATE TABLE applications (
            id INTEGER PRIMARY KEY,
            application_id TEXT UNIQUE,
            credit_score INTEGER,
            annual_income REAL,
            debt_to_income REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            application_id TEXT,
            risk_score REAL,
            probability_default REAL,
            model_version TEXT,
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (application_id) REFERENCES applications(application_id)
        )
    ''')
    
    conn.commit()
    yield conn
    conn.close()


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_analytics_engine():
    """Create a mock analytics engine."""
    engine = Mock(spec=AnalyticsEngine)
    
    # Configure mock methods
    engine.calculate_risk_metrics.return_value = {
        'portfolio_risk': 0.15,
        'concentration_risk': 0.08,
        'diversification_score': 0.85,
        'risk_adjusted_return': 0.12
    }
    
    engine.generate_portfolio_report.return_value = {
        'total_exposure': 1000000,
        'expected_loss': 50000,
        'value_at_risk': 150000,
        'sector_breakdown': {
            'technology': 0.3,
            'healthcare': 0.2,
            'finance': 0.25,
            'retail': 0.25
        }
    }
    
    return engine


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager."""
    manager = Mock(spec=DataManager)
    
    # Configure mock methods
    manager.validate_data.return_value = True
    manager.preprocess_data.return_value = pd.DataFrame({
        'credit_score_normalized': [0.75],
        'income_log': [11.0],
        'dti_ratio': [0.25]
    })
    
    manager.save_application.return_value = 'APP_12345'
    manager.save_prediction.return_value = True
    manager.get_application.return_value = {
        'application_id': 'APP_12345',
        'credit_score': 750,
        'annual_income': 85000
    }
    
    return manager


@pytest.fixture
def mock_ml_manager():
    """Create a mock ML model manager."""
    manager = Mock(spec=MLModelManager)
    
    # Configure mock methods
    manager.predict.return_value = {
        'risk_score': 0.23,
        'probability_default': 0.047,
        'confidence': 0.95
    }
    
    manager.ensemble_predict.return_value = {
        'ensemble_score': 0.233,
        'individual_scores': {
            'xgboost': 0.23,
            'lightgbm': 0.25,
            'neural_network': 0.22
        },
        'confidence': 0.95,
        'variance': 0.012
    }
    
    manager.get_model_metrics.return_value = {
        'accuracy': 0.94,
        'auc_score': 0.95,
        'precision': 0.89,
        'recall': 0.87,
        'f1_score': 0.88
    }
    
    manager.validate_input.return_value = True
    
    return manager


@pytest.fixture
def mock_visualization_manager():
    """Create a mock visualization manager."""
    manager = Mock(spec=VisualizationManager)
    
    manager.create_risk_distribution_chart.return_value = 'chart_data_base64'
    manager.create_feature_importance_plot.return_value = 'plot_data_base64'
    manager.create_performance_dashboard.return_value = 'dashboard_html'
    
    return manager


@pytest.fixture
def mock_external_services():
    """Create mocks for external services."""
    services = {
        'credit_bureau': Mock(),
        'email_service': Mock(),
        'powerbi_connector': Mock(),
        'redis_cache': Mock()
    }
    
    # Configure credit bureau mock
    services['credit_bureau'].get_credit_score.return_value = {
        'credit_score': 750,
        'credit_history_length': 8,
        'recent_inquiries': 2
    }
    
    # Configure email service mock
    services['email_service'].send_notification.return_value = True
    
    # Configure Power BI connector mock
    services['powerbi_connector'].push_data.return_value = {
        'status': 'success',
        'rows_updated': 100
    }
    
    # Configure Redis cache mock
    services['redis_cache'].get.return_value = None
    services['redis_cache'].set.return_value = True
    services['redis_cache'].delete.return_value = True
    
    return services


# ============================================================================
# Performance and Monitoring Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Create a performance monitoring fixture."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.start_cpu = None
            self.end_cpu = None
        
        def start_monitoring(self):
            self.start_time = time.time()
            self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            self.start_cpu = psutil.cpu_percent(interval=None)
        
        def stop_monitoring(self):
            self.end_time = time.time()
            self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            self.end_cpu = psutil.cpu_percent(interval=None)
        
        def get_metrics(self):
            return {
                'execution_time_ms': (self.end_time - self.start_time) * 1000 if self.end_time else None,
                'memory_usage_mb': self.end_memory - self.start_memory if self.end_memory else None,
                'cpu_usage_percent': self.end_cpu if self.end_cpu else None
            }
    
    return PerformanceMonitor()


@pytest.fixture
def load_test_data():
    """Generate data for load testing."""
    def generate_load_data(size=100):
        """Generate a specified amount of test data."""
        data = []
        for i in range(size):
            application = {
                'application_id': f'LOAD_TEST_{i:05d}',
                'credit_score': np.random.randint(300, 851),
                'annual_income': np.random.randint(25000, 200000),
                'debt_to_income': np.random.uniform(0.1, 0.5),
                'employment_length': np.random.randint(0, 11),
                'loan_amount': np.random.randint(5000, 50000)
            }
            data.append(application)
        return data
    
    return generate_load_data


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def test_utilities():
    """Provide test utility functions."""
    
    class TestUtilities:
        @staticmethod
        def assert_valid_response_structure(response_data, required_fields):
            """Assert that response has required structure."""
            for field in required_fields:
                assert field in response_data, f"Missing required field: {field}"
        
        @staticmethod
        def assert_valid_risk_score(score):
            """Assert that risk score is valid."""
            assert isinstance(score, (int, float)), "Risk score must be numeric"
            assert 0 <= score <= 1, "Risk score must be between 0 and 1"
        
        @staticmethod
        def assert_valid_timestamp(timestamp_str):
            """Assert that timestamp is valid ISO format."""
            try:
                datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {timestamp_str}")
        
        @staticmethod
        def generate_test_data_variations(base_data, variations):
            """Generate test data variations."""
            test_cases = []
            for variation in variations:
                test_case = base_data.copy()
                test_case.update(variation)
                test_cases.append(test_case)
            return test_cases
    
    return TestUtilities()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code here if needed
    pass


@pytest.fixture(scope='session', autouse=True)
def cleanup_after_session():
    """Automatic cleanup after test session."""
    yield
    # Session-level cleanup code here if needed
    pass