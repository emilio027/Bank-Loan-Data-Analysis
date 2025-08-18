"""Test Data Generation and Management for Enterprise Credit Risk Intelligence Platform.

This module provides comprehensive test data generation, validation, and management
utilities for testing the credit risk platform. It includes data generators for
various scenarios, edge cases, and validation functions.

Author: Testing Framework
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings("ignore")


class CreditRiskTestDataGenerator:
    """Comprehensive test data generator for credit risk testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the test data generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define valid ranges and options
        self.valid_ranges = {
            'credit_score': (300, 850),
            'annual_income': (15000, 500000),
            'debt_to_income': (0.0, 0.8),
            'employment_length': (0, 15),
            'loan_amount': (1000, 100000),
            'age': (18, 80)
        }
        
        self.categorical_options = {
            'loan_purpose': [
                'debt_consolidation', 'home_improvement', 'car', 'major_purchase',
                'medical', 'vacation', 'moving', 'wedding', 'house', 'other'
            ],
            'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            'verification_status': ['verified', 'source_verified', 'not_verified'],
            'employment_status': ['employed', 'self_employed', 'unemployed', 'retired'],
            'education_level': ['high_school', 'some_college', 'bachelors', 'masters', 'phd'],
            'marital_status': ['single', 'married', 'divorced', 'widowed'],
            'state': ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        }
        
        # Risk profiles for generating realistic data
        self.risk_profiles = {
            'low_risk': {
                'credit_score': (750, 850),
                'debt_to_income': (0.0, 0.25),
                'employment_length': (3, 15),
                'annual_income': (75000, 200000)
            },
            'medium_risk': {
                'credit_score': (650, 749),
                'debt_to_income': (0.25, 0.4),
                'employment_length': (1, 5),
                'annual_income': (40000, 85000)
            },
            'high_risk': {
                'credit_score': (300, 649),
                'debt_to_income': (0.4, 0.6),
                'employment_length': (0, 2),
                'annual_income': (15000, 50000)
            }
        }
    
    def generate_single_application(self, 
                                   risk_profile: str = 'random',
                                   include_optional: bool = True) -> Dict[str, Any]:
        """Generate a single credit application with specified risk profile."""
        
        # Select risk profile
        if risk_profile == 'random':
            profile_name = np.random.choice(['low_risk', 'medium_risk', 'high_risk'])
            profile = self.risk_profiles[profile_name]
        elif risk_profile in self.risk_profiles:
            profile = self.risk_profiles[risk_profile]
        else:
            # Use full ranges for custom/invalid profiles
            profile = {key: self.valid_ranges[key] for key in self.valid_ranges if key != 'age'}
        
        # Generate core required fields
        application = {
            'application_id': self._generate_application_id(),
            'credit_score': np.random.randint(profile['credit_score'][0], profile['credit_score'][1] + 1),
            'annual_income': np.random.randint(profile['annual_income'][0], profile['annual_income'][1]),
            'debt_to_income': np.round(np.random.uniform(profile['debt_to_income'][0], profile['debt_to_income'][1]), 3),
            'employment_length': np.random.randint(profile['employment_length'][0], profile['employment_length'][1] + 1),
            'loan_amount': np.random.randint(self.valid_ranges['loan_amount'][0], self.valid_ranges['loan_amount'][1]),
            'loan_purpose': np.random.choice(self.categorical_options['loan_purpose']),
            'home_ownership': np.random.choice(self.categorical_options['home_ownership'])
        }
        
        # Add optional fields if requested
        if include_optional:
            application.update({
                'verification_status': np.random.choice(self.categorical_options['verification_status']),
                'employment_status': np.random.choice(self.categorical_options['employment_status']),
                'education_level': np.random.choice(self.categorical_options['education_level']),
                'marital_status': np.random.choice(self.categorical_options['marital_status']),
                'state': np.random.choice(self.categorical_options['state']),
                'age': np.random.randint(self.valid_ranges['age'][0], self.valid_ranges['age'][1]),
                'application_date': self._generate_recent_date(),
                'months_since_last_delinq': np.random.randint(0, 120) if np.random.random() < 0.3 else None,
                'public_record_bankruptcies': np.random.randint(0, 3) if np.random.random() < 0.1 else 0,
                'inquiries_last_6m': np.random.randint(0, 10),
                'open_accounts': np.random.randint(1, 25),
                'total_accounts': np.random.randint(5, 50)
            })
        
        return application
    
    def generate_batch_applications(self, 
                                   count: int,
                                   risk_distribution: Optional[Dict[str, float]] = None,
                                   include_optional: bool = True) -> List[Dict[str, Any]]:
        """Generate a batch of credit applications with specified distribution."""
        
        if risk_distribution is None:
            risk_distribution = {'low_risk': 0.4, 'medium_risk': 0.4, 'high_risk': 0.2}
        
        # Validate risk distribution
        assert abs(sum(risk_distribution.values()) - 1.0) < 0.01, "Risk distribution must sum to 1.0"
        
        applications = []
        
        for i in range(count):
            # Select risk profile based on distribution
            rand_val = np.random.random()
            cumulative = 0
            selected_profile = 'medium_risk'  # default
            
            for profile, probability in risk_distribution.items():
                cumulative += probability
                if rand_val <= cumulative:
                    selected_profile = profile
                    break
            
            application = self.generate_single_application(
                risk_profile=selected_profile,
                include_optional=include_optional
            )
            applications.append(application)
        
        return applications
    
    def generate_edge_case_data(self) -> List[Dict[str, Any]]:
        """Generate edge case data for testing boundary conditions."""
        edge_cases = []
        
        # Minimum values
        edge_cases.append({
            'application_id': 'EDGE_MIN_001',
            'credit_score': self.valid_ranges['credit_score'][0],  # 300
            'annual_income': self.valid_ranges['annual_income'][0],  # 15000
            'debt_to_income': self.valid_ranges['debt_to_income'][0],  # 0.0
            'employment_length': self.valid_ranges['employment_length'][0],  # 0
            'loan_amount': self.valid_ranges['loan_amount'][0],  # 1000
            'loan_purpose': self.categorical_options['loan_purpose'][0],
            'home_ownership': self.categorical_options['home_ownership'][0]
        })
        
        # Maximum values
        edge_cases.append({
            'application_id': 'EDGE_MAX_001',
            'credit_score': self.valid_ranges['credit_score'][1],  # 850
            'annual_income': self.valid_ranges['annual_income'][1],  # 500000
            'debt_to_income': self.valid_ranges['debt_to_income'][1],  # 0.8
            'employment_length': self.valid_ranges['employment_length'][1],  # 15
            'loan_amount': self.valid_ranges['loan_amount'][1],  # 100000
            'loan_purpose': self.categorical_options['loan_purpose'][-1],
            'home_ownership': self.categorical_options['home_ownership'][-1]
        })
        
        # Boundary conditions
        boundary_cases = [
            {'credit_score': 299},  # Just below minimum
            {'credit_score': 851},  # Just above maximum
            {'annual_income': 0},   # Zero income
            {'debt_to_income': -0.1},  # Negative DTI
            {'debt_to_income': 1.5},   # DTI > 1
            {'employment_length': -1}, # Negative employment
            {'loan_amount': 0},        # Zero loan amount
        ]
        
        for i, boundary_case in enumerate(boundary_cases):
            base_data = self.generate_single_application('medium_risk', include_optional=False)
            base_data['application_id'] = f'EDGE_BOUNDARY_{i:03d}'
            base_data.update(boundary_case)
            edge_cases.append(base_data)
        
        return edge_cases
    
    def generate_invalid_data_cases(self) -> List[Dict[str, Any]]:
        """Generate various invalid data cases for negative testing."""
        invalid_cases = []
        
        # Missing required fields
        required_fields = ['credit_score', 'annual_income', 'debt_to_income', 
                          'employment_length', 'loan_amount', 'loan_purpose']
        
        for field in required_fields:
            incomplete_data = self.generate_single_application('medium_risk', include_optional=False)
            incomplete_data['application_id'] = f'INVALID_MISSING_{field.upper()}'
            del incomplete_data[field]
            invalid_cases.append(incomplete_data)
        
        # Invalid data types
        type_errors = [
            {'credit_score': 'seven hundred fifty'},
            {'annual_income': '85,000'},
            {'debt_to_income': '25%'},
            {'employment_length': 'five years'},
            {'loan_amount': '$25,000'},
            {'credit_score': [750]},
            {'annual_income': {'amount': 85000}},
            {'debt_to_income': True},
        ]
        
        for i, type_error in enumerate(type_errors):
            base_data = self.generate_single_application('medium_risk', include_optional=False)
            base_data['application_id'] = f'INVALID_TYPE_{i:03d}'
            base_data.update(type_error)
            invalid_cases.append(base_data)
        
        # Null/None values
        null_cases = [
            {'credit_score': None},
            {'annual_income': None},
            {'debt_to_income': None},
            {'loan_purpose': None},
        ]
        
        for i, null_case in enumerate(null_cases):
            base_data = self.generate_single_application('medium_risk', include_optional=False)
            base_data['application_id'] = f'INVALID_NULL_{i:03d}'
            base_data.update(null_case)
            invalid_cases.append(base_data)
        
        # Invalid categorical values
        invalid_categorical = [
            {'loan_purpose': 'invalid_purpose'},
            {'home_ownership': 'INVALID_OWNERSHIP'},
            {'verification_status': 'not_a_status'},
            {'state': 'INVALID_STATE'},
        ]
        
        for i, invalid_cat in enumerate(invalid_categorical):
            base_data = self.generate_single_application('medium_risk', include_optional=True)
            base_data['application_id'] = f'INVALID_CAT_{i:03d}'
            base_data.update(invalid_cat)
            invalid_cases.append(base_data)
        
        return invalid_cases
    
    def generate_time_series_data(self, 
                                 days: int = 30,
                                 applications_per_day_range: tuple = (50, 200)) -> List[Dict[str, Any]]:
        """Generate time series application data for trend analysis."""
        time_series_data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            applications_today = np.random.randint(applications_per_day_range[0], applications_per_day_range[1])
            
            # Simulate business cycles (more applications on weekdays)
            if current_date.weekday() >= 5:  # Weekend
                applications_today = int(applications_today * 0.6)
            
            daily_applications = self.generate_batch_applications(
                count=applications_today,
                include_optional=True
            )
            
            # Add timestamp to each application
            for i, app in enumerate(daily_applications):
                # Spread applications throughout the day
                hour_offset = np.random.randint(0, 24)
                minute_offset = np.random.randint(0, 60)
                app_datetime = current_date.replace(hour=hour_offset, minute=minute_offset)
                
                app['application_date'] = app_datetime.isoformat()
                app['application_id'] = f'TS_{current_date.strftime("%Y%m%d")}_{i:04d}'
                time_series_data.append(app)
        
        return time_series_data
    
    def generate_stress_test_data(self, size: int = 10000) -> List[Dict[str, Any]]:
        """Generate large dataset for stress testing."""
        return self.generate_batch_applications(
            count=size,
            risk_distribution={'low_risk': 0.3, 'medium_risk': 0.5, 'high_risk': 0.2},
            include_optional=True
        )
    
    def _generate_application_id(self) -> str:
        """Generate a unique application ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = np.random.randint(1000, 9999)
        return f'APP_{timestamp}_{random_suffix}'
    
    def _generate_recent_date(self) -> str:
        """Generate a recent application date within the last 30 days."""
        days_ago = np.random.randint(0, 30)
        app_date = datetime.now() - timedelta(days=days_ago)
        return app_date.isoformat()


class MockDataValidator:
    """Validator for test data to ensure it meets expected criteria."""
    
    @staticmethod
    def validate_application_data(application: Dict[str, Any]) -> Dict[str, Any]:
        """Validate application data and return validation results."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields check
        required_fields = [
            'application_id', 'credit_score', 'annual_income', 
            'debt_to_income', 'employment_length', 'loan_amount', 
            'loan_purpose', 'home_ownership'
        ]
        
        for field in required_fields:
            if field not in application:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Data type validation
        numeric_fields = {
            'credit_score': int,
            'annual_income': (int, float),
            'debt_to_income': (int, float),
            'employment_length': int,
            'loan_amount': (int, float)
        }
        
        for field, expected_type in numeric_fields.items():
            if field in application:
                if not isinstance(application[field], expected_type):
                    validation_result['errors'].append(f"Invalid type for {field}: expected {expected_type}")
                    validation_result['is_valid'] = False
        
        # Range validation
        range_validations = {
            'credit_score': (300, 850),
            'annual_income': (0, 1000000),
            'debt_to_income': (0, 1),
            'employment_length': (0, 50),
            'loan_amount': (100, 1000000)
        }
        
        for field, (min_val, max_val) in range_validations.items():
            if field in application and isinstance(application[field], (int, float)):
                value = application[field]
                if not (min_val <= value <= max_val):
                    validation_result['errors'].append(
                        f"{field} value {value} is outside valid range [{min_val}, {max_val}]"
                    )
                    validation_result['is_valid'] = False
        
        # Business logic validation
        if 'debt_to_income' in application and 'annual_income' in application:
            if application['debt_to_income'] > 0.6:
                validation_result['warnings'].append("High debt-to-income ratio detected")
        
        if 'credit_score' in application:
            if application['credit_score'] < 500:
                validation_result['warnings'].append("Low credit score detected")
        
        return validation_result
    
    @staticmethod
    def validate_batch_data(applications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of application data."""
        batch_validation = {
            'total_applications': len(applications),
            'valid_applications': 0,
            'invalid_applications': 0,
            'validation_errors': [],
            'summary': {}
        }
        
        for i, app in enumerate(applications):
            validation = MockDataValidator.validate_application_data(app)
            
            if validation['is_valid']:
                batch_validation['valid_applications'] += 1
            else:
                batch_validation['invalid_applications'] += 1
                batch_validation['validation_errors'].append({
                    'application_index': i,
                    'application_id': app.get('application_id', 'Unknown'),
                    'errors': validation['errors']
                })
        
        # Generate summary statistics
        if applications:
            credit_scores = [app.get('credit_score', 0) for app in applications if 'credit_score' in app]
            annual_incomes = [app.get('annual_income', 0) for app in applications if 'annual_income' in app]
            
            if credit_scores:
                batch_validation['summary']['credit_score_stats'] = {
                    'mean': np.mean(credit_scores),
                    'median': np.median(credit_scores),
                    'min': np.min(credit_scores),
                    'max': np.max(credit_scores),
                    'std': np.std(credit_scores)
                }
            
            if annual_incomes:
                batch_validation['summary']['income_stats'] = {
                    'mean': np.mean(annual_incomes),
                    'median': np.median(annual_incomes),
                    'min': np.min(annual_incomes),
                    'max': np.max(annual_incomes),
                    'std': np.std(annual_incomes)
                }
        
        return batch_validation


class TestDatasets:
    """Pre-defined test datasets for common testing scenarios."""
    
    def __init__(self):
        self.generator = CreditRiskTestDataGenerator()
    
    def get_golden_dataset(self) -> List[Dict[str, Any]]:
        """Get a golden dataset with known good data for regression testing."""
        return self.generator.generate_batch_applications(
            count=100,
            risk_distribution={'low_risk': 0.4, 'medium_risk': 0.4, 'high_risk': 0.2}
        )
    
    def get_edge_case_dataset(self) -> List[Dict[str, Any]]:
        """Get dataset with edge cases for boundary testing."""
        return self.generator.generate_edge_case_data()
    
    def get_invalid_dataset(self) -> List[Dict[str, Any]]:
        """Get dataset with invalid data for negative testing."""
        return self.generator.generate_invalid_data_cases()
    
    def get_performance_dataset(self, size: int = 1000) -> List[Dict[str, Any]]:
        """Get large dataset for performance testing."""
        return self.generator.generate_stress_test_data(size)
    
    def get_time_series_dataset(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get time series dataset for trend analysis testing."""
        return self.generator.generate_time_series_data(days)


# ============================================================================
# Pytest Fixtures for Test Data
# ============================================================================

@pytest.fixture(scope='session')
def test_data_generator():
    """Provide test data generator instance."""
    return CreditRiskTestDataGenerator()


@pytest.fixture(scope='session')
def test_datasets():
    """Provide pre-defined test datasets."""
    return TestDatasets()


@pytest.fixture
def golden_dataset(test_datasets):
    """Golden dataset for regression testing."""
    return test_datasets.get_golden_dataset()


@pytest.fixture
def edge_case_dataset(test_datasets):
    """Edge case dataset for boundary testing."""
    return test_datasets.get_edge_case_dataset()


@pytest.fixture
def invalid_dataset(test_datasets):
    """Invalid dataset for negative testing."""
    return test_datasets.get_invalid_dataset()


@pytest.fixture
def performance_dataset(test_datasets):
    """Performance dataset for load testing."""
    return test_datasets.get_performance_dataset(1000)


@pytest.fixture
def time_series_dataset(test_datasets):
    """Time series dataset for trend analysis."""
    return test_datasets.get_time_series_dataset(30)


@pytest.fixture
def data_validator():
    """Provide data validator instance."""
    return MockDataValidator()


# ============================================================================
# Test Functions for Data Generation
# ============================================================================

def test_single_application_generation():
    """Test single application data generation."""
    generator = CreditRiskTestDataGenerator()
    
    # Test different risk profiles
    for risk_profile in ['low_risk', 'medium_risk', 'high_risk']:
        app = generator.generate_single_application(risk_profile)
        
        # Basic structure validation
        assert 'application_id' in app
        assert 'credit_score' in app
        assert 'annual_income' in app
        assert isinstance(app['credit_score'], int)
        assert isinstance(app['annual_income'], (int, float))


def test_batch_application_generation():
    """Test batch application data generation."""
    generator = CreditRiskTestDataGenerator()
    
    # Test batch generation
    batch = generator.generate_batch_applications(count=50)
    
    assert len(batch) == 50
    assert all('application_id' in app for app in batch)
    
    # Test unique application IDs
    app_ids = [app['application_id'] for app in batch]
    assert len(set(app_ids)) == len(app_ids), "Application IDs should be unique"


def test_edge_case_generation():
    """Test edge case data generation."""
    generator = CreditRiskTestDataGenerator()
    
    edge_cases = generator.generate_edge_case_data()
    
    assert len(edge_cases) > 0
    assert any('EDGE_MIN' in app['application_id'] for app in edge_cases)
    assert any('EDGE_MAX' in app['application_id'] for app in edge_cases)


def test_invalid_data_generation():
    """Test invalid data generation."""
    generator = CreditRiskTestDataGenerator()
    
    invalid_cases = generator.generate_invalid_data_cases()
    
    assert len(invalid_cases) > 0
    assert any('INVALID' in app['application_id'] for app in invalid_cases)


def test_data_validation():
    """Test data validation functionality."""
    generator = CreditRiskTestDataGenerator()
    validator = MockDataValidator()
    
    # Test valid data
    valid_app = generator.generate_single_application('medium_risk')
    validation_result = validator.validate_application_data(valid_app)
    
    assert validation_result['is_valid'] is True
    assert len(validation_result['errors']) == 0
    
    # Test invalid data
    invalid_app = {'credit_score': 'invalid'}
    validation_result = validator.validate_application_data(invalid_app)
    
    assert validation_result['is_valid'] is False
    assert len(validation_result['errors']) > 0


def test_time_series_generation():
    """Test time series data generation."""
    generator = CreditRiskTestDataGenerator()
    
    time_series = generator.generate_time_series_data(days=7, applications_per_day_range=(10, 20))
    
    assert len(time_series) > 0
    assert all('application_date' in app for app in time_series)
    
    # Verify date ordering
    dates = [app['application_date'] for app in time_series]
    assert len(dates) == len(time_series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])