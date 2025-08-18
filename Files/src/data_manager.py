"""
Data Manager Module

Handles all data operations including extraction, transformation, loading,
and caching for the AI Law Firm Dashboard.

Author: Emilio Cardenas
Email: ec@emiliocardenas.io
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
from pathlib import Path
import asyncio
import aiohttp
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import redis
from functools import lru_cache, wraps
import time

logger = logging.getLogger(__name__)

def cache_result(expiration: int = 300):
    """
    Decorator for caching function results.
    
    Args:
        expiration: Cache expiration time in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache
            try:
                if hasattr(wrapper, '_cache') and cache_key in wrapper._cache:
                    cached_time, result = wrapper._cache[cache_key]
                    if time.time() - cached_time < expiration:
                        return result
            except:
                pass
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = {}
            wrapper._cache[cache_key] = (time.time(), result)
            
            return result
        return wrapper
    return decorator

class DataManager:
    """
    Comprehensive data management system for legal operations.
    
    Handles data extraction from multiple sources, transformation,
    validation, and caching for optimal performance.
    """
    
    def __init__(self, config):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration object containing database and API settings
        """
        self.config = config
        self.db_engine = None
        self.redis_client = None
        self._connection_status = False
        
        # Initialize connections
        self._initialize_connections()
        
        # Sample data for demo purposes
        self._initialize_sample_data()
        
        logger.info("DataManager initialized successfully")
    
    def _initialize_connections(self) -> None:
        """Initialize database and cache connections."""
        try:
            # Database connection
            if hasattr(self.config, 'DATABASE_URL'):
                self.db_engine = create_engine(self.config.DATABASE_URL)
                self._connection_status = True
            
            # Redis connection for caching
            if hasattr(self.config, 'REDIS_URL'):
                self.redis_client = redis.from_url(self.config.REDIS_URL)
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.warning(f"Database connection failed, using demo data: {str(e)}")
            self._connection_status = False
    
    def _initialize_sample_data(self) -> None:
        """Initialize sample data for demonstration purposes."""
        np.random.seed(42)
        
        # Generate sample matters data
        self.sample_matters = self._generate_sample_matters()
        self.sample_attorneys = self._generate_sample_attorneys()
        self.sample_clients = self._generate_sample_clients()
        self.sample_financial = self._generate_sample_financial()
        
        logger.info("Sample data initialized")
    
    def _generate_sample_matters(self) -> pd.DataFrame:
        """Generate realistic sample matters data."""
        n_matters = 150
        
        practice_areas = [
            'Corporate Law', 'Litigation', 'Real Estate', 'Employment Law',
            'Intellectual Property', 'Tax Law', 'Family Law', 'Criminal Defense'
        ]
        
        statuses = ['Active', 'On Hold', 'Completed', 'Cancelled']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        
        # Generate base data
        matters_data = {
            'matter_id': [f"M{1000 + i}" for i in range(n_matters)],
            'client_name': np.random.choice([
                'TechCorp Inc.', 'Global Industries', 'Smith & Associates',
                'Metro Real Estate', 'Innovation Labs', 'Healthcare Partners',
                'Financial Services Group', 'Manufacturing Co.', 'Retail Chain',
                'Energy Solutions', 'Transportation LLC', 'Media Group'
            ], n_matters),
            'practice_area': np.random.choice(practice_areas, n_matters),
            'status': np.random.choice(statuses, n_matters, p=[0.6, 0.1, 0.25, 0.05]),
            'priority': np.random.choice(priorities, n_matters, p=[0.3, 0.4, 0.25, 0.05]),
            'assigned_attorney': np.random.choice([
                'Sarah Johnson', 'Michael Chen', 'Emily Rodriguez', 'David Kim',
                'Lisa Thompson', 'Robert Wilson', 'Jennifer Lee', 'Mark Davis'
            ], n_matters),
            'start_date': pd.date_range(
                start='2023-01-01', 
                end='2024-12-01', 
                periods=n_matters
            ),
            'estimated_hours': np.random.gamma(2, 50, n_matters).round(1),
            'actual_hours': np.random.gamma(2, 45, n_matters).round(1),
            'hourly_rate': np.random.choice([250, 350, 450, 550, 650], n_matters),
            'estimated_value': None,
            'actual_value': None,
            'expenses': np.random.exponential(2000, n_matters).round(2),
            'description': [
                f"Legal matter involving {area.lower()} for {client}"
                for area, client in zip(
                    np.random.choice(practice_areas, n_matters),
                    np.random.choice(['contract review', 'compliance audit', 'litigation support', 
                                    'regulatory filing', 'due diligence', 'negotiation'], n_matters)
                )
            ]
        }
        
        df = pd.DataFrame(matters_data)
        
        # Calculate derived fields
        df['estimated_value'] = df['estimated_hours'] * df['hourly_rate']
        df['actual_value'] = df['actual_hours'] * df['hourly_rate']
        df['profit_margin'] = ((df['actual_value'] - df['expenses']) / df['actual_value'] * 100).round(2)
        df['days_active'] = (datetime.now() - df['start_date']).dt.days
        
        # Add deadline information
        df['deadline'] = df['start_date'] + pd.to_timedelta(
            np.random.randint(30, 365, n_matters), unit='D'
        )
        df['days_to_deadline'] = (df['deadline'] - datetime.now()).dt.days
        
        return df
    
    def _generate_sample_attorneys(self) -> pd.DataFrame:
        """Generate sample attorney data."""
        attorneys = [
            {'name': 'Sarah Johnson', 'practice_area': 'Corporate Law', 'level': 'Partner', 'hourly_rate': 650},
            {'name': 'Michael Chen', 'practice_area': 'Litigation', 'level': 'Senior Associate', 'hourly_rate': 450},
            {'name': 'Emily Rodriguez', 'practice_area': 'Real Estate', 'level': 'Partner', 'hourly_rate': 550},
            {'name': 'David Kim', 'practice_area': 'Employment Law', 'level': 'Associate', 'hourly_rate': 350},
            {'name': 'Lisa Thompson', 'practice_area': 'IP Law', 'level': 'Senior Associate', 'hourly_rate': 500},
            {'name': 'Robert Wilson', 'practice_area': 'Tax Law', 'level': 'Partner', 'hourly_rate': 600},
            {'name': 'Jennifer Lee', 'practice_area': 'Family Law', 'level': 'Associate', 'hourly_rate': 300},
            {'name': 'Mark Davis', 'practice_area': 'Criminal Defense', 'level': 'Senior Associate', 'hourly_rate': 400}
        ]
        
        df = pd.DataFrame(attorneys)
        
        # Add performance metrics
        df['target_hours'] = 1800
        df['actual_hours'] = np.random.normal(1750, 200, len(df)).round(0)
        df['utilization_rate'] = (df['actual_hours'] / df['target_hours'] * 100).round(1)
        df['revenue_generated'] = df['actual_hours'] * df['hourly_rate']
        df['client_satisfaction'] = np.random.normal(4.2, 0.3, len(df)).round(1)
        
        return df
    
    def _generate_sample_clients(self) -> pd.DataFrame:
        """Generate sample client data."""
        clients = [
            'TechCorp Inc.', 'Global Industries', 'Smith & Associates',
            'Metro Real Estate', 'Innovation Labs', 'Healthcare Partners',
            'Financial Services Group', 'Manufacturing Co.', 'Retail Chain',
            'Energy Solutions', 'Transportation LLC', 'Media Group'
        ]
        
        client_data = {
            'client_name': clients,
            'industry': np.random.choice([
                'Technology', 'Manufacturing', 'Healthcare', 'Finance',
                'Real Estate', 'Energy', 'Transportation', 'Media'
            ], len(clients)),
            'relationship_start': pd.date_range(
                start='2020-01-01', 
                end='2023-01-01', 
                periods=len(clients)
            ),
            'total_revenue': np.random.exponential(500000, len(clients)).round(2),
            'outstanding_balance': np.random.exponential(50000, len(clients)).round(2),
            'payment_terms': np.random.choice([30, 45, 60], len(clients)),
            'credit_rating': np.random.choice(['A', 'B', 'C'], len(clients), p=[0.6, 0.3, 0.1])
        }
        
        df = pd.DataFrame(client_data)
        df['relationship_years'] = ((datetime.now() - df['relationship_start']).dt.days / 365).round(1)
        df['annual_revenue'] = (df['total_revenue'] / df['relationship_years']).round(2)
        
        return df
    
    def _generate_sample_financial(self) -> pd.DataFrame:
        """Generate sample financial data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        
        financial_data = {
            'date': dates,
            'daily_revenue': np.random.exponential(15000, len(dates)),
            'daily_expenses': np.random.exponential(8000, len(dates)),
            'billable_hours': np.random.poisson(120, len(dates)),
            'collection_amount': np.random.exponential(12000, len(dates))
        }
        
        df = pd.DataFrame(financial_data)
        df['daily_profit'] = df['daily_revenue'] - df['daily_expenses']
        df['profit_margin'] = (df['daily_profit'] / df['daily_revenue'] * 100).round(2)
        
        return df
    
    @cache_result(expiration=300)
    def get_filtered_data(self, filters: Dict) -> pd.DataFrame:
        """
        Get filtered matters data based on provided filters.
        
        Args:
            filters: Dictionary containing filter criteria
            
        Returns:
            Filtered DataFrame
        """
        try:
            df = self.sample_matters.copy()
            
            # Apply date range filter
            if 'date_range' in filters and filters['date_range']:
                start_date, end_date = filters['date_range']
                df = df[
                    (df['start_date'].dt.date >= start_date) & 
                    (df['start_date'].dt.date <= end_date)
                ]
            
            # Apply practice area filter
            if 'practice_areas' in filters and filters['practice_areas']:
                df = df[df['practice_area'].isin(filters['practice_areas'])]
            
            # Apply attorney filter
            if 'attorneys' in filters and filters['attorneys']:
                df = df[df['assigned_attorney'].isin(filters['attorneys'])]
            
            # Apply client filter
            if 'clients' in filters and filters['clients']:
                df = df[df['client_name'].isin(filters['clients'])]
            
            logger.info(f"Filtered data: {len(df)} records returned")
            return df
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            return pd.DataFrame()
    
    @cache_result(expiration=600)
    def get_practice_areas(self) -> List[str]:
        """Get list of available practice areas."""
        return sorted(self.sample_matters['practice_area'].unique().tolist())
    
    @cache_result(expiration=600)
    def get_attorneys(self) -> List[str]:
        """Get list of available attorneys."""
        return sorted(self.sample_matters['assigned_attorney'].unique().tolist())
    
    @cache_result(expiration=600)
    def get_top_clients(self, limit: int = 20) -> List[str]:
        """
        Get top clients by revenue.
        
        Args:
            limit: Maximum number of clients to return
            
        Returns:
            List of client names
        """
        client_revenue = self.sample_matters.groupby('client_name')['actual_value'].sum()
        return client_revenue.nlargest(limit).index.tolist()
    
    def get_matters_data(self, filters: Dict) -> pd.DataFrame:
        """
        Get detailed matters data for display.
        
        Args:
            filters: Applied filters
            
        Returns:
            Formatted matters DataFrame
        """
        df = self.get_filtered_data(filters)
        
        if df.empty:
            return df
        
        # Select and format columns for display
        display_columns = [
            'matter_id', 'client_name', 'practice_area', 'status',
            'assigned_attorney', 'actual_hours', 'hourly_rate',
            'actual_value', 'profit_margin', 'days_to_deadline'
        ]
        
        result = df[display_columns].copy()
        result.columns = [
            'Matter ID', 'Client', 'Practice Area', 'Status',
            'Attorney', 'Hours', 'Rate', 'Value', 'Margin %', 'Days to Deadline'
        ]
        
        return result.sort_values('Value', ascending=False)
    
    def get_upcoming_deadlines(self, filters: Dict, days_ahead: int = 30) -> pd.DataFrame:
        """
        Get matters with upcoming deadlines.
        
        Args:
            filters: Applied filters
            days_ahead: Number of days to look ahead
            
        Returns:
            DataFrame with upcoming deadlines
        """
        df = self.get_filtered_data(filters)
        
        if df.empty:
            return df
        
        # Filter for upcoming deadlines
        upcoming = df[
            (df['days_to_deadline'] >= 0) & 
            (df['days_to_deadline'] <= days_ahead)
        ].copy()
        
        if upcoming.empty:
            return upcoming
        
        # Format for display
        result = upcoming[[
            'matter_id', 'client_name', 'practice_area', 
            'assigned_attorney', 'deadline', 'days_to_deadline'
        ]].copy()
        
        result.columns = [
            'Matter ID', 'Client', 'Practice Area', 
            'Attorney', 'Deadline', 'Days Remaining'
        ]
        
        return result.sort_values('Days Remaining')
    
    def get_financial_data(self, filters: Dict) -> pd.DataFrame:
        """
        Get financial data based on filters.
        
        Args:
            filters: Applied filters
            
        Returns:
            Financial DataFrame
        """
        df = self.sample_financial.copy()
        
        # Apply date range filter
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            df = df[
                (df['date'].dt.date >= start_date) & 
                (df['date'].dt.date <= end_date)
            ]
        
        return df
    
    def get_attorney_performance(self, filters: Dict) -> pd.DataFrame:
        """
        Get attorney performance data.
        
        Args:
            filters: Applied filters
            
        Returns:
            Attorney performance DataFrame
        """
        matters_df = self.get_filtered_data(filters)
        
        if matters_df.empty:
            return pd.DataFrame()
        
        # Aggregate by attorney
        performance = matters_df.groupby('assigned_attorney').agg({
            'matter_id': 'count',
            'actual_hours': 'sum',
            'actual_value': 'sum',
            'profit_margin': 'mean'
        }).round(2)
        
        performance.columns = ['Matters', 'Hours', 'Revenue', 'Avg Margin %']
        performance['Avg Revenue/Matter'] = (performance['Revenue'] / performance['Matters']).round(2)
        
        return performance.sort_values('Revenue', ascending=False)
    
    def refresh_data(self) -> None:
        """Refresh all cached data."""
        try:
            # Clear function caches
            if hasattr(self.get_filtered_data, '_cache'):
                self.get_filtered_data._cache.clear()
            if hasattr(self.get_practice_areas, '_cache'):
                self.get_practice_areas._cache.clear()
            if hasattr(self.get_attorneys, '_cache'):
                self.get_attorneys._cache.clear()
            if hasattr(self.get_top_clients, '_cache'):
                self.get_top_clients._cache.clear()
            
            # Clear Redis cache if available
            if self.redis_client:
                self.redis_client.flushdb()
            
            # Regenerate sample data with new random seed
            np.random.seed(int(time.time()))
            self._initialize_sample_data()
            
            logger.info("Data refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}")
            raise
    
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self._connection_status
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Get data quality metrics for monitoring.
        
        Returns:
            Dictionary with data quality metrics
        """
        try:
            df = self.sample_matters
            
            metrics = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'data_freshness': (datetime.now() - df['start_date'].max()).days,
                'completeness_rate': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {str(e)}")
            return {}
    
    async def async_data_fetch(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Asynchronously fetch data from external APIs.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            
        Returns:
            API response data
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error in async data fetch: {str(e)}")
            return {}
    
    def export_data(self, data: pd.DataFrame, format: str = 'csv') -> bytes:
        """
        Export data in specified format.
        
        Args:
            data: DataFrame to export
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            Exported data as bytes
        """
        try:
            if format.lower() == 'csv':
                return data.to_csv(index=False).encode('utf-8')
            elif format.lower() == 'excel':
                import io
                buffer = io.BytesIO()
                data.to_excel(buffer, index=False)
                return buffer.getvalue()
            elif format.lower() == 'json':
                return data.to_json(orient='records').encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise

