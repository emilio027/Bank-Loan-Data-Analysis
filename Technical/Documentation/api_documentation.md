# Enterprise Credit Risk Intelligence Platform
## API Documentation

### Version 2.0.0 Enterprise
### Author: API Documentation Team
### Date: August 2025

---

## Overview

The Enterprise Credit Risk Intelligence Platform provides comprehensive RESTful APIs for credit risk assessment, portfolio management, and regulatory compliance. All APIs support JSON data exchange and implement enterprise-grade security with OAuth 2.0 authentication.

**Base URL**: `https://api.creditrisk.enterprise.com/v2`
**Authentication**: Bearer Token (OAuth 2.0)
**Rate Limiting**: 1000 requests/minute per API key

## Authentication

### OAuth 2.0 Bearer Token

All API requests require a valid bearer token in the Authorization header:

```bash
Authorization: Bearer {access_token}
```

### Get Access Token

**Endpoint**: `POST /auth/token`

**Request**:
```json
{
  "grant_type": "client_credentials",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "scope": "credit-risk:read credit-risk:write portfolio:manage"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "credit-risk:read credit-risk:write portfolio:manage"
}
```

## Core Credit Risk APIs

### 1. Credit Risk Assessment

#### Score Individual Loan Application

**Endpoint**: `POST /credit-risk/score`

**Description**: Assess credit risk for a single loan application using advanced ensemble ML models.

**Request Headers**:
```
Content-Type: application/json
Authorization: Bearer {access_token}
X-Request-ID: unique-request-identifier
```

**Request Body**:
```json
{
  "application_id": "APP-2025-001234",
  "applicant": {
    "loan_amount": 250000,
    "annual_income": 85000,
    "credit_score": 720,
    "employment_length": 5.2,
    "debt_to_income": 28.5,
    "loan_term": 60,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "debt_consolidation",
    "delinq_2yrs": 0,
    "open_accounts": 12,
    "total_accounts": 28,
    "revolving_balance": 15420,
    "revolving_utilization": 45.2
  },
  "additional_data": {
    "employment_verified": true,
    "income_verified": true,
    "property_value": 420000,
    "down_payment": 60000
  }
}
```

**Response** (200 OK):
```json
{
  "request_id": "REQ-2025-08-001234",
  "application_id": "APP-2025-001234",
  "risk_assessment": {
    "overall_risk_score": 0.143,
    "risk_grade": "B+",
    "probability_of_default": 0.087,
    "confidence_interval": {
      "lower_bound": 0.064,
      "upper_bound": 0.112
    },
    "risk_category": "MEDIUM_LOW",
    "recommended_action": "APPROVE_WITH_CONDITIONS"
  },
  "model_predictions": {
    "xgboost": {
      "score": 0.139,
      "confidence": 0.924
    },
    "lightgbm": {
      "score": 0.147,
      "confidence": 0.918
    },
    "random_forest": {
      "score": 0.141,
      "confidence": 0.897
    },
    "neural_network": {
      "score": 0.145,
      "confidence": 0.903
    },
    "ensemble": {
      "score": 0.143,
      "confidence": 0.911
    }
  },
  "feature_importance": {
    "credit_score": 0.234,
    "debt_to_income": 0.187,
    "loan_to_income_ratio": 0.156,
    "revolving_utilization": 0.123,
    "employment_length": 0.089,
    "total_accounts": 0.067,
    "other_features": 0.144
  },
  "risk_factors": [
    {
      "factor": "High revolving utilization",
      "impact": "NEGATIVE",
      "severity": "MEDIUM",
      "value": 45.2,
      "benchmark": "<30%"
    },
    {
      "factor": "Strong credit score",
      "impact": "POSITIVE",
      "severity": "HIGH",
      "value": 720,
      "benchmark": ">680"
    }
  ],
  "pricing_recommendation": {
    "base_rate": 0.0675,
    "risk_adjustment": 0.0045,
    "recommended_rate": 0.0720,
    "margin_vs_benchmark": 0.0025
  },
  "compliance": {
    "fair_lending_review": "PASSED",
    "adverse_action_indicators": [],
    "model_explainability_score": 0.94
  },
  "timestamp": "2025-08-18T10:30:45Z",
  "processing_time_ms": 42
}
```

**Error Response** (400 Bad Request):
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: credit_score",
    "details": {
      "field": "applicant.credit_score",
      "requirement": "Integer between 300 and 850"
    }
  },
  "request_id": "REQ-2025-08-001234",
  "timestamp": "2025-08-18T10:30:45Z"
}
```

#### Batch Risk Scoring

**Endpoint**: `POST /credit-risk/batch-score`

**Description**: Process multiple loan applications simultaneously for high-volume origination.

**Request Body**:
```json
{
  "batch_id": "BATCH-2025-001",
  "applications": [
    {
      "application_id": "APP-2025-001235",
      "applicant": { /* same structure as individual scoring */ }
    },
    {
      "application_id": "APP-2025-001236",
      "applicant": { /* same structure as individual scoring */ }
    }
  ],
  "processing_options": {
    "priority": "HIGH",
    "callback_url": "https://your-system.com/webhook/credit-results",
    "include_explanations": true
  }
}
```

**Response** (202 Accepted):
```json
{
  "batch_id": "BATCH-2025-001",
  "status": "PROCESSING",
  "total_applications": 250,
  "estimated_completion": "2025-08-18T10:35:00Z",
  "progress_url": "/credit-risk/batch-score/BATCH-2025-001/status",
  "results_url": "/credit-risk/batch-score/BATCH-2025-001/results"
}
```

### 2. Portfolio Risk Management

#### Portfolio Risk Analysis

**Endpoint**: `GET /portfolio/risk-analysis`

**Description**: Comprehensive portfolio-level risk metrics and concentration analysis.

**Query Parameters**:
- `portfolio_id` (string): Portfolio identifier
- `as_of_date` (string): Analysis date (YYYY-MM-DD)
- `include_scenarios` (boolean): Include stress test scenarios

**Example Request**:
```bash
GET /portfolio/risk-analysis?portfolio_id=PORT-CONSUMER-2025&as_of_date=2025-08-18&include_scenarios=true
```

**Response**:
```json
{
  "portfolio_id": "PORT-CONSUMER-2025",
  "as_of_date": "2025-08-18",
  "portfolio_summary": {
    "total_exposure": 2450000000,
    "number_of_loans": 12450,
    "average_loan_size": 196787,
    "weighted_average_score": 685,
    "portfolio_risk_grade": "B+"
  },
  "risk_metrics": {
    "expected_loss": {
      "amount": 22050000,
      "percentage": 0.009,
      "confidence_level": 0.95
    },
    "value_at_risk": {
      "var_95": 45200000,
      "var_99": 67800000,
      "var_99_9": 89400000
    },
    "concentration_risk": {
      "herfindahl_index": 0.0234,
      "top_10_exposure": 0.187,
      "single_name_limit_breaches": 0
    }
  },
  "sector_exposure": {
    "technology": 0.285,
    "healthcare": 0.187,
    "manufacturing": 0.142,
    "retail": 0.098,
    "real_estate": 0.076,
    "other": 0.212
  },
  "geographic_distribution": {
    "california": 0.245,
    "texas": 0.156,
    "new_york": 0.134,
    "florida": 0.098,
    "other": 0.367
  },
  "stress_scenarios": [
    {
      "scenario_name": "Economic Downturn",
      "gdp_shock": -0.03,
      "unemployment_shock": 0.025,
      "expected_loss_impact": 0.0156,
      "portfolio_loss": 38220000
    },
    {
      "scenario_name": "Interest Rate Shock",
      "rate_shock": 0.02,
      "expected_loss_impact": 0.0089,
      "portfolio_loss": 21805000
    }
  ],
  "basel_metrics": {
    "risk_weighted_assets": 1960000000,
    "capital_requirement": 156800000,
    "capital_adequacy_ratio": 0.142,
    "tier_1_ratio": 0.098
  }
}
```

#### Portfolio Optimization

**Endpoint**: `POST /portfolio/optimize`

**Description**: Optimize portfolio allocation for risk-adjusted returns.

**Request Body**:
```json
{
  "portfolio_id": "PORT-CONSUMER-2025",
  "optimization_objective": "MAXIMIZE_SHARPE_RATIO",
  "constraints": {
    "max_sector_concentration": 0.25,
    "min_diversification_score": 0.75,
    "target_duration": 36,
    "max_single_exposure": 0.05
  },
  "risk_tolerance": "MODERATE",
  "time_horizon_months": 60
}
```

**Response**:
```json
{
  "optimization_id": "OPT-2025-001",
  "recommended_allocation": {
    "consumer_loans": 0.45,
    "auto_loans": 0.25,
    "mortgage_loans": 0.20,
    "commercial_loans": 0.10
  },
  "expected_metrics": {
    "expected_return": 0.087,
    "portfolio_volatility": 0.124,
    "sharpe_ratio": 0.642,
    "max_drawdown": 0.078
  },
  "rebalancing_trades": [
    {
      "action": "BUY",
      "asset_class": "consumer_loans",
      "amount": 25000000,
      "reason": "Underweight vs optimal"
    },
    {
      "action": "SELL", 
      "asset_class": "commercial_loans",
      "amount": 15000000,
      "reason": "Overweight vs optimal"
    }
  ]
}
```

### 3. Regulatory Compliance APIs

#### Basel III Capital Calculation

**Endpoint**: `POST /compliance/basel-capital`

**Description**: Calculate Basel III capital requirements and ratios.

**Request Body**:
```json
{
  "bank_id": "BANK-001",
  "calculation_date": "2025-08-18",
  "portfolios": [
    {
      "portfolio_id": "PORT-CONSUMER-2025",
      "exposure_amount": 2450000000,
      "risk_weight": 0.75
    },
    {
      "portfolio_id": "PORT-COMMERCIAL-2025", 
      "exposure_amount": 1850000000,
      "risk_weight": 1.0
    }
  ],
  "capital_components": {
    "common_equity_tier1": 450000000,
    "additional_tier1": 125000000,
    "tier2": 75000000
  }
}
```

**Response**:
```json
{
  "calculation_id": "BASEL-2025-08-001",
  "bank_id": "BANK-001",
  "calculation_date": "2025-08-18",
  "risk_weighted_assets": {
    "credit_risk": 4137500000,
    "operational_risk": 456000000,
    "market_risk": 123000000,
    "total_rwa": 4716500000
  },
  "capital_ratios": {
    "cet1_ratio": 0.0954,
    "tier1_ratio": 0.1219,
    "total_capital_ratio": 0.1378
  },
  "minimum_requirements": {
    "cet1_minimum": 0.045,
    "tier1_minimum": 0.06,
    "total_capital_minimum": 0.08,
    "capital_conservation_buffer": 0.025,
    "countercyclical_buffer": 0.01
  },
  "compliance_status": {
    "cet1_compliant": true,
    "tier1_compliant": true,
    "total_capital_compliant": true,
    "buffer_compliant": true,
    "overall_status": "COMPLIANT"
  },
  "recommendations": [
    "Capital ratios exceed minimum requirements",
    "Consider optimizing capital allocation for higher returns"
  ]
}
```

#### Stress Testing

**Endpoint**: `POST /compliance/stress-test`

**Description**: Perform regulatory stress testing scenarios (CCAR/DFAST).

**Request Body**:
```json
{
  "test_id": "STRESS-2025-SEVERELY-ADVERSE",
  "scenario_type": "SEVERELY_ADVERSE",
  "portfolios": ["PORT-CONSUMER-2025", "PORT-COMMERCIAL-2025"],
  "stress_parameters": {
    "gdp_growth": [-0.085, -0.045, 0.012, 0.024],
    "unemployment_rate": [0.102, 0.087, 0.065, 0.058],
    "house_price_index": [-0.285, -0.145, 0.023, 0.045],
    "equity_prices": [-0.455, -0.225, 0.087, 0.124]
  },
  "time_horizon_quarters": 9
}
```

**Response**:
```json
{
  "test_id": "STRESS-2025-SEVERELY-ADVERSE", 
  "scenario_type": "SEVERELY_ADVERSE",
  "stress_results": {
    "pre_provision_net_revenue": [
      {"quarter": 1, "amount": 45000000},
      {"quarter": 2, "amount": 42000000},
      {"quarter": 3, "amount": 38000000}
    ],
    "provision_for_credit_losses": [
      {"quarter": 1, "amount": 78000000},
      {"quarter": 2, "amount": 85000000}, 
      {"quarter": 3, "amount": 67000000}
    ],
    "net_income_before_tax": [
      {"quarter": 1, "amount": -33000000},
      {"quarter": 2, "amount": -43000000},
      {"quarter": 3, "amount": -29000000}
    ]
  },
  "capital_projections": {
    "starting_cet1_ratio": 0.0954,
    "minimum_cet1_ratio": 0.0687,
    "ending_cet1_ratio": 0.0745,
    "capital_action_required": false
  },
  "portfolio_losses": {
    "PORT-CONSUMER-2025": {
      "total_losses": 234000000,
      "loss_rate": 0.0955
    },
    "PORT-COMMERCIAL-2025": {
      "total_losses": 167000000,
      "loss_rate": 0.0903
    }
  }
}
```

### 4. Model Management APIs

#### Model Performance Monitoring

**Endpoint**: `GET /models/performance`

**Description**: Monitor ML model performance and drift metrics.

**Query Parameters**:
- `model_name` (string): Specific model identifier
- `start_date` (string): Performance period start (YYYY-MM-DD)
- `end_date` (string): Performance period end (YYYY-MM-DD)

**Response**:
```json
{
  "model_name": "ensemble_credit_risk_v2.0",
  "performance_period": {
    "start_date": "2025-07-01",
    "end_date": "2025-08-18"
  },
  "accuracy_metrics": {
    "auc_roc": 0.947,
    "precision": 0.912,
    "recall": 0.938,
    "f1_score": 0.925,
    "gini_coefficient": 0.894
  },
  "drift_analysis": {
    "feature_drift_score": 0.034,
    "prediction_drift_score": 0.021,
    "drift_status": "STABLE",
    "last_retrain_date": "2025-07-15"
  },
  "performance_trend": [
    {"date": "2025-07-01", "auc": 0.945},
    {"date": "2025-07-15", "auc": 0.948},
    {"date": "2025-08-01", "auc": 0.947},
    {"date": "2025-08-18", "auc": 0.947}
  ],
  "recommendations": [
    "Model performance remains stable",
    "Next scheduled retrain: 2025-09-15"
  ]
}
```

#### Model Explainability

**Endpoint**: `POST /models/explain`

**Description**: Generate SHAP explanations for specific predictions.

**Request Body**:
```json
{
  "model_name": "ensemble_credit_risk_v2.0",
  "application_id": "APP-2025-001234",
  "explanation_type": "SHAP_VALUES",
  "include_global_importance": true
}
```

**Response**:
```json
{
  "application_id": "APP-2025-001234",
  "prediction": 0.143,
  "baseline_prediction": 0.087,
  "shap_values": [
    {
      "feature": "credit_score",
      "value": 720,
      "shap_value": -0.045,
      "contribution": "POSITIVE"
    },
    {
      "feature": "debt_to_income",
      "value": 28.5,
      "shap_value": 0.021,
      "contribution": "NEGATIVE"
    },
    {
      "feature": "revolving_utilization", 
      "value": 45.2,
      "shap_value": 0.034,
      "contribution": "NEGATIVE"
    }
  ],
  "feature_interactions": [
    {
      "features": ["credit_score", "debt_to_income"],
      "interaction_value": -0.012
    }
  ],
  "explanation_confidence": 0.94
}
```

## Data Models

### Loan Application Schema

```json
{
  "type": "object",
  "properties": {
    "application_id": {"type": "string"},
    "loan_amount": {"type": "number", "minimum": 1000, "maximum": 10000000},
    "annual_income": {"type": "number", "minimum": 10000},
    "credit_score": {"type": "integer", "minimum": 300, "maximum": 850},
    "employment_length": {"type": "number", "minimum": 0},
    "debt_to_income": {"type": "number", "minimum": 0, "maximum": 100},
    "loan_term": {"type": "integer", "enum": [36, 60, 84]},
    "home_ownership": {"type": "string", "enum": ["RENT", "OWN", "MORTGAGE"]},
    "loan_purpose": {"type": "string", "enum": ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"]},
    "delinq_2yrs": {"type": "integer", "minimum": 0},
    "open_accounts": {"type": "integer", "minimum": 0},
    "total_accounts": {"type": "integer", "minimum": 0},
    "revolving_balance": {"type": "number", "minimum": 0},
    "revolving_utilization": {"type": "number", "minimum": 0, "maximum": 100}
  },
  "required": ["loan_amount", "annual_income", "credit_score", "employment_length", "debt_to_income"]
}
```

## Error Handling

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 201 | Created | Resource created successfully |
| 202 | Accepted | Request accepted for processing |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "specific_field_name",
      "requirement": "validation requirement"
    }
  },
  "request_id": "REQ-2025-08-001234",
  "timestamp": "2025-08-18T10:30:45Z"
}
```

## Rate Limiting

- **Default Rate Limit**: 1000 requests/minute per API key
- **Burst Limit**: 100 requests/10 seconds
- **Headers**:
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Window reset time (Unix timestamp)

## Webhooks

### Risk Score Notification

**URL**: Your configured webhook endpoint
**Method**: POST
**Headers**: 
- `X-Webhook-Signature`: HMAC-SHA256 signature
- `X-Webhook-Event`: Event type

**Payload Example**:
```json
{
  "event_type": "risk_score_completed",
  "timestamp": "2025-08-18T10:30:45Z",
  "data": {
    "application_id": "APP-2025-001234",
    "risk_score": 0.143,
    "risk_grade": "B+",
    "processing_time_ms": 42
  }
}
```

## SDK Examples

### Python SDK

```python
from creditrisk_api import CreditRiskClient

# Initialize client
client = CreditRiskClient(
    api_key="your_api_key",
    base_url="https://api.creditrisk.enterprise.com/v2"
)

# Score individual application
application = {
    "loan_amount": 250000,
    "annual_income": 85000,
    "credit_score": 720,
    # ... other fields
}

result = client.score_application(application)
print(f"Risk Score: {result.risk_assessment.overall_risk_score}")
```

### JavaScript SDK

```javascript
const CreditRiskAPI = require('@enterprise/creditrisk-api');

const client = new CreditRiskAPI({
  apiKey: 'your_api_key',
  baseURL: 'https://api.creditrisk.enterprise.com/v2'
});

// Score application
const application = {
  loan_amount: 250000,
  annual_income: 85000,
  credit_score: 720
  // ... other fields
};

const result = await client.scoreApplication(application);
console.log(`Risk Score: ${result.risk_assessment.overall_risk_score}`);
```

## Support & Resources

- **API Status Page**: https://status.creditrisk.enterprise.com
- **Developer Portal**: https://developers.creditrisk.enterprise.com
- **Support Email**: api-support@enterprise.com
- **Documentation**: https://docs.creditrisk.enterprise.com
- **Rate Limit Increase Requests**: contact your account manager

---

This API documentation provides comprehensive guidance for integrating with the Enterprise Credit Risk Intelligence Platform. All endpoints are designed for high-volume production use with enterprise-grade security, performance, and compliance capabilities.