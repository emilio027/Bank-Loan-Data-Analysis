# Makefile for Enterprise Credit Risk Intelligence Platform
# Development automation and CI/CD support

.PHONY: help install install-dev test test-unit test-integration test-api test-performance
.PHONY: test-security test-all lint format format-check type-check security-scan
.PHONY: coverage coverage-report clean build docker-build docker-run docker-stop
.PHONY: docs serve-docs deploy-staging deploy-production backup restore
.PHONY: setup-pre-commit update-deps check-deps benchmark stress-test

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER_IMAGE := enterprise-credit-risk-platform
DOCKER_TAG := latest
PORT := 8000

# Colors for output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# Default target
help: ## Show this help message
	@echo "$(BOLD)Enterprise Credit Risk Intelligence Platform$(RESET)"
	@echo "$(CYAN)Development and Deployment Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'

# ============================================================================
# Installation and Setup
# ============================================================================

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Production dependencies installed$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,test,docs]"
	@echo "$(GREEN)✅ Development dependencies installed$(RESET)"

setup-pre-commit: ## Setup pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit hooks installed$(RESET)"

update-deps: ## Update all dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) freeze > requirements.lock
	@echo "$(GREEN)✅ Dependencies updated$(RESET)"

check-deps: ## Check for dependency vulnerabilities
	@echo "$(GREEN)Checking dependencies for vulnerabilities...$(RESET)"
	safety check
	@echo "$(GREEN)✅ Dependency check completed$(RESET)"

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	$(PYTEST) Files/tests/ -v --cov=Files/src --cov-report=term-missing
	@echo "$(GREEN)✅ All tests completed$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	$(PYTEST) Files/tests/test_main.py Files/tests/test_data.py -v --cov=Files/src
	@echo "$(GREEN)✅ Unit tests completed$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	$(PYTEST) Files/tests/test_integration.py -v --cov=Files/src
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

test-api: ## Run API tests only
	@echo "$(GREEN)Running API tests...$(RESET)"
	$(PYTEST) Files/tests/test_api.py -v --cov=Files/src
	@echo "$(GREEN)✅ API tests completed$(RESET)"

test-performance: ## Run performance tests (excluding slow tests)
	@echo "$(GREEN)Running performance tests...$(RESET)"
	$(PYTEST) Files/tests/test_performance.py -m "not slow" -v --benchmark-only
	@echo "$(GREEN)✅ Performance tests completed$(RESET)"

test-security: ## Run security-focused tests
	@echo "$(GREEN)Running security tests...$(RESET)"
	$(PYTEST) Files/tests/ -m "security" -v
	@echo "$(GREEN)✅ Security tests completed$(RESET)"

test-all: ## Run comprehensive test suite
	@echo "$(GREEN)Running comprehensive test suite...$(RESET)"
	$(PYTEST) Files/tests/ -v \
		--cov=Files/src \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--junitxml=test-results.xml \
		--durations=10
	@echo "$(GREEN)✅ Comprehensive test suite completed$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	$(PYTEST) Files/tests/test_performance.py \
		--benchmark-only \
		--benchmark-json=benchmark-results.json \
		--benchmark-sort=mean
	@echo "$(GREEN)✅ Benchmarks completed$(RESET)"

stress-test: ## Run stress tests (may take a long time)
	@echo "$(YELLOW)Running stress tests (this may take a while)...$(RESET)"
	$(PYTEST) Files/tests/test_performance.py -m "slow" -v \
		--timeout=1800 \
		--benchmark-json=stress-test-results.json
	@echo "$(GREEN)✅ Stress tests completed$(RESET)"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run all linting tools
	@echo "$(GREEN)Running linting tools...$(RESET)"
	flake8 Files/src Files/tests --max-line-length=100 --extend-ignore=E203,W503
	pylint Files/src --max-line-length=100
	@echo "$(GREEN)✅ Linting completed$(RESET)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	black Files/src Files/tests
	isort Files/src Files/tests
	@echo "$(GREEN)✅ Code formatting completed$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	black --check --diff Files/src Files/tests
	isort --check-only --diff Files/src Files/tests
	@echo "$(GREEN)✅ Code formatting check completed$(RESET)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(RESET)"
	mypy Files/src --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)✅ Type checking completed$(RESET)"

security-scan: ## Run security scans
	@echo "$(GREEN)Running security scans...$(RESET)"
	bandit -r Files/src
	safety check
	@echo "$(GREEN)✅ Security scans completed$(RESET)"

# ============================================================================
# Coverage
# ============================================================================

coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(RESET)"
	$(PYTEST) Files/tests/ --cov=Files/src --cov-report=html --cov-report=xml
	@echo "$(GREEN)✅ Coverage report generated$(RESET)"

coverage-report: ## Open coverage report in browser
	@echo "$(GREEN)Opening coverage report...$(RESET)"
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "$(YELLOW)Please open htmlcov/index.html manually$(RESET)"; \
	fi

# ============================================================================
# Development Server
# ============================================================================

serve: ## Start development server
	@echo "$(GREEN)Starting development server on port $(PORT)...$(RESET)"
	$(PYTHON) Files/src/main.py

serve-debug: ## Start development server in debug mode
	@echo "$(GREEN)Starting development server in debug mode...$(RESET)"
	DEBUG=true $(PYTHON) Files/src/main.py

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(RESET)"

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(RESET)"
	docker run -d \
		--name $(DOCKER_IMAGE) \
		-p $(PORT):$(PORT) \
		-e APP_PORT=$(PORT) \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Container running on port $(PORT)$(RESET)"

docker-stop: ## Stop Docker container
	@echo "$(GREEN)Stopping Docker container...$(RESET)"
	docker stop $(DOCKER_IMAGE) || true
	docker rm $(DOCKER_IMAGE) || true
	@echo "$(GREEN)✅ Container stopped$(RESET)"

docker-logs: ## View Docker container logs
	@echo "$(GREEN)Viewing container logs...$(RESET)"
	docker logs -f $(DOCKER_IMAGE)

docker-shell: ## Access Docker container shell
	@echo "$(GREEN)Accessing container shell...$(RESET)"
	docker exec -it $(DOCKER_IMAGE) /bin/bash

# ============================================================================
# Documentation
# ============================================================================

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	sphinx-build -W -b html Files/docs docs/_build
	@echo "$(GREEN)✅ Documentation built$(RESET)"

serve-docs: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(RESET)"
	cd docs/_build && $(PYTHON) -m http.server 8080

# ============================================================================
# Database Operations
# ============================================================================

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(RESET)"
	$(PYTHON) -c "from Files.src.data_manager import DataManager; DataManager().migrate()"
	@echo "$(GREEN)✅ Database migrations completed$(RESET)"

db-seed: ## Seed database with test data
	@echo "$(GREEN)Seeding database with test data...$(RESET)"
	$(PYTHON) -c "from Files.src.data_manager import DataManager; DataManager().seed_test_data()"
	@echo "$(GREEN)✅ Database seeded$(RESET)"

backup: ## Create database backup
	@echo "$(GREEN)Creating database backup...$(RESET)"
	mkdir -p backups
	$(PYTHON) -c "from Files.src.data_manager import DataManager; DataManager().backup('backups/backup_$(shell date +%Y%m%d_%H%M%S).sql')"
	@echo "$(GREEN)✅ Database backup created$(RESET)"

restore: ## Restore database from backup (requires BACKUP_FILE=path)
	@echo "$(GREEN)Restoring database from backup...$(RESET)"
	$(PYTHON) -c "from Files.src.data_manager import DataManager; DataManager().restore('$(BACKUP_FILE)')"
	@echo "$(GREEN)✅ Database restored$(RESET)"

# ============================================================================
# Model Operations
# ============================================================================

train-models: ## Train ML models
	@echo "$(GREEN)Training ML models...$(RESET)"
	$(PYTHON) -c "from Files.src.ml_models import MLModelManager; MLModelManager().train_all_models()"
	@echo "$(GREEN)✅ Model training completed$(RESET)"

validate-models: ## Validate ML models
	@echo "$(GREEN)Validating ML models...$(RESET)"
	$(PYTHON) -c "from Files.src.ml_models import MLModelManager; MLModelManager().validate_all_models()"
	@echo "$(GREEN)✅ Model validation completed$(RESET)"

# ============================================================================
# Deployment
# ============================================================================

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging environment...$(RESET)"
	@echo "$(YELLOW)Note: Configure staging deployment commands$(RESET)"
	# Add staging deployment commands here
	@echo "$(GREEN)✅ Staging deployment completed$(RESET)"

deploy-production: ## Deploy to production environment
	@echo "$(GREEN)Deploying to production environment...$(RESET)"
	@echo "$(YELLOW)Note: Configure production deployment commands$(RESET)"
	# Add production deployment commands here
	@echo "$(GREEN)✅ Production deployment completed$(RESET)"

# ============================================================================
# Monitoring and Health
# ============================================================================

health-check: ## Perform health check
	@echo "$(GREEN)Performing health check...$(RESET)"
	curl -f http://localhost:$(PORT)/health || echo "$(RED)Health check failed$(RESET)"

monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring dashboard...$(RESET)"
	$(PYTHON) -c "from Files.src.monitoring import start_monitoring; start_monitoring()"

# ============================================================================
# Utilities
# ============================================================================

clean: ## Clean up temporary files and caches
	@echo "$(GREEN)Cleaning up temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf test-results.xml
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf build
	rm -rf dist
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

build: ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ Distribution packages built$(RESET)"

install-package: ## Install package in development mode
	@echo "$(GREEN)Installing package in development mode...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ Package installed$(RESET)"

version: ## Show version information
	@echo "$(BOLD)Enterprise Credit Risk Intelligence Platform$(RESET)"
	@echo "Version: $(shell python -c "import sys; sys.path.insert(0, 'Files/src'); from main import __version__; print(__version__)" 2>/dev/null || echo "Unknown")"
	@echo "Python: $(shell python --version)"
	@echo "Platform: $(shell python -c "import platform; print(platform.platform())")"

# ============================================================================
# CI/CD Support
# ============================================================================

ci-install: ## Install dependencies for CI
	@echo "$(GREEN)Installing CI dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy bandit safety
	@echo "$(GREEN)✅ CI dependencies installed$(RESET)"

ci-test: ## Run tests for CI environment
	@echo "$(GREEN)Running CI test suite...$(RESET)"
	$(PYTEST) Files/tests/ \
		--cov=Files/src \
		--cov-report=xml \
		--junitxml=test-results.xml \
		--timeout=300 \
		-v
	@echo "$(GREEN)✅ CI tests completed$(RESET)"

ci-quality: ## Run quality checks for CI
	@echo "$(GREEN)Running CI quality checks...$(RESET)"
	black --check Files/src Files/tests
	flake8 Files/src Files/tests --max-line-length=100
	mypy Files/src --ignore-missing-imports
	bandit -r Files/src
	safety check
	@echo "$(GREEN)✅ CI quality checks completed$(RESET)"

# ============================================================================
# Help and Information
# ============================================================================

env-info: ## Show environment information
	@echo "$(BOLD)Environment Information$(RESET)"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repository')"

dependencies: ## Show current dependencies
	@echo "$(BOLD)Current Dependencies$(RESET)"
	$(PIP) list

# Quick development setup
quick-setup: install-dev setup-pre-commit ## Quick setup for development
	@echo "$(GREEN)✅ Quick development setup completed$(RESET)"
	@echo "$(CYAN)Next steps:$(RESET)"
	@echo "  1. Run 'make test' to verify everything works"
	@echo "  2. Run 'make serve' to start the development server"
	@echo "  3. Run 'make help' to see all available commands"