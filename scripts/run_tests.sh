#!/bin/bash
set -e

# Ensure scripts directory exists
mkdir -p scripts

# Default arguments
COV_REPORT="--cov-report=term"
TEST_TYPE="all"
EXTRA_ARGS=""

# Display help message
function show_help {
    echo "Usage: ./scripts/run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo "  -u, --unit               Run only unit tests"
    echo "  -i, --integration        Run only integration tests"
    echo "  -a, --api                Run only API tests"
    echo "  -c, --cov                Generate HTML coverage report"
    echo "  -v, --verbose            Increase verbosity"
    echo "  -x, --exitfirst          Exit on first failure"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_tests.sh --unit          # Run unit tests"
    echo "  ./scripts/run_tests.sh --cov           # Run all tests with coverage report"
    echo "  ./scripts/run_tests.sh -i -v           # Run integration tests with verbose output"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -u|--unit)
            TEST_TYPE="unit"
            shift
            ;;
        -i|--integration)
            TEST_TYPE="integration"
            shift
            ;;
        -a|--api)
            TEST_TYPE="api"
            shift
            ;;
        -c|--cov)
            COV_REPORT="--cov-report=term --cov-report=html"
            shift
            ;;
        -v|--verbose)
            EXTRA_ARGS="$EXTRA_ARGS -v"
            shift
            ;;
        -x|--exitfirst)
            EXTRA_ARGS="$EXTRA_ARGS -x"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            show_help
            ;;
    esac
done

# Set up test paths based on test type
case $TEST_TYPE in
    unit)
        echo "Running unit tests..."
        PYTEST_ARGS="tests/test_memory.py tests/test_embeddings.py tests/test_vector_db.py"
        COV_MODULE="--cov=./memory"
        ;;
    integration)
        echo "Running integration tests..."
        PYTEST_ARGS="tests/test_memory_pipeline.py tests/test_semantic_search.py"
        COV_MODULE="--cov=./"
        ;;
    api)
        echo "Running API tests..."
        PYTEST_ARGS="tests/test_api_*.py"
        COV_MODULE="--cov=./api"
        ;;
    all)
        echo "Running all tests..."
        PYTEST_ARGS="tests/"
        COV_MODULE="--cov=./"
        ;;
esac

# Make sure our virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -d "venv" ]]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found. Consider creating one with:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
    fi
fi

# Run the tests
python -m pytest $PYTEST_ARGS $COV_MODULE $COV_REPORT $EXTRA_ARGS

# If we're generating HTML coverage, print the report location
if [[ "$COV_REPORT" =~ "html" ]]; then
    echo "Coverage report generated at htmlcov/index.html"
fi 