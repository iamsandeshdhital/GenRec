.PHONY: install test run demo clean lint

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python -m pytest tests/ -v

# Run the web application
run:
	python app.py

# Run the demo
demo:
	python demo.py

# Clean up
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf data/
	rm -rf outputs/
	rm -rf cache/

# Lint code
lint:
	flake8 genrec/ --max-line-length=120

# Format code
format:
	black genrec/ --line-length=120

# Run all checks
check: lint test
