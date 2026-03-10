.PHONY: install run test lint setup

install:
	pip install -r requirements.txt

setup: install
	python -c "from core.embeddings import get_embeddings; get_embeddings()"

run:
	streamlit run app.py

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .
