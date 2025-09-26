
.PHONY: install test lint build docker-build docker-run notebook

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	flake8 src --max-line-length=100

build:
	python main.py build

docker-build:
	docker build -t multimodal-rag .

docker-run:
	docker run -p 8888:8888 multimodal-rag

notebook:
	jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
