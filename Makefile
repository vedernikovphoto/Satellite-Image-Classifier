install:
	pip install -r requirements.txt

download_dataset:
	curl -L -o dataset.zip "https://www.kaggle.com/api/v1/datasets/download/nikitarom/planets-dataset?datasetVersionNumber=3"
	unzip dataset.zip -d data
	rm dataset.zip

train:
	set PYTHONPATH=%CD% && python src/train.py config/config.yaml

inference:
	set PYTHONPATH=%CD% && python src/inference.py
