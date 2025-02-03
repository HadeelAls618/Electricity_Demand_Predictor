.PHONE: init features training inference frontend monitoring

# downloads Poetry and installs all dependencies from pyproject.toml
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

# generates new batch of features and stores them in the feature store
features:
	poetry run python src/piplines/feature_pipline.py

# trains a new model and stores it in the model registry
training:
	poetry run python src/piplines/training_pipline.py

# generates predictions and stores them in the feature store
inference:
	poetry run python src/piplines/inference_pipline.py

# backfills the feature store with historical data
backfill:
	poetry run python src/component/backfill_feature_group.py

# starts the Streamlit app
frontend-app:
	poetry run streamlit run src/frontend.py

monitoring-app:
	poetry run streamlit run src/monitoring_frontend.py

lint:
	@echo "Fixing linting issues..."
	poetry run ruff check --fix .

format:
	echo "Formatting Python code..."
	poetry run ruff format .