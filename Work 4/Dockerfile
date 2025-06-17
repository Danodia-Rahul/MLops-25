FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY Model.py .

RUN pip install --no-cache-dir pandas scikit-learn==1.6.1 pyarrow fastparquet

ENTRYPOINT ["python", "Model.py"]
