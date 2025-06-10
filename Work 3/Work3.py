import mlflow
import mlflow.sklearn
import sklearn
import pandas as pd
import prefect
import pickle

from prefect import flow, task
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer


@task
def setup_mlflow():
    tracking_uri = 'http://127.0.0.1:5000'
    experiment_name = 'NYC Taxi March 23'
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@task(log_prints=True)
def info():
    print('Tool Chosen: PREFECT')
    print(f'Prefect Version: {prefect.__version__}')


@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(month, year):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f'Number of records loaded: {df.shape[0]}')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f'Size of data after preprocessing: {df.shape[0]}')

    return df


@task(log_prints=True)
def train_model(dataframe):
    features = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()
    X_train = dataframe[features]
    train_dict = X_train.to_dict(orient='records')
    feature_matrix = dv.fit_transform(train_dict)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(feature_matrix, dataframe.duration)

        mlflow.sklearn.log_model(model, artifact_path="models")

        with open('dict_vectorizer.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('dict_vectorizer.b')

        print(f"Model intercept: {model.intercept_}")

    return dv, model


@flow
def run():
    setup_mlflow()
    info()
    df = read_dataframe(3, 2023)
    train_model(df)


if __name__ == '__main__':
    run()
