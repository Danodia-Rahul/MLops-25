import pickle
import pandas as pd
import argparse


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

def read_data(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def predict(dv, model, df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Prediction mean: {y_pred.mean()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict taxi trip duration for a given year and month.')
    parser.add_argument('--year', type=int, required=True, help='Year of the dataset')
    parser.add_argument('--month', type=int, required=True, help='Month of the dataset')

    args = parser.parse_args()

    dv, model = load_model()
    df = read_data(args.year, args.month)
    predict(dv, model, df)
