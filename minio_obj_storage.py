import os
from minio import Minio
import io
import torch
import numpy as np
import json
import pickle
from io import BytesIO

def download_blob_to_stream(minio_client: Minio, bucket_name: str, blob: str):
    response = minio_client.get_object(bucket_name, blob)
    stream = io.BytesIO(response.read())
    response.close()
    response.release_conn()
    stream.seek(0)
    return stream

def get_connection_details():
    with open("credentials.json", 'r') as f:
        data = json.load(f)
    
    return data['endpoint'], data['accessKey'], data['secretKey']

def upload_blob_file(minio_client: Minio, bucket_name: str, path, blob_name, file_name):
    with open(os.path.join(path, file_name), 'rb') as data:
        minio_client.put_object(bucket_name, blob_name, data, os.path.getsize(os.path.join(path, file_name)))

def get_model_from_minio_blob(bucket_name, object_name):
    endpoint, access_key, secret_key = get_connection_details()

    # Create the Minio client object
    minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
    print(f'Getting model {object_name} from {bucket_name}')
    buffer = download_blob_to_stream(minio_client, bucket_name, object_name)
    model_params = torch.load(buffer, map_location='cpu')
    return model_params

def save_to_cloud(model_state_dict, bucket_name, object_name, overwrite=False):
    # Serialize the model state_dict to bytes
    buffer = BytesIO()
    torch.save(model_state_dict, buffer)
    buffer.seek(0)

    endpoint, access_key, secret_key = get_connection_details()

    # Create the Minio client object
    minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # Upload the object
    minio_client.put_object(bucket_name, object_name, buffer, buffer.getbuffer().nbytes)

def upload_numpy_as_blob(bucket_name, dir_name, file_name, numpy_array, overwrite=False):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=numpy_array)
    buffer.seek(0)

    endpoint, access_key, secret_key = get_connection_details()

    # Create the Minio client object
    minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # Upload the object
    minio_client.put_object(bucket_name, f"{dir_name}/{file_name}", buffer, buffer.getbuffer().nbytes)

def get_numpy_from_cloud(bucket_name, dir_name, file_name):
    endpoint, access_key, secret_key = get_connection_details()

    # Create the Minio client object
    minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    buffer = download_blob_to_stream(minio_client, bucket_name, f"{dir_name}/{file_name}")
    buffer.seek(0)
    loaded_array = np.load(buffer, allow_pickle=True)

    if 'data' in loaded_array:
        return loaded_array['data']

    return loaded_array
