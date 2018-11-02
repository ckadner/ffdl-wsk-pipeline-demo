#!/usr/bin/env python

import boto3
import json
import os
import sys
import yaml
import zipfile

training_files_folder = "training_files"

script_file   = os.path.join(training_files_folder, "gender_classification.py")
archive_file  = os.path.join(training_files_folder, "model.zip")
manifest_file = os.path.join(training_files_folder, "manifest.yml")
dataset_file  = os.path.join(training_files_folder, "UTKFace.tar.gz")


def parse_manifest():
    with open(manifest_file, 'r') as stream:
        try:
            manifest_data = yaml.load(stream)
            # print(json.dumps(manifest_data, sort_keys=True, indent=4))
            return manifest_data
        except yaml.YAMLError as e:
            print(e)
    return dict()


def create_cos_connection(manifest):
    connection = manifest["data_stores"][0]["connection"]
    cos = boto3.resource("s3",
                         endpoint_url=connection["auth_url"],
                         aws_access_key_id = connection["user_name"],
                         aws_secret_access_key = connection["password"])
    return cos


def create_bucket(cos, bucket_name):
    bucket = cos.Bucket(bucket_name)
    if bucket.creation_date:
        print('Bucket already exists "{}"'.format(bucket_name))
    else:
        print('Creating bucket "{}" ...'.format(bucket_name))
        try:
            bucket = cos.create_bucket(Bucket=bucket_name)
        except boto3.exceptions.botocore.client.ClientError as e:
            print('Error: {}.'.format(e.response['Error']['Message']))
    return bucket


def download_training_data(dataset_file):
    pass


def create_model_zip():
    zipfile.ZipFile(archive_file, mode='w').write(script_file, os.path.basename(script_file))


def upload_files_to_bucket(bucket, files=[manifest_file, archive_file, dataset_file]):
    print('Uploading files to bucket "{}":'.format(bucket.name))
    for filename in files:
        print('- {}'.format(filename))
        bucket.upload_file(filename, os.path.basename(filename))


def print_bucket_contents(bucket):
    print(bucket.name)
    for obj in bucket.objects.all():
        print("  File: {}, {:4.2f}kB".format(obj.key, obj.size/1024))


def generate_parameters_file(manifest):
    data_store = manifest["data_stores"][0]
    parameters = {
        "aws_endpoint_url" : data_store["connection"]["auth_url"],
        "aws_access_key_id" : data_store["connection"]["user_name"],
        "aws_secret_access_key" : data_store["connection"]["password"],
        "training_data_bucket" : data_store["training_data"]["container"],
        "training_results_bucket": data_store["training_results"]["container"]
    }
    with open('parameters.json', 'w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4, )


def main(args):
    manifest_data = parse_manifest()
    cos = create_cos_connection(manifest_data)
    data_bucket = create_bucket(cos, bucket_name=manifest_data["data_stores"][0]["training_data"]["container"])
    result_bucket = create_bucket(cos, bucket_name=manifest_data["data_stores"][0]["training_results"]["container"])
    download_training_data(dataset_file)
    create_model_zip()
    upload_files_to_bucket(data_bucket, files=[manifest_file, archive_file, dataset_file])
    print_bucket_contents(data_bucket)
    generate_parameters_file(manifest_data)


if __name__ == "__main__":
    main(sys.argv)
