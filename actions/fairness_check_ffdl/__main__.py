# OpenWhisk action to perform a model fairness check with AIF360 on FfDL

import json, requests


def run_safe(args):
    try:
        import boto3
        import traceback
        import zipfile
        from ruamel.yaml import YAML

        yaml = YAML()

        script_file = "fairness_check.py"
        archive_file = "model.zip"
        manifest_file = "manifest.yml"

        def create_model_zip():
            zipfile.ZipFile(archive_file, mode='w').write(script_file)

        def create_cos_connection(params):
            cos = boto3.resource("s3",
                                 endpoint_url=params["aws_endpoint_url"],
                                 aws_access_key_id=params["aws_access_key_id"],
                                 aws_secret_access_key=params["aws_secret_access_key"])
            return cos

        def get_or_create_bucket(cos, bucket_name):
            bucket = cos.Bucket(bucket_name)
            if not bucket.creation_date:
                bucket = cos.create_bucket(Bucket=bucket_name)
            return bucket

        def get_data_bucket_name(params):
            return params["fairnesscheck_data_bucket"]

        def get_result_bucket_name(params):
            return params["fairnesscheck_results_bucket"]

        def copy_file(source_bucket, source_file, target_bucket, target_file):
            if source_bucket != target_bucket or source_file != target_file:
                target_bucket.copy({"Bucket": source_bucket.name, "Key": source_file}, target_file)

        def copy_training_result_files(cos, params):
            model_id = params["model_id"]
            tdb = get_or_create_bucket(cos, params["training_data_bucket"])
            trb = get_or_create_bucket(cos, params["training_results_bucket"])
            fcb = get_or_create_bucket(cos, get_data_bucket_name(params))  # fairness-check data bucket
            for out_file in ["y_test.out", "p_test.out", "y_pred.out"]:
                src_path = "%s/%s" % (model_id, out_file)
                copy_file(trb, src_path, fcb, out_file)

        def create_manifest(params):
            training_command = "\
                pip install aif360 pandas; \
                python -u fairness_check.py --label_dir $DATA_DIR --model_dir $DATA_DIR/model.pt"

            manifest_dict = {
                "name": params.get("training_job_name", "fairnesscheck_%s" % params["model_id"]),
                "description": "Pytorch gender classification model fairness check",
                "version": "1.0",
                "memory": params.get("memory", "2Gb"),
                "gpus": int(params.get("gpus", 0)),
                "cpus": float(params.get("cpus", 2)),
                "learners": 1,
                "data_stores": [
                    {
                        "id": "fairness-check",
                        "type": "mount_cos",
                        "training_data": {
                            "container": get_data_bucket_name(params)
                        },
                        "training_results": {
                            "container": get_result_bucket_name(params)
                        },
                        "connection": {
                            "auth_url":  params["aws_endpoint_url"],
                            "user_name": params["aws_access_key_id"],
                            "password":  params["aws_secret_access_key"]
                        }
                    }
                ],
                "framework": {
                    "name": "pytorch",
                    "version": "latest",
                    "command": training_command.strip()
                }
            }
            with open(manifest_file, "w") as f:
                yaml.dump(manifest_dict, f)

        def start_fairness_check(params):
            url = "%s/v1/models?version=2017-02-13" % params["ffdl_service_url"]
            headers = {"Accept": "application/json",
                       "Authorization": params["basic_authtoken"],
                       "X-Watson-Userinfo": params["watson_auth_token"]}
            files = {'manifest': open('manifest.yml', 'rb'),
                     'model_definition': open('model.zip', 'rb')}
            response = requests.post(url, headers=headers, files=files)
            return json.loads(response.text or response.content or "{}")

        cos = create_cos_connection(args)
        get_or_create_bucket(cos, get_data_bucket_name(args))
        get_or_create_bucket(cos, get_result_bucket_name(args))
        create_model_zip()
        create_manifest(args)
        copy_training_result_files(cos, args)
        response = start_fairness_check(args)
        # OpenWhisk/IBM Cloud Functions does not return JSON when the returned dict contains key "error"
        if "error" in response:
            print(response)
            return {
                "Status": "Error",
                "Details": response
            }
        else:
            return response
    except Exception as e:
        print('%s: %s\n%s' % (e.__class__.__name__, str(e), traceback.format_exc()))
        return {
            "Status": "Error",
            "Details": {
                "Error": e.__class__.__name__,
                "Message": str(e),
                "Trace": traceback.format_exc()
            }
        }


# main() method will be run when this action gets invoked
# @param a JSON string or JSON file converted to a dict by OpenWhisk
# @return The output of this action, which must be of type dict (converted to a JSON object by OpenWhisk)
def main(parameters):
    # run "safe" catching any exceptions and return errors as part of the result to help debugging
    return run_safe(parameters)


# --------------------------------
# Helper methods for local testing
# --------------------------------


def _load_json_args():
    import sys, json
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    if arg1 and arg1.endswith(".json"):
        with open(arg1) as json_file:
            json_data = json.load(json_file)
    elif arg1:
        json_data = json.loads(arg1)
    else:
        raise ValueError("Expected JSON file or JSON string as argument.")
    return json_data


if __name__ == "__main__":
    params = _load_json_args()
    result = main(params)
    print(result)
