# OpenWhisk action to perform a model robustness check with ART on FfDL

import json, requests


def run_safe(args):
    try:
        import boto3
        import traceback
        import zipfile
        from ruamel.yaml import YAML

        yaml = YAML()

        script_file = "robustness_check.py"
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
            # return "robustness-check-data" + params["model_id"][8:].lower()
            return params["robustnesscheck_data_bucket"]

        def get_result_bucket_name(params):
            # return "robustness-check-results" + params["model_id"][8:].lower()
            return params["robustnesscheck_results_bucket"]

        def copy_file(source_bucket, source_file, target_bucket, target_file):
            if source_bucket != target_bucket or source_file != target_file:
                target_bucket.copy({"Bucket": source_bucket.name, "Key": source_file}, target_file)

        def copy_training_result_files(cos, params):
            model_id = params["model_id"]
            tdb = get_or_create_bucket(cos, params["training_data_bucket"])
            trb = get_or_create_bucket(cos, params["training_results_bucket"])
            rcb = get_or_create_bucket(cos, get_data_bucket_name(params))
            copy_file(tdb, params["dataset_file"], rcb, params["dataset_file"])
            copy_file(trb, "%s/%s" % (model_id, params["networkdefinition_file"]), rcb, params["networkdefinition_file"])
            copy_file(trb, "%s/%s" % (model_id, params["weights_file"]), rcb, params["weights_file"])

        def create_manifest(params):
            training_command = "\
                pip3 install keras; \
                pip3 install https://github.com/IBM/adversarial-robustness-toolbox/zipball/master; \
                python3 robustness_check.py \
                  --epsilon 0.2 \
                  --data ${DATA_DIR}/%s \
                  --networkdefinition ${DATA_DIR}/%s \
                  --weights ${DATA_DIR}/%s"\
                .replace("\
                ", "") % (params["dataset_file"],
                          params["networkdefinition_file"],
                          params["weights_file"])

            manifest_dict = {
                "name": params.get("training_job_name", "robustnesscheck_%s" % params["model_id"]),
                "description": "Generates adversarial samples to check model robustness using FGM",
                "version": "1.0",
                "memory": params.get("memory", "2Gb"),
                "gpus": int(params.get("gpus", 0)),
                "cpus": float(params.get("cpus", 2)),
                "data_stores": [
                    {
                        "id": "robustness-check",
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
                    "name": "tensorflow",
                    "version": "1.5.0-py3",
                    "command": training_command
                },
                "evaluation_metrics": {
                    "type": "tensorboard",
                    "in": "$JOB_STATE_DIR/logs/tb"
                }
            }
            #yaml.dump(manifest_dict, open("manifest.yml", "w"), default_flow_style=False)
            with open(manifest_file, "w") as f:
                yaml.dump(manifest_dict, f)

        def start_robustness_check(params):
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
        response = start_robustness_check(args)
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
