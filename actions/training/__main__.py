# OpenWhisk action to train a deep learning model with FfDL


def run_safe(parameters):
    try:
        import boto3, json, requests, os, sys, traceback

        def download_files_from_s3(params):
            cos = boto3.resource("s3",
                                 aws_access_key_id=params["aws_access_key_id"],
                                 aws_secret_access_key=params["aws_secret_access_key"],
                                 endpoint_url=params["aws_endpoint_url"])
            bucket = cos.Bucket(params["training_data_bucket"])
            bucket.download_file("model.zip", "model.zip")
            bucket.download_file("manifest.yml", "manifest.yml")

        def train_model(params):
            url = "%s/v1/models?version=2017-02-13" % params["ffdl_service_url"]
            headers = {
                "Accept": "application/json",
                "Authorization": params["basic_authtoken"],
                "X-Watson-Userinfo": params["watson_auth_token"]
            }
            files = {'manifest': open('manifest.yml', 'rb'),
                     'model_definition': open('model.zip', 'rb')}
            response = requests.post(url, headers=headers, files=files)
            result = json.loads(response.text or response.content or "{}")
            return result

        download_files_from_s3(parameters)
        result = train_model(parameters)
        return result or dict()
    except Exception as e:
        # print('%s: %s\n%s' % (e.__class__.__name__, str(e), traceback.format_exc()))
        return {
            "Status": "Error",
            "Details": {
                e.__class__.__name__: str(e),
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
