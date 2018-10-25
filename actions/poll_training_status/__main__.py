# OpenWhisk action to poll the status of a FfDL training job


def run_safe(params):
    try:
        import requests
        url = "%s/v1/models/%s?version=2017-02-13" % (params["ffdl_service_url"], params["training_id"])
        headers = {
            "Accept": "application/json",
            "Authorization": params["basic_authtoken"],
            "X-Watson-Userinfo": params["watson_auth_token"]
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return {
                "status": "NOT FOUND",
                "training_id": params['training_id'],
                "request": url,
                "response": response.json()
            }
        result = response.json()  # json.loads(response.text or response.content or "{}")
        return {
            "training_id": result['model_id'],
            "status": result['training']['training_status']['status'],
            "model_name": result['name']
        }

    except Exception as e:
        import traceback
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
    import json
    params = _load_json_args()
    result = main(params)
    print(json.dumps(result, sort_keys=True, indent=4))

