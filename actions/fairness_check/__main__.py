# OpenWhisk action to perform a model fairness check with AIF360


def create_cos_connection(params):
    import boto3
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


def copy_file(source_bucket, source_file, target_bucket, target_file):
    if source_bucket != target_bucket or source_file != target_file:
        target_bucket.copy({"Bucket": source_bucket.name, "Key": source_file}, target_file)


def copy_training_result_files(cos, params):
    model_id = params["model_id"]
    # tdb = get_or_create_bucket(cos, params["training_data_bucket"])
    trb = get_or_create_bucket(cos, params["training_results_bucket"])
    for out_file in ["y_test.out", "p_test.out", "y_pred.out"]:
        src_path = "%s/%s" % (model_id, out_file)
        trb.download_file(src_path, out_file)


def get_fairness_check_metrics(params):
    from fairness_check import fairness_check
    metrics = fairness_check(label_dir=".", model_dir=".")
    return metrics


def run_safe(args):
    try:
        cos = create_cos_connection(args)
        copy_training_result_files(cos, args)
        metrics = get_fairness_check_metrics(args)
        return metrics
    except Exception as e:
        import traceback
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
    # result = get_fairness_check_metrics(params)
    print(result)
