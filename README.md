# Deep Learning Pipeline with Fabric for Deep Learning and Apache OpenWhisk on IBM Cloud Functions

An AI pipeline demo using OpenWhisk actions to train deep learning models with FfDL, perform model robustness check with ART and deploy the trained model with Seldon on Kubernetes


## Prerequisites

### Kubernetes Cluster with FfDL
You need to have [Fabric for Deep Learning](https://github.com/IBM/FfDL/) deployed on a Kubernetes Cluster with at least 
2 CPUs and 4 Gb Memory.

### Cloud Object Storage
To store model and training data, this notebook requires access to a Cloud Object Storage (COS) instance.
[BlueMix Cloud Object Storage](https://console.bluemix.net/catalog/services/cloud-object-storage) offers a free 
*lite plan*. 
Follow [these instructions](https://dataplatform.ibm.com/docs/content/analyze-data/ml_dlaas_object_store.html)
to create your COS instance and generate [service credentials](https://console.bluemix.net/docs/services/cloud-object-storage/iam/service-credentials.html#service-credentials)
with [HMAC keys](https://console.bluemix.net/docs/services/cloud-object-storage/hmac/credentials.html#using-hmac-credentials).
Then go to the COS dashboard:
- Get the `cos_service_endpoint` from the **Endpoint** tab
- In the **Service credentials** tab, click **New Credential +** 
  - Add the "[HMAC](https://console.bluemix.net/docs/services/cloud-object-storage/hmac/credentials.html#using-hmac-credentials)"
    **inline configuration parameter**: `{"HMAC":true}`, click **Add**
  - Get the `access_key_id` (*AWS_ACCESS_KEY_ID*) and `secret_access_key` (*AWS_SECRET_ACCESS_KEY*) 
    from the `cos_hmac_keys` section of the instance credentials:
    ```
      "cos_hmac_keys": {
          "access_key_id": "1234567890abcdefghijklmnopqrtsuv",
          "secret_access_key": "0987654321zxywvutsrqponmlkjihgfedcba1234567890ab"
       }
    ```


## Setup

### Environment Variables

## Dataset

Fashion-MNIST is a [dataset of clothing images](https://github.com/zalandoresearch/fashion-mnist) provided by 
[Zalando Research](https://research.zalando.com/). It is intended to serve as a direct drop-in replacement for the 
original MNIST dataset of hand-written digits for benchmarking Machine Learning algorithms. The Fashion-MNIST dataset 
is split into 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated 
with a label from 10 classes.

![Fashion-MNIST](https://github.com/IBM/Fashion-MNIST-using-FfDL/blob/master/fashion-mnist-webapp/static/img/p1.png)


## License
[Apache 2.0](LICENSE)
