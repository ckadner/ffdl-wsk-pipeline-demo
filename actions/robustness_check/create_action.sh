#!/usr/bin/env bash

WSK_CLI="bx wsk"
#WSK_CLI="~/Projects/incubator-openwhisk-devtools/docker-compose/openwhisk-master/bin/wsk -i"

ACTION_NAME="robustness_check"

if [ ! -d virtualenv ]; then
    virtualenv --python=python3 virtualenv
fi

source virtualenv/bin/activate
pip install -r requirements.txt

# only include packages that are not available on IBM Cloud Functions:
# https://raw.githubusercontent.com/ibm-functions/runtime-python/7b87b88/python3/requirements.txt
zip -r "${ACTION_NAME}.zip" \
    virtualenv/bin/activate_this.py \
    virtualenv/lib/python3.6/site-packages/boto3 \
    virtualenv/lib/python3.6/site-packages/s3transfer \
    virtualenv/lib/python3.6/site-packages/ruamel \
    robustness_check.py \
    __main__.py \
    -x '*/examples/*' \
    -x '*/__pycache__/*'

# delete existing action
${WSK_CLI} action list | grep -q "${ACTION_NAME}" && ${WSK_CLI} action delete "${ACTION_NAME}"

echo "creating action ..."
${WSK_CLI} action create "${ACTION_NAME}" --kind python-jessie:3 --main main "${ACTION_NAME}.zip" --param-file "parameters_default.json"

# clean up
rm -f "${ACTION_NAME}.zip"

# print command to invoke the action
echo
echo "To invoke the action '${ACTION_NAME}', run the following command:"
echo
echo "    ${WSK_CLI} action invoke ${ACTION_NAME} --blocking --result --param-file parameters.json"
echo
