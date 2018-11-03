#!/usr/bin/env bash

WSK_CLI="bx wsk"
#WSK_CLI="~/Projects/incubator-openwhisk-devtools/docker-compose/openwhisk-master/bin/wsk -i"

ACTION_NAME="fairness_check_ffdl"

WEB_SECRET="fiddle"

if [ ! -d virtualenv ]; then
    virtualenv --python=python3 virtualenv
fi
source virtualenv/bin/activate

echo "Installing Python dependencies into virtual environment ..."
pip install -q -r requirements.txt

echo "Creating ${ACTION_NAME}.zip ..."
# only include packages that are not available on IBM Cloud Functions:
# https://raw.githubusercontent.com/ibm-functions/runtime-python/7b87b88/python3/requirements.txt
echo "  adding: virtualenv/lib/python3.6/site-packages/..."
zip -r -q "${ACTION_NAME}.zip" \
    virtualenv/bin/activate_this.py \
    virtualenv/lib/python3.6/site-packages/boto3 \
    virtualenv/lib/python3.6/site-packages/s3transfer \
    virtualenv/lib/python3.6/site-packages/ruamel \
    -x '*/examples/*' \
    -x '*/__pycache__/*'
zip -r "${ACTION_NAME}.zip" \
    fairness_check.py \
    __main__.py

# delete existing action
${WSK_CLI} action list | grep -q "${ACTION_NAME}" && ${WSK_CLI} action delete "${ACTION_NAME}"

echo "creating action ..."
${WSK_CLI} action create "${ACTION_NAME}" \
    --kind python-jessie:3 \
    --main main "${ACTION_NAME}.zip" \
    --web true #--web-secure "${WEB_SECRET}"   # TODO: web-secure action after fixing Web UI Angular CORS issues
#    --param-file "parameters_default.json"   # for web action default parameters are locked down (not overridable)

# clean up
rm -f "${ACTION_NAME}.zip"

# print command to invoke the action
echo
echo "To invoke the action '${ACTION_NAME}', run the following command:"
echo
echo "    ${WSK_CLI} action invoke ${ACTION_NAME} --blocking --result --param-file parameters.json"
echo

echo

echo "To invoke it as a web action, run:"
echo
echo "    curl -X POST ${OW_HOST}/api/v1/web/${OW_NAMESPACE}/default/${ACTION_NAME}.json \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d @parameters.json"
echo