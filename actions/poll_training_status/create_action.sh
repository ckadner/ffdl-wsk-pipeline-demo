#!/usr/bin/env bash

WSK_CLI="bx wsk"
#WSK_CLI="~/Projects/incubator-openwhisk-devtools/docker-compose/openwhisk-master/bin/wsk -i"

ACTION_NAME="poll_training_status"

WEB_SECRET="fiddle"

zip -r "${ACTION_NAME}.zip" __main__.py

# delete existing action
${WSK_CLI} action list | grep -q "${ACTION_NAME}" && ${WSK_CLI} action delete "${ACTION_NAME}"

echo "creating action ..."
${WSK_CLI} action create "${ACTION_NAME}" \
    --kind python-jessie:3 \
    --main main "${ACTION_NAME}.zip" \
    --web true # --web-secure "${WEB_SECRET}"  # TODO: web-secure action
#    --param-file "parameters_default.json"    # for web action default parameters are locked down (not overridable)

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