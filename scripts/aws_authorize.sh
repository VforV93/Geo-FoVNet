#! /bin/bash

# This script allows for AWS autorization using aws-vault, without the need to execute authorization
# in the same terminal session as IDE / python script.

if [[ -z $AWS_CONFIG_FILE ]];
then
    echo "AWS_CONFIG_FILE variable is not set please run devops bootstrap script";
fi

mkdir -p ~/.aws
AWS_CRED_FILE=~/.aws/credentials

# Check if AWS_CRED_FILE is empty and if not, ask for overwrite. If yes, delete the existing file.
if [ -s "$AWS_CRED_FILE" ]; then
    read -p "AWS credentials file is not empty. Do you want to overwrite it? [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$AWS_CRED_FILE"
    else
        exit 1
    fi
fi

unset AWS_VAULT
aws-vault clear
envs=$(aws-vault exec hubs-prod_rnd -- env)

touch "$AWS_CRED_FILE"
echo "[default]" > "$AWS_CRED_FILE"
echo "$envs" | grep AWS_DEFAULT_REGION >> "$AWS_CRED_FILE"
echo "$envs" | grep AWS_ACCESS_KEY_ID >> "$AWS_CRED_FILE"
echo "$envs" | grep AWS_SECRET_ACCESS_KEY >> "$AWS_CRED_FILE"
echo "$envs" | grep AWS_SESSION_TOKEN >> "$AWS_CRED_FILE"

exit 0