#!/bin/bash

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script uploads a Tunix docker image to GCR.
# It assumes you have already built the image using docker_build_dependency_image.sh.
# Example usage:
# bash docker_upload_runner.sh image_type=base
# bash docker_upload_runner.sh image_type=base project=<your-gcp-project-id>

set -e

# Set default values
IMAGE_TYPE="base"
GCR_PROJECT_ID=""
DOCKER_IMAGE_TAG="latest"

# Parse command-line arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        image_type) IMAGE_TYPE=${VALUE} ;;
        project)    GCR_PROJECT_ID=${VALUE} ;;
        tag)        DOCKER_IMAGE_TAG=${VALUE} ;;
        *)
    esac
done

if [[ "$IMAGE_TYPE" == "base" ]]; then
    LOCAL_IMAGE_NAME=tunix_base_image
elif [[ "$IMAGE_TYPE" == "deepswe" ]]; then
    LOCAL_IMAGE_NAME=tunix_deepswe
else
    echo "Unknown image type: $IMAGE_TYPE"
    exit 1
fi

if [[ -z "$GCR_PROJECT_ID" ]]; then
    echo "GCP project ID not provided. Querying for current project..."
    GCR_PROJECT_ID=$(gcloud config get-value project)
    if [[ -z "$GCR_PROJECT_ID" ]]; then
        echo "Error: No GCP project is set. Please provide one with project=<project-id> or run 'gcloud config set project <project-id>'."
        exit 1
    fi
fi

echo "Preparing to upload ${LOCAL_IMAGE_NAME} to ${GCR_PROJECT_ID} with tag ${DOCKER_IMAGE_TAG}"

gcloud auth configure-docker

REMOTE_IMAGE_NAME="gcr.io/${GCR_PROJECT_ID}/${LOCAL_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME}
docker push ${REMOTE_IMAGE_NAME}

echo "Successfully pushed ${REMOTE_IMAGE_NAME}"