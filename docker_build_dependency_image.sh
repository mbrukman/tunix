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

# This scripts takes a docker image that already contains the Tunix dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Script to buid a Tunix base image locally, example cmd is:
# bash docker_build_dependency_image.sh image_type=base
# bash docker_build_dependency_image.sh image_type=deepswe
set -e

# Set default values
IMAGE_TYPE="base"

# Parse command-line arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        image_type) IMAGE_TYPE=${VALUE} ;;
        *)
    esac
done

echo "Building image of type: $IMAGE_TYPE"

if [[ "$IMAGE_TYPE" == "base" ]]; then
    export LOCAL_IMAGE_NAME=tunix_base_image
    DOCKERFILE=./tunix_dependencies.Dockerfile
elif [[ "$IMAGE_TYPE" == "deepswe" ]]; then
    export LOCAL_IMAGE_NAME=tunix_deepswe
    DOCKERFILE=./tunix/experimental/deep_swe/images/tunix_deepswe_dependencies.Dockerfile
else
    echo "Unknown image type: $IMAGE_TYPE"
    exit 1
fi

echo "Building to $LOCAL_IMAGE_NAME using $DOCKERFILE"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

build_ai_image() {
    if [[ -z ${LOCAL_IMAGE_NAME+x} ]]; then
        echo "Error: LOCAL_IMAGE_NAME is unset, please set it!"
        exit 1
    fi
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo "Building Tunix Image at commit hash ${COMMIT_HASH}..."

    docker build \
        --network=host \
        -t ${LOCAL_IMAGE_NAME} \
        -f ${DOCKERFILE} .
}

build_ai_image

echo ""
echo "*************************"
echo ""

echo "Built your base docker image and named it ${LOCAL_IMAGE_NAME}.
It only has the dependencies installed. Assuming you're on a TPUVM, to run the
docker image locally and mirror your local working directory run:"
echo "docker run -it ${LOCAL_IMAGE_NAME}"
echo ""
echo "You can run tunix and your development tests inside of the docker image. Changes to your workspace will automatically
be reflected inside the docker container."
echo "Once you want you upload your docker container to GCR, take a look at docker_upload_runner.sh"