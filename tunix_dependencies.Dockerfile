FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    gnupg \
    ca-certificates && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*


# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.10:${PATH}"


WORKDIR /workspace

# Copy and install local tunix
COPY . /workspace/tunix
RUN pip install -e ./tunix

CMD ["bash"]