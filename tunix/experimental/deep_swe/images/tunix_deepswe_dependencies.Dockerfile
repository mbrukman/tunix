From tunix_base_image

# Clone R2E-Gym and install
RUN git clone https://github.com/R2E-Gym/R2E-Gym.git && \
    pip install -e ./R2E-Gym

CMD ["bash"]