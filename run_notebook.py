import gc
import os
import time

from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from qwix import lora
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
import sys
import os

# add the parent directory (one level up) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

# ! pip install -r ../../maxtext/requirements.txt

import MaxText as mt
from MaxText import pyconfig

# Data
BATCH_SIZE = 16

# Model
MESH = [(1, 8), ("fsdp", "tp")]
# LoRA
RANK = 16
ALPHA = 2.0

# Train
MAX_STEPS = 100
EVAL_EVERY_N_STEPS = 20
NUM_EPOCHS = 3


# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/content/intermediate_ckpt/"
CKPT_DIR = "/content/ckpts/"
PROFILING_DIR = "/content/profiling/"

def get_ref_maxtext_model():

  #python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=${FINETUNE_RUN_NAME} max_target_length=8192 steps=10 async_checkpointing=false model_name=gemma-2b checkpoint_period=5

  #TODO: @mazumdera: change this to use Gemma2-2b-it
  config = pyconfig.initialize(
      ["", "MaxText/configs/base.yml"], #TODO: @mazumdera: why decode.py?
      base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
      run_name="test-tunix-maxtext-gemma-2b",
      # dataset_path=we use Tunix's dataset
      load_parameters_path="gs://maxtext-gemma/2b/",
      tokenizer_path="maxtext/assets/tokenizer.gemma",
      per_device_batch_size=1,
      max_target_length=8192,
      steps=10,
      async_checkpointing="false",
      model_name="gemma-2b",
      checkpoint_period=5,
      skip_jax_distributed_system="true"

  )
  model = mt.from_pretrained(config)
  mesh  = model.mesh

  # We can continue to use Tunix's model_config
  model_config = gemma_lib.TransformerConfig.gemma2_2b()

  return model, mesh, model_config

def get_base_model(ckpt_path):

  model_config = gemma_lib.TransformerConfig.gemma_2b()
  mesh = jax.make_mesh(*MESH)
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh, model_config

gemma, mesh, model_config = get_ref_maxtext_model()
gemma_maxtext_nnx = nnx.bridge.ToNNX(gemma)

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.globals import PKG_DIR

gemma_tokenizer = data_lib.GemmaTokenizer(
)

sampler = sampler_lib.Sampler(
    transformer=gemma_maxtext_nnx,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

input_batch = [
    "Translate this into French:\nHello,, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=10,  # The number of steps performed when generating a response.
)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"----------------------")
  print(f"Prompt:\n{input_string}")
  print(f"Output:\n{out_string}")
