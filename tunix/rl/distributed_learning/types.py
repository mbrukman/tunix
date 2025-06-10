"""Type definitions and utilities for distributed learning."""

from typing import Union, Any, TypeVar
import numpy as np
import flax
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from flax.serialization import to_bytes, from_bytes

ArrayType = TypeVar('ArrayType', np.ndarray, jax.Array)
ArrayLike = Union[np.ndarray, jax.Array]

@flax.struct.dataclass(frozen=True)
class TrainExample:
  prompt_ids: jax.Array
  prompt_mask: jax.Array
  completion_ids: jax.Array
  completion_mask: jax.Array
  advantages: jax.Array
  ref_per_token_logps: jax.Array | None
  old_per_token_logps: jax.Array | None

@dataclass
class DeviceArrayPayload:
    """A class to handle serialization and deserialization of arrays between NumPy and JAX.
    
    This class provides a consistent way to serialize arrays (either NumPy or JAX) for gRPC
    transmission and deserialize them back into JAX arrays on the receiver side.
    """
    
    array_data: bytes
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_array(cls, array: ArrayLike) -> 'DeviceArrayPayload':
        """Create a DeviceArrayPayload from a NumPy or JAX array.
        
        Args:
            array: Input array (either NumPy or JAX)
            
        Returns:
            DeviceArrayPayload containing the serialized array data
        """
        # Get array info
        shape = array.shape
        dtype = str(array.dtype)
        
        # Serialize the array using Flax's to_bytes
        array_data = to_bytes(array)
        
        return cls(
            array_data=array_data,
            shape=shape,
            dtype=dtype
        )
    
    def to_jax(self) -> jax.Array:
        """Convert the payload to a JAX array.
        
        Returns:
            The deserialized JAX array
        """
        # Create a template array with the expected shape and dtype
        dtype = jax.dtypes.canonicalize_dtype(self.dtype)
        template = jnp.empty(self.shape, dtype=dtype)
        
        # Deserialize into the template
        jax_array = from_bytes(template, self.array_data)
        
        return jax_array

    def to_proto(self) -> Any:
        """Convert to proto message format.
        
        Returns:
            The proto message representation of this payload
        """
        from tunix.rl.distributed_learning.proto import worker_pb2
        
        return worker_pb2.DeviceArrayPayload(
            array_data=self.array_data,
            shape=list(self.shape),  # Convert tuple to list for proto
            dtype=self.dtype,
        )
    
    @classmethod
    def from_proto(cls, proto: Any) -> 'DeviceArrayPayload':
        """Create a DeviceArrayPayload from a proto message.
        
        Args:
            proto: The proto message containing array data
            
        Returns:
            DeviceArrayPayload containing the deserialized data
        """
        return cls(
            array_data=proto.array_data,
            shape=tuple(proto.shape),  # Convert list back to tuple
            dtype=proto.dtype,
        ) 