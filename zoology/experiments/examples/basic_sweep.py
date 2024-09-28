import numpy as np
from zoology.config import TrainConfig, LoggerConfig
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig

from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig

configs = []



for lr in np.logspace(-4, -2, 4):
   input_seq_len = 64
   num_kv_pairs = 4
   d_model = 128

   config = TrainConfig(
      max_epochs=3,
      data=DataConfig(
         train_configs=[MQARConfig(
            num_examples=20_000,
            vocab_size=256,
            input_seq_len=input_seq_len,  # 难度
            num_kv_pairs=num_kv_pairs  # 难度
         )],
         test_configs=[MQARConfig(
            num_examples=1_000,
            vocab_size=256,
            input_seq_len=input_seq_len,
            num_kv_pairs=num_kv_pairs
         )],
         batch_size=64
      ),
      model=ModelConfig(
         vocab_size=256,
         max_position_embeddings=256,
         sequence_mixer=ModuleConfig(
            name="zoology.mixers.mamba.Mamba",
            # name="zoology.mixers.attention.MHA",
            # kwargs={"dropout": 0.1, "num_heads": 1},
         ),
         d_model=d_model,
         block_type="MambaBlock",
         name="mamba-1",
      ),
      logger=LoggerConfig(
         project_name="zoo-3",
         entity="udem-malr",
      ),
      learning_rate=lr,
      run_id=f"lr_{lr:.5f}_d_{d_model}_kv_{num_kv_pairs}_seq_{input_seq_len}"
   )
   configs.append(config)

