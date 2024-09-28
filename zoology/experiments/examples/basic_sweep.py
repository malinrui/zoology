import numpy as np
from zoology.config import TrainConfig, LoggerConfig
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig

from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig

configs = []

for lr in np.logspace(-4, -2, 4):

   config = TrainConfig(
      max_epochs=64,
      data=DataConfig(
         train_configs=[MQARConfig(
            num_examples=20_000,
            vocab_size=256,
            input_seq_len=64,  # 难度
            num_kv_pairs=16  # 难度
         )],
         test_configs=[MQARConfig(
            num_examples=1_000,
            vocab_size=256,
            input_seq_len=64,
            num_kv_pairs=16
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
         d_model=128,
      ),
      logger=LoggerConfig(
         project_name="zoo-3",
         entity="udem-malr",
      ),
      learning_rate=lr
   )
   configs.append(config)


for lr in np.logspace(-4, -2, 4):

   config = TrainConfig(
      max_epochs=64,
      data=DataConfig(
         train_configs=[MQARConfig(
            num_examples=20_000,
            vocab_size=256,
            input_seq_len=64,  # 难度
            num_kv_pairs=4  # 难度
         )],
         test_configs=[MQARConfig(
            num_examples=1_000,
            vocab_size=256,
            input_seq_len=64,
            num_kv_pairs=4
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
         d_model=128,
      ),
      logger=LoggerConfig(
         project_name="zoo-3",
         entity="udem-malr",
      ),
      learning_rate=lr
   )
   configs.append(config)
