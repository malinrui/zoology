from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
#
# config = TrainConfig(
#     data=DataConfig(
#         # cache_dir="/path/to/cache/dir"  TODO: add this
#         vocab_size=256,
#         input_seq_len=64,
#         num_train_examples=10_000,
#         num_test_examples=1_000,
#         builder=FunctionConfig(
#             name="zoology.data.associative_recall.multiquery_ar",
#             kwargs={"num_kv_pairs": 4}
#         ),
#
#     ),
#     model=ModelConfig(
#         vocab_size=256,
#         max_position_embeddings=64,
#         sequence_mixer=ModuleConfig(
#             name="zoology.mixers.attention.MHA",
#             kwargs={"dropout": 0.1, "num_heads": 1}
#         )
#     ),
#
# )

from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig, FunctionConfig
import numpy as np


input_seq_len = 256
num_kv_pairs = 16
d_model = 128

# lr = 0.00046416 # for attn
# lr = 0.0021544 # for mamba

configs = []

for lr in [0.0021544, 0.0021544]:
    config = TrainConfig(
        max_epochs=64,
        data=DataConfig(
            train_configs=[MQARConfig(
                num_examples=20_000,
                vocab_size=512,
                input_seq_len=input_seq_len,  # 难度
                num_kv_pairs=num_kv_pairs  # 难度
            )],
            test_configs=[MQARConfig(
                num_examples=1_000,
                vocab_size=512,
                input_seq_len=input_seq_len,  # 难度
                num_kv_pairs=num_kv_pairs  # 难度
            )],
            batch_size=64
        ),
        model=ModelConfig(
            vocab_size=512,
            max_position_embeddings=256,


            sequence_mixer=[
                # ModuleConfig(
                #     name="zoology.mixers.mamba.Mamba",
                # ),
                ModuleConfig(
                    name="zoology.mixers.attention.MHA",
                    kwargs={"dropout": 0.1, "num_heads": 1}
                ) for _ in range(4)
            ],

            d_model=d_model,
            block_type="TransformerBlock",
            name="mha-1",

            n_layers=4,
        ),
        logger=LoggerConfig(
            project_name="zoo-3",
            entity="udem-malr",
        ),
        learning_rate=lr,
        run_id=f"TransformerBlock_lr_{lr:.5f}_d_{d_model}_kv_{num_kv_pairs}_seq_{input_seq_len}",

        # load_from_pretrained_path = "trained_models/TransformerBlock_lr_0.00046_d_128_kv_16_seq_256.pth",
        # save_model = True,
        #
        mix_with_mamba = True,
        mamba_layers = [0, 1, 2, 3],
        # init_from_attention_weights = True,

        # freeze_attn = True,

        # fake_mamba = True,

        # teacher_model_path = "trained_models/TransformerBlock_lr_0.00046_d_128_kv_16_seq_256.pth",
    )

    configs.append(config)

