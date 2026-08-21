import os
import json

from kv_compression.best_bit_analysis_kv_cache import export_initial_compress_parameters

class model_default_config:
    def __init__(self):
        self.model_root_dir = '/opt/data/private/'
        self.block_size=128
        self.config_dict = None
        self.max_new_tokens = 128
        # valiable in diffusion model
        self.compress_text_encoder = False

def convert_json_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            try:
                if '.' in key:
                    new_key = float(key)
                else:
                    new_key = int(key)
            except (ValueError, AttributeError):
                new_key = key           
            new_dict[new_key] = value
        return new_dict
    elif isinstance(obj, list):
        return [convert_json_keys(item) for item in obj]
    return obj

class qwen3_4B_config(model_default_config):
    def __init__(self, config_path='./model_configs/Qwen3-4B_weights.json', kv_cache_path='./model_configs/kv_test_Qwen3-4B_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/Qwen/Qwen3-4B"
        self.df11_model_id = f'{self.model_root_dir}/DFloat11/Qwen3-4B-DF11'
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class mistral_7B_config(model_default_config):
    def __init__(self, config_path='./model_configs/Mistral-7B-Instruct-v0.3_weights.json', kv_cache_path='./model_configs/kv_test_Mistral-7B-Instruct-v0.3_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/mistralai/Mistral-7B-Instruct-v0.3"
        # self.df11_model_id = f'{self.model_root_dir}/DFloat11/Qwen3-4B-DF11'
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class qwen1p5_moe_config(model_default_config):
    def __init__(self, config_path='./model_configs/Qwen1.5-MoE-A2.7B-Chat_weights.json', kv_cache_path='./model_configs/kv_test_Qwen1.5-MoE-A2.7B-Chat_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/Qwen/Qwen1.5-MoE-A2.7B-Chat"
        # self.df11_model_id = f'{self.model_root_dir}/DFloat11/Qwen3-4B-DF11'
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class huihui_moe_config(model_default_config):
    def __init__(self, config_path='./model_configs/Huihui-MoE-1.2B-A0.6B_weights.json', kv_cache_path='./model_configs/kv_test_Huihui-MoE-1.2B-A0.6B_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/huihui-ai/Huihui-MoE-1.2B-A0.6B"
        # self.df11_model_id = f'{self.model_root_dir}/DFloat11/Qwen3-4B-DF11'
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class qwen2_14B_config(model_default_config):
    def __init__(self, config_path='./model_configs/Qwen2.5-14B-Instruct_weights.json', kv_cache_path='./model_configs/kv_test_Qwen2.5-14B-Instruct_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/Qwen/Qwen2.5-14B-Instruct"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Qwen2.5-14B-Instruct-DF11"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class llama_3p1_8B_config(model_default_config):
    def __init__(self, config_path='./model_configs/Meta-Llama-3.1-8B-Instruct_weights.json', kv_cache_path='./model_configs/kv_test_Meta-Llama-3.1-8B-Instruct_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/LLM-Research/Meta-Llama-3.1-8B-Instruct"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Llama-3.1-8B-Instruct-DF11"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)


class phi_4_config(model_default_config):
    def __init__(self, config_path='./model_configs/Phi-4-reasoning-plus_weights.json', kv_cache_path='./model_configs/kv_test_Phi-4-reasoning-plus_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/LLM-Research/Phi-4-reasoning-plus"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Phi-4-reasoning-plus-DF11"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class ds_dis_llama_8b_config(model_default_config):
    def __init__(self, config_path='./model_configs/DeepSeek-R1-Distill-Llama-8B_weights.json', kv_cache_path='./model_configs/kv_test_DeepSeek-R1-Distill-Llama-8B_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/DeepSeek-R1-Distill-Llama-8B-DF11"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class ds_dis_qwen_7b_config(model_default_config):
    def __init__(self, config_path='./model_configs/DeepSeek-R1-Distill-Qwen-7B_weights.json', kv_cache_path='./model_configs/kv_test_DeepSeek-R1-Distill-Qwen-7B_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # kv cahce config
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class deepseek_v2_lite_config(model_default_config):
    def __init__(self, config_path='./model_configs/DeepSeek-V2-Lite_weights.json', kv_cache_path='./model_configs/kv_test_DeepSeek-V2-Lite_exp.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}deepseek-ai/deepseek-ai/DeepSeek-V2-Lite"
        self.config_dict = None
        self.block_size=128
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = json.load(file)
                self.config_dict = convert_json_keys(config_dict)
        # mla cache config: latent/rope parameters
        self.rights, self.nbits = export_initial_compress_parameters(kv_cache_path)

class Dolphin3_config(model_default_config):
    def __init__(self, config_path='./model_configs/Phi-4-reasoning-plus_weights.json'):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/cognitivecomputations/Dolphin3.0-R1-Mistral-24B"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Dolphin3.0-R1-Mistral-24B-DF11"
        self.config_dict = None
        self.block_size=128
        # if os.path.exists(config_path):
        #     with open(config_path, 'r', encoding='utf-8') as file:
        #         config_dict = json.load(file)
        #         self.config_dict = convert_json_keys(config_dict)


def get_config_dict(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = json.load(file)
            config_dict = convert_json_keys(config_dict)
        return config_dict
    return None

class black_flux_transformer_config(model_default_config):
    def __init__(self):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/black-forest-labs/FLUX.1-dev/"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/FLUX.1-dev-DF11/"
        self.config_dict = None
        self.block_size=128
        self.transformer_dict = get_config_dict('./model_configs/black_flux_dev_transformer.json')
        self.text_encoder_dict = get_config_dict('./model_configs/black_flux_dev_text_encoder_2.json')

class black_flux_kontext_transformer_config(model_default_config):
    def __init__(self):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/black-forest-labs/FLUX.1-Kontext-dev"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/FLUX.1-Kontext-dev-DF11"
        self.config_dict = None
        self.block_size=128
        self.transformer_dict = get_config_dict('./model_configs/black_flux_Kontext_transformer.json')
        self.text_encoder_dict = get_config_dict('./model_configs/black_flux_Kontext_text_encoder_2.json')

class stable_diffusion_config(model_default_config):
    def __init__(self):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/stabilityai/stable-diffusion-3.5-large"
        self.df11_model_id = f"{self.model_root_dir}/stable-diffusion-3.5-large-DF11"
        self.config_dict = None
        self.block_size=128
        self.transformer_dict = get_config_dict('./model_configs/StableDiffusion_transformer.json')
        self.text_encoder_dict = get_config_dict('./model_configs/StableDiffusion_text_encoder_3.json')

class Wan2p1_T2V_14B_config(model_default_config):
    def __init__(self):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/Wan-AI/Wan2.1-T2V-14B-Diffusers"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Wan2.1-T2V-14B-Diffusers-DF11"
        self.config_dict = None
        self.block_size=128
        self.transformer_dict = get_config_dict('./model_configs/wan_transformer.json')
        self.text_encoder_dict = get_config_dict('./model_configs/wan_text_encoder.json')

class Chroma_config(model_default_config):
    def __init__(self):
        super().__init__()
        self.model_id = f"{self.model_root_dir}/lodestones/Chroma"
        self.df11_model_id = f"{self.model_root_dir}/DFloat11/Chroma-DF11"
        self.config_dict = None
        self.block_size=128
        self.transformer_dict = get_config_dict('./model_configs/chroma_transformer.json')
        self.text_encoder_dict = get_config_dict('./model_configs/chroma_text_encoder.json')

model_config_map = {
    'qwen2_14B_config':qwen2_14B_config,
    'phi_4_config':phi_4_config,
    'ds_dis_llama_8b_config':ds_dis_llama_8b_config,
    'ds_dis_qwen_7b_config':ds_dis_qwen_7b_config,
    'qwen3_4B_config': qwen3_4B_config,
    'mistral_7B_config':mistral_7B_config,
    'qwen1p5_moe_config':qwen1p5_moe_config,
    'huihui_moe_config':huihui_moe_config,
    'deepseek_v2_lite_config':deepseek_v2_lite_config,
}
