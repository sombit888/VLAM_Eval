import argparse
import os 
import torch
import transformers
import yaml 
from src.configuration_utils import TrajectoryVLAConfig
from src.model_utils import TrajectoryVLA
def remove_waypointer_prefix(ckpt):
    new_state_dict = {}
    for key, value in ckpt.items():
        # Remove the 'waypointer.' prefix if it exists
        if key.startswith('waypointer.'):
            new_key = key[len('waypointer.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def model_inference(ckpt_path,config_path):
    config_yaml = open(config_path, "r")    
    config = yaml.load(config_yaml, Loader = yaml.FullLoader)
    model_config = config['training_module_config']['model_config']
    waypointer_config = config['training_module_config']['model_config']['waypointer_config']
        
    timestep_proj_config = waypointer_config['timestep_proj_config']
    timestep_proj_config['num_tokens'] = 3 ####### HOT FIX, yanl file does not have num_tokens key
    token_proj_config = waypointer_config['token_proj_config'] 
    token_proj_config['llm_image_tokens_layers'] = []####### HOT FIX, yanl file does not have num_tokens key
    transformer_config = waypointer_config['transformer_config']
    pos_embed_config  = transformer_config['pos_embed_config']
    decoder_config = transformer_config['decoder_block_config']
    encoder_config = transformer_config['encoder_block_config']
    ## delete autoclass key 

    del waypointer_config['autoclass'],timestep_proj_config['autoclass'] , 
    del pos_embed_config['autoclass'],transformer_config['autoclass'],decoder_config['autoclass'],
    del encoder_config['autoclass']
    
    ### Hardcoded  Prismatic 
    
    prismatic_config_dict = {
        "vision_backbone_id":"dinosiglip-vit-so-224px",
        "llm_backbone_id":"llama2-7b-pure",
        "arch_specifier": "no-align+gelu-mlp",
        "use_fused_vision_backbone" :True, 
        "image_resize_strategy" : "letterbox",
        "text_config" : None,
        "llm_max_length"  : 2048,
        "pad_token_id" :32000,
        "pad_to_multiple_of" : 64,
        "output_projector_states" : False,
        "return_dict": False,
    }

    transformer_config = {
        "pos_embed_config": pos_embed_config,
        "encoder_block_config": encoder_config,
        "decoder_block_config": decoder_config,
        "num_blocks": 2
    }
    TrajectoryVlaConfig_dict = {
        "prismatic_config":prismatic_config_dict,
        "token_size": 1024,
        "cheat": False,
        "num_timesteps": 6,
        "rotation_components": 9,
        "seperate_control_proj": True,
        "timestep_proj_config": timestep_proj_config,
        "token_proj_config": token_proj_config,
        "transformer_config": transformer_config,
        "num_timestep_tokens": 3,
    }

    trajectoryvla_config = TrajectoryVLAConfig(**TrajectoryVlaConfig_dict)
    model = TrajectoryVLA(trajectoryvla_config)

    ckpt_params = torch.load(ckpt_path, map_location='cpu', mmap= True)
    ckpt_params = remove_waypointer_prefix(ckpt_params)

    model.load_state_dict(ckpt_params, strict=True)

    model = model.to(dtype=torch.bfloat16).cuda()
    model.eval()
    del ckpt_params
    
    return model

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description = "Eval the model")
    parser.add_argument("--ckpt_path", type = str, help = "Model path")
    parser.add_argument("--config_path", type = str, help = "Model Config path")
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    # assert os.path.exists(ckpt_path), "Model path does not exist"
    
    model = model_inference(ckpt_path,config_path)
    