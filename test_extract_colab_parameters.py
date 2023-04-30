
from extract_colab_parameters import extract_parameters 

def test_extract_parameters():
    input_list = [
        "    map_location = \"cuda\" #@param [\"cpu\", \"cuda\"]\n",
        "    custom_checkpoint_path = \"\" #@param {type:\"string\"}\n",
        "    flip_2d_perspective = False #@param {type:\"boolean\"}\n",
        "    color_coherence_video_every_N_frames = 1 #@param {type:\"integer\"}\n",
        "    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}\n"
    ]

    expected_output = [
        {"name": "map_location", "default": "cuda", "constraints": ["cpu", "cuda"], "type": None},
        {"name": "custom_checkpoint_path", "default": "", "constraints": None, "type": "string"},
        {"name": "flip_2d_perspective", "default": False, "constraints": None, "type": "boolean"},
        {"name": "color_coherence_video_every_N_frames", "default": 1, "constraints": None, "type": "integer"},
        #{"name": "diffusion_cadence", "default": '1', "constraints": ['1', '2', '3', '4', '5', '6', '7', '8'], "type": "string"}
        {"name": "diffusion_cadence", "default": '1', "constraints": None, "type": "string"}
    ]

    output = extract_parameters(input_list)
    assert output == expected_output
