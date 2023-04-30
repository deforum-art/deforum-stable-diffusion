# Self-contained utility for extracting parameters from a Colab notebook
# and saving them to a file.
# #@param metadata turns a Python variable into a form field that users can enter. 
# If you want your ipynb to be the one source of truth, then 
# use this to generate the schema it uses. 

import json
import re
#pattern = re.compile(r'\s*(\w+)\s*=\s*(.+)\s*#@param(?:.*type:([\'\"])(\w+)\3)?(?:\s*\[(.*)\])?')
pattern = re.compile(r'\s*(\w+)\s*=\s*([^#\n]+)\s*#@param(?:.*type:\s*([\'\"])(\w+)\3)?(?:\s*\[(.*)\])?')



def extract_parameters(input_list):
    output = []
    for line in input_list:
        match = pattern.match(line)
        if match:
            variable_name, value, _, var_type, possible_values = match.groups()
            # if var_type:
            #     print(f"### {variable_name}: {value} (type: {var_type})", end="")
            # else:
            #     print(f"### {variable_name}: {value}", end="")
            
            if possible_values:
                var_type = "enum"
                # print(f" (possible values: [{possible_values}])")
            # else:
            #     print()
            output.append( {
                "name": variable_name,
                "default": value,
                "constraints": possible_values.split(",") if possible_values else None,
                "type": var_type
            })


    # print(f"Extracted {len(output)} variables from {input_list}")
    return output


def extract_colab_params(ipynb_file):
    variables = []

    with open(ipynb_file) as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # filter out list entries from cell['source'] that do not have the #@param annotation
            params = [s for s in cell['source'] if "#@param" in s]
            # print(f"## found {len(params)} parameters in cell id {cell['metadata']['id']}")
            variables.extend(extract_parameters(params))

    # print(f"## extracted {len(variables)} variables from {ipynb_file}")
    return variables

# take ipynb file from --notebook parameter given in command line
# and extract parameters from it
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract parameters from a Colab notebook')

    parser.add_argument('--notebook', type=str, help='Colab notebook file')
    args = parser.parse_args()

    variables = extract_colab_params(args.notebook)
    print(f"Extracted {len(variables)} variables from {args.notebook}")

    # json pretty print the variables
    print(json.dumps(variables, indent=2))


if __name__ == "__main__":
   main()
   
