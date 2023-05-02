# Template to take Google colab parameters and and generate a parameters.py file with python type hinting 
# and scoping based methods they're instantiated in.
# 'parameters.py' will be automatically generated. Do not edit -- rather, rerun the file "extract_colab_parameters.py"
# This is from a jinja2 template.
# example input file: Deforum_Stable_Diffusion.ipynb

{% for method in methods -%} 
class {{ method }}:
{%- for obj in objects -%}
  {%- if obj.method == method %}
    {%- if obj.type in ["bool","float","int"] %}
      {{ obj.name }}: {{ obj.type }} = {{ obj.default }}
    {%- elif obj.type is none %}
      {{ obj.name }} = "{{ obj.default }}"{% if obj.constraints %}  # {{ obj.constraints }}{% endif %}
    {%- else %}
      {{ obj.name }}: {{ obj.type }} = "{{ obj.default }}"{% if obj.constraints %}  # {{ obj.constraints }}{% endif %}
    {%- endif %}
  {%- endif %}
{%- endfor %}
{% endfor %}