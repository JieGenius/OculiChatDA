import sys
import os
import ruamel.yaml

mstr = ruamel.yaml.scalarstring.DoubleQuotedScalarString

MODEL_DIR = "../merged/"
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
file_name = 'metafile.yml'
# 读取YAML文件内容
with open(file_name, 'r') as file:
    data = yaml.load(file)
data['Models'] = []
for file in os.listdir(MODEL_DIR):
    if file == "metafile.yml": continue
    data['Models'].append({
        "Name": mstr(file),
        "Results": [dict(Task=mstr("Text Generation"), Dataset=mstr("none"))],
        "Weights": mstr(file),
    })

# 将修改后的数据写回文件
with open(os.path.join(MODEL_DIR, file_name), 'w') as file:
    yaml.dump(data, file)

print("Modifications saved to the file.")
