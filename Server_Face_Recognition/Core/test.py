import json

jf = open("../Data/csdl.json")
data = json.load(jf)
name = "Hung"
for key, value in data.items():
    if key == name:
        print(value)
        #break
jf.close()
