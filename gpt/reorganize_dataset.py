import json


path_to_use = r'C:\Users\16028\Downloads\gpt3_data'
path = "/Users/alex/Downloads/naturalproofs_proofwiki.json"
path =  path_to_use + r'\naturalproofs_proofwiki.json'
f = open(path, "r")

data = json.load(f)
fdata = []
Theorems = data['dataset']['theorems']

miss = 0
cnt = 0
# for i in range(len(Theorems)):
for i in range(500):
	theorem = Theorems[i]
	content = theorem['contents']
	cnt += 1
	if len(theorem['proofs']) == 0:
		miss += 1
		continue
	proof = theorem['proofs'][0]['contents']

	content_string = ""
	proof_string = ""
	for x in content:
		if "\\displaystyle" in x:
			x = x.replace("\\displaystyle", "")
		content_string += "\'" + str(x) + "\'" + ", "
	for x in proof:
		if "\\displaystyle" in x:
			x = x.replace("\\displaystyle", "")
		proof_string += "\'" + str(x) + "\'" + ", "

	ent = {"prompt": content_string, "completion": proof_string}
	fdata.append(ent)

json_string = json.dumps(fdata, indent=4)
print(cnt, miss)

output_path = path_to_use + r'\naturalproofs_subset.json'
with open('output_path', 'w', encoding='utf-8') as f:
	f.write('\n'.join(json.dumps(i) for i in fdata))
