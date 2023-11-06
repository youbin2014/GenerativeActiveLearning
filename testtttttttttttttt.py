import json

with open('tiny_imagenet_labels.json', 'r') as file:
    labels = json.load(file)


for id in labels:
    prompt = labels[id].split(", ")
    print(id, prompt)

for i in range(200):
    label_list = list(labels.keys())
    label_list.sort()
    label = label_list[i]

    print(label)



