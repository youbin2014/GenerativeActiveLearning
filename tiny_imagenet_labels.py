import os
import json


# # Directory containing Tiny ImageNet
# tiny_imagenet_dir = '../'
#
# # The 'wnids.txt' file contains the list of all unique class IDs in the Tiny ImageNet dataset
# wnids_file = os.path.join(tiny_imagenet_dir, 'wnids.txt')
#
# # The 'words.txt' file contains a mapping from class ID to human-readable labels
# words_file = os.path.join(tiny_imagenet_dir, 'words.txt')
#
# # Read class IDs
# with open(wnids_file, 'r') as f:
#     wnids = f.readlines()
# wnids = [x.strip() for x in wnids]
#
# # Read human-readable labels
# with open(words_file, 'r') as f:
#     words_data = f.readlines()
# words_data = [x.strip().split('\t') for x in words_data]
# words_dict = {x[0]: x[1] for x in words_data if x[0] in wnids}
#
# # Save to JSON
# output_file = 'tiny_imagenet_labels.json'
# with open(output_file, 'w') as f:
#     json.dump(words_dict, f)
#
# print(f"Labels saved to {output_file}")



wnidsfilename = 'data/TinyImageNet/tiny-imagenet-200/wnids.txt'
with open(wnidsfilename, 'r') as f:
    wnids = [line.strip() for line in f]

wordsfilename = 'data/TinyImageNet/tiny-imagenet-200/words.txt'
with open(wordsfilename, 'r') as f:
    wordnet_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in f}

wnid_label_dict = {wnid: wordnet_dict[wnid] for wnid in wnids if wnid in wordnet_dict}

output_file = 'tiny_imagenet_labels.json'
with open(output_file, 'w') as f:
    json.dump(wnid_label_dict, f)

# print(f"Labels saved to {output_file}")
