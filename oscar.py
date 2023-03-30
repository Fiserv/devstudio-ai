# example with OSCAR 2201
from datasets import load_dataset


dataset = load_dataset("oscar-corpus/OSCAR-2301",
                       use_auth_token=True,  # required
                       language="eo",
                       streaming=True,  # optional
                       split="train")  # optional

for d in dataset:
    print(d)  # prints documents
