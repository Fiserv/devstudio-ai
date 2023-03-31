import os
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import redis

openai.organization = "org-VCfNuxZVxCD6rYF47bMOBxls"
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    
    
    
    embedding_results = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=[text]
        )
    
    return embedding_results