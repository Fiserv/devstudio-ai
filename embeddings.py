import os
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

"""
- make api call to openai => make it a function
- take input for embedding
- stub fuction to read from elastic to compare embedding
- return results.  with link to markdown / api explorer
"""

openai.organization = "org-VCfNuxZVxCD6rYF47bMOBxls"
openai.api_key = os.getenv("OPENAI_API_KEY")

#data values for testing
ch_doc_input = "Apple Pay is a digital wallet platform and online payment system developed by Apple to power in-app and tap-to-pay purchases on mobile devices, enabling users to make payments with supported devices like iPhones, iPads, and Apple Watches."
ch_api_input = "Use this payload to originate a financial transaction based on the captureFlag * Pre-Auth = false * Sale = true * Capture = true with a transaction identifier"

bh_doc_input = "Fiserv core account processing solutions provide the functionality to support the full range of deposit accounts offered by financial institutions. This includes demand deposit accounts, which allow the accountholder to withdraw money from the account without advance notice, as well as accounts on which the accountholder earns interest such as savings accounts, certificates of deposit, health savings accounts and individual retirement accounts. \
    Our solutions enable users to set up and maintain deposit accounts in a manner that complies with applicable regulations. They make it possible to track transactions (debits and credits), calculate balances, pay interest, assess overdraft or other fees, generate necessary notices and statements, produce tax reporting documents, and record and modify information about parties related to an account such as names, addresses, tax IDs, contact information and the nature of the party’s relationship to the account. \
    Our account processing solutions accommodate deposit accounts for individuals, small business and large enterprises. To accommodate the special needs of business accounts, they offer functionality such as nightly sweeps and repurchase agreements."
bh_api_input = "/cardservice/cards/cards"

fvapac_doc_input = "The final step is to get the newly activated card to be added to the ApplePay wallet. The push provisioning APIs can be used to generate the payload to initiate the green channel provisioning of newly created digital card on the ApplePay mobile-app."
fvapac_api_input = "This service is used to activate the card after successful verification of the cardholder."


search_query = "Can you do apple pay?"

def search(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

def main():
    print("Hello OpenAI Embeddings!")
    
    embedding_results = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[ch_doc_input, ch_api_input, bh_doc_input, bh_api_input, fvapac_doc_input, fvapac_api_input]
    )
    
    
    print(embedding_results)

    #df = pd.read_json(embedding_results)

    res2 = search(df, 'How do I use apple pay?', n=3)
    print(res2)


    # datafile_path = "data/data.csv"

    # df = pd.read_csv(datafile_path)
    # df["embedding"] = df.embedding.apply(eval).apply(np.array)
    
    # res1 = search(df, 'delicious beans', n=3)
    # print(res1)

if __name__ == "__main__":
    main()