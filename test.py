import os
import openai
openai.organization = "org-VCfNuxZVxCD6rYF47bMOBxls"
openai.api_key = os.getenv("OPENAI_API_KEY")

print(openai.Model.list())

print("happy")
