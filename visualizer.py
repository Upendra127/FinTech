import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import pickle
import os
import subprocess
#
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

texts = []
directory = "/Users/chinu/Desktop/FinTech copy/data/mda"
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    with open(filepath, "r") as file:
        texts.append(file.read())

def get_embed(text):
    inputs1 = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs1 = model(**inputs1)
    return outputs1.last_hidden_state[:, 0, :]

new_text = "reported a banner year in 2024, achieving a staggering 100 percent growth in revenue, reaching a record-breaking 70 billion dollars. This phenomenal performance was driven by a remarkable 400% increase in sales, fueled by strategic global expansion, innovative new products (10 launched in 2024), and exceptional customer demand. Maintaining a strong 70 percent gross margin, the company  demonstrated its commitment to efficient operations and profitability.  This success story is further reinforced by a 100 percent rise in investor and stakeholder confidence, reflecting their belief in the company's long-term potential (based on 2024 10-K filing)."

vals = []
for t in texts:
    vals.append((torch.matmul(get_embed(t), get_embed(new_text).T))[0].item())

#checking if static dir is there or not
static_dir = "/Users/chinu/Desktop/FinTech copy/static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Generate a unique filename for each plot
graph_name = "graph_{}.png".format(len(os.listdir(static_dir)) + 1)
graph_path = os.path.join(static_dir, graph_name)

plt.plot(vals)
plt.xlabel("year")
plt.ylabel("Performance")
plt.savefig(graph_path)  # Save the image in the 'static' folder
plt.close() 

subprocess.run(["python", "deleter.py"])