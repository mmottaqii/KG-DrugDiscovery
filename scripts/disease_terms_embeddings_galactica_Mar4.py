import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# Load Galactica 6.7B tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
model = AutoModel.from_pretrained("facebook/galactica-1.3b")
model.to("cpu")


# Load the phrases
with open('/home/vmottaqi/kg/stitch/disease_phrases.pkl', 'rb') as f:
    phrases = pickle.load(f)


# Compute embeddings in a loop (as shown in previous examples)
embeddings_list = []
for phrase in phrases:
    inputs = tokenizer(phrase, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # Remove unwanted keys
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Mean pooling over tokens
    token_embeddings = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]
    phrase_embedding = token_embeddings.mean(dim=1)  # shape: [1, hidden_dim]
    # L2 normalize for cosine similarity usage
    phrase_embedding = torch.nn.functional.normalize(phrase_embedding, p=2, dim=1)
    embeddings_list.append(phrase_embedding.cpu().numpy())

# Concatenate all embeddings into a single numpy array (shape: [n_phrases, hidden_dim])
embeddings = np.concatenate(embeddings_list, axis=0)
print("Embeddings shape:", embeddings.shape)


# # Save the embeddings as a .npy file
np.save('/home/vmottaqi/kg/stitch/disease_phrases_emb_Mar4.npy', embeddings)
print("Embeddings saved to 'disease_phrases_emb_Mar4.npy'")