import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

class DEBATERModel(nn.Module):
    def __init__(self, model_name="openbmb/MiniCPM-2B-dpo-bf16", cod_length=8):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cod_length = cod_length
        
        # Add special tokens
        cod_tokens = [f"[CoD{i}]" for i in range(1, cod_length + 1)]
        self.tokenizer.add_tokens(cod_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.eos_token = self.tokenizer.eos_token
        
    def encode_query(self, query_texts):
        """Encode a batch of queries to get h^q."""
        input_texts = [f"Query: {q} {self.eos_token}" for q in query_texts]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        eos_pos = (inputs["input_ids"] == eos_id).nonzero(as_tuple=True)[1]
        h_q = outputs.last_hidden_state[torch.arange(len(query_texts)), eos_pos, :]
        return h_q
    
    def encode_document(self, doc_texts):
        """Encode a batch of documents to get CoD embeddings h_1^d to h_m^d."""
        cod_tokens = " ".join([f"[CoD{i}]" for i in range(1, self.cod_length + 1)])
        input_texts = [f"Document: {d} {cod_tokens}" for d in doc_texts]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        cod_ids = self.tokenizer.convert_tokens_to_ids([f"[CoD{i}]" for i in range(1, self.cod_length + 1)])
        h_d = []
        for cod_id in cod_ids:
            pos = (inputs["input_ids"] == cod_id).nonzero(as_tuple=True)[1]
            h_d_i = outputs.last_hidden_state[torch.arange(len(doc_texts)), pos, :]
            h_d.append(h_d_i)
        h_d = torch.stack(h_d, dim=1)  # Shape: (batch, m, dim)
        return h_d
    
    def embed_query(self, query_texts):
        """Generate query embedding for retrieval (h^q)."""
        h_q = self.encode_query(query_texts if isinstance(query_texts, list) else [query_texts])
        return h_q if isinstance(query_texts, list) else h_q[0]
    
    def embed_document(self, doc_texts):
        """Generate document embedding for retrieval (h_m^d)."""
        h_d = self.encode_document(doc_texts if isinstance(doc_texts, list) else [doc_texts])
        h_m = h_d[:, -1, :]  # Take the final CoD embedding
        return h_m if isinstance(doc_texts, list) else h_m[0]
    
    def forward(self, query_texts, doc_texts):
        """Compute training losses: contrastive loss and self-distillation loss."""
        # Encode queries and documents
        h_q = self.encode_query(query_texts)  # Shape: (B, dim)
        h_d = self.encode_document(doc_texts)  # Shape: (B, m, dim)
        
        # Normalize embeddings for cosine similarity
        h_q_norm = F.normalize(h_q, p=2, dim=1)
        h_d_norm = F.normalize(h_d, p=2, dim=2)
        
        # Compute similarity matrices
        S_all = torch.stack([h_q_norm @ h_d_norm[:, k, :].T for k in range(self.cod_length)], dim=0)  # Shape: (m, B, B)
        S_max, _ = S_all.max(dim=0)  # Shape: (B, B)
        S_final = h_q_norm @ h_d_norm[:, -1, :].T  # Shape: (B, B)
        
        # Contrastive loss (in-batch negatives)
        labels = torch.arange(len(query_texts)).to(h_q.device)
        loss_c = F.cross_entropy(S_max, labels)
        
        # Self-distillation loss (KL divergence)
        P = F.softmax(S_max, dim=1)
        Q = F.softmax(S_final, dim=1)
        loss_t = (P * (P.log() - Q.log())).sum(dim=1).mean()
        
        return loss_c + loss_t

class Retriever:
    def __init__(self, embedding_model, dimension=None):
        self.embedding_model = embedding_model
        self.dimension = dimension or embedding_model.base_model.config.hidden_size
        self.index = None
        self.documents = []
    
    def build_index(self, documents):
        """Build FAISS index over document embeddings."""
        with torch.no_grad():
            embeddings = self.embedding_model.embed_document(documents).cpu().numpy()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.documents = documents
    
    def retrieve(self, query, top_k=5):
        """Retrieve top-k documents for a query."""
        with torch.no_grad():
            q_emb = self.embedding_model.embed_query(query).cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(q_emb, top_k)
        return [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

# Example training loop
def train_debater(model, train_data, epochs=3, batch_size=16, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(train_data["queries"]), batch_size):
            batch_queries = train_data["queries"][i:i+batch_size]
            batch_docs = train_data["docs"][i:i+batch_size]
            optimizer.zero_grad()
            loss = model(batch_queries, batch_docs)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item():.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DEBATERModel().to(device)
    
    # Sample training data
    train_data = {
        "queries": ["What is AI?", "How does retrieval work?"],
        "docs": ["AI is intelligence by machines.", "Retrieval finds relevant documents."]
    }
    train_debater(model, train_data)
    
    # Initialize retriever
    retriever = Retriever(model)
    documents = ["AI is intelligence by machines.", "Retrieval finds relevant documents.", "Random text."]
    retriever.build_index(documents)
    
    # Retrieve
    query = "What is AI?"
    results = retriever.retrieve(query, top_k=2)
    for doc, score in results:
        print(f"Doc: {doc}, Score: {score:.4f}")
