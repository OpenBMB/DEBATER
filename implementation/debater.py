import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DEBATERModel(nn.Module):
    def __init__(self, model_name="openbmb/MiniCPM-2B-dpo-bf16", cod_length=8):
        super().__init__()
        self.cod_length = cod_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Verify tokenizer supports adding tokens
        if not hasattr(self.tokenizer, "add_tokens"):
            raise ValueError(f"Tokenizer for {model_name} does not support adding tokens.")
        
        # Add CoD tokens
        cod_tokens = [f"[CoD{i}]" for i in range(1, cod_length + 1)]
        num_added = self.tokenizer.add_tokens(cod_tokens)
        if num_added != cod_length:
            logger.warning(f"Only {num_added}/{cod_length} CoD tokens added.")
        
        self.eos_token = self.tokenizer.eos_token
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Verify model supports embedding resizing
        try:
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        except AttributeError:
            raise ValueError(f"Model {model_name} does not support resizing token embeddings.")
        
        logger.info(f"Initialized DEBATERModel with {cod_length} CoD tokens.")

    def encode_query(self, query_texts):
        """Encode a batch of queries to get h^q."""
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        input_texts = [f"Query: {q} {self.eos_token}" for q in query_texts]
        max_length = max(len(self.tokenizer.encode(t)) for t in input_texts)
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        eos_pos = (inputs["input_ids"] == eos_id).nonzero(as_tuple=True)[1]
        h_q = outputs.last_hidden_state[torch.arange(len(query_texts)), eos_pos, :]
        return h_q

    def encode_document(self, doc_texts):
        """Encode a batch of documents to get CoD embeddings h_1^d to h_m^d."""
        if isinstance(doc_texts, str):
            doc_texts = [doc_texts]
        cod_tokens = " ".join([f"[CoD{i}]" for i in range(1, self.cod_length + 1)])
        input_texts = [f"Document: {d} {cod_tokens}" for d in doc_texts]
        
        # Dynamic max_length to prevent CoD token truncation
        max_length = max(len(self.tokenizer.encode(t)) for t in input_texts)
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        
        # Verify CoD tokens are present
        cod_ids = self.tokenizer.convert_tokens_to_ids([f"[CoD{i}]" for i in range(1, self.cod_length + 1)])
        if not all(any(inputs["input_ids"][i] == cod_id for cod_id in cod_ids) for i in range(len(doc_texts))):
            raise ValueError("CoD tokens were truncated in some documents.")
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        
        h_d = []
        for cod_id in cod_ids:
            pos = (inputs["input_ids"] == cod_id).nonzero(as_tuple=True)[1]
            if pos.numel() != len(doc_texts):
                raise ValueError(f"CoD token [CoD{cod_ids.index(cod_id)+1}] not found in all sequences.")
            h_d_i = outputs.last_hidden_state[torch.arange(len(doc_texts)), pos, :]
            h_d.append(h_d_i)
        h_d = torch.stack(h_d, dim=1)  # Shape: (batch, m, dim)
        return h_d

    def embed_query(self, query_texts):
        """Generate query embedding for retrieval (h^q)."""
        h_q = self.encode_query(query_texts)
        return h_q if isinstance(query_texts, list) else h_q[0]

    def embed_document(self, doc_texts):
        """Generate document embedding for retrieval (h_m^d)."""
        h_d = self.encode_document(doc_texts)
        h_m = h_d[:, -1, :]  # Final CoD embedding
        return h_m if isinstance(doc_texts, list) else h_m[0]

    def forward(self, query_texts, doc_texts):
        """Compute training losses: contrastive loss and self-distillation loss."""
        h_q = self.encode_query(query_texts)  # Shape: (B, dim)
        h_d = self.encode_document(doc_texts)  # Shape: (B, m, dim)
        
        # Normalize embeddings for cosine similarity
        h_q_norm = F.normalize(h_q, p=2, dim=1)
        h_d_norm = F.normalize(h_d, p=2, dim=2)
        
        # Compute similarity matrices
        S_all = torch.stack([h_q_norm @ h_d_norm[:, k, :].T for k in range(self.cod_length)], dim=0)  # Shape: (m, B, B)
        S_max, _ = S_all.max(dim=0)  # Shape: (B, B)
        S_final = h_q_norm @ h_d_norm[:, -1, :].T  # Shape: (B, B)
        
        # Contrastive loss
        labels = torch.arange(len(query_texts)).to(h_q.device)
        loss_c = F.cross_entropy(S_max, labels)
        
        # Self-distillation loss with numerical stability
        epsilon = 1e-10
        P = F.softmax(S_max, dim=1) + epsilon
        Q = F.softmax(S_final, dim=1) + epsilon
        loss_t = (P * (P.log() - Q.log())).sum(dim=1).mean()
        
        total_loss = loss_c + loss_t
        logger.debug(f"Loss: Contrastive={loss_c.item():.4f}, Self-Distillation={loss_t.item():.4f}, Total={total_loss.item():.4f}")
        return total_loss

class Retriever:
    def __init__(self, embedding_model, dimension=None):
        self.embedding_model = embedding_model
        self.dimension = dimension or embedding_model.base_model.config.hidden_size
        self.index = None
        self.documents = []
    
    def build_index(self, documents):
        """Build FAISS index over document embeddings."""
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        embeddings = []
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            with torch.no_grad():
                batch_emb = self.embedding_model.embed_document(batch_docs).cpu().numpy()
            embeddings.append(batch_emb)
        embeddings = np.concatenate(embeddings, axis=0)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.documents = documents
        logger.info("FAISS index built.")
    
    def retrieve(self, query, top_k=5):
        """Retrieve top-k documents for a query."""
        with torch.no_grad():
            q_emb = self.embedding_model.embed_query(query).cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(q_emb, top_k)
        return [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

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
            logger.info(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
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
