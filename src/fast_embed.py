import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ONNX_PATH = Path(r"C:\Users\javie\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41\onnx\model.onnx")

class FastEmbedder:
    def __init__(self, batch_size=128, max_length=256):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(ONNX_PATH), opts, providers=providers)
        
        # Warm up
        dummy = self.tokenizer(["warmup"], return_tensors="np", padding=True, truncation=True, max_length=16)
        feed = {k: v for k, v in dummy.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        self.session.run(None, feed)
        print("FastEmbedder initialized with providers:", self.session.get_providers())

    def encode(self, texts, show_progress=True):
        all_embeddings = []
        n = len(texts)
        t0 = time.time()
        
        for i in range(0, n, self.batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            feed = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            outputs = self.session.run(None, feed)
            token_embeddings = outputs[0]  # shape: (B, L, 384)
            attention_mask = inputs["attention_mask"]  # shape: (B, L)
            
            # Mean pooling
            mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask
            
            # L2 normalization
            norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norm = np.clip(norm, a_min=1e-12, a_max=None)
            normalized = (mean_pooled / norm).astype(np.float32)
            all_embeddings.append(normalized)
            
            if show_progress and (i // self.batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                speed = i / elapsed
                print(f"  Progreso: {i:,}/{n:,} ({i/n*100:.1f}%) a {speed:.1f} textos/s")
                
        result = np.vstack(all_embeddings)
        print(f"  Codificados {len(result):,} textos en {time.time()-t0:.1f}s ({len(result)/(time.time()-t0):.1f} textos/s).")
        return result

if __name__ == "__main__":
    from src import load
    tr = load.load_split("train")
    sample_texts = tr["resp_a_text"].iloc[:500].tolist()
    embedder = FastEmbedder()
    embs = embedder.encode(sample_texts)
    print("Resultado shape:", embs.shape)
