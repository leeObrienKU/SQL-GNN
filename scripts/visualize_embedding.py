import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# === Paths ===
ARCH = "gcn"
EMBEDDING_PATH = f"output/results/graph/{ARCH}/employee_embeddings.pt"
LABELS_PATH = f"output/results/graph/{ARCH}/employee_labels.pt"
OUTPUT_PLOT_PATH = f"output/results/graph/{ARCH}/embedding_tsne.png"

# === Load embeddings and labels ===
print(f"📥 Loading embeddings from {EMBEDDING_PATH}")
embeddings = torch.load(EMBEDDING_PATH)

if os.path.exists(LABELS_PATH):
    labels = torch.load(LABELS_PATH)
else:
    print("⚠️ No labels found, generating random ones.")
    labels = torch.randint(0, 5, (embeddings.size(0),))

# === Dimensionality reduction with t-SNE ===
print("🔍 Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())

# === Plotting ===
print("📊 Plotting...")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=5)
plt.colorbar(scatter, label="Title Label")
plt.title("t-SNE of Employee Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
print(f"✅ Plot saved to: {OUTPUT_PLOT_PATH}")
plt.show()
