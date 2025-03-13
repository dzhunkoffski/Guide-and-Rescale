import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import PIL
from PIL import Image

def visualize_self_attn(attention_matrix):
    n_heads, seq_len, emb_size = attention_matrix.shape
    image_size = int(math.sqrt(seq_len))
    attention_matrix = attention_matrix.permute(1, 0, 2).reshape(seq_len, -1)
    attention_matrix = attention_matrix.cpu().detach().numpy()
    pca = PCA(n_components=3)
    attn_img = pca.fit_transform(attention_matrix).reshape(image_size, image_size, 3)
    attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min())
    return Image.fromarray((attn_img * 255).astype(np.uint8))
