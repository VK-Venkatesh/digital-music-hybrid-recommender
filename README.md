# 🎵 digital-music-hybrid-recommender
This project implements a hybrid recommender system trained on the Amazon Reviews 2023 Digital Music dataset by McAuley Lab. It integrates content-based (TF-IDF + cosine), collaborative (item-based + SVD), and popularity-based (Bayesian weighted) recommendation techniques to produce personalized digital music recommendations.

The system provides an interactive Streamlit web interface that allows users to upload .jsonl review and metadata files, explore top-rated songs or albums, and adjust model weights dynamically to balance content, collaborative, and popularity signals.

Built for reproducible RecSys benchmarking, it features:

* Fine-grained data cleaning and merging of reviews and metadata

* TF-IDF text vectorization on titles, categories, and descriptions

* Sparse cosine similarity and latent SVD embeddings for collaborative filtering

* Weighted hybrid blending with user-adjustable sliders

* Tabbed visualization for Popular, Content-based, Collaborative, and Hybrid recommendations

  Excellent — I’ve reviewed your uploaded **`Time_Series_Model.py`** script.
It’s a **Streamlit-based hybrid recommender system** (Content-based + Collaborative + Popularity + Hybrid blend) built for the **Amazon Digital Music dataset (2023)** from McAuley Lab.

---

# 🎵 Digital Music Hybrid Recommender System

[![Streamlit App](https://img.shields.io/badge/Launch-App-brightgreen?logo=streamlit)](https://share.streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

A **hybrid recommendation system** for **Amazon Digital Music Reviews (2023)**, built using **Streamlit**, **TF-IDF**, and **Collaborative Filtering**.  
This project blends **popularity**, **content-based**, and **collaborative** approaches to generate personalized digital music recommendations.

---

## 📘 Dataset

This model uses the **Amazon Reviews 2023 — Digital Music** dataset from [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/):

| Feature | Description |
|----------|--------------|
| **Reviews** | 571M+ user-item interactions (May 1996 – Sep 2023) |
| **Metadata** | Rich item info (title, category, description, etc.) |
| **Granularity** | Timestamps at the second-level |
| **Splits** | Standard RecSys train/test splits |

---

## ⚙️ Features

✅ Load and clean `.jsonl` reviews and metadata  
✅ TF–IDF based content similarity model  
✅ Item–item collaborative filtering (cosine + SVD)  
✅ Popularity (Bayesian weighted mean) ranking  
✅ Hybrid weighted recommender (configurable weights)  
✅ Fully interactive **Streamlit** interface  
✅ User-controlled tuning sliders and data uploads

---

## 🧠 Model Architecture

| Module | Technique | Description |
|---------|------------|-------------|
| **Popularity** | Bayesian average | Weighted rating using vote counts |
| **Content-based** | TF-IDF + Cosine similarity | Textual similarity on title + category + description |
| **Collaborative** | Sparse cosine NN + SVD | Item–item matrix factorization |
| **Hybrid** | Weighted fusion | Combines all scores via linear weights |

---

## 🧩 Tech Stack

- **Python 3.9+**
- **Streamlit** for deployment/UI
- **scikit-learn** (TF-IDF, SVD, NearestNeighbors)
- **pandas**, **NumPy**, **SciPy**
- **McAuley Lab Amazon Dataset (2023)**

---

### Launch the app

```bash
streamlit run Time_Series_Model.py
```

### Upload Data

You can:

* Use default file paths (if placed in `data/`)
* Or upload `.jsonl` review and meta files via the Streamlit sidebar

---

## 🧪 Example Weights for Hybrid Blend

| Component     | Default Weight |
| ------------- | -------------- |
| Content-based | 0.45           |
| Collaborative | 0.45           |
| Popularity    | 0.10           |

---

## 📈 Output Tabs

| Tab               | Description                           |
| ----------------- | ------------------------------------- |
| **Popular**       | Top-N Bayesian weighted items         |
| **Content-based** | TF-IDF cosine recommendations         |
| **Collaborative** | Item–item cosine + SVD latent factors |
| **Hybrid**        | Final blended recommendations         |

---

## 🧾 Citation

If you use this dataset or pipeline, please cite:

> He, Ruining, McAuley, Julian.
> *Amazon Product Data 2023* (McAuley Lab, UCSD).
> [📑 Paper]([https://cseweb.ucsd.edu/~jmcauley/datasets.html](https://amazon-reviews-2023.github.io/))

---

## 📸 Demo Screenshot

![Streamlit Demo](assets/Screenshot%202025-11-04%20151318.png)

---

## 🧰 Folder Structure

```
.
├── Time_Series_Model.py     # Streamlit app
├── requirements.txt
├── data/                    # Example .jsonl data
├── assets/                  # Screenshots or demo
├── notebooks/               # Optional exploration
└── models/                  # Saved models (TF-IDF, SVD, etc.)
```

---

## 🪄 Future Improvements

* ✅ Integrate HuggingFace embeddings for better semantic similarity
* ✅ Add session-based or sequence models (e.g., SASRec, BERT4Rec)
* 🔜 Deploy on Streamlit Cloud or HuggingFace Spaces
* 🔜 Add user profiling and A/B testing

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

### 👨‍💻 Author

**Venkatesh**
Data Science | Machine Learning | Recommender Systems
📧 [venkateshvarada56@gmail.com](mailto:venkateshvarada56@gmail.com)
🌐 [LinkedIn Profile](www.linkedin.com/in/venkatesh-ds25)

```
