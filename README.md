# 🎮 Steam-200K Game Recommender System (BM25-Weighted Implicit ALS)

## 📘 Overview
This project builds a **personalized game recommender system** on the [Steam-200K dataset], using **implicit feedback signals** (playtime and purchases) to infer player interests.  
The model applies **BM25-weighted Alternating Least Squares (ALS)** to learn latent user–game representations and generate Top-N recommendations.  

---

## 🧩 Dataset & Motivation
Dataset: `steam-200k.csv`  
Each record contains:
- `id`: Steam user ID  
- `game_name`: game title  
- `action`: either `purchase` or `play`  
- `play_time`: total playtime (in minutes)

### 🔍 Observation
- Every “purchase” action in the dataset has `play_time = 1`, which is not actual playtime.  
- Real engagement should be captured from `action == 'play'`.  
- We treat **“playtime” as a proxy for interest**, and **“purchase” as a weak positive signal (weighted 0.3)** — this ensures the recommender still recognizes a game as interesting when a user has purchased it but hasn’t played much yet.  
  - The 0.3 weight introduces a mild boost to such items, slightly improving diversity and recall, while preventing overfitting toward only high-playtime items.

---

## ⚙️ Technical Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.12 |
| **Data Manipulation** | pandas, numpy |
| **Visualization** | seaborn, matplotlib |
| **Sparse Matrix** | scipy.sparse (COO / CSR) |
| **Recommendation Model** | [implicit](https://github.com/benfred/implicit) |
| **Weighting Strategy** | BM25 weighting |
| **Algorithm** | AlternatingLeastSquares (implicit feedback) |

---

## 🧠 Recommendation Strategy

### 1️⃣ Implicit Preference Weight
Each `(user, game)` pair is assigned a **preference weight** proportional to interest intensity:

```python
purchase_weight = 0.3
df_copy["weight"] = np.where(
    df_copy["action"] == "play",
    np.sqrt(df_copy["play_time"]),   # diminishing returns for long sessions
    df_copy["play_time"] * purchase_weight
)
```

- Playtime → stronger implicit signal  
- Purchase only → weak positive feedback (0.3× weight)  
- Aggregated per user–game pair using:
  ```python
  df_copy.groupby(["id", "game_name"])["weight"].sum().reset_index()
  ```

---

### 2️⃣ Cold-Start Filtering
Users or games with only one interaction are filtered out to improve collaborative signal reliability:
```python
df_copy = df_copy[
    (df_copy["uses_played_games_count"] >= 1) &
    (df_copy["games_be_played_count"] >= 1)
]
```

---

### 3️⃣ Sparse Encoding
- Encode users and games with `pd.factorize`  
- Construct a sparse **users×items** matrix with `scipy.sparse.coo_matrix`  

This matrix serves as the input to the **implicit ALS** model, where latent factors for users and games are learned through alternating optimization.

---

## 🧮 Example Output
```
✅ Top-10 recommendations for user 92842632:
 1. Team Fortress 2 (score=0.5148)
 2. Portal 2 (score=0.5027)
 3. Left 4 Dead 2 (score=0.4931)
 ...
```

---
## 📦 Setup & 🚀 How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/Eriq7/ML-Steamgames-Recommendations-model.git
   cd ML-Steamgames-Recommendations-model


## 🧑‍💻 Author
**Qun Li**  

