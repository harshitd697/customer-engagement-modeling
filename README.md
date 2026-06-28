# Customer Engagement & Click Probability Modeling

A machine learning project to predict customer engagement probability on e-commerce platforms, enabling personalized item rankings and maximizing user interaction.

---

## Objective

Predict whether a user will **engage** (add to cart / purchase) with an item based on their behavioral and temporal patterns — enabling personalized item recommendations ranked by engagement probability.

---

## Dataset

**Retailrocket E-Commerce Dataset** (Public, Kaggle)

- 100,000 user interaction events sampled from 2.75M+ total events
- Event types: `view`, `addtocart`, `transaction`
- Natural class imbalance: ~96.75% views vs ~3.25% engagement events
- Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

---

## Project Structure

```
customer-engagement-modeling/
│
├── customer_engagement_modeling.py   # Main project code
├── recommendations.csv               # Output: top 5 items per visitor
├── eda_plots.png                     # EDA visualizations
├── model_results.png                 # Model comparison & feature importance
└── README.md                         # Project documentation
```

---

## Approach

### 1. Data Loading & Cleaning
- Loaded 100K user interaction events from parquet/csv
- Converted Unix timestamps to datetime objects
- Created binary target variable: `1` if addtocart/transaction, `0` if view only

### 2. Feature Engineering

| Feature Type | Features Created |
|---|---|
| **Temporal** | hour of day, day of week, month |
| **Behavioral** | visitor total views, visitor unique items explored |
| **Contextual** | item total views, item unique visitors (popularity) |

### 3. Class Imbalance Handling
- Dataset has severe imbalance: 96.75% non-engagement vs 3.25% engagement
- Used `scale_pos_weight` parameter in XGBoost to penalize majority class
- Applied `stratify=y` in train/test split to maintain class ratio

### 4. Model Training & Comparison

| Model | AUC Score |
|---|---|
| **XGBoost** | **0.8730** Winner |
| CatBoost | 0.8524 |

### 5. Ranking Output
- Generated predicted engagement probability for each visitor-item pair
- Ranked items per visitor by predicted probability
- Output top 5 recommended items per visitor

---

## Results

- **XGBoost AUC: 0.8730** on held-out validation set (20% split)
- XGBoost outperformed CatBoost by ~2%
- Most important features: `item_total_views`, `visitor_total_views`, `hour`



## Author

**Harshit** | [GitHub](https://github.com/harshitd697)
