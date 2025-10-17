import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import joblib

# ====== データ読み込み ======
dataset_path = "dataset.csv"
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"エラー: 指定されたファイルが見つかりません: {dataset_path}")
    exit(1)

print("=== データ読み込み完了 ===")
print(f"データ件数: {len(df)}")
print("カラム:", ", ".join(df.columns))
print()

# ====== 特徴量抽出関数 ======
def extract_features(domain: str):
    """ドメイン文字列から数値特徴量を抽出"""
    domain = str(domain).lower()
    
    # 基本情報
    length = len(domain)
    letters = sum(c.isalpha() for c in domain)
    digits = sum(c.isdigit() for c in domain)
    alnum_ratio = (letters + digits) / length if length > 0 else 0
    digit_ratio = digits / length if length > 0 else 0

    # サブドメイン情報
    parts = domain.split('.')
    num_subdomains = len(parts)
    avg_sub_len = np.mean([len(p) for p in parts]) if parts else 0

    # ビッグラム特徴量（上位10個の出現頻度を使って平均化）
    bigrams = [domain[i:i+2] for i in range(len(domain)-1)]
    unique_bigrams = len(set(bigrams))
    bigram_ratio = unique_bigrams / len(bigrams) if len(bigrams) > 0 else 0

    return [
        length,
        alnum_ratio,
        digit_ratio,
        num_subdomains,
        avg_sub_len,
        bigram_ratio
    ]

# ====== 特徴量生成 ======
print("=== 特徴量抽出中... ===")
X_features = df["dns.qry.name"].apply(extract_features)
X = np.array(X_features.tolist())
y = df["label"].values

# 特徴量確認
print("サンプル特徴量（最初の3件）:")
for i, feats in enumerate(X[:3]):
    print(f"  [{i}] {feats}")
print()

# ====== データ分割 ======
print("=== データ分割 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"トレーニングデータ: {len(X_train)}件, テストデータ: {len(X_test)}件")
print()

# ====== モデル作成・学習 ======
print("=== モデル学習中... ===")
nb = GaussianNB()
nb.fit(X_train, y_train)
print("学習完了")
print()

# ====== 推論 ======
print("=== 推論および評価 ===")
y_pred = nb.predict(X_test)

# ====== 精度評価 ======
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("--- 結果 ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print()

# ====== モデル保存 ======
#os.makedirs("dataset/05-Naive-Bayes", exist_ok=True)
#version = datetime.now().strftime("%Y%m%d_%H%M%S")
#model_path = f"dataset/05-Naive-Bayes/naive_bayes_model_{version}.pkl"
#joblib.dump(nb, model_path)
#print(f"モデルを保存しました: {model_path}")
#print("=== 処理完了 ===")

# === ランダムドメイン検知テスト ===
print("\n=== ランダムドメイン検知テスト ===")

# テスト用のドメイン一覧
test_domains = [
    "a1b2c3d4e5f6.example.com",     # ランダムっぽい
    "normal.login.microsoft.com",   # 正常っぽい
    "sdmksfmafmdsmkmffrmrk.cloudflare.net",  # 長いランダム
    "news.yahoo.co.jp",             # 正常
    "1a2b3c4d5e6f7g8h9i.test.net",   # ランダムっぽい
    "njecsvnds.google.com",
    "wwwsdafewqdf.example.com",
    "123.example.com",
    "qwjgkgwwwrsrd.example.com"
]

# 特徴量抽出
X_custom_test = np.array([extract_features(domain) for domain in test_domains])

# 予測（nbを使う）
preds = nb.predict(X_custom_test)
probs = nb.predict_proba(X_custom_test)

# 結果表示
for domain, pred, prob in zip(test_domains, preds, probs):
    print(f"ドメイン: {domain}")
    print(f"→ 予測ラベル: {pred}（0=正常, 1=異常）")
    print(f"→ 確率: 正常={prob[0]:.3f}, 異常={prob[1]:.3f}")
    print("-" * 40)
