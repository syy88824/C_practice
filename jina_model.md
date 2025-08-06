# Hugging Face中的embedding model

## 1. 實驗目的

將篩選後的 Windows PE 執行檔中的 `.text` 段進行 disassembly 後，使用 transformer-based embedding 模型對其 disassembled text 進行向量化，以利後續如家族分類等應用。

## 2. Chunk 切割策略

由於 embedding 模型的輸入長度有限，我們將每個檔案的 disassembled text 切割為多個 chunk，並逐一產生 embedding。

**malware**

(預留malware的ins_clean的說明)

**goodware**

以 **2048 tokens** 為單位，切成多個chunk

## 3. 選用模型介紹：`jinaai/jina-embeddings-v2-base-code`

### 模型特性

- 支援語言：english, 前、後端程式語言, assembly
- 特徵維度：768-dim
- encoding function :
  ```python
  for text_id, text in enumerate(assembly_list):
        embd = model.encode(text, add_special_tokens=False)
  ```
  ※ 依 hugging face 上的說明文件，encode function 的內容為：
  
  ```python
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    encoded_input = tokenizer(assembly_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    ```

## 4. 降維與可視化方法

為了可視化高維度的 embedding，我們嘗試了PCA, UMAP, t-SNE
在僅取每個檔案的前三個chunks來畫圖的情況下，PCA 和 UMAP 降維後畫出的圖皆無法將各 family 的資料聚集成clusters，僅有 t-SNE能看出較明顯的分群效果 
以 t-SNE 作為降維方法

### 示例圖（t-SNE）

> 插入可視化圖例，例如：
> ![t-SNE](./tsne_example.png)

## 5. embedding 模型比較策略

### 候選模型
| 模型 | 訓練資料集 | max length |
|------|------|------|
| `jinaai/jina-embeddings-v2-base-code` | github-code、allenai/c4 | 8192 |
| `microsoft/codebert-base` | github/CodeSearchNet | 以 masked LM 訓練，對自然語言友善 |
| `Salesforce/codet5-base` | Code summarization, translation | seq2seq 模型，適用於生成任務 |

- 固定若干個樣本（如每類 malware 隨機選取 5 個檔案）
- 將這些樣本的所有 chunks 個別送入不同模型取得 embeddings
- 將 embeddings 降維（t-SNE）後可視化於同一平面
- 計算群內/群間距離、Silhouette score 等
  ### Embedding 分布特性分析

- 降維畫圖
- 評估指標：Silhouette Score, Cluster Visualization

### 分類任務評估

- 分類器：
   ```python
  models_all = {
      "Logistic Regression": LogisticRegression(max_iter=300),
      "Random Forest": RandomForestClassifier(n_estimators=100),
      "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(256,32,), max_iter=500, early_stopping=True),
      "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
  }
  ```
不同模型產生的 embedding 訓練出的分類器 accuracy :
| 模型 | Logistic Regression | Random Forest | MLP Neural Network | XGBoost |
|------|------|------|------|------|
| `jinaai/jina-embeddings-v2-base-code` | 0.80 | 0.79 | 0.79 | 0.80 |
| `microsoft/codebert-base` | 0.62 | 0.58 | 0.65 | 0.59 |

## 6. 後續應用

- 

## 附錄：chunk 數據統計（擬填）

| 檔案總數 | 平均 chunk 數 | 最大 chunk 數 | 句子平均長度 |
|----------|----------------|----------------|----------------|
| xxxx     | xx             | xx             | xxxx tokens    |
