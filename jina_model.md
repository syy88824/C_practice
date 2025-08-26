# Hugging Face中的encoder

## 1. 實驗目的

將篩選後的 Windows PE 執行檔中的 `.text` 段進行 disassembly 後，使用 transformer-based embedding 模型對其 disassembled text 進行向量化，以利後續如家族分類等應用。

## 2. Chunk 切割策略
### jina model
由於 embedding 模型的輸入長度有限，我們將每個檔案的 disassembled text 切割為多個 chunk，並逐一產生 embedding。

**malware**

- 如有大量重複的 disassembled text，只留五筆完全相同的資料
- 以 **250 個 instructions** 為一個 chunk -> 避免一條 instruction 被分隔在兩個 chunks 中

**goodware**

以 **1000 tokens** 為單位，切成多個chunk

**chunk 數據統計**

| label | 檔案總數 (有多少資料被切成chunks) | 平均 chunk 數/檔 (μ±σ) | 最大 chunk 數 | unk比例 |
|----------|----------|----------------|----------------|----------------|
| xxxx     | xxxx     | xx             | xx             | xxxx tokens    |

### codebert model

## 3. 選用模型介紹：`jinaai/jina-embeddings-v2-base-code`

### 模型特性
- 為sentence transformer
- 支援語言：english, 前、後端程式語言, assembly
- 特徵維度：768-dim
- tokenizer切割：
  ```text
  Instruction: 'xor eax eax'
  Decoded Tokens: ['<s>', 'xor', ' eax', ' eax', '</s>']
  --------------------
  Instruction: 'mov dword ptr esi + hex eax'
  Decoded Tokens: ['<s>', 'mov', ' dword', ' ptr', ' esi', ' +', ' hex', ' eax', '</s>']
  ```
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
| `microsoft/codebert-base` | github/CodeSearchNet | 512 |
| `microsoft/unixcoder-base` | github/CodeSearchNet、allenai/c4 | 1024 |

  ### Embedding 分布特性分析

- 降維畫圖
  
**`jinaai/jina-embeddings-v2-base-code`**

**`microsoft/codebert-base`**

**`microsoft/unixcoder-base`**
  
<img width="1347" height="587" alt="unnamed" src="https://github.com/user-attachments/assets/ca7d0f7d-45c0-48ed-b184-4ea73e053810" />

### 下游任務評估 (分類效果)

- 分類器：
   ```python
  models = {
      "Logistic Regression": LogisticRegression(max_iter=300),
      "Random Forest": RandomForestClassifier(n_estimators=100),
      "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(256,32,), max_iter=500, early_stopping=True),
      "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
  }
  ```
- 不同模型產生的 embedding 訓練出的分類器 accuracy :
  | 模型 | Logistic Regression | Random Forest | MLP Neural Network | XGBoost |
  |------|------|------|------|------|
  | `jinaai/jina-embeddings-v2-base-code` | 0.80 | 0.79 | 0.79 | 0.80 |
  | `microsoft/codebert-base` | 0.62 | 0.58 | 0.65 | 0.59 |
  | `microsoft/unixcoder-base` | 0.49 | 0.50 | 0.50 | 0.52 |

## 6. 後續應用

### 惡意程式辨識、惡意程式家族分類
- 以 4:1 切割訓練集、測試集
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
  ```
- 訓練結果：
  | 模型 | Accuracy (Testing set) | Accuracy (Cross-Validation) | Macro f1-score | Weighted f1-score |
  |------|------|------|------|------|
  | Logistic Regression | 0.80 | 0.7808 ± 0.0173 | 0.71 | 0.78 |
  | Random Forest | 0.79 | 0.7749 ± 0.0163 | 0.70 | 0.78 |
  | MLP Neural Network | 0.79 | 0.7814 ± 0.0132 | 0.71 | 0.79 |
  | XGBoost | 0.80 | 0.7816 ± 0.0142 | 0.72 | 0.79 |
