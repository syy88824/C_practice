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

| label                 | 檔案總數 | 平均 chunk 數/檔 (μ±σ) | 最大 chunk 數 |
| --------------------- | ---- | ------------------ | ---------- |
| ADWARE.SCREENSAVER    | 33   | 80.70 ± 28.30      | 99         |
| DOWNLOADER.MORSTAR    | 24   | 33.13 ± 0.34       | 33         |
| TROJAN.AUTOIT         | 18   | 30.83 ± 28.66      | 98         |
| DOWNLOADER.LMN        | 17   | 37.24 ± 10.63      | 40         |
| TROJAN.GENERIC        | 15   | 27.48 ± 39.31      | 99         |
| TROJAN.LECNA          | 14   | 20.50 ± 7.37       | 29         |
| WEBTOOLBAR.ZANGO      | 11   | 84.55 ± 17.76      | 99         |
| WORM.AUTOIT           | 11   | 41.09 ± 36.61      | 81         |
| TROJAN.GRAFTOR        | 10   | 39.82 ± 35.64      | 99         |
| EMAIL-WORM.MYDOOM     | 10   | 24.50 ± 15.28      | 38         |
| goodware              | 9    | 68.11 ± 33.71      | 99         |
| TROJAN.DOINA          | 9    | 70.56 ± 30.38      | 85         |
| TROJAN.AGENT          | 7    | 30.44 ± 40.61      | 99         |
| ADWARE.GENERIC        | 6    | 88.50 ± 28.17      | 99         |
| TROJAN.STRICTOR       | 6    | 64.33 ± 33.74      | 99         |
| TROJAN-DROPPER.DINWOD | 6    | 9.00 ± 2.45        | 9          |
| DOWNLOADER.MEDIAGET   | 6    | 100.00 ± 0.00      | 99         |
| TROJAN.DUMP           | 6    | 15.17 ± 5.53       | 21         |
| ADWARE.GATOR          | 5    | 100.00 ± 0.00      | 99         |
| WORM.AUTORUN          | 5    | 71.00 ± 0.00       | 70         |
| TROJAN.ANDROM         | 5    | 70.83 ± 42.53      | 99         |
| TROJAN-DROPPER.ROXER  | 5    | 86.00 ± 0.00       | 85         |
| TROJAN.MACHETE        | 5    | 72.00 ± 0.00       | 71         |
| TROJAN.BARYS          | 4    | 34.67 ± 50.62      | 99         |
| TROJAN-SPY.ZBOT       | 4    | 4.29 ± 6.97        | 19         |
| TROJAN.FAREIT         | 3    | 25.80 ± 42.10      | 99         |
| TROJAN.ZBOT           | 3    | 30.50 ± 45.85      | 99         |
| TROJAN-RANSOM.BLOCKER | 1    | 2.50 ± 3.67        | 9          |
| TROJAN-PSW\.TEPFER    | 1    | 13.50 ± 30.62      | 75         |
| TROJAN.VBKRYPT        | 1    | 1.23 ± 0.83        | 3          |

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
在僅取每個檔案的第1-30個chunks來畫圖的情況下，PCA 和 UMAP 降維後畫出的圖皆無法將各 family 的資料聚集成clusters，僅有 t-SNE能看出較明顯的分群效果 
以 t-SNE 作為降維方法

**t-SNE**

<img width="1096" height="470" alt="image" src="https://github.com/user-attachments/assets/153e08ec-665e-4980-8a3b-dbd2a5e6b30b" />


**PCA**

<img width="1090" height="477" alt="image" src="https://github.com/user-attachments/assets/ce068c6d-cb70-4a38-87a2-ecaa86cbfc30" />

**Umap**

<img width="1102" height="483" alt="image" src="https://github.com/user-attachments/assets/7aef29a0-47af-4d31-9c8e-d448993ef0aa" />


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

<img width="1096" height="470" alt="image" src="https://github.com/user-attachments/assets/4c0fd44c-c98b-42a8-9fbb-5f8fa259b489" />

**`microsoft/codebert-base`**

<img width="1091" height="479" alt="image" src="https://github.com/user-attachments/assets/00c2e1ae-7d33-4446-a219-de4dd7c44627" />


**`microsoft/unixcoder-base`**
  
<img width="1347" height="587" alt="unnamed" src="https://github.com/user-attachments/assets/ca7d0f7d-45c0-48ed-b184-4ea73e053810" />

### 下游任務評估 (分類效果)

- 分類器：
   ```python
  models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(
      n_estimators=10,
      warm_start=True,
      bootstrap=True,
      n_jobs=-1,
      random_state=42,
      max_depth=None,
      min_samples_leaf=1),
    "MLP Neural Network": MLPClassifier(
      hidden_layer_sizes=(256, 32),
      activation="relu",
      learning_rate_init=1e-3,
      batch_size=256,
      max_iter=1,
      warm_start=False,
      random_state=42),
    "XGBoost": xgb.XGBClassifier(
      objective='multi:softprob',
      num_class=len(class_names),
      n_estimators=500,
      random_state=42)
  }

  ```
- 不同模型產生的 embedding 訓練出的分類器 accuracy :
  | 模型 | Logistic Regression | Random Forest | MLP Neural Network | XGBoost |
  |------|------|------|------|------|
  | `jinaai/jina-embeddings-v2-base-code` | 0.80 | 0.79 | 0.79 | 0.80 |
  | `microsoft/codebert-base` | 0.62 | 0.58 | 0.65 | 0.59 |
  | `microsoft/unixcoder-base` | 0.65 | 0.60 | 0.64 | 0.61 |

## 6. 後續應用

### 惡意程式辨識、惡意程式家族分類
- 資料切分 (training : validation : testing = 7 : 1 : 2)：
  ```python
  X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)  
  # 0.3*2/3=0.2 → test 20%，val 10%
  
  # 標準化
  scaler = StandardScaler().fit(X_train_full)
  X_tr_s  = scaler.transform(X_train_full)
  X_val_s = scaler.transform(X_val)
  X_test_s = scaler.transform(X_test)

  ```
- 訓練結果：
  
  | 模型 | Accuracy (Testing set) | Accuracy (Cross-Validation) | Macro f1-score | Weighted f1-score |
  |------|------|------|------|------|
  | Logistic Regression | 0.80 | 0.7808 ± 0.0173 | 0.71 | 0.78 |
  | Random Forest | 0.79 | 0.7749 ± 0.0163 | 0.70 | 0.78 |
  | MLP Neural Network | 0.79 | 0.7814 ± 0.0132 | 0.71 | 0.79 |
  | XGBoost | 0.80 | 0.7816 ± 0.0142 | 0.72 | 0.79 |

<img width="1095" height="749" alt="image" src="https://github.com/user-attachments/assets/29939aca-0988-4941-bd0d-1e3fccaf3c3b" />
<img width="1020" height="423" alt="image" src="https://github.com/user-attachments/assets/d137d71e-ac6f-4e50-b2f9-24307f9244e4" />


