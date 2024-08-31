# 练习 05 - 使用 Azure Databricks 和 Azure OpenAI 评估大型语言模型

## 目标
在本练习中，你将了解如何使用 Azure Databricks 和 GPT-4 OpenAI 模型评估大型语言模型 (LLM)。 这包括设置环境、定义评估指标，以及分析模型在特定任务上的表现。

## 要求
一个有效的 Azure 订阅。 如果没有该帐户，可以注册[免费试用版](https://azure.microsoft.com/en-us/free/)。

## 步骤 1：预配 Azure Databricks
- 登录到 Azure 门户：
    1. 转到 Azure 门户，然后使用凭据登录。
- 创建 Databricks 服务：
    1. 导航到“创建资源”>“分析”>“Azure Databricks”。
    2. 输入所需的详细信息，例如工作区名称、订阅、资源组（新建或选择现有），以及位置。
    3. 选择定价层（为此实验室选择标准定价层）。
    4. 单击“查看 + 创建”，然后在通过验证后单击“创建”。

## 步骤 2：启动工作区并创建群集
- 启动 Databricks 工作区：
    1. 部署完成后，转到资源并单击“启动工作区”。
- 创建 Spark 群集：
    1. 在 Databricks 工作区中，单击边栏上的“计算”，然后单击“创建计算”。
    2. 指定群集名称并选择 Spark 的运行时版本。
    3. 根据可用选项选择将辅助角色类型作为“标准”和节点类型（选择较小的节点以提高成本效益）。
    4. 单击“创建计算”。

## 步骤 3：安装所需的库

- 登录到你的 Azure Databricks 工作区。
- 创建新的笔记本并选择默认群集。
- 要安装所需的 Python 库，请运行以下命令：

```python
%pip install openai
%pip install transformers
%pip install datasets
```

- 配置 OpenAI API 密钥：
    1. 将 Azure OpenAI API 密钥添加到笔记本：

    ```python
    import openai
    openai.api_key = "your-openai-api-key"
    ```

## 步骤 4：定义评估指标
- 定义常见评估指标：
    1. 在此步骤中，你将根据任务定义评估指标，例如 Perplexity、BLEU 分数、ROUGE 分数和准确度。

    ```python
    from datasets import load_metric

    # Example: Load BLEU metric
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")

    def compute_bleu(predictions, references):
        return bleu_metric.compute(predictions=predictions, references=references)

    def compute_rouge(predictions, references):
        return rouge_metric.compute(predictions=predictions, references=references)
    ```

- 定义特定于任务的指标：
    1. 根据用例，定义其他相关指标。 例如，对于情绪分析，请定义准确度：

    ```python
    from sklearn.metrics import accuracy_score

    def compute_accuracy(predictions, references):
        return accuracy_score(references, predictions)
    ```

## 步骤 2：准备数据集
- 加载数据集
    1. 使用数据集库加载预定义的数据集。 对于本实验室，你可以使用简单的数据集（如 IMDB 电影评论数据集）进行情绪分析：

    ```python
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    test_data = dataset["test"]
    ```

- 预处理数据
    1. 标记并预处理数据集，使其与 GPT-4 模型兼容：

    ```python
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = test_data.map(preprocess_function, batched=True)
    ```

## 步骤 6：评估 GPT-4 模型
- 生成预测：
    1. 使用 GPT-4 模型对测试数据集生成预测

    ```python
    def generate_predictions(input_texts):
    predictions = []
    for text in input_texts:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=text,
            max_tokens=50
        )
        predictions.append(response.choices[0].text.strip())
    return predictions

    input_texts = tokenized_data["text"]
    predictions = generate_predictions(input_texts)
    ```

- 计算评估指标
    1. 根据 GPT-4 模型生成的预测计算评估指标

    ```python
    # Example: Compute BLEU and ROUGE scores
    bleu_score = compute_bleu(predictions, tokenized_data["text"])
    rouge_score = compute_rouge(predictions, tokenized_data["text"])

    print("BLEU Score:", bleu_score)
    print("ROUGE Score:", rouge_score)
    ```

    2. 如果要评估特定任务（如情绪分析），请计算准确度

    ```python
    # Assuming binary sentiment labels (positive/negative)
    actual_labels = test_data["label"]
    predicted_labels = [1 if "positive" in pred else 0 for pred in predictions]

    accuracy = compute_accuracy(predicted_labels, actual_labels)
    print("Accuracy:", accuracy)
    ```

## 步骤 7：分析和解释结果

- 解释结果
    1. 分析 BLEU、ROUGE 或准确度分数，以确定 GPT-4 模型针对你的任务表现如何。
    2. 讨论任何差异的潜在原因，并考虑改进模型性能的方法（例如微调、更多数据预处理）。

- 可视化结果
    1. 或者，可以使用 Matplotlib 或任何其他可视化工具可视化结果。

    ```python
    import matplotlib.pyplot as plt

    # Example: Plot accuracy scores
    plt.bar(["Accuracy"], [accuracy])
    plt.ylabel("Score")
    plt.title("Model Evaluation Metrics")
    plt.show()
    ```

## 步骤 8：试验不同方案

- 使用不同的提示进行试验
    1. 修改提示结构以查看它如何影响模型的性能。

- 评估不同的数据集
    1. 尝试使用不同的数据集评估 GPT-4 模型在不同任务中的多功能性。

- 优化评估指标
    1. 使用温度、最大标记等超参数进行试验，以优化评估指标。

## 步骤 9：清理资源
- 终止群集：
    1. 返回到“计算”页，选择群集，然后单击“终止”以停止群集。

- 可选：删除 Databricks 服务：
    1. 为了避免产生进一步的费用，如果此实验室不属于大型项目或学习路径，请考虑删除 Databricks 工作区。

本练习指导你完成使用 Azure Databricks 和 GPT-4 OpenAI 模型评估大型语言模型的过程。 完成本练习后，你将深入了解模型的性能，并了解如何针对特定任务改进和微调模型。