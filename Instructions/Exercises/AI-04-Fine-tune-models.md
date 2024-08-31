# 练习 04 - 使用 Azure Databricks 和 Azure OpenAI 微调大型语言模型

## 目标
本练习将指导你完成使用 Azure Databricks 和 Azure OpenAI 微调大型语言模型 (LLM) 的过程。 你将了解如何设置环境、预处理数据和微调自定义数据的 LLM，以实现特定的 NLP 任务。

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
- 在群集的“库”选项卡中，单击“安装新建”。
- 安装以下 Python 包：
    1. transformers
    2. datasets
    3. azure-ai-openai
- 或者，还可以安装任何其他必需的包，例如 torch。

### 新建笔记本
- 转到“工作区”部分，然后单击“创建”>“笔记本”。
- 为笔记本命名（例如，Fine-Tuning-GPT4），并选择 Python 作为默认语言。
- 将笔记本附加到你的群集。

## 步骤 4 - 准备数据集

- 加载数据集
    1. 可以使用适合微调任务的任何文本数据集。 例如，让我们使用 IMDB 数据集进行情绪分析。
    2. 在笔记本中，运行以下代码来加载数据集

    ```python
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    ```

- 预处理数据集
    1. 使用转换器库中的 tokenizer 标记文本数据。
    2. 在笔记本中，添加以下代码：

    ```python
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    ```

- 准备数据进行微调
    1. 将数据拆分为训练数据集和验证数据集。
    2. 在笔记本中，添加：

    ```python
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
    ```

## 步骤 5 - 微调 GPT-4 模型

- 设置 OpenAI API
    1. 需要 Azure OpenAI API 密钥和终结点。
    2. 在笔记本中，设置 API 凭据：

    ```python
    import openai

    openai.api_type = "azure"
    openai.api_key = "YOUR_AZURE_OPENAI_API_KEY"
    openai.api_base = "YOUR_AZURE_OPENAI_ENDPOINT"
    openai.api_version = "2023-05-15"
    ```
- 微调模型
    1. 可以通过调整超参数并对特定数据集继续执行训练过程来实施 GPT-4 微调。
    2. 微调可能更为复杂，可能需要批处理数据、自定义训练循环等。
    3. 使用以下内容作为基本模板：

    ```python
    from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
    )

    trainer.train()
    ```
    4. 此代码提供用于训练的基本框架。 需要针对特定情况调整参数和数据集。

- 监视训练过程
    1. Databricks 允许通过笔记本界面和集成工具（如 MLflow）监视训练过程，以便进行跟踪。

## 步骤 6：评估微调模型

- 生成预测
    1. 微调后，对评估数据集生成预测。
    2. 在笔记本中，添加：

    ```python
    predictions = trainer.predict(small_eval_dataset)
    print(predictions)
    ```

- 评估模型性能
    1. 使用准确度、精准率、召回率和 F1 分数等指标来评估模型。
    2. 示例：

    ```python
    from sklearn.metrics import accuracy_score

    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy}")
    ```

- 保存经过微调的模型
    1. 将微调后的模型保存到 Azure Databricks 环境或 Azure 存储中，以供将来使用。
    2. 示例：

    ```python
    model.save_pretrained("/dbfs/mnt/fine-tuned-gpt4/")
    ```

## 步骤 7：部署微调模型
- 打包模型以进行部署
    1. 将模型转换为与 Azure OpenAI 或其他部署服务兼容的格式。

- 部署模型
    1. 通过 Azure 机器学习或直接向 OpenAI 终结点注册模型，使用 Azure OpenAI 进行部署。

- 测试已部署的模型
    1. 运行测试，确保部署的模型按预期运行，并与应用程序顺利集成。

## 步骤 8：清理资源
- 终止群集：
    1. 返回到“计算”页，选择群集，然后单击“终止”以停止群集。

- 可选：删除 Databricks 服务：
    1. 为了避免产生进一步的费用，如果此实验室不属于大型项目或学习路径，请考虑删除 Databricks 工作区。

本练习提供了有关使用 Azure Databricks 和 Azure OpenAI 微调 GPT-4 等大型语言模型的综合指南。 通过执行这些步骤，你将能够针对特定任务微调模型，评估其性能，并在实际应用中进行部署。

