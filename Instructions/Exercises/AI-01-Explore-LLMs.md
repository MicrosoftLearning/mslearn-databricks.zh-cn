---
lab:
  title: 使用 Azure Databricks 探索大型语言模型
---

# 使用 Azure Databricks 探索大型语言模型

大型语言模型 (LLM) 与 Azure Databricks 和 Hugging Face Transformer 集成后，可以成为自然语言处理 (NLP) 任务的强大资产。 Azure Databricks 提供访问、微调和部署 LLM（包括 Hugging Face 广泛库中预先训练的模型）的无缝平台。 对于模型推理，Hugging Face 的管道类简化了预先训练模型的使用，支持在 Databricks 环境中执行各种 NLP 任务。

完成本实验室大约需要 30 分钟。

> **备注**：Azure Databricks 用户界面可能会不断改进。 自编写本练习中的说明以来，用户界面可能已更改。

## 开始之前

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。
2. 使用页面顶部搜索栏右侧的 **[\>_]** 按钮在 Azure 门户中创建新的 Cloud Shell，选择 ***PowerShell*** 环境。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > **备注**：如果以前创建了使用 *Bash* 环境的 Cloud Shell，请将其切换到 ***PowerShell***。

3. 请注意，可以通过拖动窗格顶部的分隔条来调整 Cloud Shell 的大小，或使用窗格右上角的 **&#8212;**、**&#10530;** 和 **X** 图标来最小化、最大化和关闭窗格。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

4. 在 PowerShell 窗格中，输入以下命令以克隆此存储库：

     ```powershell
    rm -r mslearn-databricks -f
    git clone https://github.com/MicrosoftLearning/mslearn-databricks
     ```

5. 克隆存储库后，请输入以下命令以运行 **setup.ps1** 脚本，以在可用区域中预配 Azure Databricks 工作区：

     ```powershell
    ./mslearn-databricks/setup.ps1
     ```

6. 如果出现提示，请选择要使用的订阅（仅当有权访问多个 Azure 订阅时才会发生这种情况）。

7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 16.4 LTS <u>ML</u> 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程****。

1. 在 Azure 门户中，浏览到已由脚本创建的 msl-xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）
1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。
1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 机器学习****：已启用
    - Databricks Runtime****：16.4 LTS
    - 使用 Photon 加速****：未选定<u></u>
    - 辅助角色类型****：Standard_D4ds_v5
    - 单节点****：已选中

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 安装所需的库

1. 在群集的页面中，选择“库”**** 选项卡。

2. 选择“新安装”****。

3. 选择“PyPI”**** 作为库源，并在“包”**** 字段中键入“`transformers==4.53.0`”。

4. 选择“安装”  。

## 加载预先训练的模型

1. 在 Databricks 工作区中，转到“工作区”**** 部分。

2. 选择“创建”****，然后选择“笔记本”****。

3. 命名笔记本并验证是否已选择 `Python` 作为语言。

4. 在“连接”**** 下拉菜单中，选择之前创建的计算资源。

5. 在第一个代码单元格中，输入并运行以下代码：

    ```python
   from transformers import pipeline

   # Load the summarization model with PyTorch weights
   summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

   # Load the sentiment analysis model
   sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

   # Load the translation model
   translator = pipeline("translation_en_to_fr", model="google-t5/t5-base", revision="a9723ea")

   # Load a general purpose model for zero-shot classification and few-shot learning
   classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="d7645e1") 
    ```
     
    这将加载本练习中显示的 NLP 任务所需的所有模型。

### 汇总文本

汇总管道可生成较长文本的简明摘要。 通过指定长度范围（`min_length`，`max_length`） 以及是否使用采样 (`do_sample`)，我们可以确定生成摘要的精确程度或创意性。 

1. 在新的代码单元格中，输入以下代码：

     ```python
    text = "Large language models (LLMs) are advanced AI systems capable of understanding and generating human-like text by learning from vast datasets. These models, which include OpenAI's GPT series and Google's BERT, have transformed the field of natural language processing (NLP). They are designed to perform a wide range of tasks, from translation and summarization to question-answering and creative writing. The development of LLMs has been a significant milestone in AI, enabling machines to handle complex language tasks with increasing sophistication. As they evolve, LLMs continue to push the boundaries of what's possible in machine learning and artificial intelligence, offering exciting prospects for the future of technology."
    summary = summarizer(text, max_length=75, min_length=25, do_sample=False)
    print(summary)
     ```

2. 运行单元格以查看汇总的文本。

### 分析情绪

情绪分析管道确定给定文本的情绪。 它将文本分为正面、负面或中性等类别。

1. 在新的代码单元格中，输入以下代码：

     ```python
    text = "I love using Azure Databricks for NLP tasks!"
    sentiment = sentiment_analyzer(text)
    print(sentiment)
     ```

2. 运行单元格以查看情绪分析结果。

### 翻译文本

翻译管道将文本从一种语言转换为另一种语言。 在本练习中，使用的任务是 `translation_en_to_fr`，这意味着它会将任何给定文本从英语翻译为法语。

1. 在新的代码单元格中，输入以下代码：

     ```python
    text = "Hello, how are you?"
    translation = translator(text)
    print(translation)
     ```

2. 运行单元格以查看法语翻译文本。

### 对文本进行分类

零样本分类管道允许模型将文本分类为训练期间未见过的类别。 因此，它需要预定义标签作为 `candidate_labels` 参数。

1. 在新的代码单元格中，输入以下代码：

     ```python
    text = "Azure Databricks is a powerful platform for big data analytics."
    labels = ["technology", "health", "finance"]
    classification = classifier(text, candidate_labels=labels)
    print(classification)
     ```

2. 运行单元格以查看零样本分类结果。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
