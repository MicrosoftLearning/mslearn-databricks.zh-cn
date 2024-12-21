---
lab:
  title: 使用 Azure Databricks 的检索增强生成
---

# 使用 Azure Databricks 的检索增强生成

检索增强生成 (RAG) 是 AI 中的一种前沿方法，可通过整合外部知识源来增强大型语言模型。 Azure Databricks 提供了用于开发 RAG 应用程序的可靠平台，允许将非结构化数据转换为适合检索和响应生成的格式。 此过程涉及一系列步骤，包括了解用户的查询、检索相关数据以及使用语言模型生成响应。 Azure Databricks 提供的框架可支持快速迭代和部署 RAG 应用程序，确保提供可能包含最新信息和专有知识且特定于域的高质量响应。

完成本实验室大约需要 40 分钟。

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

7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的 [Delta Lake 简介](https://docs.microsoft.com/azure/databricks/delta/delta-intro)一文。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML**<u></u>** 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 msl-xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）
1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。
1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - Databricks Runtime 版本****：选择最新非 beta 版本运行时的 ML***<u></u>** 版本（不是****标准运行时版本），该版本符合以下条件：*
        - 不使用 GPU**
        - 包括 Scala > 2.11
        - *包括 Spark > **3.4***
    - 使用 Photon 加速****：未选定<u></u>
    - **节点类型**：Standard_D4ds_v5
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 安装所需的库

1. 在群集的页面中，选择“库”**** 选项卡。

2. 选择“新安装”****。

3. 选择“PyPI”**** 作为库源，并在“包”**** 字段中键入“`transformers==4.44.0`”。

4. 选择“安装”  。

5. 重复上述步骤以安装 `databricks-vectorsearch==0.40`。
   
## 创建笔记本并引入数据

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。 在“连接”**** 下拉列表中，选择群集（如果尚未选择）。 如果群集未运行，可能需要一分钟左右才能启动。

2. 在笔记本的第一个单元格中输入以下代码，该代码使用 *shell* 命令将数据文件从 GitHub 下载到群集使用的文件系统中。

     ```python
    %sh
    rm -r /dbfs/RAG_lab
    mkdir /dbfs/RAG_lab
    wget -O /dbfs/RAG_lab/enwiki-latest-pages-articles.xml https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/enwiki-latest-pages-articles.xml
     ```

3. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。

4. 在新单元格中，运行以下代码以根据原始数据创建数据帧：

     ```python
    from pyspark.sql import SparkSession

    # Create a Spark session
    spark = SparkSession.builder \
        .appName("RAG-DataPrep") \
        .getOrCreate()

    # Read the XML file
    raw_df = spark.read.format("xml") \
        .option("rowTag", "page") \
        .load("/RAG_lab/enwiki-latest-pages-articles.xml")

    # Show the DataFrame
    raw_df.show(5)

    # Print the schema of the DataFrame
    raw_df.printSchema()
     ```

5. 在新单元格中，运行以下代码，将 `<catalog_name>` 替换为 Unity 目录的名称（包含工作区名称和唯一后缀的目录），以便清理和预处理数据以提取相关文本字段：

     ```python
    from pyspark.sql.functions import col

    clean_df = raw_df.select(col("title"), col("revision.text._VALUE").alias("text"))
    clean_df = clean_df.na.drop()
    clean_df.write.format("delta").mode("overwrite").saveAsTable("<catalog_name>.default.wiki_pages")
    clean_df.show(5)
     ```

如果打开目录 (CTRL + Alt + C)**** 资源管理器并刷新其窗格，则会看到在默认 Unity 目录中创建的 Delta 表。

## 生成嵌入并实现矢量搜索

Databricks 的 Mosaic AI 矢量搜索是在 Azure Databricks 平台中集成的矢量数据库解决方案。 它利用分层可导航小世界 (HNSW) 算法优化嵌入的存储和检索。 它支持高效的最近的邻域搜索，其混合关键字相似性搜索功能通过组合基于矢量的搜索和基于关键字的搜索技术提供更相关的结果。

1. 在新单元格中，运行以下 SQL 查询以在创建增量同步索引之前在源表中启用“更改数据馈送”功能。

     ```python
    %sql
    ALTER TABLE <catalog_name>.default.wiki_pages SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
     ```

2. 在新单元格中，运行以下代码以创建矢量搜索索引。

     ```python
    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient()

    client.create_endpoint(
        name="vector_search_endpoint",
        endpoint_type="STANDARD"
    )

    index = client.create_delta_sync_index(
      endpoint_name="vector_search_endpoint",
      source_table_name="<catalog_name>.default.wiki_pages",
      index_name="<catalog_name>.default.wiki_index",
      pipeline_type="TRIGGERED",
      primary_key="title",
      embedding_source_column="text",
      embedding_model_endpoint_name="databricks-gte-large-en"
     )
     ```
     
如果打开目录 (CTRL + Alt + C)**** 资源管理器并刷新其窗格，则会看到在默认 Unity 目录中创建的索引。

> **备注：** 在运行下一个代码单元格之前，请验证是否已成功创建索引。 为此，请在“目录”窗格中右键单击索引，然后选择“在目录资源管理器中打开”****。 等待索引状态变为“在线”****。

3. 在新单元格中，运行以下代码以基于查询矢量搜索相关文档。

     ```python
    results_dict=index.similarity_search(
        query_text="Anthropology fields",
        columns=["title", "text"],
        num_results=1
    )

    display(results_dict)
     ```

验证输出是否找到与查询提示相关的相应 Wiki 页面。

## 使用检索的数据增强提示：

现在，我们可以通过为大型语言模型提供来自外部数据源的其他上下文来增强其功能。 这样一来，模型就可以生成更准确且上下文相关的响应。

1. 在新单元格中运行以下代码，将检索到的数据与用户的查询相结合，为 LLM 创建丰富的提示。

     ```python
    # Convert the dictionary to a DataFrame
    results = spark.createDataFrame([results_dict['result']['data_array'][0]])

    from transformers import pipeline

    # Load the summarization model
    summarizer = pipeline("summarization")

    # Extract the string values from the DataFrame column
    text_data = results.select("_2").rdd.flatMap(lambda x: x).collect()

    # Pass the extracted text data to the summarizer function
    summary = summarizer(text_data, max_length=512, min_length=100, do_sample=True)

    def augment_prompt(query_text):
        context = " ".join([item['summary_text'] for item in summary])
        return f"Query: {query_text}\nContext: {context}"

    prompt = augment_prompt("Explain the significance of Anthropology")
    print(prompt)
     ```

3. 在新单元格中运行以下代码，以使用 LLM 生成响应。

     ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=300, 
        num_return_sequences=1, 
        repetition_penalty=2.0, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
     ```

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
