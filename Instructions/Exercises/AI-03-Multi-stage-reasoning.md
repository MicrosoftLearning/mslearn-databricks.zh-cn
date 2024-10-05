---
lab:
  title: 使用 Azure Databricks 和 Azure OpenAI 通过 LangChain 进行多阶段推理
---

# 使用 Azure Databricks 和 Azure OpenAI 通过 LangChain 进行多阶段推理

多阶段推理是 AI 中的一种前沿方法，涉及将复杂问题分解为更小、更易于管理的阶段。 LangChain 是一个软件框架，有助于创建利用大型语言模型 (LLM) 的应用程序。 与 Azure Databricks 集成时，LangChain 允许无缝数据加载、模型包装和开发复杂的 AI 代理。 这种组合特别强大，用于处理复杂的任务，这些任务需要深入了解上下文以及跨多个步骤推理的能力。

完成本实验室大约需要 30 分钟。

## 开始之前

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure OpenAI 资源

如果还没有 Azure OpenAI 资源，请在 Azure 订阅中预配 Azure OpenAI 资源。

1. 登录到 Azure 门户，地址为 ****。
2. 请使用以下设置创建 Azure OpenAI 资源：
    - 订阅****：*选择已被批准访问 Azure OpenAI 服务的 Azure 订阅*
    - **资源组**：*创建或选择资源组*
    - 区域****：从以下任何区域中进行随机选择******\*
        - 澳大利亚东部
        - 加拿大东部
        - 美国东部
        - 美国东部 2
        - 法国中部
        - 日本东部
        - 美国中北部
        - 瑞典中部
        - 瑞士北部
        - 英国南部
    - **名称**：所选项的唯一名称**
    - **定价层**：标准版 S0

> \* Azure OpenAI 资源受区域配额约束。 列出的区域包括本练习中使用的模型类型的默认配额。 在与其他用户共享订阅的情况下，随机选择一个区域可以降低单个区域达到配额限制的风险。 如果稍后在练习中达到配额限制，你可能需要在不同的区域中创建另一个资源。

3. 等待部署完成。 然后在 Azure 门户中转至部署的 Azure OpenAI 资源。

4. 在左窗格的“**资源管理**”下，选择“**密钥和终结点**”。

5. 复制终结点和其中一个可用密钥，因为稍后将在本练习中使用它。

## 部署所需的模块

Azure 提供了一个名为 **Azure AI Studio** 的基于 Web 的门户，可用于部署、管理和探索模型。 你将通过使用 Azure OpenAI Studio 部署模型，开始探索 Azure OpenAI。

> **备注**：使用 Azure AI Studio 时，可能会显示建议你执行任务的消息框。 可以关闭这些消息框并按照本练习中的步骤进行操作。

1. 在 Azure 门户中的 Azure OpenAI 资源的“**概述**”页上，向下滚动到“**开始**”部分，然后选择转到 **Azure AI Studio** 的按钮。
   
1. 在 Azure AI Studio 的左侧窗格中，选择“**部署**”页并查看现有模型部署。 如果没有模型部署，请使用以下设置创建新的“gpt-35-turbo-16k”**** 模型部署：
    - **部署名称**：*gpt-35-turbo-16k*
    - **模型**：gpt-35-turbo-16k *（如果 16k 模型不可用，请选择 gpt-35-turbo 并相应地对部署进行命名）*
    - **模型版本**：*使用默认版本*
    - **部署类型**：标准
    - **每分钟令牌速率限制**：5K\*
    - **内容筛选器**：默认
    - **启用动态配额**：已禁用
    
1. 返回到“**部署**”页，使用以下设置创建“**text-embedding-ada-002**”模型的新部署：
    - **部署名称**：*text-embedding-ada-002*
    - **模型**：text-embedding-ada-002
    - **模型版本**：*使用默认版本*
    - **部署类型**：标准
    - **每分钟令牌速率限制**：5K\*
    - **内容筛选器**：默认
    - **启用动态配额**：已禁用

> \*每分钟 5,000 个令牌的速率限制足以完成此练习，同时也为使用同一订阅的其他人留出容量。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

1. 登录到 Azure 门户，地址为 ****。
2. 请使用以下设置创建 **Azure Databricks** 资源：
    - **订阅**：*选择用于创建 Azure OpenAI 资源的同一 Azure 订阅*
    - **资源组**：*在其中创建了 Azure OpenAI 资源的同一资源组*
    - **区域**：*在其中创建 Azure OpenAI 资源的同一区域*
    - **名称**：所选项的唯一名称**
    - **定价层**：*高级*或*试用版*

3. 选择“**查看 + 创建**”，然后等待部署完成。 然后转到资源并启动工作区。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML**<u></u>** 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在Azure 门户中，浏览到创建 Azure Databricks 工作区的资源组。
2. 单击 Azure Databricks 服务资源。
3. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

> 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

4. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。
5. 在“新建群集”页中，使用以下设置创建新群集：
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

6. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。

## 安装所需的库

1. 在 Databricks 工作区中，转到“工作区”**** 部分。

2. 选择“创建”****，然后选择“笔记本”****。

3. 为笔记本命名，然后选择“`Python`”作为语言。

4. 在第一个代码单元格中，输入并运行以下代码以安装所需的库：
   
     ```python
    %pip install langchain openai langchain_openai faiss-cpu
     ```

5. 安装完成后，在新单元格中重启内核：

     ```python
    %restart_python
     ```

6. 在新单元格中，定义将用于初始化 OpenAI 模型的身份验证参数，并将 `your_openai_endpoint` 和 `your_openai_api_key` 替换为之前从 OpenAI 资源复制的终结点和密钥：

     ```python
    endpoint = "your_openai_endpoint"
    key = "your_openai_api_key"
     ```
     
## 创建矢量索引和存储嵌入

矢量索引是一种专用数据结构，用于高效存储和检索高维矢量数据，这对于执行快速相似性搜索和最近的邻域查询至关重要。 另一方面，嵌入是对象的数字表示形式，这些对象以矢量形式捕获其含义，使计算机能够处理和理解各种类型的数据，包括文本和图像。

1. 在新单元格中，运行以下代码以加载示例数据集：

     ```python
    from langchain_core.documents import Document

    documents = [
         Document(page_content="Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform.", metadata={"date_created": "2024-08-22"}),
         Document(page_content="LangChain is a framework designed to simplify the creation of applications using large language models.", metadata={"date_created": "2024-08-22"}),
         Document(page_content="GPT-4 is a powerful language model developed by OpenAI.", metadata={"date_created": "2024-08-22"})
    ]
    ids = ["1", "2", "3"]
     ```
     
1. 在新单元格中，运行以下代码以使用 `text-embedding-ada-002` 模型生成嵌入：

     ```python
    from langchain_openai import AzureOpenAIEmbeddings
     
    embedding_function = AzureOpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        azure_endpoint=endpoint,
        openai_api_key=key,
        chunk_size=1
    )
     ```
     
1. 在新单元格中，运行以下代码，使用第一个文本示例创建矢量索引作为矢量维度的引用：

     ```python
    import faiss
      
    index = faiss.IndexFlatL2(len(embedding_function.embed_query("Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform.")))
     ```

## 生成基于检索器的链

检索器组件基于查询提取相关文档或数据。 这在需要集成大量数据以供分析的应用程序（例如在检索扩充的生成系统中）中特别有用。

1. 在新单元格中，运行以下代码来创建一个检索器，该检索器可以搜索最相似的文本的矢量索引。

     ```python
    from langchain.vectorstores import FAISS
    from langchain_core.vectorstores import VectorStoreRetriever
    from langchain_community.docstore.in_memory import InMemoryDocstore

    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=documents, ids=ids)
    retriever = VectorStoreRetriever(vectorstore=vector_store)
     ```

1. 在新单元格中，运行以下代码，使用检索器和 `gpt-35-turbo-16k` 模型创建 QA 系统：
    
     ```python
    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
     
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k",
        model_name="gpt-35-turbo-16k",
        azure_endpoint=endpoint,
        api_version="2023-03-15-preview",
        openai_api_key=key,
    )

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    prompt1 = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)

    qa_chain1 = create_retrieval_chain(retriever, chain)
     ```

1. 在新单元格中，运行以下代码来测试 QA 系统：

     ```python
    result = qa_chain1.invoke({"input": "What is Azure Databricks?"})
    print(result)
     ```

结果输出应根据示例数据集中存在的相关文档以及 LLM 生成的生成式文本显示答案。

## 将链合并到多链系统中

Langchain 是一种通用的工具，允许将多个链组合到多链系统中，从而增强语言模型的功能。 此过程涉及将各种组件串在一起，这些组件可以并行或按顺序处理输入，最终合成最终响应。

1. 在新单元格中，运行以下代码以创建第二个链

     ```python
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt2 = ChatPromptTemplate.from_template("Create a social media post based on this summary: {summary}")

    qa_chain2 = ({"summary": qa_chain1} | prompt2 | llm | StrOutputParser())
     ```

1. 在新单元格中，运行以下代码以调用具有给定输入的多阶段链：

     ```python
    result = qa_chain2.invoke({"input": "How can we use LangChain?"})
    print(result)
     ```

第一个链基于提供的示例数据集提供输入答案，而第二个链基于第一个链的输出创建社交媒体帖子。 此方法允许你通过将多个步骤链接在一起来处理更复杂的文本处理任务。

## 清理

使用完 Azure OpenAI 资源后，请记得在位于 `https://portal.azure.com` 的 **Azure 门户** 中删除部署或整个资源。

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
