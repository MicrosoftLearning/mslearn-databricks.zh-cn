# 练习 03 - 使用 Azure Databricks 和 GPT-4 通过 LangChain 进行多阶段推理

## 目标
本练习旨在指导你在 Azure Databricks 上使用 LangChain 构建多阶段推理系统。 你将了解如何创建矢量索引、存储嵌入、生成基于检索器的链、构造图像生成链，以及最后使用 GPT-4 OpenAI 模型将它们合并到多链系统中。

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

- 在工作区中打开一个新笔记本。
- 使用以下命令安装必要的库：

```python
%pip install langchain openai faiss-cpu
```

- 配置 OpenAI API

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

## 步骤 4：创建矢量索引和存储嵌入

- 加载数据集
    1. 加载要为其生成嵌入内容的示例数据集。 对于本实验室，我们将使用小型文本数据集。

    ```python
    sample_texts = [
        "Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform.",
        "LangChain is a framework designed to simplify the creation of applications using large language models.",
        "GPT-4 is a powerful language model developed by OpenAI."
    ]
    ```
- 生成嵌入内容
    1. 使用 OpenAI GPT-4 模型为这些文本生成嵌入内容。

    ```python
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(sample_texts)
    ``` 

- 使用 FAISS 存储嵌入内容
    1. 使用 FAISS 创建矢量索引，以提高检索效率。

    ```python
    import faiss
    import numpy as np

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    ```

## 步骤 5：生成基于检索器的链
- 定义检索器
    1. 创建一个检索器，该检索器可以搜索最相似文本的向量索引。

    ```python
    from langchain.chains import RetrievalQA
    from langchain.vectorstores.faiss import FAISS

    vector_store = FAISS(index, embeddings_model)
    retriever = vector_store.as_retriever()  
    ```

- 生成 RetrievalQA 链
    1. 使用检索器和 GPT-4 模型创建 QA 系统。
    
    ```python
    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain

    llm = OpenAI(model_name="gpt-4")
    qa_chain = load_qa_chain(llm, retriever)
    ```

- 测试 QA 系统
    1. 提出与嵌入的文本相关的问题

    ```python
    result = qa_chain.run("What is Azure Databricks?")
    print(result)
    ```

## 步骤 6：生成图像生成链

- 设置图像生成模型
    1. 使用 GPT-4 配置图像生成功能。

    ```python
    from langchain.chains import SimpleChain

    def generate_image(prompt):
        # Assuming you have an endpoint or a tool to generate images from text.
        return f"Generated image for prompt: {prompt}"

    image_generation_chain = SimpleChain(input_variables=["prompt"], output_variables=["image"], transform=generate_image)
    ```

- 测试图像生成链
    1. 基于文本提示生成图像。

    ```python
    prompt = "A futuristic city with flying cars"
    image_result = image_generation_chain.run(prompt=prompt)
    print(image_result)
    ```

## 步骤 7：将链合并到多链系统中
- 合并链
    1. 将基于检索器的 QA 链和图像生成链整合到多链系统中。

    ```python
    from langchain.chains import MultiChain

    multi_chain = MultiChain(
        chains=[
            {"name": "qa", "chain": qa_chain},
            {"name": "image_generation", "chain": image_generation_chain}
        ]
    )
    ```

- 运行多链系统
    1. 传递同时涉及文本检索和图像生成的任务。

    ```python
    multi_task_input = {
        "qa": {"question": "Tell me about LangChain."},
        "image_generation": {"prompt": "A conceptual diagram of LangChain in use"}
    }

    multi_task_output = multi_chain.run(multi_task_input)
    print(multi_task_output)
    ```

## 步骤 8：清理资源
- 终止群集：
    1. 返回到“计算”页，选择群集，然后单击“终止”以停止群集。

- 可选：删除 Databricks 服务：
    1. 为了避免产生进一步的费用，如果此实验室不属于大型项目或学习路径，请考虑删除 Databricks 工作区。

以上就是使用 Azure Databricks 通过 LangChain 进行多阶段推理的练习。