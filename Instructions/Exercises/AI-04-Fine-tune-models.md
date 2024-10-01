---
lab:
  title: 使用 Azure Databricks 和 Azure OpenAI 微调大型语言模型
---

# 使用 Azure Databricks 和 Azure OpenAI 微调大型语言模型

借助 Azure Databricks，用户现在可以通过使用自己的数据微调 LLM 来利用 LLM 的强大功能完成专门任务，从而增强特定于域的性能。 若要使用 Azure Databricks 微调语言模型，可以利用 Mosaic AI 模型训练界面，从而简化完整模型微调过程。 通过此功能，可以使用自定义数据微调模型，并将检查点保存到 MLflow，确保对微调的模型保持完全控制。

完成本实验室大约需要 60 分钟。

## 开始之前

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure OpenAI 资源

如果还没有 Azure OpenAI 资源，请在 Azure 订阅中预配 Azure OpenAI 资源。

1. 登录到 Azure 门户，地址为 ****。
2. 请使用以下设置创建 Azure OpenAI 资源：
    - 订阅****：*选择已被批准访问 Azure OpenAI 服务的 Azure 订阅*
    - **资源组**：*创建或选择资源组*
    - 区域****：从以下任何区域中进行随机选择******\*
        - 美国东部 2
        - 美国中北部
        - 瑞典中部
        - 瑞士西部
    - **名称**：所选项的唯一名称**
    - **定价层**：标准版 S0

> \* Azure OpenAI 资源受区域配额约束。 列出的区域包括本练习中使用的模型类型的默认配额。 在与其他用户共享订阅的情况下，随机选择一个区域可以降低单个区域达到配额限制的风险。 如果稍后在练习中达到配额限制，你可能需要在不同的区域中创建另一个资源。

3. 等待部署完成。 然后在 Azure 门户中转至部署的 Azure OpenAI 资源。

4. 在左窗格的“**资源管理**”下，选择“**密钥和终结点**”。

5. 复制终结点和其中一个可用密钥，因为稍后将在本练习中使用它。

6. 启动 Cloud Shell 并运行 `az account get-access-token` 以获取用于 API 测试的临时授权令牌。 将其与之前复制的终结点和密钥放在一起。

## 部署所需的模型

Azure 提供了一个名为 **Azure AI Studio** 的基于 Web 的门户，可用于部署、管理和探索模型。 你将通过使用 Azure OpenAI Studio 部署模型，开始探索 Azure OpenAI。

> **备注**：使用 Azure AI Studio 时，可能会显示建议你执行任务的消息框。 可以关闭这些消息框并按照本练习中的步骤进行操作。

1. 在 Azure 门户中的 Azure OpenAI 资源的“**概述**”页上，向下滚动到“**开始**”部分，然后选择转到 **Azure AI Studio** 的按钮。
   
1. 在 Azure AI Studio 的左侧窗格中，选择“**部署**”页并查看现有模型部署。 如果没有模型部署，请使用以下设置创建新的 **gpt-35-turbo** 模型部署：
    - **部署名称**：*gpt-35-turbo-0613*
    - 模型：gpt-35-turbo
    - **模型版本**：0613
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

1. 在群集的页面中，选择“库”**** 选项卡。

2. 选择“新安装”****。

3. 选择 **PyPI** 作为库源并安装以下 Python 包：
   - `numpy==2.1.0`
   - `requests==2.32.3`
   - `openai==1.42.0`
   - `tiktoken==0.7.0`

## 创建笔记本并引入数据

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
   
1. 为笔记本命名并在“**连接**”下拉列表中，选择群集（如果尚未选择）。 如果群集未运行，可能需要一分钟左右才能启动。

2. 在笔记本的第一个单元格中输入以下代码，该代码使用 *shell* 命令将数据文件从 GitHub 下载到群集使用的文件系统中。

     ```python
    %sh
    rm -r /dbfs/fine_tuning
    mkdir /dbfs/fine_tuning
    wget -O /dbfs/fine_tuning/training_set.jsonl https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/training_set.jsonl
    wget -O /dbfs/fine_tuning/validation_set.jsonl https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/validation_set.jsonl
     ```

3. 在新单元格中，使用本练习开始时复制的访问信息运行以下代码，以便在使用 Azure OpenAI 资源时分配用于身份验证的持久性环境变量：

     ```python
    import os

    os.environ["AZURE_OPENAI_API_KEY"] = "your_openai_api_key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "your_openai_endpoint"
    os.environ["TEMP_AUTH_TOKEN"] = "your_access_token"
     ```
     
## 验证令牌计数

`training_set.jsonl` 和 `validation_set.jsonl` 都是由 `user` 和 `assistant` 之间的不同对话示例组成，它们将用作训练和验证微调模型的数据点。 各个示例需要保持在 `gpt-35-turbo` 模型的 4096 个令牌的输入令牌限制内。

1. 在新单元格中，运行以下代码来验证每个文件的令牌计数：

   ```python
    import json
    import tiktoken
    import numpy as np
    from collections import defaultdict

    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n##### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")

    files = ['/dbfs/fine_tuning/training_set.jsonl', '/dbfs/fine_tuning/validation_set.jsonl']

    for file in files:
        print(f"File: {file}")
        with open(file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        total_tokens = []
        assistant_tokens = []

        for ex in dataset:
            messages = ex.get("messages", {})
            total_tokens.append(num_tokens_from_messages(messages))
            assistant_tokens.append(num_assistant_tokens_from_messages(messages))

        print_distribution(total_tokens, "total tokens")
        print_distribution(assistant_tokens, "assistant tokens")
        print('*' * 75)
   ```

## 将微调文件上传到 Azure OpenAI

在开始微调模型之前，需要初始化 OpenAI 客户端并将微调文件添加到其环境中，生成将用于初始化作业的文件 ID。

1. 在新单元格中，运行以下代码：

     ```python
    import os
    from openai import AzureOpenAI

    client = AzureOpenAI(
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),
      api_version = "2024-05-01-preview"  # This API version or later is required to access seed/events/checkpoint features
    )

    training_file_name = '/dbfs/fine_tuning/training_set.jsonl'
    validation_file_name = '/dbfs/fine_tuning/validation_set.jsonl'

    training_response = client.files.create(
        file = open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file = open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)
     ```

## 提交微调作业

现在微调文件已成功上传，可以提交微调训练作业。 需要一个多小时才能完成训练的情况并不罕见。 训练完成后，可以通过在左窗格中选择“**微调**”选项，就可以在 Azure AI Studio 中看到结果。

1. 在新单元格中，运行以下代码以启动微调训练作业：

     ```python
    response = client.fine_tuning.jobs.create(
        training_file = training_file_id,
        validation_file = validation_file_id,
        model = "gpt-35-turbo-0613",
        seed = 105 # seed parameter controls reproducibility of the fine-tuning job. If no seed is specified one will be generated automatically.
    )

    job_id = response.id
     ```

`seed` 参数控制微调作业的可重现性。 传入相同的种子和作业参数应会产生相同的结果，但在极少数情况下可能会有差异。 如果未指定种子，则会自动生成一个种子。

2. 在新单元格中，可以运行以下代码来监控微调作业的状态：

     ```python
    print("Job ID:", response.id)
    print("Status:", response.status)
     ```

3. 作业状态更改为 `succeeded` 后，运行以下代码以获取最终结果：

     ```python
    response = client.fine_tuning.jobs.retrieve(job_id)

    print(response.model_dump_json(indent=2))
    fine_tuned_model = response.fine_tuned_model
     ```
   
## 部署微调的模型

现在已经有了经过微调的模型，可以将其部署为自定义模型，然后就像使用任何其他已部署的模型一样，在 Azure AI Studio 的“**聊天**操场”中或通过聊天补全 API 使用它。

1. 在新单元格中，运行以下代码以部署经过微调的模型：
   
     ```python
    import json
    import requests

    token = os.getenv("TEMP_AUTH_TOKEN")
    subscription = "<YOUR_SUBSCRIPTION_ID>"
    resource_group = "<YOUR_RESOURCE_GROUP_NAME>"
    resource_name = "<YOUR_AZURE_OPENAI_RESOURCE_NAME>"
    model_deployment_name = "gpt-35-turbo-ft"

    deploy_params = {'api-version': "2023-05-01"}
    deploy_headers = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}

    deploy_data = {
        "sku": {"name": "standard", "capacity": 1},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": "<YOUR_FINE_TUNED_MODEL>",
                "version": "1"
            }
        }
    }
    deploy_data = json.dumps(deploy_data)

    request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

    print('Creating a new deployment...')

    r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

    print(r)
    print(r.reason)
    print(r.json())
     ```

2. 在新单元格中，运行以下代码，以在聊天补全调用中使用自定义模型：
   
     ```python
    import os
    from openai import AzureOpenAI

    client = AzureOpenAI(
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),
      api_version = "2024-02-01"
    )

    response = client.chat.completions.create(
        model = "gpt-35-turbo-ft", # model = "Custom deployment name you chose for your fine-tuning model"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure AI services support this too?"}
        ]
    )

    print(response.choices[0].message.content)
     ```
 
## 清理

使用完 Azure OpenAI 资源后，请记得在位于 `https://portal.azure.com` 的 **Azure 门户** 中删除部署或整个资源。

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
