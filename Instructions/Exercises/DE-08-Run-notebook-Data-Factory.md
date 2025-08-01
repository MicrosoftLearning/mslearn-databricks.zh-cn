---
lab:
  title: 使用 Azure 数据工厂自动化 Azure Databricks 笔记本
---

# 使用 Azure 数据工厂自动化 Azure Databricks 笔记本

可以使用 Azure Databricks 中的笔记本来执行数据工程任务，例如处理数据文件以及将数据加载到表中。 如果需要将这些任务作为数据工程管道的一部分进行协调，可以使用 Azure 数据工厂。

完成此练习大约需要 40 分钟。

> **备注**：Azure Databricks 用户界面可能会不断改进。 自编写本练习中的说明以来，用户界面可能已更改。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。
2. 使用页面顶部搜索栏右侧的 **[\>_]** 按钮在 Azure 门户中创建新的 Cloud Shell，选择 ***PowerShell*** 环境。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > **备注**：如果以前创建了使用 *Bash* 环境的 Cloud Shell，请将其切换到 ***PowerShell***。

3. 请注意，可以通过拖动窗格顶部的分隔条来调整 Cloud Shell 的大小，或使用窗格右上角的 **&#8212;**、**&#10530;** 和 **X** 图标来最小化、最大化和关闭窗格。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

4. 在 PowerShell 窗格中，输入以下命令以克隆此存储库：

    ```
    rm -r mslearn-databricks -f
    git clone https://github.com/MicrosoftLearning/mslearn-databricks
    ```

5. 克隆存储库后，请输入以下命令以运行 **setup.ps1** 脚本，以在可用区域中预配 Azure Databricks 工作区：

    ```
    ./mslearn-databricks/setup.ps1
    ```

6. 如果出现提示，请选择要使用的订阅（仅当有权访问多个 Azure 订阅时才会发生这种情况）。
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看[什么是 Azure 数据工厂？](https://docs.microsoft.com/azure/data-factory/introduction)。

## 创建 Azure 数据工厂资源

除了 Azure Databricks 工作区之外，还需要在订阅中预配 Azure 数据工厂资源。

1. 在 Azure 门户中，关闭云 shell 窗格并浏览到由安装脚本创建的 msl-*xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）。
1. 在工具栏中，选择“+ 创建”**** 并搜索 `Data Factory`。 然后使用以下设置创建新的“数据工厂”资源****：
    - **订阅：***你的订阅*
    - 资源组****：msl-xxxxxxx**（或包含现有 Azure Databricks 工作区的资源组）
    - **名称**：唯一名称，例如 adf-xxxxxxx******
    - 区域****：与 Azure databricks 工作区相同的区域（如果未列出，则为任何其他可用区域）**
    - **版本**：V2
1. 创建新资源后，验证资源组是否包含 Azure Databricks 工作区和 Azure 数据工厂资源。

## 创建笔记本

可以在 Azure Databricks 工作区中创建笔记本，运行用一系列编程语言编写的代码。 在本练习中，你将创建一个简单的笔记本，用于从文件中引入数据并将其保存在 Databricks 文件系统 (DBFS) 的文件夹中。

1. 在 Azure 门户中，浏览到已由脚本创建的 msl-xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）
1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 查看 Azure Databricks 工作区门户，请注意，左侧边栏包含可执行的各种任务的图标。
1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称 (**Untitled Notebook *[date]***) 更改为 `Process Data`。
1. 在笔记本的第一个单元格中，输入（但不运行）以下代码，为供此笔记本在其中保存数据的文件夹设置一个变量。

    ```python
   # Use dbutils.widget define a "folder" variable with a default value
   dbutils.widgets.text("folder", "data")
   
   # Now get the parameter value (if no value was passed, the default set above will be used)
   folder = dbutils.widgets.get("folder")
    ```

1. 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入（但不运行）以下代码，以便下载数据并将其保存到文件夹中：

    ```python
   import urllib3
   
   # Download product data from GitHub
   response = urllib3.PoolManager().request('GET', 'https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv')
   data = response.data.decode("utf-8")
   
   # Save the product data to the specified folder
   path = "dbfs:/{0}/products.csv".format(folder)
   dbutils.fs.put(path, data, True)
    ```

1. 在左侧边栏中，选择“工作区”**** 并确保列出 Process Data **** 笔记本。 你将使用 Azure 数据工厂将笔记本作为管道的一部分运行。

    > **注意**：笔记本几乎可以包含所需的任何数据处理逻辑。 此简单示例旨在展示关键原则。

## 启用 Azure Databricks 与 Azure 数据工厂的集成

若要从 Azure 数据工厂管道使用 Azure Databricks，需要在 Azure 数据工厂中创建一个链接服务，以便能够访问 Azure Databricks 工作区。

### 生成访问令牌

1. 在 Azure Databricks 门户的右上方菜单栏中，选择用户名，然后从下拉列表中选择“用户设置”。
1. 在“用户设置”页中，选择“开发人员” 。 然后在“访问令牌”旁边，选择“管理” 。
1. 选择“生成新令牌”，并使用注释“数据工厂”和空白生存期生成新令牌（这样令牌不会过期）。 在选择“完成”之前，请注意在显示令牌时复制令牌<u></u>。
1. 将复制的令牌粘贴到文本文件中，以便稍后在本练习中使用。

## 使用管道运行 Azure Databricks 笔记本

创建链接服务后，可以在管道中使用它来运行之前查看的笔记本。

### 创建管道

1. 在 Azure 数据工厂工作室的导航窗格中，选择“创作”。
2. 在“创作”页上的“工厂资源”窗格中，使用 + 图标添加“管道”   。
3. 在新管道的“**属性**”窗格中，将其名称更改为 `Process Data with Databricks`。 然后使用工具栏右侧的“属性”按钮（看起来类似于“&#128463;”<sub>*</sub>）隐藏“属性”窗格  。
4. 在“活动”窗格中展开“Databricks”，将“笔记本”活动拖动到管道设计器图面  。
5. 选择新的“Notebook1”活动后，在底部窗格中设置以下属性：
    - 常规：
        - **名称**：`Process Data`
    - Azure Databricks：
        - Databricks 链接服务：选择之前创建的 AzureDatabricks 链接服务**
    - 设置：
        - 笔记本路径****：浏览到“Users/your_user_name”文件夹，然后选择“处理数据”笔记本**********
        - 基参数****：** 添加名为 `folder` 且值为 `product_data` 的新参数
6. 使用管道设计器图面上方的“验证”按钮验证管道。 然后使用“全部发布”按钮发布（保存）。

### 在 Azure 数据工厂中创建链接服务

1. 返回到 Azure 门户，在 msl-*xxxxxxx* 资源组中，选择 Azure 数据工厂资源 adf*xxxxxxx*。********
2. 在“概述”页上，选择“启动工作室”以打开 Azure 数据工厂工作室 。 根据提示登录。
3. 在 Azure 数据工厂工作室中，使用 >> 图标展开左侧的导航窗格。 然后选择“管理”页。
4. 在“管理”页的“链接服务”选项卡中，选择“+ 新建”添加新的链接服务  。
5. 在“新建链接服务”窗格中，选择顶部的“计算”选项卡 。 然后选择“Azure Databricks”。
6. 继续，使用以下设置创建链接服务：
    - **名称**：`AzureDatabricks`
    - **说明**：`Azure Databricks workspace`
    - 通过集成运行时连接：AutoResolveIntegrationRuntime
    - 帐户选择方式：从 Azure 订阅
    - Azure 订阅：选择你的订阅
    - Databricks 工作区：选择你的 databricksxxxxxxx 工作区**
    - 选择群集：新建作业群集
    - Databrick 工作区 URL：自动设置为 Databricks 工作区 URL
    - 身份验证类型：访问令牌
    - 访问令牌：粘贴访问令牌
    - **群集版本**：13.3 LTS（Spark 3.4.1、Scala 2.12）
    - **群集节点类型**：Standard_D4ds_v5
    - Python 版本：3
    - 辅助角色选项：已修复
    - 辅助角色：1

### 运行管道

1. 在管道设计器图面上方，选择“添加触发器”，然后选择“立即触发” 。
2. 在“管道运行”窗格中，选择“确定”运行管道 。
3. 在左侧导航窗格中，选择“监视”，并在“管道运行”选项卡上观察“使用 Databricks 处理数据”管道。运行可能需要一段时间，因为它会动态创建 Spark 群集并运行笔记本  。 可以使用“管道运行”页上的“&#8635;刷新”按钮刷新状态 。

    > 注意：如果管道运行失败，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足，无法创建作业群集。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./setup.ps1 eastus`

4. 运行成功后，选择其名称，查看运行详细信息。 然后，在“使用 Databricks 处理数据”页的“活动运行”部分，选择“处理数据”活动，并使用其“输出”图标查看该活动的输出 JSON，应如下所示  ：

    ```json
    {
        "runPageUrl": "https://adb-..../run/...",
        "effectiveIntegrationRuntime": "AutoResolveIntegrationRuntime (East US)",
        "executionDuration": 61,
        "durationInQueue": {
            "integrationRuntimeQueue": 0
        },
        "billingReference": {
            "activityType": "ExternalActivity",
            "billableDuration": [
                {
                    "meterType": "AzureIR",
                    "duration": 0.03333333333333333,
                    "unit": "Hours"
                }
            ]
        }
    }
    ```

## 清理

如果已完成对 Azure Databricks 的探索，则现在可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
