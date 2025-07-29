---
lab:
  title: 使用 AutoML 训练模型
---

# 使用 AutoML 训练模型

AutoML 是 Azure Databricks 的一项功能，它会尝试将多种算法和参数用于你的数据来训练最佳的机器学习模型。

完成此练习大约需要 45 分钟。

> **备注**：Azure Databricks 用户界面可能会不断改进。 自编写本练习中的说明以来，用户界面可能已更改。

## 开始之前

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **注意**：就本练习来说，你需要一个**高级** Azure Databricks 工作区，该工作区位于某个支持*模型服务*的区域中。 有关区域 Azure Databricks 功能的详细信息，请参阅 [Azure Databricks 区域](https://learn.microsoft.com/azure/databricks/resources/supported-regions)。 如果你已在合适的区域拥有*高级*或*试用* Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

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
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的[什么是 AutoML？](https://learn.microsoft.com/azure/databricks/machine-learning/automl/)一文。

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

## 将训练数据上传到 SQL 仓库

若要使用 AutoML 训练机器学习模型，需要上传训练数据。 在本练习中，你将训练一个模型，根据观察（包括位置和身体测量）将企鹅分为三个物种之一。 你要将包含物种标签的训练数据加载到 Azure Databricks 数据仓库中的表中。

1. 在 Azure Databricks 门户中你的工作区的边栏中，选择“SQL”下的“SQL 仓库”。********
1. 请注意，工作区已包含名为“**无服务器初学者仓库**”的 SQL 仓库。
1. 在 SQL 仓库的“操作”(&#8285;) 菜单中，选择“编辑”  。 然后将“群集大小”属性设置为 2X-Small 并保存更改 。
1. 使用“开始”按钮启动 SQL 仓库（可能需要一两分钟）。

    > **注意**：如果 SQL 仓库无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 有关详细信息，请参阅[所需的 Azure vCPU 配额](https://docs.microsoft.com/azure/databricks/sql/admin/sql-endpoints#required-azure-vcpu-quota)。 如果发生这种情况，可以在仓库启动失败时尝试请求增加配额，如错误消息中所示。 或者，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

1. 将 [**penguins.csv**](https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv) 文件从 `https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv` 下载到本地计算机，并将其另存为 **penguins.csv**。
1. 在 Azure Databricks 工作区门户的边栏中，选择“**(+)新建**”，然后选择“**添加或数据**”。 在“**添加数据**”页中，选择“**创建或修改表**”，并将下载的 **penguins.csv** 文件上传到计算机。
1. 在“**从文件上传创建或修改表**”页中，选择 **default** 架构并将表名设置为 **penguins**。 然后选择“创建表”。
1. 创建表后，查看其详细信息。

## 创建 AutoML 试验

有了一些数据后，就可以将其与 AutoML 一起使用来训练模型。

1. 在左侧边栏中，选择“试验”****。
1. 在“**试验**”页上，找到“**分类**”图块并选择“**开始训练**”。
1. 配置 AutoML 试验，设置如下：
    - **群集**：*选择群集*
    - **输入训练数据集**：*浏览到“**default**”数据库并选择“**penguins**”表*
    - **预测目标**：物种
    - **试验名称**：Penguin-classification
    - **高级配置**：
        - **评估指标**：精准率
        - **训练框架**：lightgbm、sklearn、xgboost
        - **Timeout**：5
        - **训练/验证/测试拆分的时间列**：留空**
        - **正标签**：留空**
        - **中间数据存储位置**：MLflow 工件
1. 使用“**启动 AutoML**”按钮启动试验。 关闭显示的任何信息对话框。
1. 等待试验完成。 可以查看在“**运行**”选项卡下生成的运行的详细信息。
1. 五分钟后，试验将结束。 刷新运行后将在列表顶部显示带来效果最佳的模型（基于所选的*精准率*指标）的运行。

## 部署性能最佳的模型

运行 AutoML 试验后，可以探索其生成的最佳性能模型。

1. 在“Penguin-classification”试验页中，选择“查看最佳模型的笔记本”，在新浏览器标签页中打开用于训练模型的笔记本。********
1. 滚动浏览笔记本中的单元格，注意用于训练模型的代码。
1. 关闭包含笔记本的浏览器选项卡，返回到“**Penguin-classification**”试验页。
1. 在运行列表中，选择第一个运行的名称（生成了最佳模型的那个）以将其打开。
1. 在“**工件**”部分中，请注意，模型已保存为 MLflow 工件。 然后使用“**注册模型**”按钮将模型注册为名为“**Penguin-Classifier**”的新模型。
1. 在左侧边栏中，切换到“**模型**”页。 然后选择刚刚注册的“**Penguin-Classifier**”模型。
1. 在“**Penguin-Classifier**”页面中，使用“**使用模型进行推理**”按钮新建一个具有以下设置的实时终结点：
    - **模型**：Penguin-Classifier
    - **模型版本**：1
    - **终结点**：classify-penguin
    - **计算大小**：Small

    服务终结点托管在新群集中，创建该群集可能需要几分钟的时间。
  
1. 创建终结点后，请使用右上角的“查询终结点”**** 按钮打开一个界面，你可以在其中测试终结点。 然后在测试界面的“浏览器”**** 选项卡上，输入以下 JSON 请求并使用“发送请求”**** 按钮来调用终结点并生成预测。

    ```json
    {
      "dataframe_records": [
      {
         "Island": "Biscoe",
         "CulmenLength": 48.7,
         "CulmenDepth": 14.1,
         "FlipperLength": 210,
         "BodyMass": 4450
      }
      ]
    }
    ```

1. 对企鹅特征的几个不同值进行试验并观察返回的结果。 然后，关闭测试界面。

## 删除终结点

当不再需要该终结点时，应将其删除，以避免不必要的成本。

在“classify-penguin”**** 终结点页面的“&#8285;”**** 菜单中，选择“删除”****。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。

> **详细信息**：有关详细信息，请参阅 Azure Databricks 文档中的“[Databricks AutoML 的工作原理](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl/how-automl-works)”。
