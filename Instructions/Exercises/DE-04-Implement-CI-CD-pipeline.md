---
lab:
  title: 使用 Azure Databricks 实现 CI/CD 工作流
---

# 使用 Azure Databricks 实现 CI/CD 工作流

使用 Azure Databricks 和 Azure DevOps 或 Azure Databricks 和 GitHub 实现持续集成 (CI) 和持续部署 (CD) 管道涉及设置一系列自动化步骤，以确保高效集成、测试和部署代码更改。 此过程通常包括连接到 Git 存储库、使用 Azure Pipelines 运行生成的作业和单元测试代码，以及部署要在 Databricks 笔记本中使用的生成项目。 此工作流支持可靠的开发周期，从而实现符合现代 DevOps 实践的持续集成和交付。

完成本实验室大约需要 40 分钟。

>**备注：** 需要 Github 帐户和 Azure DevOps 访问权限才能完成本练习。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。

2. 使用页面顶部搜索栏右侧的 [\>_] 按钮在 Azure 门户中创建新的 Cloud Shell，在出现提示时选择“PowerShell”环境并创建存储。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > 注意：如果以前创建了使用 Bash 环境的 Cloud shell，请使用 Cloud Shell 窗格左上角的下拉菜单将其更改为“PowerShell”。

3. 请注意，可以通过拖动窗格顶部的分隔条或使用窗格右上角的 &#8212;、&#9723; 或 X 图标来调整 Cloud Shell 的大小，以最小化、最大化和关闭窗格  。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

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

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 **msl-xxxxxxx*** 资源组（或包含现有 Azure Databricks 工作区的资源组）

1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。

1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。

1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - **Databricks 运行时版本**：13.3 LTS（Spark 3.4.1、Scala 2.12）或更高版本
    - 使用 Photon 加速：已选择
    - 节点类型：Standard_DS3_v2
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

    > 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 创建笔记本并引入数据

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。 在“连接”**** 下拉列表中，选择群集（如果尚未选择）。 如果群集未运行，可能需要一分钟左右才能启动。

2. 在笔记本的第一个单元格中输入以下代码，该代码使用 *shell* 命令将数据文件从 GitHub 下载到群集使用的文件系统中。

     ```python
    %sh
    rm -r /dbfs/FileStore
    mkdir /dbfs/FileStore
    wget -O /dbfs/FileStore/sample_sales.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales.csv
     ```

3. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。
   
## 设置 GitHub 存储库和 Azure DevOps 项目

将 GitHub 存储库连接到 Azure DevOps 项目后，可以设置 CI 管道，对存储库所做的任何更改都会触发这些管道。

1. 转到 GitHub 帐户[](https://github.com/)，为项目创建新存储库。

2. 使用 `git clone` 将存储库克隆到本地计算机。

3. 将 CSV 文件下载到本地存储库并提交更改。[](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales.csv)

4. 下载 Databricks 笔记本[](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales_notebook.dbc)，该笔记本将用于读取 CSV 文件并执行数据转换。 提交更改。

5. 转到 Azure DevOps 门户[](https://azure.microsoft.com/en-us/products/devops/)并创建新项目。

6. 在 Azure DevOps 项目中，转到 Repos**** 部分，然后选择“导入”**** 以将其连接到 GitHub 存储库。

7. 在左侧栏中，导航到“项目设置”>“服务连接”****。

8. 选择“新建服务连接”，然后选择“Azure 资源管理器”。********

9. 在“身份验证方法”窗格中，选择“工作负载联合身份验证（自动）”********。 选择**下一步**。

10. 在“范围级别”中，选择“订阅”。******** 选择在其中创建了 Databricks 工作区的订阅和资源组。

11. 输入服务连接名称，然后选中“授予所有管道访问权限”选项。**** 选择“保存”。

现在，DevOps 项目有权访问 Databricks 工作区，并且可以将其连接到管道。

## 配置 CI 管道

1. 在左侧栏中，导航到“管道”并选择“创建管道”。********

2. 选择“GitHub”作为源，然后选择存储库。****

3. 在“配置管道”**** 窗格中，选择“启动管道”****，并为 CI 管道使用以下 YAML 配置：

```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.x'
    addToPath: true

- script: |
    pip install databricks-cli
  displayName: 'Install Databricks CLI'

- script: |
    databricks configure --token <<EOF
    <your-databricks-host>
    <your-databricks-token>
    EOF
  displayName: 'Configure Databricks CLI'

- script: |
    databricks fs cp dbfs:/FileStore/sample_sales.csv . --overwrite
  displayName: 'Download Sample Data from DBFS'
```

4. 将 `<your-databricks-host>` 和 `<your-databricks-token>` 替换为实际的 Databricks 主机 URL 和令牌。 这会在尝试使用 Databricks CLI 之前对其进行配置。

5. 选择**保存并运行**。

此 YAML 文件将设置一个 CI 管道，该管道由存储库 `main` 分支的更改而触发。 管道设置 Python 环境，安装 Databricks CLI，从 Databricks 工作区下载示例数据。 这是 CI 工作流的常见设置。

## 配置 CD 管道

1. 在左侧栏中，导航到“管道”>“发布”并选择“创建发布”。********

2. 选择生成管道作为项目源。

3. 添加阶段并配置要部署到 Azure Databricks 的任务：

```yaml
stages:
- stage: Deploy
  jobs:
  - job: DeployToDatabricks
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.x'
        addToPath: true

    - script: |
        pip install databricks-cli
      displayName: 'Install Databricks CLI'

    - script: |
        databricks configure --token <<EOF
        <your-databricks-host>
        <your-databricks-token>
        EOF
      displayName: 'Configure Databricks CLI'

    - script: |
        databricks workspace import_dir /path/to/notebooks /Workspace/Notebooks
      displayName: 'Deploy Notebooks to Databricks'
```

运行此管道之前，请将 `/path/to/notebooks` 替换为存储库中笔记本所在的目录的路径，以及将 `/Workspace/Notebooks` 替换为要将笔记本保存在 Databricks 工作区中的文件路径。

4. 选择**保存并运行**。

## 运行管道

1. 在本地存储库中，在 `sample_sales.csv` 文件末尾添加以下行：

     ```sql
    2024-01-01,ProductG,1,500
     ```

2. 提交更改并将其推送到远程 GitHub 存储库。

3. 存储库中的更改将触发 CI 管道。 验证 CI 管道是否已成功完成。

4. 在发布管道中创建新版本，并将笔记本部署到 Databricks。 验证笔记本是否已在 Databricks 工作区中成功部署和执行。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。







