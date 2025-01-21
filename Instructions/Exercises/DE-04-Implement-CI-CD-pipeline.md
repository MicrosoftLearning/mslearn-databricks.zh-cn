---
lab:
  title: 使用 Azure Databricks 实现 CI/CD 工作流
---

# 使用 Azure Databricks 实现 CI/CD 工作流

使用 GitHub Actions 和 Azure Databricks 实现 CI/CD 工作流可以简化开发过程并增强自动化。 GitHub Actions 提供一个功能强大的平台，用于自动执行软件工作流，包括持续集成 (CI) 和持续交付 (CD)。 与 Azure Databricks 集成时，这些工作流可以执行复杂的数据任务，例如运行笔记本或将更新部署到 Databricks 环境。 例如，可以使用 GitHub Actions 自动部署 Databricks 笔记本、管理 Databricks 文件系统上传，并在工作流中设置 Databricks CLI。 这种集成有助于实现更高效且更不易出错的开发周期，尤其是对于数据驱动的应用程序。

完成本实验室大约需要 30 分钟。

> **备注**：Azure Databricks 用户界面可能会不断改进。 自编写本练习中的说明以来，用户界面可能已更改。

> **备注：** 要完成本练习，需要 GitHub 帐户并在本地计算机上安装 Git 客户端（例如 Git 命令行工具）。

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

7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待期间，请查看 Azure Databricks 文档中的[使用 Databricks 资产捆绑包和 GitHub Actions 运行 CI/CD 工作流](https://learn.microsoft.com/azure/databricks/dev-tools/bundles/ci-cd-bundles)一文。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 **msl-xxxxxxx*** 资源组（或包含现有 Azure Databricks 工作区的资源组）

1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。

1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+)新建**”任务，然后选择“**群集**”（可能需要查看“**更多**”子菜单）。

1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - **Databricks 运行时版本**：13.3 LTS（Spark 3.4.1、Scala 2.12）或更高版本
    - 使用 Photon 加速：已选择
    - **节点类型**：Standard_D4ds_v5
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

    > 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`
   
## 设置 Github 存储库

将 GitHub 存储库连接到 Azure Databricks 工作区后，可以在 GitHub Actions 中设置 CI/CD 管道，对存储库所做的任何更改都会触发这些管道。

1. 转到 [GitHub 帐户](https://github.com/)，并使用合适的名称（例如 *databricks-cicd-repo*）创建一个新的专用存储库。

1. 使用 [git clone](https://git-scm.com/docs/git-clone) 命令将空存储库克隆到本地计算机。

1. 将本练习所需的文件下载到存储库的本地文件夹：
   - [CSV 文件](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales.csv)
   - [Databricks Notebook](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales_notebook.py)
   - [作业配置文件](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/job-config.json)

1. 在本地 Git 存储库克隆中，[添加](https://git-scm.com/docs/git-add)这些文件。 现在，[提交](https://git-scm.com/docs/git-commit)更改并将其[推送](https://git-scm.com/docs/git-push)到存储库。

## 设置存储库密码

机密是在组织、存储库或存储库环境中创建的变量。 创建的机密可在 GitHub Actions 工作流中使用。 仅当在工作流中显式包含机密时，GitHub Actions 才能读取该机密。

由于 GitHub Actions 工作流需要从 Azure Databricks 访问资源，因此身份验证凭据将存储为要用于 CI/CD 管道的加密变量。

在创建存储库密码之前，需要在 Azure Databricks 中生成个人访问令牌：

1. 在 Azure Databricks 工作区中，选择顶部栏中的*用户*图标，然后从下拉列表中选择“**设置**”。

1. 在“**开发人员**”页上，选择“**访问令牌**”旁边的“**管理**”。

1. 选择“**生成新令牌**”，然后选择“**生成**”。

1. 复制显示的令牌并将其粘贴到稍后可引用的某个位置。 然后选择“完成”。

1. 现在，在 GitHub 存储库页中，选择“**设置**”选项卡。

   ![GitHub 设置选项卡](./images/github-settings.png)

1. 在左侧边栏中，选择“**密码和变量**”，然后选择“**操作**”。

1. 选择“**新增存储库密码**”，然后添加以下每个变量：
   - **名称：** DATABRICKS_HOST **密码：** 添加 Databricks 工作区的 URL。
   - **名称：** DATABRICKS_TOKEN **密码：** 添加之前生成的访问令牌。

## 设置 CI 管道

现在，你已存储了从 GitHub 访问 Azure Databricks 工作区所需的凭据，接下来将创建工作流来自动执行数据引入。 每当存储库的主分支推送了提交或合并了拉取请求时，都会部署此工作流。 此工作流将确保 Azure Databricks 工作区中使用的数据源始终是最新的。

1. 在存储库页，选择“**操作**”选项卡。

    ![GitHub Actions 选项卡](./images/github-actions.png)

1. 选择**自己设置工作流**并输入以下代码：

     ```yaml
    name: CI Pipeline for Azure Databricks

    on:
      push:
        branches:
          - main
      pull_request:
        branches:
          - main

    jobs:
      deploy:
        runs-on: ubuntu-latest

        steps:
        - name: Checkout code
          uses: actions/checkout@v3

        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.x'

        - name: Install Databricks CLI
          run: |
            pip install databricks-cli

        - name: Configure Databricks CLI
          run: |
            databricks configure --token <<EOF
            ${{ secrets.DATABRICKS_HOST }}
            ${{ secrets.DATABRICKS_TOKEN }}
            EOF

        - name: Upload sample data to DBFS
          run: databricks fs cp sample_sales.csv dbfs:/FileStore/sample_sales.csv --overwrite
     ```

    以上代码将安装和配置 Databricks CLI，并将示例数据从你的存储库复制到工作区。

1. 将工作流命名为 **CI_pipeline.yml**，然后选择“**提交更改**”。 管道将自动运行，可以在“**操作**”选项卡中检查其状态。

1. 工作流完成后，转到工作区页，选择“**+ 新建**”并创建新笔记本。
  
1. 在第一个代码单元格中，运行以下代码：

     ```python
    %fs
    ls FileStore
     ``` 

    在输出中，可以验证示例数据现在是否存在于 Databricks 文件系统中并可在工作区中使用。

## 设置 CD 管道

设置 CI 工作流以自动执行数据引入后，将创建第二个工作流来自动执行数据处理。 CD 工作流将笔记本作为作业运行执行，并在 Azure Databricks 工作区的“**作业运行**”页中注册其输出。 笔记本包含数据在使用之前所需的所有转换步骤。

1. 转到工作区页，选择“**计算**”，然后选择群集。

1. 在群集的页面中，打开“**终止**”按钮左侧的选项，然后选择“**查看 JSON**”。 复制群集的 ID，因为在工作流中设置作业运行时将需要该 ID。

1. 打开存储库中的 **job-config.json**，并将 *your_cluster_id* 替换为刚刚复制的群集 ID。 此外，将 */Workspace/Users/your_username/your_notebook* 替换为工作区中希望存储管道中使用的笔记本的路径。 提交更改。

    > **备注：** 如果转到“**操作**”选项卡，将会看到 CI 管道再次开始运行。 由于它应该在推送提交时触发，因此更改 *job-config.json* 将按预期部署管道。

1. 在“**操作**”选项卡中，创建名为 **CD_pipeline.yml** 的新工作流，并输入以下代码：

     ```yaml
    name: CD Pipeline for Azure Databricks

    on:
      push:
        branches:
          - main

    jobs:
      deploy:
        runs-on: ubuntu-latest

        steps:
        - name: Checkout code
          uses: actions/checkout@v3

        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.x'

        - name: Install Databricks CLI
          run: pip install databricks-cli

        - name: Configure Databricks CLI
          run: |
            databricks configure --token <<EOF
            ${{ secrets.DATABRICKS_HOST }}
            ${{ secrets.DATABRICKS_TOKEN }}
            EOF
     
        - name: Import Notebook to Workspace
          run: databricks workspace import sample_sales_notebook.py /Workspace/Users/your_username/your_notebook -l python --overwrite

          env:
            DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}

        - name: Run Databricks Job
          run: |
            databricks jobs create --json-file job-config.json
            databricks jobs run-now --job-id $(databricks jobs list | grep -m 1 'CD pipeline' | awk '{print $1}')
          env:
            DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
     ```

    在提交更改之前，请将 `/Workspace/Users/your_username/your_notebook` 替换为要在 Azure Databricks 工作区导入笔记本的文件路径。

1. 提交更改。

    此代码将再次安装和配置 Databricks CLI，将笔记本导入工作区，并创建一个将执行该笔记本的作业运行。 可以在工作区的“**工作流**”页中监视作业运行的进度。 检查输出并验证数据示例是否已加载到数据帧并进行修改以用于进一步分析。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
