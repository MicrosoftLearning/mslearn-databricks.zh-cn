---
lab:
  title: 在 Azure Databricks 中为机器学习优化超参数
---

# 在 Azure Databricks 中为机器学习优化超参数

在本练习中，你将使用 Hyperopt**** 库在 Azure Databricks 中优化机器学习模型训练的超参数。

完成此练习大约需要 30 分钟。

## 准备工作

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。
2. 使用页面顶部搜索栏右侧的 [\>_] 按钮在 Azure 门户中创建新的 Cloud Shell，在出现提示时选择“PowerShell”环境并创建存储。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > 注意：如果以前创建了使用 Bash 环境的 Cloud shell，请使用 Cloud Shell 窗格左上角的下拉菜单将其更改为“PowerShell”。

3. 请注意，可以通过拖动窗格顶部的分隔条或使用窗格右上角的 &#8212;、&#9723; 或 X 图标来调整 Cloud Shell 的大小，以最小化、最大化和关闭窗格  。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

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
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的[超参数优化](https://learn.microsoft.com/azure/databricks/machine-learning/automl-hyperparam-tuning/)一文。

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

## 创建笔记本

你将运行使用 Spark MLLib 库来训练机器学习模型的代码，因此第一步是在工作区中创建一个新笔记本。

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称（**Untitled Notebook *[日期]***）更改为“Hyperparameter Tuning”，然后在“连接”下拉列表中选择群集（如果尚未选中）****。**** 如果群集未运行，可能需要一分钟左右才能启动。

## 引入数据

本练习的场景基于对南极洲企鹅的观察，目的是训练一个机器学习模型，用于根据观察到的企鹅的位置和身体度量来预测其种类。

> **引文**：本练习中使用的企鹅数据集是 [Kristen Gorman 博 士](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)和[长期生态研究网络](https://lternet.edu/)成员[南极洲帕默站](https://pal.lternet.edu/)收集并提供的数据的子集。

1. 在笔记本的第一个单元格中输入以下代码，该代码使用 shell** 命令将企鹅数据从 GitHub 下载到群集使用的文件系统中。

    ```bash
    %sh
    rm -r /dbfs/hyperopt_lab
    mkdir /dbfs/hyperopt_lab
    wget -O /dbfs/hyperopt_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv
    ```

1. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。
1. 现在为机器学习准备数据。 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，其用途为：
    - 删除任何不完整的行
    - 应用适当的数据类型
    - 查看数据的随机样本
    - 将数据拆分成两个数据集：一个用于训练，另一个用于测试。


    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   
   data = spark.read.format("csv").option("header", "true").load("/hyperopt_lab/penguins.csv")
   data = data.dropna().select(col("Island").astype("string"),
                             col("CulmenLength").astype("float"),
                             col("CulmenDepth").astype("float"),
                             col("FlipperLength").astype("float"),
                             col("BodyMass").astype("float"),
                             col("Species").astype("int")
                             )
   display(data.sample(0.2))
   
   splits = data.randomSplit([0.7, 0.3])
   train = splits[0]
   test = splits[1]
   print ("Training Rows:", train.count(), " Testing Rows:", test.count())
    ```

## 优化用于训练模型的超参数值

可以通过将特征拟合为一种计算最可能标签的算法来训练机器学习模型。 算法将训练数据作为参数，并尝试计算特征和标签之间的数学关系。 除了数据之外，大多数算法还使用一个或多个超参数** 来影响关系的计算方式；确定最优超参数值是迭代模型训练过程的重要组成部分。

为了帮助你确定最优超参数值，Azure Databricks 支持 Hyperopt****，该库使你能够尝试多个超参数值并找到最适合数据的组合。

使用 Hyperopt 的第一步是创建一个具有以下特点的函数：

- 使用一个或多个作为参数传递给函数的超参数值来训练模型。
- 计算可用于度量“损失”**（模型距完美预测性能的距离）的性能指标
- 返回损失值，以便通过尝试不同的超参数值以迭代方式对损失值进行优化（最小化）

1. 添加一个新单元格并使用以下代码创建一个函数，该函数使用企鹅数据训练一个分类模型，该模型根据企鹅的位置和度量来预测企鹅的种类：

    ```python
   from hyperopt import STATUS_OK
   import mlflow
   from pyspark.ml import Pipeline
   from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
   from pyspark.ml.classification import DecisionTreeClassifier
   from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
   def objective(params):
       # Train a model using the provided hyperparameter value
       catFeature = "Island"
       numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
       catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
       numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
       numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
       featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
       mlAlgo = DecisionTreeClassifier(labelCol="Species",    
                                       featuresCol="Features",
                                       maxDepth=params['MaxDepth'], maxBins=params['MaxBins'])
       pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, mlAlgo])
       model = pipeline.fit(train)
       
       # Evaluate the model to get the target metric
       prediction = model.transform(test)
       eval = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName="accuracy")
       accuracy = eval.evaluate(prediction)
       
       # Hyperopt tries to minimize the objective function, so you must return the negative accuracy.
       return {'loss': -accuracy, 'status': STATUS_OK}
    ```

1. 添加一个新单元格并使用以下代码来执行下列操作：
    - 定义一个搜索空间，该空间指定要用于一个或多个超参数的值的范围（如需更多详细信息，请参阅 Hyperopt 文档中的[定义搜索空间](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/)）。
    - 指定要使用的 Hyperopt 算法（如需更多详细信息，请参阅 Hyperopt 文档中的[算法](http://hyperopt.github.io/hyperopt/#algorithms)）。
    - 使用 hyperopt.fmin**** 函数重复调用训练函数并尝试最大程度地减少损失。

    ```python
   from hyperopt import fmin, tpe, hp
   
   # Define a search space for two hyperparameters (maxDepth and maxBins)
   search_space = {
       'MaxDepth': hp.randint('MaxDepth', 10),
       'MaxBins': hp.choice('MaxBins', [10, 20, 30])
   }
   
   # Specify an algorithm for the hyperparameter optimization process
   algo=tpe.suggest
   
   # Call the training function iteratively to find the optimal hyperparameter values
   argmin = fmin(
     fn=objective,
     space=search_space,
     algo=algo,
     max_evals=6)
   
   print("Best param values: ", argmin)
    ```

1. 在代码以迭代方式运行训练函数 6 次（基于 max_evals**** 设置）的过程中进行观察。 每次运行都会由 MLflow 记录，你可以使用 &#9656;**** 切换按钮展开代码单元格下的“MLflow 运行”**** 输出，然后选择用于查看它们的“试验”**** 超链接。 每次运行都会分配一个随机名称，你可以在 MLflow 运行查看器中查看每个运行，以了解已记录参数和指标的详细信息。
1. 当所有运行都完成后，你会观察到代码显示找到的最佳超参数值（导致损失最小的组合）的详细信息。 在本例中，MaxBins**** 参数被定义为从三个可能值（10、20 和 30）的列表中进行的选择 - 最佳值表示列表中从零开始的项（因此，0=10，1=20，2=30）。 MaxDepth**** 参数被定义为 0 到 10 之间的随机整数，将显示提供最佳结果的整数值。 若要详细了解如何为搜索空间指定超参数值范围，请参阅 Hyperopt 文档中的[参数表达式](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions)。

## 使用 Trials 类记录运行详细信息

除了使用 MLflow 试验运行来记录每次迭代的详细信息之外，还可以使用 hyperopt.Trials**** 类来记录和查看每次运行的详细信息。

1. 添加新单元格并使用以下代码查看 Trials**** 类记录的每次运行的详细信息：

    ```python
   from hyperopt import Trials
   
   # Create a Trials object to track each run
   trial_runs = Trials()
   
   argmin = fmin(
     fn=objective,
     space=search_space,
     algo=algo,
     max_evals=3,
     trials=trial_runs)
   
   print("Best param values: ", argmin)
   
   # Get details from each trial run
   print ("trials:")
   for trial in trial_runs.trials:
       print ("\n", trial)
    ```

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则现在可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。

> **详细信息**：有关详细信息，请参阅 Azure Databricks 文档中的[超参数优化](https://learn.microsoft.com/azure/databricks/machine-learning/automl-hyperparam-tuning/)。