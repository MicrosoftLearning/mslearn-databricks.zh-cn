---
lab:
  title: 在 Azure Databricks 中使用 MLflow
---

# 在 Azure Databricks 中使用 MLflow

在本练习中，你将探索如何使用 MLflow 在 Azure Databricks 中训练和提供机器学习模型。

完成此练习大约需要 45 分钟。

## 准备工作

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **注意**：就本练习来说，你需要一个高级**** Azure Databricks 工作区，该工作区位于某个支持模型服务** 的区域中。 有关区域 Azure Databricks 功能的详细信息，请参阅 [Azure Databricks 区域](https://learn.microsoft.com/azure/databricks/resources/supported-regions)。 如果你已在合适的区域拥有高级** 或试用** Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本尝试在一个区域中创建高级** 层 Azure Databricks 工作区资源，而该区域中的 Azure 订阅具有足够的用于本练习所需计算核心的配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足而失败，可以尝试在 Azure 门户中以交互方式创建 Azure Databricks 工作区。

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

5. 克隆存储库后，请输入以下命令以运行 setup.ps1**** 脚本，该脚本会在可用区域中预配 Azure Databricks 工作区：

    ```
    ./mslearn-databricks/setup.ps1
    ```

6. 如果出现提示，请选择要使用的订阅（仅当有权访问多个 Azure 订阅时才会发生这种情况）。
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的 [MLflow 指南](https://learn.microsoft.com/azure/databricks/mlflow/)一文。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML**<u></u>** 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 msl-xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）
1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 databricks-xxxxxxx******）。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“(+)新建”任务，然后选择“群集”********。
1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - Databricks Runtime 版本****：选择最新非 beta 版本运行时的 ML***<u></u>** 版本（不是****标准运行时版本），该版本符合以下条件：*
        - 不使用 GPU**
        - 包括 Scala > 2.11
        - 包括 Spark (> 3.4)******
    - 使用 Photon 加速****：未选定<u></u>
    - 节点类型：Standard_DS3_v2
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 创建笔记本

你将运行使用 Spark MLLib 库来训练机器学习模型的代码，因此第一步是在工作区中创建一个新笔记本。

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称（**Untitled Notebook *[日期]***）更改为“MLflow”，然后在“连接”下拉列表中选择群集（如果尚未选中）****。**** 如果群集未运行，可能需要一分钟左右才能启动。

## 引入和准备数据

本练习的场景基于对南极洲企鹅的观察，目的是训练一个机器学习模型，用于根据观察到的企鹅的位置和身体度量来预测其种类。

> **引文**：本练习中使用的企鹅数据集是 [Kristen Gorman 博 士](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)和[长期生态研究网络](https://lternet.edu/)成员[南极洲帕默站](https://pal.lternet.edu/)收集并提供的数据的子集。

1. 在笔记本的第一个单元格中输入以下代码，该代码使用 shell** 命令将企鹅数据从 GitHub 下载到群集使用的 Databricks 文件系统 (DBFS) 中。

    ```bash
    %sh
    rm -r /dbfs/mlflow_lab
    mkdir /dbfs/mlflow_lab
    wget -O /dbfs/mlflow_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv
    ```

1. 使用以下单元格右上角的“&#9656; 运行单元格”菜单选项来运行它****。 然后等待代码运行的 Spark 作业完成。

1. 现在为机器学习准备数据。 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，其用途为：
    - 删除任何不完整的行
    - 应用适当的数据类型
    - 查看数据的随机样本
    - 将数据拆分成两个数据集：一个用于训练，另一个用于测试。


    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   
   data = spark.read.format("csv").option("header", "true").load("/mlflow_lab/penguins.csv")
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

## 运行 MLflow 试验

使用 MLflow，你能够运行跟踪模型训练过程和记录评估指标的试验。 这种记录模型训练运行细节的功能在创建有效的机器学习模型的迭代过程中非常有用。

可以使用通常用于训练和评估模型的相同库和技术（在本例中，我们将使用 Spark MLLib 库），但需在 MLflow 试验的上下文中执行此操作，其中包括用于在此过程中记录重要指标和信息的其他命令。

1. 添加一个新单元格并在其中输入以下代码：

    ```python
   import mlflow
   import mlflow.spark
   from pyspark.ml import Pipeline
   from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
   from pyspark.ml.classification import LogisticRegression
   from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   import time
   
   # Start an MLflow run
   with mlflow.start_run():
       catFeature = "Island"
       numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
     
       # parameters
       maxIterations = 5
       regularization = 0.5
   
       # Define the feature engineering and model steps
       catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
       numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
       numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
       featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
       algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=maxIterations, regParam=regularization)
   
       # Chain the steps as stages in a pipeline
       pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
       # Log training parameter values
       print ("Training Logistic Regression model...")
       mlflow.log_param('maxIter', algo.getMaxIter())
       mlflow.log_param('regParam', algo.getRegParam())
       model = pipeline.fit(train)
      
       # Evaluate the model and log metrics
       prediction = model.transform(test)
       metrics = ["accuracy", "weightedRecall", "weightedPrecision"]
       for metric in metrics:
           evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName=metric)
           metricValue = evaluator.evaluate(prediction)
           print("%s: %s" % (metric, metricValue))
           mlflow.log_metric(metric, metricValue)
   
           
       # Log the model itself
       unique_model_name = "classifier-" + str(time.time())
       mlflow.spark.log_model(model, unique_model_name, mlflow.spark.get_default_conda_env())
       modelpath = "/model/%s" % (unique_model_name)
       mlflow.spark.save_model(model, modelpath)
       
       print("Experiment run complete.")
    ```

1. 试验运行完成后，如有必要，请在代码单元格下使用 &#9656;**** 切换按钮展开“MLflow 运行”**** 详细信息。 然后使用此处显示的**试验**超链接打开列出试验运行的 MLflow 页面。 每个运行都会被分配一个独一无二的名称。
1. 选择最近的运行并查看其详细信息。 请注意，可以展开各个部分来查看已记录的“参数”**** 和“指标”****，并且可以查看已训练和保存的模型的详细信息。

    > **提示**：还可以使用此笔记本右侧边栏菜单中的“MLflow 试验”**** 图标来查看试验运行的详细信息。

## 创建函数

在机器学习项目中，数据科学家会经常尝试使用不同参数来训练模型，每次都会记录结果。 为此，通常需要创建一个封装此训练过程的函数，并使用你想要尝试的参数来调用它。

1. 在新单元格中运行以下代码，以根据之前使用的训练代码来创建函数：

    ```python
   def train_penguin_model(training_data, test_data, maxIterations, regularization):
       import mlflow
       import mlflow.spark
       from pyspark.ml import Pipeline
       from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
       from pyspark.ml.classification import LogisticRegression
       from pyspark.ml.evaluation import MulticlassClassificationEvaluator
       import time
   
       # Start an MLflow run
       with mlflow.start_run():
   
           catFeature = "Island"
           numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
           # Define the feature engineering and model steps
           catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
           numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
           numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
           featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
           algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=maxIterations, regParam=regularization)
   
           # Chain the steps as stages in a pipeline
           pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
           # Log training parameter values
           print ("Training Logistic Regression model...")
           mlflow.log_param('maxIter', algo.getMaxIter())
           mlflow.log_param('regParam', algo.getRegParam())
           model = pipeline.fit(training_data)
   
           # Evaluate the model and log metrics
           prediction = model.transform(test_data)
           metrics = ["accuracy", "weightedRecall", "weightedPrecision"]
           for metric in metrics:
               evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName=metric)
               metricValue = evaluator.evaluate(prediction)
               print("%s: %s" % (metric, metricValue))
               mlflow.log_metric(metric, metricValue)
   
   
           # Log the model itself
           unique_model_name = "classifier-" + str(time.time())
           mlflow.spark.log_model(model, unique_model_name, mlflow.spark.get_default_conda_env())
           modelpath = "/model/%s" % (unique_model_name)
           mlflow.spark.save_model(model, modelpath)
   
           print("Experiment run complete.")
    ```

1. 在新单元格中，使用以下代码来调用函数：

    ```python
   train_penguin_model(train, test, 10, 0.2)
    ```

1. 查看第二次运行的 MLflow 试验的详细信息。

## 使用 MLflow 注册并部署模型

除了跟踪训练试验运行的详细信息之外，还可以使用 MLflow 来管理已训练的机器学习模型。 你已经记录了每次试验运行所训练的模型。 你还可以注册** 模型并部署它们，以便将它们提供给客户端应用程序。

> **注意**：模型服务仅在 Azure Databricks 高级** 工作区中受支持，并且仅限于[某些区域](https://learn.microsoft.com/azure/databricks/resources/supported-regions)。

1. 查看最新试验运行的详细信息页面。
1. 使用“注册模型”**** 按钮注册已在该试验中记录的模型，并在出现提示时创建一个名为“Penguin Predictor”**** 的新模型。
1. 注册模型后，查看“模型”**** 页面（位于左侧导航栏中）并选择“Penguin Predictor”**** 模型。
1. 在“Penguin Predictor”**** 模型的页面中，通过“使用模型进行推理”**** 按钮创建一个具有以下设置的新实时终结点：
    - **模型**：Penguin Predictor
    - 模型版本****：1
    - 终结点****：predict-penguin
    - 计算大小****：Small

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

在“predict-penguin”**** 终结点页面的“&#8285;”**** 菜单中，选择“删除”****。

## 清理

在 Azure Databricks 门户的“计算”页面上，选择你的群集，然后选择“&#9632; 终止”以将其关闭。********

如果已完成对 Azure Databricks 的探索，则现在可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。

> **详细信息**：有关详细信息，请参阅 [Spark MLLib 文档](https://spark.apache.org/docs/latest/ml-guide.html)。
