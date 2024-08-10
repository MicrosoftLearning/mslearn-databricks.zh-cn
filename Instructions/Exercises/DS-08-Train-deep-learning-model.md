---
lab:
  title: 训练深度学习模型
---

# 训练深度学习模型

在本练习中，你将使用 **PyTorch** 库在 Azure Databricks 中训练深度学习模型。 然后，你将使用 **Horovod** 库在群集中的多个工作器节点之间分配深度学习训练。

完成此练习大约需要 45 分钟。

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
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的[分布式训练](https://learn.microsoft.com/azure/databricks/machine-learning/train-model/distributed-training/)一文。

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
    - 节点类型：Standard_DS3_v2
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 创建笔记本

你将运行使用 Spark MLLib 库来训练机器学习模型的代码，因此第一步是在工作区中创建一个新笔记本。

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称（**Untitled Notebook *[日期]***）更改为“深度学习”，然后在“连接”下拉列表中选择群集（如果尚未选中）****。**** 如果群集未运行，可能需要一分钟左右才能启动。

## 引入和准备数据

本练习的场景基于对南极洲企鹅的观察，目的是训练一个机器学习模型，用于根据观察到的企鹅的位置和身体度量来预测其种类。

> **引文**：本练习中使用的企鹅数据集是 [Kristen Gorman 博 士](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)和[长期生态研究网络](https://lternet.edu/)成员[南极洲帕默站](https://pal.lternet.edu/)收集并提供的数据的子集。

1. 在笔记本的第一个单元格中输入以下代码，该代码使用 shell** 命令将企鹅数据从 GitHub 下载到群集使用的文件系统中。

    ```bash
    %sh
    rm -r /dbfs/deepml_lab
    mkdir /dbfs/deepml_lab
    wget -O /dbfs/deepml_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv
    ```

1. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。
1. 现在为机器学习准备数据。 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，其用途为：
    - 删除任何不完整的行
    - 将（字符串）岛屿名称编码为整数
    - 应用适当的数据类型
    - 将数值数据规范化为类似的尺度
    - 将数据拆分成两个数据集：一个用于训练，另一个用于测试。

    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   from sklearn.model_selection import train_test_split
   
   # Load the data, removing any incomplete rows
   df = spark.read.format("csv").option("header", "true").load("/deepml_lab/penguins.csv").dropna()
   
   # Encode the Island with a simple integer index
   # Scale FlipperLength and BodyMass so they're on a similar scale to the bill measurements
   islands = df.select(collect_set("Island").alias('Islands')).first()['Islands']
   island_indexes = [(islands[i], i) for i in range(0, len(islands))]
   df_indexes = spark.createDataFrame(island_indexes).toDF('Island', 'IslandIdx')
   data = df.join(df_indexes, ['Island'], 'left').select(col("IslandIdx"),
                      col("CulmenLength").astype("float"),
                      col("CulmenDepth").astype("float"),
                      (col("FlipperLength").astype("float")/10).alias("FlipperScaled"),
                       (col("BodyMass").astype("float")/100).alias("MassScaled"),
                      col("Species").astype("int")
                       )
   
   # Oversample the dataframe to triple its size
   # (Deep learning techniques like LOTS of data)
   for i in range(1,3):
       data = data.union(data)
   
   # Split the data into training and testing datasets   
   features = ['IslandIdx','CulmenLength','CulmenDepth','FlipperScaled','MassScaled']
   label = 'Species'
      
   # Split data 70%-30% into training set and test set
   x_train, x_test, y_train, y_test = train_test_split(data.toPandas()[features].values,
                                                       data.toPandas()[label].values,
                                                       test_size=0.30,
                                                       random_state=0)
   
   print ('Training Set: %d rows, Test Set: %d rows \n' % (len(x_train), len(x_test)))
    ```

## 安装和导入 PyTorch 库

PyTorch 是用于创建机器学习模型的框架，包括深度神经网络 (DNN)。 由于我们计划使用 PyTorch 创建企鹅分类器，因此我们需要导入要使用的 PyTorch 库。 PyTorch 已安装在具有 ML Databricks 运行时的 Azure databricks 群集上（PyTorch 的特定安装取决于群集是否有可用于通过 *cuda* 进行高性能处理的图形处理单元 (GPU)）。

1. 添加新代码单元并运行以下代码以准备使用 PyTorch：

    ```python
   import torch
   import torch.nn as nn
   import torch.utils.data as td
   import torch.nn.functional as F
   
   # Set random seed for reproducability
   torch.manual_seed(0)
   
   print("Libraries imported - ready to use PyTorch", torch.__version__)
    ```

## 创建数据加载程序

PyTorch 利用*数据加载程序*分批加载训练和验证数据。 我们已经将数据加载到 numpy 数组中，但我们需要将它们包装到 PyTorch 数据集中（其中数据转换为 PyTorch *tensor* 对象），并创建加载程序以从这些数据集读取批次。

1. 添加单元格并运行以下代码来准备数据加载程序：

    ```python
   # Create a dataset and loader for the training data and labels
   train_x = torch.Tensor(x_train).float()
   train_y = torch.Tensor(y_train).long()
   train_ds = td.TensorDataset(train_x,train_y)
   train_loader = td.DataLoader(train_ds, batch_size=20,
       shuffle=False, num_workers=1)

   # Create a dataset and loader for the test data and labels
   test_x = torch.Tensor(x_test).float()
   test_y = torch.Tensor(y_test).long()
   test_ds = td.TensorDataset(test_x,test_y)
   test_loader = td.DataLoader(test_ds, batch_size=20,
                                shuffle=False, num_workers=1)
   print('Ready to load data')
    ```

## 定义神经网络

现在，我们已准备好定义神经网络。 在这种情况下，我们将创建一个由 3 个完全连接的层组成的网络：

- 输入层，接收每个特征的输入值（在本例中，即岛屿索引和四个企鹅度量值）并生成 10 个输出。
- 从输入层接收十个输入并将十个输出发送到下一层的隐藏层。
- 输出层，为三种可能的企鹅物种中的每一种生成概率矢量。

通过传递数据来训练网络时，**forward** 函数会将 *RELU* 激活函数应用于前两个层（以将结果限制为正数），并返回一个最终输出层，该层使用 *log_softmax* 函数返回一个值，该值表示三个可能的类中每一个类的概率分数。

1. 运行以下代码来定义神经网络：

    ```python
   # Number of hidden layer nodes
   hl = 10
   
   # Define the neural network
   class PenguinNet(nn.Module):
       def __init__(self):
           super(PenguinNet, self).__init__()
           self.fc1 = nn.Linear(len(features), hl)
           self.fc2 = nn.Linear(hl, hl)
           self.fc3 = nn.Linear(hl, 3)
   
       def forward(self, x):
           fc1_output = torch.relu(self.fc1(x))
           fc2_output = torch.relu(self.fc2(fc1_output))
           y = F.log_softmax(self.fc3(fc2_output).float(), dim=1)
           return y
   
   # Create a model instance from the network
   model = PenguinNet()
   print(model)
    ```

## 创建用于训练和测试神经网络模型的函数

若要训练模型，我们需要反复通过网络向前馈送训练值，使用损失函数来计算损失，使用优化器来反向传播权重和偏置值调整，并使用我们保留的测试数据来验证模型。

1. 为此，请使用以下代码创建一个函数来训练和优化模型，另一个函数用来测试模型。

    ```python
   def train(model, data_loader, optimizer):
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model.to(device)
       # Set the model to training mode
       model.train()
       train_loss = 0
       
       for batch, tensor in enumerate(data_loader):
           data, target = tensor
           #feedforward
           optimizer.zero_grad()
           out = model(data)
           loss = loss_criteria(out, target)
           train_loss += loss.item()
   
           # backpropagate adjustments to the weights
           loss.backward()
           optimizer.step()
   
       #Return average loss
       avg_loss = train_loss / (batch+1)
       print('Training set: Average loss: {:.6f}'.format(avg_loss))
       return avg_loss
              
               
   def test(model, data_loader):
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model.to(device)
       # Switch the model to evaluation mode (so we don't backpropagate)
       model.eval()
       test_loss = 0
       correct = 0
   
       with torch.no_grad():
           batch_count = 0
           for batch, tensor in enumerate(data_loader):
               batch_count += 1
               data, target = tensor
               # Get the predictions
               out = model(data)
   
               # calculate the loss
               test_loss += loss_criteria(out, target).item()
   
               # Calculate the accuracy
               _, predicted = torch.max(out.data, 1)
               correct += torch.sum(target==predicted).item()
               
       # Calculate the average loss and total accuracy for this epoch
       avg_loss = test_loss/batch_count
       print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           avg_loss, correct, len(data_loader.dataset),
           100. * correct / len(data_loader.dataset)))
       
       # return average loss for the epoch
       return avg_loss
    ```

## 训练模型

现在可以使用**训练**和**测试**函数来训练神经网络模型。 跨多个*纪元*以迭代方式训练神经网络，记录每个纪元的损失和准确性统计信息。

1. 使用以下代码训练模型：

    ```python
   # Specify the loss criteria (we'll use CrossEntropyLoss for multi-class classification)
   loss_criteria = nn.CrossEntropyLoss()
   
   # Use an optimizer to adjust weights and reduce loss
   learning_rate = 0.001
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   optimizer.zero_grad()
   
   # We'll track metrics for each epoch in these arrays
   epoch_nums = []
   training_loss = []
   validation_loss = []
   
   # Train over 100 epochs
   epochs = 100
   for epoch in range(1, epochs + 1):
   
       # print the epoch number
       print('Epoch: {}'.format(epoch))
       
       # Feed training data into the model
       train_loss = train(model, train_loader, optimizer)
       
       # Feed the test data into the model to check its performance
       test_loss = test(model, test_loader)
       
       # Log the metrics for this epoch
       epoch_nums.append(epoch)
       training_loss.append(train_loss)
       validation_loss.append(test_loss)
    ```

    当训练过程正在运行时，让我们尝试了解正在发生的事情：

    - 在每个*纪元*，整个训练数据集通过网络向前传递。 每个观察都有五个特征，对应输入层中的五个节点，因此每个观察的特征会作为五个值的矢量传递给该层。 但是，为了提高效率，特征矢量会组成批：因此，实际上每次都会馈送一个包含多个特征矢量的矩阵。
    - 特征值的矩阵由一个函数处理，该函数使用初始化的权重和偏置值执行加权求和。 然后，此函数的结果由输入层的激活函数处理，以约束传递到下一层中的节点的值。
    - 加权求和与激活函数会在每个层中重复。 请注意，这些函数对矢量和矩阵而非单个标量值进行操作。 换句话说，向前传递本质上是一系列嵌套线性代数函数。 这是数据科学家更喜欢使用具有图形处理单元 (GPU) 的计算机的原因，因为这些计算机已针对矩阵和矢量计算进行优化。
    - 在网络的最后一层中，输出矢量包含每个可能的类的计算值（在本例中，为类 0、1 和 2）。 此矢量由*损失函数*处理，该函数会根据实际类确定它们与预期值之间的距离 - 例如，假设巴布亚企鹅（类 1）的输出为 \[0.3, 0.4, 0.3\]。 正确的预测应该是 \[0.0, 1.0, 0.0\]，因此预测值与实际值之间的方差（每个预测值与实际值之间的距离）为 \[0.3, 0.6, 0.3\]。 此方差在每个批次中聚合，并作为运行聚合进行维护，以计算纪元中训练数据产生的总误差（*损失*）。
    - 在每个纪元结束时，验证数据会通过网络传递，同时也会计算其损失和准确性（根据输出矢量中最高的概率值得出的正确预测比例）。 这样做很有用，因为它使我们能够在每个纪元后使用未投入训练的数据比较模型的性能，帮助我们确定它是否能通用于新数据，或者对训练数据*过度拟合*。
    - 通过网络向前传递所有数据后，*训练*数据（<u>并非</u>*验证*数据）的丢失函数的输出将传递给优化器。 优化器处理损失的确切细节因所使用的特定优化算法而异，但从根本上讲，可以将整个网络从输入层到损失函数视为一个大型嵌套（*复合*）函数。 优化器会应用一些微分来计算函数相对于网络中使用的每个权重和偏置值的*偏导数*。 得益于*链式法则*，可以为嵌套函数高效地执行此操作，这使你能够通过内部函数和外部函数的导数来确定复合函数的导数。 你不需要担心此处的数学细节（优化器会为你搞定），但最终结果是，偏导数会告诉我们损失函数相对于每个权重和偏置值的斜率（或*梯度*），换句话说，我们可以确定是要增加还是减少权重和偏置值，以尽量减少损失。
    - 在确定了调整权重和偏置的方向后，优化器使用*学习速率*来确定调整它们的量，然后在一个名为*反向传播*的进程中通过网络向后工作，将新值分配给每个层的权重和偏置。
    - 现在，下一个纪元将重复整个训练、验证和反向传播过程，从来自上一纪元的修改后的权重和偏置开始，这有望降低损失。
    - 此过程会持续如此 100 个纪元。

## 检查训练和验证损失

训练完成后，我们可以检查在训练和验证模型时记录的损失指标。 我们真正要找的是两件事：

- 损失应该随纪元而减少，表明模型正在学习正确的权重和偏置来预测正确的标签。
- 训练损失和验证损失应遵循类似的趋势，表明模型未过度拟合训练数据。

1. 使用以下代码绘制损失：

    ```python
   %matplotlib inline
   from matplotlib import pyplot as plt
   
   plt.plot(epoch_nums, training_loss)
   plt.plot(epoch_nums, validation_loss)
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.legend(['training', 'validation'], loc='upper right')
   plt.show()
    ```

## 查看习得的权重和偏置

训练后的模型包含训练过程中由优化器确定的最终权重和偏置。 根据我们的网络模型，我们应期望每个层有以下值：

- 第 1 层 (*fc1*)：有 5 个输入值前往 10 个输出节点，因此应该有 10 x 5 的权重和 10 个偏置值。
- 第 2 层 (*fc2*)：有 10 个输入值前往 10 个输出节点，因此应该有 10 x 10 的权重和 10 个偏置值。
- 第 3 层 (*fc3*)：有 10 个输入值前往 3 个输出节点，因此应该有 3 x 10 的权重和 3 个偏置值。

1. 使用以下代码查看训练后的模型中的层：

    ```python
   for param_tensor in model.state_dict():
       print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())
    ```

## 保存和使用训练后的模型

现在，我们有一个训练后的模型，我们可以保存其训练的权重供以后使用。

1. 使用以下代码保存模型：

    ```python
   # Save the model weights
   model_file = '/dbfs/penguin_classifier.pt'
   torch.save(model.state_dict(), model_file)
   del model
   print('model saved as', model_file)
    ```

1. 使用以下代码加载模型权重并预测新观察到的企鹅的物种：

    ```python
   # New penguin features
   x_new = [[1, 50.4,15.3,20,50]]
   print ('New sample: {}'.format(x_new))
   
   # Create a new model class and load weights
   model = PenguinNet()
   model.load_state_dict(torch.load(model_file))
   
   # Set model to evaluation mode
   model.eval()
   
   # Get a prediction for the new data sample
   x = torch.Tensor(x_new).float()
   _, predicted = torch.max(model(x).data, 1)
   
   print('Prediction:',predicted.item())
    ```

## 使用 Horovod 进行分布式训练

之前的模型训练是在群集的单个节点上执行的。 实际上，最好在单个计算机上跨多个 CPU（或最好是 GPU）缩放深度学习模型训练，但在某些情况下，需要通过多层深度学习模型传递大量训练数据，则可以通过跨多个群集节点分配训练工作来实现一些效率。

Horovod 是一个开源库，可用于在 Spark 群集中的多个节点上分配深度学习训练，就像在 Azure Databricks 工作区中预配的那样。

### 创建训练函数

若要使用 Horovod，需封装代码以配置训练设置并在新函数中调用**训练**函数，你将使用 **HorovodRunner** 类来运行它，从而跨多个节点分配执行。 在训练包装器函数中，可以使用各种 Horovod 类来定义分布式数据加载程序，以便每个节点都可以处理整个数据集的子集），将模型权重和优化器的初始状态广播到所有节点，确定正在使用的节点数，以及代码正在哪个节点上运行。

1. 运行以下代码以创建使用 Horovod 训练模型的函数：

    ```python
   import horovod.torch as hvd
   from sparkdl import HorovodRunner
   
   def train_hvd(model):
       from torch.utils.data.distributed import DistributedSampler
       
       hvd.init()
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       if device.type == 'cuda':
           # Pin GPU to local rank
           torch.cuda.set_device(hvd.local_rank())
       
       # Configure the sampler so that each worker gets a distinct sample of the input dataset
       train_sampler = DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())
       # Use train_sampler to load a different sample of data on each worker
       train_loader = torch.utils.data.DataLoader(train_ds, batch_size=20, sampler=train_sampler)
       
       # The effective batch size in synchronous distributed training is scaled by the number of workers
       # Increase learning_rate to compensate for the increased batch size
       learning_rate = 0.001 * hvd.size()
       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       
       # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
       optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
   
       # Broadcast initial parameters so all workers start with the same parameters
       hvd.broadcast_parameters(model.state_dict(), root_rank=0)
       hvd.broadcast_optimizer_state(optimizer, root_rank=0)
   
       optimizer.zero_grad()
   
       # Train over 50 epochs
       epochs = 100
       for epoch in range(1, epochs + 1):
           print('Epoch: {}'.format(epoch))
           # Feed training data into the model to optimize the weights
           train_loss = train(model, train_loader, optimizer)
   
       # Save the model weights
       if hvd.rank() == 0:
           model_file = '/dbfs/penguin_classifier_hvd.pt'
           torch.save(model.state_dict(), model_file)
           print('model saved as', model_file)
    ```

1. 使用以下代码从 **HorovodRunner** 对象调用函数：

    ```python
   # Reset random seed for PyTorch
   torch.manual_seed(0)
   
   # Create a new model
   new_model = PenguinNet()
   
   # We'll use CrossEntropyLoss to optimize a multiclass classifier
   loss_criteria = nn.CrossEntropyLoss()
   
   # Run the distributed training function on 2 nodes
   hr = HorovodRunner(np=2, driver_log_verbosity='all') 
   hr.run(train_hvd, model=new_model)
   
   # Load the trained weights and test the model
   test_model = PenguinNet()
   test_model.load_state_dict(torch.load('/dbfs/penguin_classifier_hvd.pt'))
   test_loss = test(test_model, test_loader)
    ```

可能需要滚动以查看所有输出，它应显示 Horovod 的一些信息性消息，后跟节点的日志输出（因为 **driver_log_verbosity** 参数设置为 **all**）。 节点输出应显示每个纪元之后的损失。 最后，**测试**函数用于测试训练后的模型。

> **提示**：如果损失在每个纪元后不会减少，请尝试再次运行单元！

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
