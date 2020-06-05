说明文档
===========

文件结构：
-----------
* __MAIN__：程序入口，编写训练和测试代码位置
    - embedding: 数据集编码入口
    - train*：训练入口
    - test*：测试入口   
* Agent：可以自行添加智能体网络或者训练算法,智能体中的网络模块在Model中
    - deepAgent: 智能体网络
    - DQN: 训练智能体的算法double-DQN  
* Configuration:数据处理，环境搭建，智能体构建与训练等全部参数的存放位置
    - config: 整个程序运行过程中的全部参数，其中manually标记部分需要人为给定
* Environment：智能体探索的环境，其基本构成要素为知识图谱所构成的有向图，
    - environment_*: 知识图谱所构成的环境类，主要需要调用的功能：
        - init(dataSet) 初始化训练数据（初始化环境）；step(act_node) 当智能体给出动作时返回下一个状态和reward
    - MCTS_NSclass_*: 通过知识图谱（图结构）定义的蒙特卡洛树搜索算法中的状态类，树定义类
    - PolicyMCTS_*: 策略加强的蒙特卡洛树搜索算法
* Metric：对实验结果的评估算法
    - mAP:平均精度均值  
    - path_explain:手动查看路径的可解释性能
    - POSRewardRate:智能体在各个数据集环境中获得的正回报率   
* Model：深度智能体网络的组件（三个子网络模块）
    - action_encoder:全动作空间编码层
    - policy_net:策略决策层
    - tcn:时域卷积网络的基类
    - TCN_encoder:基于时域卷积网络的状态编码层
* PreDataSets：处理后的数据存放，其中主要分为两个文件夹：
    - embedding: 文件包含实体和关系的编码文件，其中包含不同数据集的编码
    - transdFile: 文件包含编码过程中的中间转换文件（当采用随机编码和transE编码时产生）
* RawDataSets：原始数据集，按统一格式处理过（注：M-walk作者处理）原始数据1G左右的文本，因此此处不上传仅提供地址，下载即可直接使用
* Result：实验结果，按数据集的名称存放tensorboardX的散点图，包含训练过程中的误差变化，回报率变化等。
* SavedModel：模型保存的位置，DQN算法定义时，提供加载初始化方法
* Script：各类脚本代码，主要包含embedding，构建graph和模型输入转换
    - node2vec：node2vec的代码包，无需改动（可单独调用其中的node2vec.py）
    - TransEmbedding：TransE的代码包，其中也包含了随机编码的代码文件，无需改动（可单独调用其中的node2vec.py）
    - buildNetworkGraph：调用networkx将数据集处理成DiGraph
    - get_embedding：读取embedding后的数据，转换为张量用以训练
    - graph_embedding：对数据进行embedding的程序入口，可调用的编码方式包含“random”“node2vec”“transE”
    - save_netGraph：保存模型结构图
    - state_tup2input：trajectory的表述方式为列表[query,entity1,entity2....-1(entityn)],转换为智能体网络的输入形式（张量）
    - Tools：一些简单方法的封装

实验过程：
-----------
1. 配置Configuration/config.py
    - 注：“manually”标签请手动配置，数据集Name，编码方式，编码维度，以及训练参数等
2. 运行Run/embedding.py 对配置数据进行编码
3. 运行Run/train.py 进行训练，（可自编写）主要说明如下：
    - env = Environment(GRAPH)： 调用Environment中的环境类，并需要初始化训练的数据
    - agentP = DQN(None)： 调用Agent训练算法（为主要过程，可自编写）,参数为加载模型  
    - train：train函数流程为DQN算法基本流程：
        - env.reset() 给出随机的环境初始状态
        - env.render() 打印状态信息
        - env.step() 智能体agentP从env给出的初始状态开始，在环境中探索。探索方法可选：“random”“MCTS”“DQN_self”“Policy_MCTS”。返回值为trajectory和reward
    - agentP.store_transition(): 智能体agentP将trajectory和reward存入路径存储池带训练，满足一定数量后开始训练
4. 运行Run/test.py 进行测试，测试过程同3类似，（可自编写）初始化数据为测试数据，测试过程无需再调用updata更新智能体Agent
