<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg"></a>
# cchess-zero
AlphaZero implemented Chinese chess. AlphaGo Zero / AlphaZero实践项目，实现中国象棋。

__Author__ chengstone

__e-Mail__ 69558140@163.com

代码详解请参见文内jupyter notebook和↓↓↓

知乎专栏：https://zhuanlan.zhihu.com/p/34433581

博客：http://blog.csdn.net/chengcheng1394/article/details/79526474

欢迎转发扩散 ^_^

这是一个AlphaZero的实践项目，实现了一个中国象棋程序，使用TensorFlow1.0和Python 3.5开发，还要安装uvloop。

因为我的模型训练的不充分，只训练了不到4K次，模型刚刚学会用象和士防守，总之仍然下棋很烂。

如果您有条件可以再多训练试试，我自从收到信用卡扣款400美元通知以后就把aws下线了：D 贫穷限制了我的想象力O(∩_∩)O

我训练的模型文件下载地址：https://pan.baidu.com/s/1dLvxFFpeWZK-aZ2Koewrvg

解压后放到项目根目录下即可，文件夹名叫做gpu_models

现在介绍下命令如何使用：

命令分为两类，一类是训练，一类是下棋。

训练专用：

 - --mode 指定是训练（train）还是下棋（play），默认是训练
 - --train_playout 指定MCTS的模拟次数，论文中是1600，我做训练时使用1200
 - --batch_size 指定训练数据达到多少时开始训练，默认512
 - --search_threads 指定执行MCTS时的线程个数，默认16
 - --processor 指定是使用cpu还是gpu，默认是cpu
 - --num_gpus 指定gpu的个数，默认是1
 - --res_block_nums 指定残差块的层数，论文中是19或39层，我默认是7

下棋专用：

 - --ai_count 指定ai的个数，1是人机对战，2是看两个ai下棋
 - --ai_function 指定ai的下棋方法，是思考（mcts，会慢），还是直觉（net，下棋快）
 - --play_playout 指定ai进行MCTS的模拟次数
 - --delay和--end_delay默认就好，两个ai下棋太快，就不知道俩ai怎么下的了：）
 - --human_color 指定人类棋手的颜色，w是先手，b是后手

训练命令举例：

python main.py --mode train --train_playout 1200 --batch_size 512 --search_threads 16 --processor gpu --num_gpus 2 --res_block_nums 7

下棋命令举例：

python main.py --mode play --ai_count 1 --ai_function mcts --play_playout 1200 --human_color w

# 许可
Licensed under the MIT License with the [`996ICU License`](https://github.com/996icu/996.ICU/blob/master/LICENSE).
