<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg"></a>
# cchess-zero
AlphaZero implemented Chinese chess. AlphaGo Zero / AlphaZero Practical project to realize Chinese chess

__Author__ chengstone

__e-Mail__ 69558140@163.com

For a detailed explanation of the code, please refer to the jupyter notebook and ↓↓↓

知乎专栏：https://zhuanlan.zhihu.com/p/34433581

Blog：http://blog.csdn.net/chengcheng1394/article/details/79526474

Welcome to forward and spread ^_^

This is a practical project of AlphaZero, which implements a Chinese chess program, developed with TensorFlow1.0 and Python 3.5, and also installs uvloop.

Because my model is not fully trained, I only trained it less than 4K times. The model has just learned to use elephants and soldiers to defend. In short, I still play chess badly.

If you have the conditions, you can try more training. I have taken aws offline since I received the 400 USD credit card charge notification: D Poverty limits my imagination O(∩_∩)O

Download address of the model file I trained: https://pan.baidu.com/s/1dLvxFFpeWZK-aZ2Koewrvg

After decompression, put it in the root directory of the project, the folder name is called gpu_models

Now introduce how to use the following command:

Commands are divided into two categories, one is training and the other is chess.

Training dedicated:

 - --mode specifies whether it is training (train) or chess (play), the default is training
 - --train_playout specifies the simulation times of MCTS, 1600 in the paper, I use 1200 when I do training
 - --batch_size specifies how much training data to start training, the default is 512
 - --search_threads specifies the number of threads to execute MCTS, the default is 16
 - --processor Specify whether to use cpu or gpu, the default is cpu
 - --num_gpus specifies the number of gpus, the default is 1
 - --res_block_nums specifies the number of layers of the residual block, in the paper it is 19 or 39 layers, my default is 7


For Chess:

 - --ai_count specifies the number of ai, 1 is human-computer battle, 2 is to watch two ai play chess
 - --ai_function Specifies the chess method of ai, whether it is thinking (mcts, slow) or intuition (net, fast chess)
 - --play_playout specifies the number of MCTS simulations performed by ai
 - --delay and --end_delay are just fine by default. The two ai play chess too fast, so I don't know how the two ai played :)
 - --human_color specifies the color of the human chess player, w is the first move, b is the second move


Examples of training commands:

python main.py --mode train --train_playout 1200 --batch_size 512 --search_threads 16 --processor gpu --num_gpus 2 --res_block_nums 7

Examples of chess commands:

python main.py --mode play --ai_count 1 --ai_function mcts --play_playout 1200 --human_color w

# Permission
Licensed under the MIT License with the [`996ICU License`](https://github.com/996icu/996.ICU/blob/master/LICENSE).