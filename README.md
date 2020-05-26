# Alphazero_dots_and_boxes

Use alpha zero algorithm to master the game of dots&boxes.

## Dependencies

* TensorFlow
* Keras

# Usage
Please run the script `test.py` to see the output.

The AlphaZero player (`SmartMCTSPlayer`) is the second player, you can change the argument `player=1` and create an AlphaZero player as the first player. Please change the list of `GameManager` at the same time.

The number of playouts is 400, you can change the argument `n_playout`.

To train your own model, please run `train.py`

# Acknowledgement
Thanks for Junxiao Song's prior work https://zhuanlan.zhihu.com/p/32089487
