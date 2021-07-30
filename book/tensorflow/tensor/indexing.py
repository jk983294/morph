import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    print(rank_1_tensor)
    print("First:", rank_1_tensor[0].numpy())  # 0
    print("Second:", rank_1_tensor[1].numpy())  # 1
    print("Last:", rank_1_tensor[-1].numpy())  # 34
    print("Everything:", rank_1_tensor[:].numpy())  # [ 0  1  1  2  3  5  8 13 21 34]
    print("Before 4:", rank_1_tensor[:4].numpy())  # [0 1 1 2]
    print("From 4 to the end:", rank_1_tensor[4:].numpy())  # [ 3  5  8 13 21 34]
    print("From 2, before 7:", rank_1_tensor[2:7].numpy())  # [1 2 3 5 8]
    print("Every other item:", rank_1_tensor[::2].numpy())  # [ 0  1  3  8 21]
    print("Reversed:", rank_1_tensor[::-1].numpy())  # [34 21 13  8  5  3  2  1  1  0]

    rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    print("Second row:", rank_2_tensor[1, :].numpy())  # [3. 4.]
    print("Second column:", rank_2_tensor[:, 1].numpy())  # [2. 4. 6.]
    print("Last row:", rank_2_tensor[-1, :].numpy())  # [5. 6.]
    print("First item in last column:", rank_2_tensor[0, -1].numpy())  # 2.0
    print("Skip the first row:")
    print(rank_2_tensor[1:, :].numpy(), "\n")

    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]], ])
    print(rank_3_tensor[:, :, 4])  # tf.Tensor(, shape=(3, 2), dtype=int32)
