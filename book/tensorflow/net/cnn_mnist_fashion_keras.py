import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)
    print(len(train_labels))
    print(len(test_labels))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('pred:', predictions[0])
    print('pred category:', np.argmax(predictions[0]))
    print('grand truth:', test_labels[0])

    # prediction about a single image
    img = test_images[1]
    img = (np.expand_dims(img, 0))
    predictions_single = probability_model.predict(img)
    print(predictions_single)
    print(np.argmax(predictions_single[0]))
