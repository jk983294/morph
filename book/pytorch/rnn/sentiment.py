import os
from collections import Counter
import torch
from string import punctuation
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def pad_features(reviews_ints, seq_length_):
    """
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length
    [:seq_length] 把多出来的去掉， features[i, -len(row):] 靠右存
    """
    row_cnt = len(reviews_ints)
    features_ = np.zeros((row_cnt, seq_length_), dtype=int)

    for i, row in enumerate(reviews_ints):
        features_[i, -len(row):] = np.array(row)[:seq_length_]
    return features_


def pre_process(reviews_, labels_):
    # get rid of punctuation
    reviews_ = reviews_.lower()
    all_text = ''.join([c for c in reviews_ if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)

    # create a list of words
    words = all_text.split()

    # Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int_ = {word: ii for ii, word in enumerate(vocab, 1)}  # 从1开始，0留给padding

    # use the dict to tokenize each review in reviews_split
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int_[word] for word in review.split()])

    print('Unique words: ', len(vocab_to_int_))

    # 1=positive, 0=negative label conversion
    labels_split = labels_.split('\n')
    y_ = np.array([1 if label == 'positive' else 0 for label in labels_split])

    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Zero-length reviews: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    print('Number of reviews before removing outliers: ', len(reviews_ints))
    # get indices of any reviews with length 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    y_ = np.array([y_[ii] for ii in non_zero_idx])
    print('Number of reviews after removing outliers: ', len(reviews_ints))

    seq_length_ = 200
    features_ = pad_features(reviews_ints, seq_length_=seq_length_)

    assert len(features_) == len(reviews_ints), "Your features should have as many rows as reviews."
    assert len(features_[0]) == seq_length_, "Each feature row should contain seq_length values."
    return features_, y_, vocab_to_int_


def split_data(split_fraction, batch_size_):
    # split data into training, validation, and test data (features and labels, x and y)
    split_idx = int(len(features) * split_fraction)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = y[:split_idx], y[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    print("\t\t\t\t\tFeature Shapes:")
    print("Train set: \t\t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # make sure the SHUFFLE your training data
    train_loader_ = DataLoader(train_data, shuffle=True, batch_size=batch_size_)
    valid_loader_ = DataLoader(valid_data, shuffle=True, batch_size=batch_size_)
    test_loader_ = DataLoader(test_data, shuffle=True, batch_size=batch_size_)
    return train_loader_, valid_loader_, test_loader_


class SentimentRNN(nn.Module):
    def __init__(self, train_on_gpu, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        vocab_size: Size of our vocabulary, word count
        output_size: the number of class we want to output (pos/neg)
        embedding_dim: Number of columns in the embedding lookup table; size of our embeddings
        hidden_dim: Number of units in the hidden layers of our LSTM cells
        n_layers: Number of LSTM layers in the network. Typically between 1-3
        """
        super(SentimentRNN, self).__init__()

        self.train_on_gpu = train_on_gpu
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        x = x.long()  # [batch, seq_len]
        embeds = self.embedding(x)  # [batch, seq_len, embedding_dim]
        lstm_out, hidden = self.lstm(embeds, hidden)  # ([batch, seq_len, hidden_dim], [n_layer, batch, hidden_dim])
        lstm_out = lstm_out[:, -1, :]  # getting the last time step output [batch, hidden_dim]
        out = self.dropout(lstm_out)  # [batch, hidden_dim]
        out = self.fc(out)  # [batch, output_n=1]
        sig_out = self.sig(out)  # [batch, output_n=1]
        return sig_out, hidden

    def init_hidden(self, batch_size_):
        # create two new tensors for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden_state = weight.new(self.n_layers, batch_size_, self.hidden_dim).zero_().cuda()
            cell_state = weight.new(self.n_layers, batch_size_, self.hidden_dim).zero_().cuda()
        else:
            hidden_state = weight.new(self.n_layers, batch_size_, self.hidden_dim).zero_()
            cell_state = weight.new(self.n_layers, batch_size_, self.hidden_dim).zero_()
        return hidden_state, cell_state


def tokenize_review(test_review, vocab_to_int):
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()  # splitting by spaces
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words])
    return test_ints


def predict(net, vocab_to_int, test_review, sequence_length=200):
    net.eval()

    test_ints = tokenize_review(test_review, vocab_to_int)

    # pad tokenized sequence
    seq_length_ = sequence_length
    features_ = pad_features(test_ints, seq_length_)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features_)
    batch_size_ = feature_tensor.size(0)

    h = net.init_hidden(batch_size_)
    if on_gpu:
        feature_tensor = feature_tensor.cuda()

    output, h = net(feature_tensor, h)
    pred = torch.round(output.squeeze())
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    if pred.item() == 1:
        print("Positive review detected!")
    else:
        print("Negative review detected.")


if __name__ == '__main__':
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        print('Training on GPU!')
    else:
        print('No GPU available')

    with open(os.path.expanduser('~/github/barn/deep/sentiment/reviews.txt'), 'r') as f:
        reviews = f.read()
    with open(os.path.expanduser('~/github/barn/deep/sentiment/labels.txt'), 'r') as f:
        labels = f.read()

    batch_size = 50
    features, y, vocab_to_int = pre_process(reviews, labels)
    train_loader, valid_loader, test_loader = split_data(0.8, batch_size)

    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
    net = SentimentRNN(on_gpu, vocab_size, output_size=1, embedding_dim=400, hidden_dim=256, n_layers=2)
    if on_gpu:
        net.cuda()
    print(net)

    criterion = nn.BCELoss()  # designed to work with a single Sigmoid output
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    epochs = 4
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    net.train()
    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            counter += 1
            if on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # new variables for the hidden state, otherwise we'd back prop through the entire training history
            h = tuple([each.data for each in h])
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])  # remove grad_fn history
                    if on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0

    h = net.init_hidden(batch_size)
    net.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])

        if on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        output, h = net(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. ' \
                      'This movie had bad acting and the dialogue was slow.'
    seq_length = 200  # good to use the length that was trained on
    predict(net, vocab_to_int, test_review_neg, seq_length)
