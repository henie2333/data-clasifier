import numpy as np
import torch
import torch.nn as nn
import torch.autograd.variable as variables
import torch.autograd.functional as F
import matplotlib.pyplot as plt
import data_processing

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)     # convert data to onehot form


# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)


# build network


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(21, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32,4),
                    nn.Softmax())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crossentropy = nn.CrossEntropyLoss()

def run_model(training_data):
    optimizer.zero_grad()
    model.zero_grad()
    in_x = variables(training_data[:,:21])
    in_y = variables(np.argmax(training_data[:,21:], axis=1))
    out = model(in_x)
    loss = crossentropy(out, in_y)  # crossentropy的input是各个状态的概率，target是应该出现的状态（即列坐标）
    loss.backward()
    optimizer.step()
    
    return loss

"""  tensorflow form:
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
"""

def test_acc():
    in_x = variables(train_data[:,:21])
    in_y = variables(np.argmax(train_data[:,21:], axis=1))
    out = model(in_x)
    accuracy = 1
    pred_ = np.argmax(out.detach().numpy(), axis=1)
    for i in range(len(pred_)):
        if pred_[i] != in_y[i]: accuracy -= 1/len(pred_)
    return pred_, accuracy

# training
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies, steps = [], []
for t in range(4000):
    # training
    batch_index = np.random.randint(len(train_data), size=32)
    loss_ = run_model(train_data[batch_index])

    if t % 50 == 0:
        # testing

        pred_, acc_ = test_acc()
        accuracies.append(acc_)
        steps.append(t)
        print("Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)

        # visualize testing
        ax1.cla()
        for c in range(4):
            bp = ax1.bar(c+0.1, height=sum((pred_ == c)), width=0.2, color='red')
            bt = ax1.bar(c-0.1, height=sum((np.argmax(train_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
        ax1.set_xticks(range(4), ["accepted", "good", "unaccepted", "very good"])
        ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
        ax1.set_ylim((0, 900))
        ax2.cla()
        ax2.plot(steps, accuracies, label="accuracy")
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)

plt.ioff()
plt.show()