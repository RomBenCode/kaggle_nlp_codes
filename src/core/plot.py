import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def plot_metrics(history):
    f1_score = history.history['f1_score']
    loss = history.history['loss']
    val_f1_score = history.history['val_f1_score']
    val_loss = history.history['val_loss']

    epochs = range(len(f1_score))

    plt.plot(epochs, f1_score, 'b', label='Training f1_score')
    plt.plot(epochs, val_f1_score, 'y', label='Validation f1_score')
    plt.title('F1 Score')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')
    plt.title('loss')
    plt.legend()

    plt.show()