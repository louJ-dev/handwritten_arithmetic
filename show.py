import json 
import matplotlib.pyplot as plt

model_name = 'model'

with open(model_name + '.json', 'r') as file: 
    data = json.load(file)
    
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(model_name, fontsize=16)

    epoch = [x for x in range(1, len(data['accuracy']) + 1)]
    
    accuracy = list(data['accuracy'])

    ax[0, 0].plot(epoch, accuracy, label=model_name, color='red')
    ax[0, 0].set_xticks(epoch)
    ax[0, 0].set_xlabel('epoch')
    ax[0, 0].set_ylabel('accuracy')
    ax[0, 0].set_title('training accuracy vs epoch')

    loss = list(data['loss'])

    ax[0, 1].plot(epoch, loss, label=model_name, color='red')
    ax[0, 1].set_xticks(epoch)
    ax[0, 1].set_xlabel('epoch')
    ax[0, 1].set_ylabel('loss')
    ax[0, 1].set_title('training loss vs epoch') 
    
    val_accuracy = list(data['val_accuracy'])

    ax[1, 0].plot(epoch, val_accuracy, label=model_name, color='red')
    ax[1, 0].set_xticks(epoch)
    ax[1, 0].set_xlabel('epoch')
    ax[1, 0].set_ylabel('accuracy')
    ax[1, 0].set_title('validation accuracy vs epoch') 
    
    val_loss = list(data['val_loss'])

    ax[1, 1].plot(epoch, val_loss, label=model_name, color='red')
    ax[1, 1].set_xticks(epoch)
    ax[1, 1].set_xlabel('epoch')
    ax[1, 1].set_ylabel('loss')
    ax[1, 1].set_title('validation loss vs epoch')

    fig.tight_layout()
    
    plt.show()
