import matplotlib.pyplot as plt




def plot(arr1, arr2, label1, label2, title, save_path, kind, figsize=(10,5)):
    """
    TODO
    """
    plt.figure(figsize=(10, 5))
    plt.plot(arr1, "o", linestyle='solid', label=label1)
    plt.plot(arr2, "o", linestyle='solid', label=label2)
    if kind.lower() == 'accuracy':
        plt.ylim(top = 1, bottom = 0.4)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()