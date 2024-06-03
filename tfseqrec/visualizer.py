import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties


def draw_predicted_text(label, pred_label, fontdict, width, height):
    label = label.replace('[UNK]', '?')
    label_length, pred_length = len(label), len(pred_label)

    if pred_label == label:
        fontdict['color'] = 'green'
        if height >= width: plt.text(width, 0, '\n'.join(pred_label), fontdict=fontdict)
        else: plt.text(0, height, pred_label, fontdict=fontdict)
        return 

    pred_start, start, end = 0, 0, 0
    while start <= end < label_length:
        pos = end * max(height, width) / label_length
        actual_token = '[UNK]' if label[end] == '?' else label[end]

        if label[start:end + 1] in pred_label[pred_start:pred_length]:
            fontdict['color'] = 'dodgerblue'
            if height >= width: plt.text(width, pos, actual_token, fontdict=fontdict)
            else: plt.text(pos, height, actual_token, fontdict=fontdict)
        else:
            if end < pred_length and end + 1 < label_length and pred_label[end] == label[end + 1]:
                fontdict['color'] = 'gray'
                if height >= width: plt.text(width, pos, actual_token, fontdict=fontdict)
                else: plt.text(pos, height, actual_token, fontdict=fontdict)
            elif end < pred_length:
                fontdict['color'] = 'red'
                if height >= width: plt.text(width, pos, pred_label[end], fontdict=fontdict)
                else: plt.text(pos, height, pred_label[end], fontdict=fontdict)
                
                fontdict['color'] = 'black'
                if height >= width: plt.text(width * 2, pos, actual_token, fontdict=fontdict)
                else: plt.text(pos, height * 2, actual_token, fontdict=fontdict)
            else: 
                fontdict['color'] = 'gray'
                if height >= width: plt.text(width, pos, actual_token, fontdict=fontdict)
                else: plt.text(pos, height, actual_token, fontdict=fontdict)
                
            pred_start = end
            start = end + 1
        end += 1


def plot_images_labels(
    img_paths, 
    labels, # shape == (batch_size, max_length)
    pred_labels = None, # shape == (batch_size, max_length)
    figsize = (15, 8),
    subplot_size = (2, 8), # tuple: (rows, columns) to display
    legend_loc = None, # Only for predictions,
    annotate_loc = None, # Only for predictions
    fontpath = None, 
    fontsize = 12
):
    nrows, ncols = subplot_size 
    num_of_labels = len(labels)
    assert len(img_paths) == num_of_labels, 'img_paths and labels must have same number of items'
    assert nrows * ncols <= num_of_labels, f'nrows * ncols must be <= {num_of_labels}'
    fontdict = {
        'fontproperties': FontProperties(fname=fontpath),
        'fontsize': fontsize,
        'color': 'black',
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    plt.figure(figsize=figsize)
    for i in range(min(nrows * ncols, num_of_labels)):
        plt.subplot(nrows, ncols, i + 1)
        image, label = plt.imread(img_paths[i]), labels[i]
        height, width, _ = image.shape
        plt.imshow(image)

        fontdict['color'] = 'black'  # Reset the color
        if pred_labels: draw_predicted_text(label, pred_labels[i], fontdict, width, height)
        elif height >= width: plt.text(width, 0, '\n'.join(label), fontdict=fontdict)
        elif width > height: plt.text(0, height, label, fontdict=fontdict)
        plt.axis('off')

    if legend_loc and annotate_loc and pred_labels:
        plt.subplots_adjust(left=0, right=0.75)
        plt.legend(handles=[
            Patch(color='green', label='Full match'),
            Patch(color='dodgerblue', label='Token match'),
            Patch(color='red', label='Wrong prediction'),
            Patch(color='black', label='Actual token'),
            Patch(color='gray', label='Missing position'),
        ], loc=legend_loc)

        annotate_text = [f'{idx + 1:02d}. {text}' for idx, text in enumerate(pred_labels)]
        plt.annotate(
            f'Model predictions:\n{chr(10).join(annotate_text)}',
            fontproperties = FontProperties(fname=fontpath),
            xycoords = 'axes fraction',
            fontsize = 14,
            xy = annotate_loc,
        )


def plot_training_results(history, save_name, figsize=(16, 14), subplot_size=(2, 2)):
    nrows, ncols = subplot_size
    if 'lr' in history.keys(): del history['lr']
    assert nrows * ncols <= len(history), f'nrows * ncols must be <= {len(history)}'
    fig = plt.figure(figsize=figsize)

    for idx, name in enumerate(history):
        if 'val' in name: continue
        plt.subplot(nrows, ncols, idx + 1)
        plt.plot(history[name], linestyle='solid', marker='o', color='crimson', label='Train')
        plt.plot(history[f'val_{name}'], linestyle='solid', marker='o', color='dodgerblue', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel(name)

        title = name.replace('acc', 'accuracy')\
                    .replace('seq_', 'sequence_')\
                    .replace('tok_', 'token_')\
                    .replace('lev_', 'levenshtein_')\
                    .replace('edit_', 'levenshtein_')\
                    .replace('_', ' ').capitalize()
        plt.title(title)
        plt.legend(loc='best')

    fig.savefig(save_name, bbox_inches='tight')
    plt.show()