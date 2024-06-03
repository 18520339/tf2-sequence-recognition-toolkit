import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties


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
    
    
class SequenceVisualizer:
    def __init__(self, fontsize=12, fontpath=None):
        self.fontdict = {
            'fontproperties': FontProperties(fname=fontpath),
            'fontsize': fontsize,
            'color': 'black',
            'verticalalignment': 'top',
            'horizontalalignment': 'left'
        }
            

    def draw_predicted_text(self, label, pred_label, width, height):
        label = label.replace('[UNK]', '?')
        label_length, pred_length = len(label), len(pred_label)

        if pred_label == label:
            self.fontdict['color'] = 'green'
            self._draw_text(pred_label, width, height)
            return

        pred_start, start, end = 0, 0, 0
        while start <= end < label_length:
            pos = end * max(height, width) / label_length
            actual_token = '[UNK]' if label[end] == '?' else label[end]

            if label[start:end + 1] in pred_label[pred_start:pred_length]:
                self.fontdict['color'] = 'dodgerblue'
                self._draw_text(actual_token, width, height, pos)
            else:
                self._handle_mismatch(label, pred_label, end, actual_token, width, height, pos)
                pred_start = end
                start = end + 1
            end += 1
            
    
    def _draw_text(self, text, width, height, pos=0, is_char=True):
        if height >= width: plt.text(width, pos, (text if is_char else '\n'.join(text)), fontdict=self.fontdict)
        else: plt.text(pos, height, text, fontdict=self.fontdict)


    def _handle_mismatch(self, label, pred_label, end, actual_token, width, height, pos):
        if end < len(pred_label) and end + 1 < len(label) and pred_label[end] == label[end + 1]:
            self.fontdict['color'] = 'gray'
            self._draw_text(actual_token, width, height, pos)
        elif end < len(pred_label):
            self.fontdict['color'] = 'red'
            self._draw_text(pred_label[end], width, height, pos)
            self.fontdict['color'] = 'black'
            self._draw_text(actual_token, width * 2, height * 2, pos)
        else:
            self.fontdict['color'] = 'gray'
            self._draw_text(actual_token, width, height, pos)


    def plot_images_labels(
        self, img_paths, 
        labels, pred_labels=None, # shape == (batch_size, max_length)
        figsize=(15, 8), subplot_size=(2, 8),
        legend_loc=None, annotate_loc=None # Only for predictions
    ):
        nrows, ncols = subplot_size
        num_of_labels = len(labels)
        assert len(img_paths) == num_of_labels, 'img_paths and labels must have same number of items'
        assert nrows * ncols <= num_of_labels, f'nrows * ncols must be <= {num_of_labels}'

        plt.figure(figsize=figsize)
        for i in range(min(nrows * ncols, num_of_labels)):
            plt.subplot(nrows, ncols, i + 1)
            image, label = plt.imread(img_paths[i]), labels[i]
            height, width, _ = image.shape
            plt.imshow(image)

            self.fontdict['color'] = 'black' # Reset the color
            if pred_labels: self.draw_predicted_text(label, pred_labels[i], width, height)
            else: self._draw_text(label, width, height, is_char=False)
            plt.axis('off')

        if legend_loc and annotate_loc and pred_labels:
            self._add_legend_and_annotation(pred_labels, legend_loc, annotate_loc)


    def _add_legend_and_annotation(self, pred_labels, legend_loc, annotate_loc):
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
            fontproperties=FontProperties(fname=self.fontpath),
            xycoords='axes fraction',
            fontsize=14,
            xy=annotate_loc,
        )