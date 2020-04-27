import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import Image
import gif

class rolling_window_validation:
    """
    This function divide the training time-series to k_folds in order to perform a validation phase
    over each of them.

    serie: the training set of the time-series.
    k_folds: the the number of folds to obtain.
    num_validation: the number of validation points to use as validation set for each fold.
    gap: the fraction of the training time-series to which generate the size of the rolling window and
       so determining the level of freedom to move the window.
    TimeSeriesSplit: split function according to sklearn package for time-series.
       
    Notes
    ----------
    If gap is 1 the rolling validation is reduced to be holdout validation.

    """
    def __init__(self, k_folds, validation_size = 0.1, gap = 0.7, TimeSeriesSplit = False):
        self.k_folds = k_folds
        self.validation_size = validation_size
        self.gap = gap
        self.TimeSeriesSplit = TimeSeriesSplit
    
    def get_splits(self, serie, show = False, path = None):
        freq = serie.index.freq
        folds = list()
        window = int((len(serie)*self.gap))
        num_validation = int(window*self.validation_size)

        def fold_funz(fold):
            folds.append(fold)
            return 0

        serie.rolling(window).apply(fold_funz, raw = False)
        # Select k_folds.
        multiples = [i*len(folds)//self.k_folds for i in range(self.k_folds - 1)]
        multiples.append(len(folds) - 1)
        folds = [fold for i, fold in enumerate(folds) if i in multiples]
        splits = list()
        for i, fold in enumerate(folds):
            if self.TimeSeriesSplit:
                train = fold[:-num_validation]
                train = pd.concat([serie[serie.index[0]:train.index[0] - 1*freq], train])
                val = fold[-num_validation:]
                splits.append((train, val))
            else:
                train = fold[:-num_validation]
                val = fold[-num_validation:]
                splits.append((train, val))
                
        if show:
            frames = list()
            for train, val in splits:
                @gif.frame
                def plot():
                    f = plt.figure(figsize = (20, 5))
                    plt.title(serie.name, color = "black")
                    plt.plot(serie, c = "gray", linestyle = ":", label = "Serie")
                    plt.plot(train, color = "#1281FF", label = "Train")
                    plt.plot(val, color = "orange", label = "Validation (%d)" % len(val))
                    plt.autoscale()
                    # Legend.
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), title = "Set", loc = "center left", 
                               bbox_to_anchor = (1.0, 0.5))
                    
                frames.append(plot())
            gif.save(frames, path + "/validation.gif", duration = 700)
            
            with open(path + "/validation.gif", "rb") as f:
                display(Image(data = f.read(), format = "png"))

        return splits