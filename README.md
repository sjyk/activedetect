# ActiveDetect
ActiveDetect is a python package that detects and prioritizes the most important data errors in a dataset.

## Installation

For Python 2.7, install the following dependencies
* gensim
* numpy
* scipy
* usaddress
* sklearn

## Example Run

### Model-Free
The first use case of ActiveDetect is *model-free* detection, i.e., find errors in a dataset independent of the subsequent analysis.
We provide a number of example datasets to test the code, one interesting dataset is a U.S Census dataset with demographic information about residents. This demographic information has several missing values, and we will use ActiveDetect to find the errors.

```
from loaders.csv_loader import CSVLoader

loadedData = c.loadFile('datasets/adult.data')
```

Then, we can run the ErrorDetector, this error detector test all possible errors in a dataset (so it's slow!):
```
detector = ErrorDetector(loadedData)
detector.fit()
```

To get all of the errors, we provide an iterator interface:
```
for error in detector:
	print error
```
Now there are only 23 remaining errors.

Errors will look like this:
```
{'cell': (32518, 10), 'cell_value': ' 99999', 'error_types': ['quantitative'], 'record_value': ['57', ' Local-gov', ' 110417', ' HS-grad', ' 9', ' Married-civ-spouse', ' Craft-repair', ' Husband', ' White', ' Male', ' 99999', ' 0', ' 40', ' United-States', ' >50K']}
{'cell': (31821, 6), 'cell_value': ' ?', 'error_types': ['semantic'], 'record_value': ['36', ' ?', ' 229533', ' HS-grad', ' 9', ' Married-civ-spouse', ' ?', ' Husband', ' White', ' Male', ' 0', ' 0', ' 40', ' United-States', ' <=50K']}

```

### Model-Based
Running the model-free detector returns over 3800 errors out of about 3200 records. However, not all errors are that important to the subsequent analysis. We show how to use a model-based selector, which identifies errors that seem to result in mispredictions downstream, to do this. 

We trained a RandomForest classifier to predict the income level from the dataset, and the mispredictions are stored in `datasets/adult-rl-misp.p`. First, load the set of mispredictions:
```
import pickle
m = pickle.load(open('datasets/adult-rl-misp.p','rb'))
```

Next, create an ErrorDetector as before, but don't run fit:
```
e = ErrorDetector(loadedData)
```

Let us apply the simplest model-based filter, restricting the errors to only mispredictions:
```
from model_based.HardFilter import HardFilter
filter = HardFilter(e, m)
filter.fit()
```

To get the filtered errors:
```
for error in filter:
	print error
```

There are now only 346! Suppose, we wanted to filter out mispredictions that were just random noise, we can alternatively apply the SafeSetFilter. This requires the actual values of the dataset:
```
unlabeleddataset = [d[0:len(d)-1] for d in loadedData] #gets all but the label
labels = np.array([int('<' in d[-1]) for d in loadedData]) #turns label into a binary vector
```

Then, we can apply the SafeSetFilter
```
from model_based.SafeSetFilter import SafeSetFilter
filter = SafeSetFilter(e, m, unlabeleddataset, labels)
filter.fit()
for error in filter:
	print error
```








