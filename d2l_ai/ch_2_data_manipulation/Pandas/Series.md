# Intro to Data Structures : Series

1. **Series** is a one-dimensional labeled array capable of holding any data type 
2. **Data** can be various things-
>1. Python dict
>2. ndarray
>3. scalar value
3. **Index** is collection of labels.

| S.no | Description                     | Sample Code                                        | Comments                                                            |
|------|---------------------------------|----------------------------------------------------|---------------------------------------------------------------------|
| 1.   | Create a series                 | s = pd.Series(data, index=index)                   | -                                                                   |
| 2.   | Series with data = ndarray      | s = pd.Series(np.random.randn(5))                  | 1. len(index) = len(data) 2. By default, index = [0,...len(data)-1] |
| 3.   | Series with data = dict         | s = pd.Series({'b': 1, 'a': 0, 'c': 2})            | 1. By default, series ordering is based on insertion order.         |
| 4.   | Series with data = scalar value | s = pd.Series(5., index=['a', 'b', 'c', 'd', 'e']) | 1. Index must be provided                                           |
