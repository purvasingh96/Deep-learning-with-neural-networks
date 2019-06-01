# Introduction to Data Structures : DataFrames

1. **DataFrame** is a 2-dimensional labeled data structure with columns of potentially different types.


| S.no. | Description                    | Sample Code                                                |
|-------|--------------------------------|------------------------------------------------------------|
| 1.    | Create dataframe               | d = {'one': [1., 2., 3., 4.],     'two': [4., 3., 2., 1.]} |
| 2.    | Selecting columns              | df[col]                                                    |
| 3.    | Deleting values in DF          | del df['two'], three = df.pop('three')                     |
| 4.    | Inserting scalar values        | df['foo'] = 'bar', df['one_trunc'] = df['one'][:2]         |
| 5.    | Select row by label            | df.loc[label]                                              |
| 6.    | Select row by integer location | df.iloc[loc]                                               |
| 7.    | Slice rows                     | df[5:10]                                                   |
| 8.    | Select rows by boolean vector  | df[bool_vec]                                               |
| 9.    | Transposing                    | df[:5].T                                                   |
