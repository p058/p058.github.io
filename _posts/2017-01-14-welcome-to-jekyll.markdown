---
layout: post
title:  "Adding Multiple Columns to Spark DataFrames"
date:   2017-01-14 18:47:51 -0600
categories: Spark
---

I have been using spark's dataframe API for quite sometime and often I would want to add many columns to a dataframe(for ex :  Creating more features from existing features for a machine learning model) and find it hard to write many withColumn statements. So I monkey patched spark dataframe to make it easy to add multiple columns
to spark dataframe.

First we create a 'udf_wrapper' decorator to keep the code concise

```python
from pyspark.sql.functions import udf

def udf_wrapper(returntype):

    def udf_func(func):

        return udf(func, returnType=returntype)

    return udf_func
```

Lets create a spark dataframe with columns, `user_id`, `app_usage (app and number of sessions of each app)`, `hours active`

```python
from  pyspark.sql import Row

hc = HiveContext(sc)

data = hc.createDataFrame([Row(user_id = 1, app_usage = {'snapchat' : 2, 'facebook' : 10, 'gmail' : 1}, active_hours = {4 : 1, 6 : 11, 22 : 1}),
                          
                          Row(user_id = 2, app_usage = {'tinder' : 100, 'zoosk' : 3, 'hotmail' : 2}, active_hours = {6 : 2, 18: 23, 23 : 80}),
                          
                          Row(user_id = 3, app_usage = {'netflix' :50, 'facebook' : 5, 'amazon' : 10}, active_hours = {10 : 4, 19 : 6, 20 : 55})])
```

Now lets add a column `total_app_usage` indicating total number of sessions

```python
from  pyspark.sql.types import *

@udf_wrapper(IntegerType())
def get_total_app_usage(x):
    
    return sum(x.values())

data_new = data.withColumn("total_app_usage", get_total_app_usage(data["app_usage"]))
    
```

Now lets add a indicator column `evening_user` indicating whether or not the user is active between 18-21 hours

```python

@udf_wrapper(StringType())
def get_evening_user(x):
 
    hours_active = set(x.keys())
    
    night_hours = set(range(18, 21))
    
    len_intersect = len(hours_active.intersection(night_hours))

    return '1' if len_intersect > 0 else '0'
    
data_new = data_new.withColumn("evening_user", get_evening_user(data["active_hours"]))
```

Instead of writing multiple withColumn statements lets create a simple util function to apply multiple functions to multiple columns

```python

from pyspark.sql import DataFrame

def add_columns(self, list_of_tuples):

    """
    :param self: Spark DataFrame
    :param list_of_tuples: [
    ("old_column_1", ["new_column_1", "new_column_2"], ["func_1", "func_2"]),
    ("old_column_2", ["new_column_3", "new_column_4"], ["func_2", "func_3"])
    (["old_column_1","old_column_2"], ["new_column_5"], ["func_4"])
    ]
    :return: Spark DataFrame with new columns
    """

    for col in list_of_tuples:

        prev_col = col[0]

        if isinstance(prev_col, list):
        
            cols = [self[j] for j in prev_col]

            for new_col, func in zip(col[1], col[2]):

                self = self.withColumn(new_col, func(*cols))

        else:

            for new_col, func in zip(col[1], col[2]):

                self = self.withColumn(new_col, func(self[prev_col]))

    return self

DataFrame.add_columns = add_columns
```

Now lets use the `add_columns` method to add multiple columns

```python

@udf_wrapper(StringType())
def most_used_key(x):
    
    """
    Accepts dict and returns the key with highest value
    
    """
    
    return sorted(x.items(), key = lambda x : x[1], reverse = True)[0][0]


@udf_wrapper(StringType())
def get_facebook_user(x):

    return '1' if 'facebook' in x.keys() else '0'

@udf_wrapper(StringType())
def total_usage_tally(x, y):

    """
    Check if the sum of sessions from app usage is same as sum of sesions from hour usage
    
    """

    return '1' if sum(x.values()) == sum(y.values()) else '0'


data_new = data.add_columns([("app_usage", ["most_used_app", "facebook_user", "total_app_usage"], [most_used_key, get_facebook_user, get_total_app_usage]),

                            ("active_hours", ["most_active_hour", "evening_user"], [most_used_key, get_evening_user]),
                            
                            (["app_usage", "active_hours"], ["usage_tally"], [total_usage_tally])
                            ])

```

You can also use builtin spark functions along with your udf's. You can also apply functions on multiple columns by passing the old columns as a list.


```python
data_new.take(3)
```

```
[Row(active_hours={10: 4, 19: 6, 20: 55}, app_usage={u'amazon': 10, u'facebook': 5, u'netflix': 50}, user_id=3, most_used_app=u'netflix', facebook_user=u'1', total_app_usage=65, most_active_hour=u'20', evening_user=u'1', usage_tally=u'1'),
 Row(active_hours={18: 23, 6: 2, 23: 80}, app_usage={u'zoosk': 3, u'hotmail': 2, u'tinder': 100}, user_id=2, most_used_app=u'tinder', facebook_user=u'0', total_app_usage=105, most_active_hour=u'23', evening_user=u'1', usage_tally=u'1'),
 Row(active_hours={4: 1, 6: 11, 22: 1}, app_usage={u'facebook': 10, u'snapchat': 2, u'gmail': 1}, user_id=1, most_used_app=u'facebook', facebook_user=u'1', total_app_usage=13, most_active_hour=u'6', evening_user=u'0', usage_tally=u'1')]
 
```

You can always switch back to a spark RDD, add columns, and convert it back to a dataframe but I don't find it to be a very neat way to do it.

Thanks for reading.