import pandas as pd
import great_expectations as gx
from icecream import ic

data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eggy", "Frank","Gerry","Hilda"],
    "age": [25, 30, 25, 30, 25, 30, 35, None],
    "salary": [70_000, 80_000, 120_000, 110_000,70_000, 80_000, 110_000, 10_000]
}

df = pd.DataFrame(data)
ic(df)
context = gx.get_context()

data_source = context.data_sources.add_pandas('pandas')
ic(data_source)
