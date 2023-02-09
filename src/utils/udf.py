import pandas as pd


def mem_usage(pandas_obj) -> str:
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        # исходим из предположения о том, что если это не DataFrame, то это Series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # преобразуем байты в мегабайты
    return "{:03.2f} MB".format(usage_mb)
