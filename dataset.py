import dataclasses

from typing import Union, Callable, Iterable, Iterator
import polars as pl
from polars.dataframe.frame import DataFrame
from polars.dataframe.group_by import GroupBy, DynamicGroupBy, RollingGroupBy
from polars.lazyframe.frame import LazyFrame
from polars.lazyframe.group_by import LazyGroupBy

DataFrameLikeClasses = (
    DataFrame,
    GroupBy,
    DynamicGroupBy,
    RollingGroupBy,
    LazyFrame,
    LazyGroupBy,
)
DataFrameLikeType = Union[
    DataFrame, GroupBy, DynamicGroupBy, RollingGroupBy, LazyFrame, LazyGroupBy
]


@dataclasses.dataclass(frozen=True)
class PolarsDataset:
    """A polars DataFrame with custom attributes.

    It works as follows:
    To create a PolarsDataset, you pass it ``data`` (a polars ``DataFrame``) and some
    additional attributes.
    When accessing attributes:
        * If the attribute is found on this class, it is returned.
        * If the attribute is not found on this class, it is searched
        for on the instance's ``data``.

    If the attribute is found, it is processed as follows:
        * If it is ``DataFrame``-like, it is cast to a ``PolarsDataset``
        using ``dataclasses.replace``.
        * If it is callable, and returns a ``DataFrame``-like object,
        the result of the callable is cast to a ``PolarsDataset`` (using a wrapper function).
        * Otherwise, the attribute is returned as-is.

    Setters are disabled: this means that both a PolarsDataset and its ``data``
    are immutable.
    """

    data: DataFrameLikeType
    extra_attr: str = "hello!"

    def __getattr__(self, __name: str):
        attr = getattr(self.data, __name)
        if isinstance(attr, DataFrameLikeClasses):
            return dataclasses.replace(self, data=attr)
        elif callable(attr):
            return self._wrap_method(attr)
        else:
            return attr

    def _wrap_method(self, __func: Callable):
        def wrapper(*args, **kwargs):
            result = __func(*args, **kwargs)
            if isinstance(result, DataFrameLikeClasses):
                return dataclasses.replace(self, data=result)

            return result

        return wrapper

    def _comp(self, other, op) -> "PolarsDataset":
        """Compare a PolarsDataset with another object."""
        if isinstance(other, PolarsDataset):
            return self._compare_to_other_df(other, op)
        else:
            return self._compare_to_non_df(other, op)

    def __eq__(self, other) -> "PolarsDataset":  # type: ignore[override]
        return self._comp(other, "eq")

    def __ne__(self, other) -> "PolarsDataset":  # type: ignore[override]
        return self._comp(other, "neq")

    def __gt__(self, other) -> "PolarsDataset":
        return self._comp(other, "gt")

    def __lt__(self, other) -> "PolarsDataset":
        return self._comp(other, "lt")

    def __ge__(self, other) -> "PolarsDataset":
        return self._comp(other, "gt_eq")

    def __le__(self, other) -> "PolarsDataset":
        return self._comp(other, "lt_eq")

    def _apply_op(self, __other, __op: str, *args, **kwargs):
        """Apply an operation to a PolarsDataset."""
        if isinstance(__other, PolarsDataset):
            return getattr(self, __op)(__other.data, *args, **kwargs)
        return dataclasses.replace(
            self,
            data=getattr(self.data, __op)(__other, *args, **kwargs),
        )

    def __floordiv__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__floordiv__")

    def __truediv__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__truediv__")

    def __mul__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__mul__")

    def __rmul__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__rmul__")

    def __add__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__add__")

    def __radd__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__radd__")

    def __sub__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__sub__")

    def __mod__(self, other) -> "PolarsDataset":
        return self._apply_op(other, "__mod__")

    def __getstate__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self._wrap_method(self.data.__getitem__)(item)

    def __iter__(self):
        return self.data.__iter__()

    def __reversed__(self):
        return self.data.__reversed__()

    def __bool__(self):
        return self.data.__bool__()

    def __len__(self):
        return len(self.data)

    def __copy__(self):
        return dataclasses.replace(self)

    def join(self, other: "PolarsDataset", *args, **kwargs):
        return self._apply_op(other, "join", *args, **kwargs)

    def join_asof(self, other: "PolarsDataset", *args, **kwargs):
        return self._apply_op(other, "join_asof", *args, **kwargs)

    def partition_by(self, *args, **kwargs):
        result = self.data.partition_by(*args, **kwargs)
        if isinstance(result, list):
            return [dataclasses.replace(self, data=new_data) for new_data in result]
        elif isinstance(result, dict):
            return {name: dataclasses.replace(self, data=new_data) for name, new_data in result.items()}
        else:
            assert False, "Unknown return type from ``partition_by``"

def concat(items: Iterable[PolarsDataset], base: PolarsDataset, **kwargs):
    new_data = pl.concat((item.data for item in items), **kwargs)
    return dataclasses.replace(base, data=new_data)
