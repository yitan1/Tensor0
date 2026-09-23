"""Storage dtype cases shared by numerical, AD and C++ tests."""


REDUCTION_DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
)


FLOAT_PAIRS = [
    ('float16', 'float32'),
    ('float32', 'complex64'),
    ('complex64', 'float32'),
    ('float64', 'complex128'),
    ('complex128', 'float64'),
    ('float32', 'float16'),
]


INTEGER_PAIRS = [('int32', 'bool'), ('int32', 'int8'), ('int32', 'int16'), ('uint32', 'uint8')]


PAIRS = FLOAT_PAIRS + INTEGER_PAIRS
