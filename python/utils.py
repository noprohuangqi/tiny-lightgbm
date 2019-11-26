


def load_data():



    
def reshape_to_c(data,is_label):

    shape0 = data.shape(0)
    shape1 = data.shape(1)

    if is_label:
        data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return data , shape0 ,shape1

    data = np.array(data.reshape(data.size), dtype=data.dtype, copy=False)
    data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    return data , shape0 ,shape1

def c_str(string):
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode('utf-8'))


def param_dict_to_str(data):
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or is_numpy_1d_array(val):
            pairs.append(str(key) + '=' + ','.join(map(str, val)))
        elif isinstance(val, string_type) or isinstance(val, numeric_types) or is_numeric(val):
            pairs.append(str(key) + '=' + str(val))
        elif val is not None:
            raise TypeError('Unknown type of parameter:%s, got:%s'
                            % (key, type(val).__name__))
    return ' '.join(pairs)