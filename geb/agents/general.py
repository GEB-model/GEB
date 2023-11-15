import numpy as np

class AgentArray:
    def __init__(self, input_array=None, n=None, max_n=None, dtype=None, fill_value=None):
        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")
        
        if input_array is not None:
            if n is None and max_n is None:
                raise ValueError("Either n or max_n must be given")
            elif n is not None and max_n is not None:
                raise ValueError("Only one of n or max_n can be given")
            if max_n:
                if input_array.ndim == 1:
                    shape = max_n
                elif input_array.ndim == 2:
                    shape = (max_n, input_array.shape[1])
                else:
                    raise ValueError("input_array can only be 1D or 2D")
                self._data = np.empty_like(input_array, shape=shape)
                n = input_array.shape[0]
                self._n = n
                self._data[:n] = input_array
            elif n:
                self._data = input_array
                self._n = n
        else:
            assert dtype is not None
            assert n is not None
            assert max_n is not None
            if fill_value:
                self._data = np.full(max_n, fill_value, dtype=dtype)
            else:
                self._data = np.empty(max_n, dtype=dtype)
            self._n = n

    @property
    def data(self):
        return self._data[:self.n]
    
    @data.setter
    def data(self, value):
        self._data[:self.n] = value
    
    @property
    def max_n(self):
        return self._data.size
    
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value > self.max_n:
            raise ValueError("n cannot exceed max_n")
        self._n = value

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self._data.__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __array_interface__(self):
        return self._data.__array_interface__()

    def __array_function__(self, func, types, args, kwargs):
        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args = tuple(
            arg.data if isinstance(arg, AgentArray) else arg for arg in args
        )
        modified_types = tuple(
            type(arg.data) if isinstance(arg, AgentArray) else type(arg) for arg in args
        )
        return self._data.__array_function__(func, modified_types, modified_args, kwargs)
    
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __repr__(self):
        return 'AgentArray(' + self.data.__str__() + ')'

    def __str__(self):
        return self.data.__str__()
    
    def __len__(self):
        return self._n
    
    def __getattr__(self, name):
        if name in ('_data', 'data', '_n', 'n'):
            return super().__getattr__(name)
        else:
            return getattr(self.data, name)
    
    def __setattr__(self, name, value):
        if name in ('_data', 'data', '_n', 'n'):
            super().__setattr__(name, value)
        else:
            setattr(self.data, name, value)

    def __getstate__(self):
        return self.data.__getstate__()
    
    def __setstate__(self, state):
        self.data.__setstate__(state)
    
    def __sizeof__(self):
        return self.data.__sizeof__()
    
    def __add__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__add__(other), max_n=self._data.shape[0])
    
    def __radd__(self, other):
        self.data = self.data.__radd__(other)
        return self
    
    def __iadd__(self, other):
        self.data = self.data.__add__(other)
        return self
    
    def __sub__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__sub__(other), max_n=self._data.shape[0])
    
    def __rsub__(self, other):
        self.data = self.data.__rsub__(other)
        return self
    
    def __isub__(self, other):
        self.data = self.data.__sub__(other)
        return self
    
    def __mul__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__mul__(other), max_n=self._data.shape[0])
    
    def __rmul__(self, other):
        self.data = self.data.__rmul__(other)
        return self
    
    def __imul__(self, other):
        self.data = self.data.__mul__(other)
        return self
    
    def __truediv__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__truediv__(other), max_n=self._data.shape[0])
    
    def __rtruediv__(self, other):
        self.data = self.data.__rtruediv__(other)
        return self
    
    def __itruediv__(self, other):
        self.data = self.data.__truediv__(other)
        return self
    
    def __floordiv__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__floordiv__(other), max_n=self._data.shape[0])
    
    def __rfloordiv__(self, other):
        self.data = self.data.__rfloordiv__(other)
        return self
    
    def __ifloordiv__(self, other):
        self.data = self.data.__floordiv__(other)
        return self
    
    def __mod__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__mod__(other), max_n=self._data.shape[0])
    
    def __rmod__(self, other):
        self.data = self.data.__rmod__(other)
        return self
    
    def __imod__(self, other):
        self.data = self.data.__mod__(other)
        return self
    
    def __pow__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[:other._n]
        return AgentArray(self.data.__pow__(other), max_n=self._data.shape[0])
    
    def __rpow__(self, other):
        self.data = self.data.__rpow__(other)
        return self
    
    def __ipow__(self, other):
        self.data = self.data.__pow__(other)
        return self
    
from numba import types
from numba.extending import typeof_impl, as_numba_type, type_callable, models, register_model, make_attribute_wrapper, lower_builtin
from numba.core import cgutils
from numba.extending import box, unbox, NativeValue
from contextlib import ExitStack
from numba.np import numpy_support

class AgentArrayType(types.Type):
    def __init__(self):
        super(AgentArrayType, self).__init__(name='AgentArray')

agent_array_type = AgentArrayType()
print(agent_array_type)

@typeof_impl.register(AgentArray)
def typeof_index(val, c):
    return agent_array_type

as_numba_type.register(AgentArray, agent_array_type)

# to fix
@type_callable(AgentArray)
def type_interval(context):
    def typer(data):
        if isinstance(data, types.Float):
            return agent_array_type
    return typer

@register_model(AgentArrayType)
class AgentArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [('data', types.float64)]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(AgentArrayType, 'data', 'data')

@lower_builtin(AgentArray, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    data, = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.data = data
    return interval._getvalue()

@unbox(AgentArrayType)
def unbox_array(typ, obj, c):
    """
    Convert a Numpy array object to a native array structure.
    """
    # This is necessary because unbox_buffer() does not work on some
    # dtypes, e.g. datetime64 and timedelta64.
    # TODO check matching dtype.
    #      currently, mismatching dtype will still work and causes
    #      potential memory corruption
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    if c.context.enable_nrt:
        errcode = c.pyapi.nrt_adapt_ndarray_from_python(obj, ptr)
    else:
        errcode = c.pyapi.numba_array_adaptor(obj, ptr)

    # TODO: here we have minimal typechecking by the itemsize.
    #       need to do better
    try:
        expected_itemsize = numpy_support.as_dtype(typ.dtype).itemsize
    except NumbaNotImplementedError:
        # Don't check types that can't be `as_dtype()`-ed
        itemsize_mismatch = cgutils.false_bit
    else:
        expected_itemsize = nativeary.itemsize.type(expected_itemsize)
        itemsize_mismatch = c.builder.icmp_unsigned(
            '!=',
            nativeary.itemsize,
            expected_itemsize,
            )

    failed = c.builder.or_(
        cgutils.is_not_null(c.builder, errcode),
        itemsize_mismatch,
    )
    # Handle error
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string("PyExc_TypeError",
                               "can't unbox array from PyObject into "
                               "native value.  The object maybe of a "
                               "different type")
    return NativeValue(c.builder.load(aryptr), is_error=failed)


@box(AgentArrayType)
def box_array(typ, val, c):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        # Steals NRT ref
        c.context.nrt.decref(c.builder, typ, val)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent