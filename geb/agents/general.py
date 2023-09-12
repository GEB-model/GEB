import numpy as np

class AgentArray(np.ndarray):
    implemented_methods = (
        'mean',
        'sum',
        'std',
        'min',
        'max',
        'n',
        '_n',
        'ndim',
        'shape',
        'dtype',
        'max_size',
        'size',
        '__class__',
        'view',
        '__repr__',
        '__str__',
    )
    def __new__(cls, input_array=None, n=None, max_size=None, dtype=None, fill_value=None):
        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")
        
        if input_array:
            if n is None and max_size is None:
                raise ValueError("Either n or max_size must be given")
            elif n is not None and max_size is not None:
                raise ValueError("Only one of n or max_size can be given")
            if max_size:
                obj = np.empty_like(input_array, shape=max_size).view(cls)
                n = input_array.size
                obj[:n] = input_array
                obj._n = n
            elif n:
                obj = input_array.view(cls)
                obj._n = n
        else:
            assert dtype is not None
            assert n is not None
            assert max_size is not None
            if fill_value:
                obj = np.full(max_size, fill_value, dtype=dtype).view(cls)
            else:
                obj = np.empty(max_size, dtype=dtype).view(cls)
            obj._n = n

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._n = getattr(obj, 'n', None)

    def __array_function__(self, func, types, args, kwargs):
        args_ = []
        for arg in args:
            if isinstance(arg, AgentArray):
                args_.append(arg[:arg.n].view(np.ndarray))
            else:
                args_.append(arg)
        return func(*args_, **kwargs)

    def __getattribute__(self, name):
        if name in AgentArray.implemented_methods:
            return super().__getattribute__(name)
        else:
            raise NotImplementedError("Attribute {} not implemented".format(name))
    
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, AgentArray):
                in_no.append(i)
                # print('i', input_, input_.n)
                args.append(input_[:input_.n].view(np.ndarray))
            else:
                args.append(input_)

        if method == 'reduce':
            return super().__array_ufunc__(ufunc, method, *args, **kwargs)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, AgentArray):
                    out_no.append(j)
                    out_args.append(output[:output.n].view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], AgentArray):
                inputs[0].n = n
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            ((np.asarray(result) if method == 'reduce' else np.asarray(result).view(AgentArray))
            if output is None else output)
            for result, output in zip(results, outputs)
        )
        for result in results:
            if isinstance(result, AgentArray):
                result.n = self._n
        return results[0] if len(results) == 1 else results

    @property
    def shape(self):
        return self[:self._n].view(np.ndarray).shape

    @property
    def size(self):
        return self[:self._n].view(np.ndarray).size

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value > self.max_size:
            raise ValueError("n cannot exceed max_size")
        self._n = value

    @property
    def max_size(self):
        return self.view(np.ndarray).size

    def __setitem__(self, key, value):
        self[:self._n].view(np.ndarray).__setitem__(key, value)

    def __getitem__(self, key):
        return self.view(np.ndarray)[:self._n].__getitem__(key)

    def __repr__(self):
        return self.view(np.ndarray)[:self._n].__repr__()

    def __str__(self):
        return self.view(np.ndarray)[:self._n].__str__()

    def __len__(self):
        return self._n