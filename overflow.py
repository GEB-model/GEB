import numpy as np

class Variable(np.ndarray):
    def __new__(cls, a, unit):
        obj = np.asarray(a).view(cls)
        obj.unit = unit
        return obj

    def split_unit(self, i):
        if self.unit == 'm':
            out = np.insert(self, i, self[i])
        else:
            raise NotImplementedError
        return out

    def __array_finalize__(self, obj):
        if obj is None: return
        self.unit = getattr(obj, 'unit', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        units = []
        for input in inputs:
            if not isinstance(input, (int, float)):
                units.append(input.unit)

        if ufunc.__name__ == 'add':
            assert len(units) == 1 or units[0] == units[1]
            unit = units[0]
        elif ufunc.__name__ == 'multiply':
            if len(units) == 1:
                unit = units[0]
            elif len(units) == 2:
                if units[0] == 'm' and units[1] == 'm':
                    unit = 'm2'
                elif units[0] == 'm' and units[1] == 'm2' or units[1] == 'm' and units[0] == 'm2':
                    unit = 'm3'
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Variable):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Variable):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], Variable):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Variable)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], Variable):
            results[0].info = info

        results[0].unit = unit

        return results[0] if len(results) == 1 else results

v = Variable(np.arange(1, 10), unit='m')
w = Variable(np.arange(1, 10), unit='m2')
print(type(v))
print(v.unit)
r = v.split_unit(3)
print(r)
print(r.unit)
r = v + 1
r = v * w
print(r)
print(r.unit)