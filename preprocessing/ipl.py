import numpy as np
import pandas as pd
from itertools import product
import itertools


class IPL(object):
    def __init__(self, original, aggregates, weight_col='weight', n=None, learning_rate=1,
                 convergence_rate=1e-5, max_iteration=500, verbose=0, rate_tolerance=1e-8):
        """
        Initialize the ipfn class

        original: numpy darray matrix or dataframe to perform the ipfn on.

        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
        They are the target values that we want along one or several axis when aggregating along one or several axes.

        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
        Preserved dimensions along which we sum to get the corresponding aggregates.

        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.

        max_iteration: Integer. Maximum number of iterations allowed.

        verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
        0: Updated matrix returned.
        1: Flag with the output status (0 for failure and 1 for success).
        2: dataframe with iteration numbers and convergence rate information at all steps.

        rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value

        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = original
        self.aggregates = aggregates
        self.weight_col = weight_col
        self.n = n or sum(aggregates[0])
        self.learning_rate = learning_rate
        self.conv_rate = convergence_rate
        self.max_itr = max_iteration
        if verbose not in [0, 1, 2]:
            raise(ValueError(f"wrong verbose input, must be either 0, 1 or 2 but got {verbose}"))
        self.verbose = verbose
        self.rate_tolerance = rate_tolerance

    @staticmethod
    def index_axis_elem(dims, axes, elems):
        inc_axis = 0
        idx = ()
        for dim in range(dims):
            if (inc_axis < len(axes)):
                if (dim == axes[inc_axis]):
                    idx += (elems[inc_axis],)
                    inc_axis += 1
                else:
                    idx += (np.s_[:],)
        return idx

    def ipfn_df(self, df, aggregates, weight_col='weight'):
        steps = len(aggregates)
        tables = [df]
        for inc in range(steps - 1):
            tables.append(df.copy())

        # Calculate the new weights for each dimension
        inc = 0
        for aggregate in aggregates:
            feature = aggregate.name
            if inc == (steps - 1):
                table_update = df
                table_current = tables[inc].copy()
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]

            count_feature = isinstance(table_current[feature].iloc[0], tuple)

            xijk = aggregates[inc]

            feat_l = []
            if count_feature:
                table_sel = table_current[feature]
                items = [item for item in list(itertools.chain(*table_sel)) if item is not None]
                unique_items = np.unique(items)
                feat_l.append(unique_items)
                table_update.set_index(feature, inplace=True)
                table_current.set_index(feature, inplace=True)

                tmp = pd.Series(
                    0,
                    index=unique_items,
                    dtype=np.float64
                )
                for idx, row in table_update.iterrows():
                    for item in idx:
                        if item:
                            tmp.loc[item] += row[weight_col]
                
            else:
                feat_l.append(np.unique(table_current[feature]))

                table_update.set_index(feature, inplace=True)
                table_current.set_index(feature, inplace=True)

                tmp = table_current.groupby(feature)[weight_col].sum()

            if count_feature:
                update_table = pd.DataFrame(1, index=table_update.index, columns=list(product(*feat_l)))
                for characteristic in product(*feat_l):
                    den = tmp.loc[characteristic]
                    if den == 0:
                        den = 1

                    mask = np.array([characteristic[0] in idx for idx in table_update.index])
                    update_table.loc[mask, (characteristic, )] = (table_current[weight_col].astype(float) * xijk.loc[characteristic] / den)[mask == True] / table_current[weight_col].astype(float)[mask == True]
                    
                old_value = table_current[weight_col]
                new_value = update_table.mean(axis=1) * table_current[weight_col].astype(float)
                table_update[weight_col] = old_value + (new_value - old_value) * self.learning_rate
            
            else:
                for characteristic in product(*feat_l):
                    den = tmp.loc[characteristic]
                    if den == 0:
                        den = 1
                    # calculate new weight for this iteration
                    mask = table_update.index == characteristic[0]

                    table_update.loc[mask, weight_col] = \
                    old_value = table_current.loc[mask, weight_col].astype(float)
                    new_value = old_value * xijk.loc[characteristic] / den
                    table_current.loc[mask, weight_col] = old_value + (new_value - old_value) * self.learning_rate

            table_update.reset_index(inplace=True)
            table_current.reset_index(inplace=True)
            inc += 1
            feat_l = []

        table_update[weight_col] = table_update[weight_col] / table_update[weight_col].sum() * self.n

        # Calculate the max convergence rate
        max_conv = 0
        inc = 0
        for aggregate in aggregates:
            feature = aggregate.name
            count_feature = isinstance(table_current[feature].iloc[0], tuple)
            if count_feature:
                tmp = pd.Series(
                    0,
                    index=unique_items,
                    dtype=np.float64
                )
                for idx, row in table_update.set_index(feature).iterrows():
                    for item in idx:
                        if item:
                            tmp.loc[item] += row[weight_col]
                
            else:
                tmp = table_update.groupby(feature)[weight_col].sum()
            ori_ijk = aggregates[inc]
            temp_conv = max(abs(tmp / ori_ijk - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv
            inc += 1
        
        return table_update, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or pandas dataframes.
        """

        i = 0
        conv = np.inf
        old_conv = -np.inf
        conv_list = []
        m = self.original

        # If the original data input is in pandas DataFrame format
        while ((i <= self.max_itr and conv > self.conv_rate) and (i <= self.max_itr and abs(conv - old_conv) > self.rate_tolerance)):
            old_conv = conv
            m, conv = self.ipfn_df(m, self.aggregates, self.weight_col)
            conv_list.append(conv)
            i += 1
        converged = 1
        if i <= self.max_itr:
            if (not conv > self.conv_rate) & (self.verbose > 1):
                print('ipfn converged: convergence_rate below threshold')
            elif not abs(conv - old_conv) > self.rate_tolerance:
                print('ipfn converged: convergence_rate not updating or below rate_tolerance')
        else:
            print('Maximum iterations reached')
            converged = 0

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        elif self.verbose == 2:
            return m, converged, pd.DataFrame({'iteration': range(i), 'conv': conv_list}).set_index('iteration')
        else:
            raise(ValueError(f'wrong verbose input, must be either 0, 1 or 2 but got {self.verbose}'))


if __name__ == '__main__':
        age = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        distance = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
        weights = [8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.]
        df = pd.DataFrame()
        df['distance'] = distance
        df['age'] = age
        df['crops'] = [('corn', 'rice'), (None, 'corn'), ('wheat', None), ('rice', 'corn'), (None, 'rice'), ('wheat', None), (None, 'rice'), ('wheat', 'rye'), ('wheat', None), (None, 'rice'), ('wheat', 'rye'), (None, 'rice')]
        df['weight'] = weights

        aggregates = [
            # pd.Series([20, 18, 22], index=[30, 40, 50], name=('age')),
            # pd.Series([10, 10, 18, 22], index=[10, 20, 30, 40], name=('distance')),
            pd.Series([30, 10, 15, 30], index=['wheat', 'rice', 'corn', 'rye'], name='crops'),
        ]

        ipl = IPL(df, aggregates, n=60, rate_tolerance=1e-1000, convergence_rate=1e-100, max_iteration=500)
        for _ in range(1):
            df = ipl.iteration()

        print('done')
        print(df)
        print(df['weight'].sum())
        # print(df.groupby('age')['weight'].sum())
        # print(df.groupby('age')['weight'].sum().sum())
        # print(df.groupby('crops')['weight'].sum())
        # print(df.groupby('crops')['weight'].sum().sum())

        crops = {
            'wheat': 0,
            'rice': 0,
            'corn': 0,
            'rye': 0,
        }
        for i, row in df.iterrows():
            crop1, crop2 = row['crops']
            if crop1:
                crops[crop1] += row['weight']
            if crop2:
                crops[crop2] += row['weight']

        print(crops)
        print(sum(crops.values()))


