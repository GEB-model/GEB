import numpy as np

from geb.agents.general import AgentArray
    
def test_agent_array():
    # Test initialization with max_size
    a = AgentArray(np.array([1, 2, 3]), max_size=10)
    a_ = AgentArray(dtype=np.int64, n=3, max_size=10)
    a_[:3] = np.array([1, 2, 3])
    assert np.array_equal(a, a_)

    assert np.array_equal(a, np.array([1, 2, 3]))
    assert a.max_size == 10
    assert a.n == 3

    # Test addition with scalar
    a += 1
    assert np.array_equal(a, np.array([2, 3, 4]))

    # Test addition with array
    b = np.array([1, 2, 3])
    a += b
    assert np.array_equal(a, np.array([3, 5, 7]))

    # Test item assignment
    a[0] = 4
    assert np.array_equal(a, np.array([4, 5, 7]))

    # Test slicing
    assert np.array_equal(a[:2], np.array([4, 5]))

    # Test array methods
    assert np.array_equal(np.unique(a, return_counts=True), (np.array([4, 5, 7]), np.array([1, 1, 1])))
    assert a.sum() == 16
    assert np.array_equal(a.mean(), 16/3)
    assert a.std() == np.std(a)
    assert a.min() == np.min(a)
    assert a.max() == np.max(a)

    # Test setting n
    a.n = 2
    assert np.array_equal(a, np.array([4, 5]))
    assert a.n == 2

    # Test setting n and adding new items
    a.n = 3
    a[2] = 6
    assert np.array_equal(a, np.array([4, 5, 6]))
    assert a.n == 3

    # Test setting n and adding new items beyond max_size
    a.n = 4
    a[3] = 7
    assert np.array_equal(a, np.array([4, 5, 6, 7]))
    assert a.n == 4
    assert a.max_size == 10
    
    # Test setting n to exceed max_size
    try:
        a.n = 11
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

    # Test size property
    assert a.size == 4

    a.fill(42)
    assert np.array_equal(a, np.array([42, 42, 42, 42]))

if __name__ == '__main__':
    test_agent_array()