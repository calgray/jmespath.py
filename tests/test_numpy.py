import numpy as np
from parameterized import parameterized

from tests import unittest

import jmespath
import jmespath.functions


class TestNumpyNumeric(unittest.TestCase):
    def setUp(self):
        self.data = {
            "a": {
                "data": np.array([[1,2],[3,4]])
            },
            "b": {
                "data": np.array([[1,2],[3,4]])
            },
            "c": {
                "data": np.array([[2,2],[3,4]])
            }
        }

    @parameterized.expand([
        ["self", "@", lambda data: data],
        ["get", "a.data", lambda data: data["a"]["data"]],
        ["slice_horizontal", "a.data[1][:]", lambda data: data["a"]["data"][1,:]],
        ["slice_vertical", "a.data[:][1]", lambda data: data["a"]["data"][:,1]],
        ["compare_self", "a.data == a.data", lambda _: True],
        ["compare_same", "a.data == b.data", lambda _: True],
        ["compare_other", "a.data == c.data", lambda _: False],
        ["compare_literal_scalar", "a.data[0][0] == `1`", lambda _: True],
        ["compare_literal_slice", "a.data[1][:] == `[3, 4]`", lambda _: True],
        ["compare_literal", "a.data == `[[1,2],[3,4]]`", lambda _: True],
    ])
    def test_search(self, _, query, expected):
        result = jmespath.search(query, self.data)
        np.testing.assert_equal(result, expected(self.data))


class TestNumpyStr(unittest.TestCase):
    def setUp(self):
        self.data = {
            "a": { 
                "data": np.array(
                    [
                        ["test", "messages"],
                        ["in", "numpy"]
                    ]
                )
            },
            "b": { 
                "data": np.array(
                    [
                        ["test", "messages"],
                        ["in", "numpy"]
                    ]
                )
            },
            "c": { 
                "data": np.array(
                    [
                        ["test", "messages"],
                        ["other", "numpy"]
                    ]
                )
            }
        }

    @parameterized.expand([
        ["self", "@", lambda data: data],
        ["get", "a.data", lambda data: data["a"]["data"]],
        ["slice_horizontal", "a.data[1][:]", lambda data: data["a"]["data"][1,:]],
        ["slice_vertical", "a.data[:][1]", lambda data: data["a"]["data"][:,1]],
        ["compare_self", "a.data == a.data", lambda _: True],
        ["compare_same", "a.data == b.data", lambda _: True],
        ["compare_other", "a.data == c.data", lambda _: False],
        ["compare_literal_scalar", "a.data[0][0] == 'test'", lambda _: True],
        ["compare_literal_slice", "a.data[1][:] == ['in', 'numpy']", lambda _: True],
        ["compare_literal", "a.data == [['test', 'messages'],['in', 'numpy']]", lambda _: True],
    ])
    def test_search(self, _, query, expected):
        result = jmespath.search(query, self.data)
        np.testing.assert_equal(result, expected(self.data))
