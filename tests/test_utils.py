from dcarte.utils import (date2iso,
                          merge_dicts)
import pytest

@pytest.mark.parametrize("test_input,expected", 
                         [('2021-06-09', '2021-06-09T00:00:00.000000Z'),
                          ('2021-06-09 08:10', '2021-06-09T08:10:00.000000Z')])
def test_date2iso(test_input,expected):
    assert date2iso(test_input) == expected
    
@pytest.mark.parametrize("d1,d2,expected", 
                         [(dict(one=1, two=2, three=3),
                            dict(one=1, four=4, three=3),
                            dict(one=1, two=2,four=4, three=3))])    
def test_merge_dicts(d1,d2,expected):
    assert dict(merge_dicts(d1,d2)) == expected