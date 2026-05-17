from simple_config.utils import merge_list


def test_add_new_item():
    l1 = [1, 2, 3]
    l2 = [4]

    merge_list(l1, l2)

    assert l1 == [1, 2, 3, 4]

def test_dont_add_existing_items():
    l1 = [1, 2, 3]
    l2 = [1, 2]

    merge_list(l1, l2)

    assert l1 == [1, 2, 3]

def test_dont_reorder_existing_items():
    l1 = [1, 2, 3]
    l2 = [2, 1, 3]

    merge_list(l1, l2)

    assert l1 == [1,2,3]