from gutenberg_aggregate import sum_stats


def test_sum_stats():

    expected = {'1': 3, '2': 6, '3': 9}
    result = sum_stats('./utf8/postprocessors/test_data/')

    print(result)

    assert expected == result['by_sentence']['char_length']


# MAYBE: an actual test framework?
if __name__ == '__main__':
    test_sum_stats()
