import logging
def imbalance_statistics(imbalance_struct, stat=None, index='simpson', sensitive_features=None):
    if index not in ['gini', 'shannon', 'simpson', 'imbalance_ratio']:
        logging.warning("index parameter not valid. Use 'gini', 'shannon', 'simpson', 'imbalance_ratio'")
        return None
    if stat == 'min':
        result = 1
        for f in imbalance_struct['results']:
            if imbalance_struct['results'][f][index] < result:
                result = imbalance_struct['results'][f][index]
    elif stat == 'max':
        result = 0
        for f in imbalance_struct['results']:
            if imbalance_struct['results'][f][index] > result:
                result = imbalance_struct['results'][f][index]
    elif stat == 'mean':
        result = 0
        for f in imbalance_struct['results']:
            result += imbalance_struct['results'][f][index]
        result /= len(imbalance_struct['results'])
    elif stat == 'median':
        # compute the median
        values = []
        for f in imbalance_struct['results']:
            values.append(imbalance_struct['results'][f][index])
        values.sort()
        n = len(values)
        if n % 2 == 0:
            result = (values[n // 2 - 1] + values[n // 2]) / 2
        else:
            result = values[n // 2]
    elif stat == 'sensitiveloss':
        #sum of differences between 1 and the index of the sensitive features
        if sensitive_features is None:
            logging.warning("sensitive_features parameter is None. Use a list of sensitive features.")
            return None
        result = 0
        for f in imbalance_struct['results']:
            if f in sensitive_features:
                result += 1 - imbalance_struct['results'][f][index]
    elif stat == 'm_change':
        #sum of the number of unique values of the dataset
        result = 0
        for f in imbalance_struct['results']:
            result += abs(len(imbalance_struct['frequencies'][f]))
    else:
        logging.warning("stat parameter not valid. Use 'min', 'max', 'mean', 'median', 'sensitiveloss', 'm_change'.")
        return None
    return result