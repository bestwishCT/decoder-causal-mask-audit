import json, math
def softmax(values):
    maximum=max(values)
    exponents=[math.exp(value-maximum) for value in values]
    return [value/sum(exponents) for value in exponents]
allowed=[[int(column<=row) for column in range(3)] for row in range(3)]
weights=[softmax([0 if keep else -math.inf for keep in row]) for row in allowed]
result={'teaching_only':True,'rows':'query','columns':'key','target':['A','B','EOS'],'input':['BOS','A','B'],'allowed':allowed,'weights':weights,'zero_is_not_mask':softmax([0,0,0]),'negative_legal_counterexample':softmax([-100,0,0]),'post_softmax_first_row_without_renorm':[1/3,0,0]}
assert all(math.isclose(sum(row),1) for row in weights)
assert weights[0]==[1,0,0] and result['negative_legal_counterexample'][0]<1e-40
print(json.dumps(result,indent=2))
