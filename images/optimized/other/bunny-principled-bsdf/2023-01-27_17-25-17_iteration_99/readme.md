### translucent principled bunny
Hyper parameters: default
Optimization parameters: default

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <rgb name="base_color" value="0.412, 0.824, 0.999"/>
    <float name="roughness" value="0.01"/>
    <float name="spec_trans" value="0.9"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized:
```
"object_bsdf.eta": [
    2.01404070854187
],
"object_bsdf.roughness.value": [
    0.0010000000474974513
],
"object_bsdf.base_color.value": "[[0.25010305643081665, 0.8068403601646423, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9990000128746033
]
```

#### Conclusion
Clearly not good enough. Index of refraction (eta) immediately pops us, since it is not even near the reference result. Other optimized parameters look better, however they can also improve. Eta seems to be the main suspect why other parameters are not near enough to corresponding reference value. Mainly because eta seems to move in the false direction -- we may put additional constraints for the eta parameter, such that it moves in the correct direction (i.e. it should decrease and near the reference value 1.49). One possible approach for this would be using max. clamp value for the eta parameter, which ensures the eta parameter to stay lower than the defined max clamp value during optimization. The provided default max. clamp value for eta is apparently not restrictive enough.