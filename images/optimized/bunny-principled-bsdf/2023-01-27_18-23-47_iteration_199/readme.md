### translucent principled bunny
Hyper parameters: iteration count 200, others default
Optimization parameters: eta clamped to 1.55, eta lr = 0.003

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
    1.5393612384796143
],
"object_bsdf.roughness.value": [
    0.0010000000474974513
],
"object_bsdf.base_color.value": "[[0.4131670296192169, 0.9194056391716003, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9541535377502441
],
```

#### Conclusion
Overall, the result looks better. This time eta is not way off the result, but it seems it did not moved too much from its initial value (1.54). This time we may have lowered the learning rate too much! We increate the learning rate of eta to 0.008. Another tool disposal at our hand is using different samples per pixel value during optimization. With the argument that the noise introduced by the monte carlo sampling may cause not near enough results, we increase the samples per pixel value from 4 to 8.

Ironically enough, the gradient-based **optimization** tool that we have developed requires also an **optimization** on the author's end.