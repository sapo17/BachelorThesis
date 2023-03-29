### translucent principled bunny
Hyper parameters: default
Optimization parameters: eta clamped to 1.55

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
1.4532604217529297
],
"object_bsdf.roughness.value": [
0.21780212223529816
],
"object_bsdf.base_color.value": "[[0.4482169449329376, 0.9465304017066956, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
0.9477763772010803
],
```

#### Conclusion
This time the results seems better. If we look at eta, we see that this time it moved into the "right direction". However, this time rougness parameter seems to be quite off. Other parameters such as base color and specular transmission are not near enough to the reference as well. Visually, we can also clearly see the difference between reference and optimized image. This time the roughness parameters seems the be the suspect. However, compared to the previous example, roughness parameter seems to move in the "right direction" (i.e. actually nearing the reference value). Thus using the same approach as in the last case, does not seems to make a lot of sense. One approach* to handle this case would be increasing the iteration count, since roughness seems to move in the right direction, but most likely did not have enough "time" (i.e. iteration count) to near the target closer. Consequently, this time we only change the iteration count -- we double it to 200, the default was 100 -- and test our approach once again.

*Another option would be increasing the learning rate of roughness from 0.03 to possibily 0.06 (or more).