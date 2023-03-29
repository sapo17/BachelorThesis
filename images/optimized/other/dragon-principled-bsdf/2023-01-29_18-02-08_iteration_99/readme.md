### translucent principled bunny
Hyper parameters: everything default except spp = 8
Optimization parameters: everything default except we use dual buffer method, spec_trans: min clamp = 0.1

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <texture name="base_color" type="bitmap">
        <string name="filename" value="gradient.jpg"/> resolution 7200x4800!
    </texture>
    <float name="roughness" value="0.001"/>
    <float name="spec_trans" value="1.0"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized last iteration:
```
    "mat-ObjectBsdf.eta": [
        2.0180864334106445
    ],
    "mat-ObjectBsdf.roughness.value": [
        0.12999707460403442
    ],
    "mat-ObjectBsdf.spec_trans.value": [
        0.10000000149011612
    ],
basecolor: see outputted texture file
```

#### Conclusion
Unfortunately, we observe no improvements in our results neither visually or analytically. The spec_trans parameter seems to stuck in its minimum clamp value throughout the optimization. Morever, the eta parameter seems to be worsened and move even farther from our previous attempt. One key thing that we learned from our previous example (TODO: ref translucent bunny) was the noise introduced by MC sampling affect the optmization procedure. In this example, the geometry of the dragon is more complex than the bunny. Therefore, we presume that the noise introduced by MC sampling is higher during rendering, since the scattering of light from the object is more complex as well. With this assumption, we test our hypothesis and increase the spp value from 8 to 16. Additionally, we increase the iteration count from 100 to 200, and bump the minimum clamp value of the spec_trans parameters from 0.1 to 0.11, with the argument that the optimization may need more steps and a gentle push to move in the right direction.