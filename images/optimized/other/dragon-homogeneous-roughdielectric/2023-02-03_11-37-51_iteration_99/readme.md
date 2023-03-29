### translucent homogeneous dragon
Hyper parameters: everything default except spp = 16
Optimization parameters: everything default except the dual buffer method

Reference:
```
<bsdf type="roughdielectric" id="mat-ObjectBsdf">
    <string name="distribution" value="ggx"/>
    <float name="alpha" value="0.01"/>
    <string name="int_ior" value="acrylic glass"/>
    <string name="ext_ior" value="air"/>
</bsdf>
<medium id="medium1" type="homogeneous">
    <float name="scale" value="5"/>
    <phase type="isotropic" />
    <float name="sigma_t" value="0.4"/>
    <volume name="albedo" type="gridvolume">
        <string name="filename" value="scenes\material-preview\textures\colored_albedo.vol"/>
    </volume>
    <boolean name="has_spectral_extinction" value="false"/>
    <boolean name="sample_emitters" value="false"/>
</medium>
```


Optimized last it:
```
    "mat-ObjectBsdf.eta": [
        1.5499999523162842
    ],
    "mat-ObjectBsdf.alpha.value": [
        0.3063618540763855
    ],
    "medium1.sigma_t.value.value": [
        0.269522100687027
    ],
    albedo: see optimized data structure
```


#### Conclusion
Interestingly our second approach also results in quite poor results. We observe that the eta parameter seems to stuck at value of 1.55, which is the maximum clamp value we defined for the eta parameter. More interestingly, after 100 iterations, the alpha parameter seems to also stuck at 0.3, which is also troublesome since it most likely affect the sigma_t and eta parameters correct convergence. Visually, the incorrectnes of the alpha parameter is quite clear, the surface of the dragon is extremly rough (TODO: cite mitsuba doc.).

In the meantime, we tried different configurations such that the optimized values near the reference values. For example we used higher spp values, larger iteration counts, lower or higher learning rates. Interestingly, on many of our attemps we observed that the alpha parameter remained around the range of values 0.2 to 0.3, which visually indicates quite rough surface material. Unfortunately, our attemps were in vain, and we were quite unsuccessful with our approach. For the sake of brevity, we omit the details regarding our unsuccessful attemps, however in the future it would be interesting to analyse which pitfalls--theoretical and practical--affects our translucent material reconstruction procedure. 

Next, we mainly focus which approach helps for the task at our hands. During the development period and testing the optimization procedure, we observed in some occasions optimizing some scene parameter separately that produce discontinuities provide better results. Consequently, as a last results, we separate the optimization into two parts: first we optimize the alpha, sigma\_t and albedo parameters defining the iteration count to 50. Next, with the optimized values from the first part, we restart the optimization by including the eta parameter with another 50 iterations. The results are shown in Figure X and Table X.

