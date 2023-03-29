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
        2.2165656089782715
    ],
    "mat-ObjectBsdf.alpha.value": [
        0.12290383875370026
    ],
    "medium1.sigma_t.value.value": [
        0.26450783014297485
    ],
    albedo: see optimized data structure
```


#### Conclusion
In our first attempt both visual and analytical results are quite poor. The first parameter that pops to eye that is again the eta parameter. Since previously the eta parameter also caused similar issues, we know roughly how our second approach should look. We expect that if the eta parameter moves in the right direction the other parameters will also improve. Consequently, in our second attempt, we test our hypothesis by setting the maximum clamp value of eta to 1.55 and leave other parameters as in our first attempt. The results are shown in Figure X and Table X.