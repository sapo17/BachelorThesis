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


Optimization first part, it #23:
```
    "mat-ObjectBsdf.alpha.value": [
        0.0010000000474974513
    ],
    "medium1.sigma_t.value.value": [
        0.35937243700027466
    ],
    albedo: see optimized data structure
```

Optimization second part, it #44:
```
    "mat-ObjectBsdf.eta": [
        1.4861127138137817
    ],
    "mat-ObjectBsdf.alpha.value": [
        0.008117001503705978
    ],
    "medium1.sigma_t.value.value": [
        0.3556882441043854
    ],
    albedo: see optimized data structure
```


#### Conclusion
With this approach we observe relatively acceptable visual and analytical results. The main drawback of this appraoch is dividing the optimization into two parts. Consequently, with this appraoch there are two steps that the user needs to manually accomplish. First, the user should find the group of parameters that the optimization works well together and start the procedure and eventually export them. Second, starting the optimization by including the parameter that cause difficulty in the first place and using the exported optimized parameters from the first half of the procedure. More specifically, In case of difficulties during translucent material reconstruction, our recommendation is including the eta parameter only in the second part of the reconstruction procedure, as we showed in Figure X.

