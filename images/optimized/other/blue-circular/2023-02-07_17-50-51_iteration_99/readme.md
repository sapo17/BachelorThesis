### blue algea
Hyper parameters: everything default except spp = 8 
Optimization parameters: everything default except the dual buffer method

Reference:
```
?
```


Optimized last it:
```
    "mat-material-bsdf.eta": [
        1.5389710664749146
    ],
    "mat-material-bsdf.roughness.value": [
        0.7054336667060852
    ],
    "mat-material-bsdf.spec_trans.value": [
        0.1552562266588211
    ],
    also bitmap texture
```

#### Conclusion
Interestingly, the visual results look identical to our previous attempt. However, analytically the results are somewhat different. This is especially the case for the \textit{spec\_trans} and \textit{roughness} parameters.

Consequently, we find these results quite troublesome since almost identical visual results provided us with different analytical results. Therefore we would like to note our approach limitation. Specifically, if the image acquisition and modeling phase are made in an amateur manner\textemdash as in our case\textemdash and if these results are accompanied by bitmap texture or grid-based volume data structure\textemdash since these types of representations may outweigh other parameters during the optimization (e.g., \textit{base\_color} represented with a bitmap texture may have a stronger influence than the \textit{spec\_trans} parameter)\textemdash the results may be \textbf{\textit{inaccurate}} and \textit{may not represent the physics-based reality}. 

However, suppose the modeling and image acquisition phase can be done more scientifically and assume that we may represent the physics-based reality with the proposed two approaches, namely the use of (1) Principled BSDF or (2) Rough dielectric BSDF with homogeneous participating media interior. In that case, we hypothesize, with the evidence that we acquired from the synthetic data results, that our approach may help the scientific and computer graphics community with different translucent material reconstruction tasks.