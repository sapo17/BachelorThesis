### translucent principled bunny
Hyper parameters: iteration count 200, spp = 16, others default
Optimization parameters: eta clamped to 1.55, eta lr = 0.008

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <rgb name="base_color" value="0.412, 0.824, 0.999"/>
    <float name="roughness" value="0.01"/>
    <float name="spec_trans" value="0.9"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized last iteration:
```
"object_bsdf.eta": [
    1.4766746759414673
],
"object_bsdf.roughness.value": [
    0.023466331884264946
],
"object_bsdf.base_color.value": "[[0.4101334810256958, 0.8309170007705688, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9098701477050781
],
```

Optimized at minimum error #192:
```
    "object_bsdf.eta": [
        1.489931344985962
    ],
    "object_bsdf.roughness.value": [
        0.02132589742541313
    ],
    "object_bsdf.base_color.value": "[[0.40548327565193176, 0.8320051431655884, 0.9990000128746033]]",
    "object_bsdf.spec_trans.value": [
        0.9084774851799011
    ],
```

#### Conclusion
This time, the results seem even much better to our previous successful attempt! It seems like this time we don't even have to use our "magic-button", the parameter error plot is quite smooth and the last iteration visually seems acceptable enough. However for the sake of completeness we use our "magic-button" and switch to the scene state with the minimum error. Visually there are not much noticable changes, the minimum error seems to be achieved in the 192. iteration, and resulting parameter values seems also quite near to the scene state in the last iteration. Once again, we further conclude that the noise introduced by MC sampling plays a role during the optimization. The higher spp value provided us with better results, but one must note that with higher spp values, the optimization procedure takes longer. In this paper, we don't go in to detail of the duration of the optimization procudere, and leave it as future point of analysis. Footnote: The difference in duration between 4, 8, 16 spp values are not grand. We direct this to the simplicity of the current scene (i.e. geometry, amount of triangles, etc.) and the power of mitsuba 3 CUDA implementation.

In their paper, Deng et al. proposes an optimization function to handle the noise introduced by the BSSRDF integral for evaluating the loss during optimization. Although we are not using the BSSRDF material model, we naively make use of their proposed method -- namely the dual buffer method -- in our procedure as the optimization function g(y). In short, the main difference of the dual buffer method introduced by Deng et al. is rather taking one differentiable rendering step, taking two differentiable rendering step for one iteration. More specifically, our default optimization function is the L2 error, TODO:math. As shown in Section X, in the dual buffer method, we put two differentiable rendering to compute the loss at any iteration i, thus TODO: math. Instinctively and intellectually the proposed method by Deng et al. makes sense, since we concluded that the noise introduced by MC plays a role during the optimization procedure, and thus using two independent renders at once should lower noise introduced by MC sampling at one iteration step. For the theoretical ramifications and derivation of the dual buffer method, we highly and kindly refer the reader to the corresponding paper (TODO: cite).

Having introduced the dual buffer method more in detail, we test the same translucent material estimation problem as in X using the Dual Buffer method. As in X, we begin with overall default optimization, and hyper parameters.