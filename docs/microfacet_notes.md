### Microfacet surfaces, notes from PBRT
Many geometric-optics-based approaches to modeling surface reflection and transmission are based on the idea that rough surfaces can be modeled as a collection of small microfacets. Surfaces comprised of microfacets are often modeled as heightfields, where the distribution of facet orientations is described statistically. Figure 8.12 shows cross sections of a relatively rough surface and a much smoother microfacet surface. When the distinction isn’t clear, we’ll use the term microsurface to describe microfacet surfaces and macrosurface to describe the underlying smooth surface (e.g., as represented by a Shape).

Microfacet-based BRDF models work by statistically modeling the scattering of light from a large collection of microfacets. If we assume that the differential area being illuminated, , is relatively large compared to the size of individual microfacets, then a large number of microfacets are illuminated and it’s their aggregate behavior that determines the observed scattering.

The two main components of microfacet models are a representation of the distribution of facets and a BRDF that describes how light scatters from individual microfacets. Given these, the task is to derive a closed-form expression giving the BRDF that describes scattering from such a surface. Perfect mirror reflection is most commonly used for the microfacet BRDF, though specular transmission is useful for modeling many translucent materials, and the Oren–Nayar model (described in the next section) treats microfacets as Lambertian reflectors.

To compute reflection from such a model, local lighting effects at the microfacet level need to be considered (Figure 8.13). Microfacets may be occluded by another facet, may lie in the shadow of a neighboring microfacet, or interreflection may cause a microfacet to reflect more light than predicted by the amount of direct illumination and the low-level microfacet BRDF. Particular microfacet-based BRDF models consider each of these effects with varying degrees of accuracy. The general approach is to make the best approximations possible, while still obtaining an easily evaluated expression.

####  Oren–Nayar Diffuse Reflection
Oren and Nayar (1994) observed that real-world objects do not exhibit perfect Lambertian reflection. Specifically, rough surfaces generally appear brighter as the illumination direction approaches the viewing direction. They developed a reflection model that describes rough surfaces by V-shaped microfacets described by a spherical Gaussian distribution with a single parameter , the standard deviation of the microfacet orientation angle.

Under the V-shape assumption, interreflection can be accounted for by only considering the neighboring microfacet; Oren and Nayar took advantage of this to derive a BRDF that models the aggregate reflection of the collection of grooves.

The resulting model, which accounts for shadowing, masking, and interreflection among the microfacets, does not have a closed-form solution, so they found the following approximation that fit it well.

#### microfacet dist. functions
Reflection models based on microfacets that exhibit perfect specular reflection and transmission have been effective at modeling light scattering from a variety of glossy materials, including metals, plastic, and frosted glass.

A widely used microfacet distribution function based on a Gaussian distribution of microfacet slopes is due to Beckmann and Spizzichino (1963); 

Another useful microfacet distribution function is due to Trowbridge and Reitz (1975). (this model is independently derived by Walter et al. (2007), who dubbed it GGX.)
In comparison to the Beckmann–Spizzichino model, Trowbridge–Reitz has higher tails—it falls off to zero more slowly for directions far from the surface normal. This characteristic matches the properties of many real-world surfaces well.

#### Masking and shadowing
The distribution of microfacet normals alone isn’t enough to fully characterize the microsurface for rendering. It’s also important to account for the fact that some microfacets will be invisible from a given viewing or illumination direction because they are back-facing (and thus, other microfacets are in front of them) and also for the fact that some of the forward-facing microfacet area will be hidden since it’s shadowed by back-facing microfacets. These effects are accounted for by Smith’s masking-shadowing function , which gives the fraction of microfacets with normal  that are visible from direction . (Note that .) In the usual case where the probability a microfacet is visible is independent of its orientation , we can write this function as .


#### The Torrance–Sparrow Model
An early microfacet model was developed by Torrance and Sparrow (1967) to model metallic surfaces. They modeled surfaces as collections of perfectly smooth mirrored microfacets.


One of the nice things about the Torrance–Sparrow model is that the derivation doesn’t depend on the particular microfacet distribution being used. Furthermore, it doesn’t depend on a particular Fresnel function, so it can be used for both conductors and dielectrics. 


### Burley 2012
We will define our BRDF and compare with measured materials in terms of the microfacet model [30,
7, 33].
In developing our new physically based reflectance model, we were cautioned by artists that we need
our shading model to be art directable and not necessarily physically correct. Because of this, our
philosophy has been to develop a “principled” model rather than a strictly physical one.
These were the principles that we decided to follow when implementing our model:
1. Intuitive rather than physical parameters should be used.
2. There should be as few parameters as possible.
3. Parameters should be zero to one over their plausible range.
4. Parameters should be allowed to be pushed beyond their plausible range where it makes sense.
5. All combinations of parameters should be as robust and plausible as possible.
We thoroughly debated the addition of each parameter. In the end we ended up with one color
parameter and ten scalar parameters described in the following section.


#### Diffuse model details
Based on our observations, we developed a novel empirical model for diffuse retroreflection that
transitions between a diffuse Fresnel shadow for smooth surfaces and an added highlight for rough
surfaces. A possible explanation for this effect may be that, for rough surfaces, light enters and exits
the sides of micro-surface features, causing an increase in refraction at grazing angles. In any event,
our artists like it, and it is similar to features we used to have in our ad-hoc model except that it is
now more plausible and has a physical basis.
In our model, we ignore the index of refraction for the diffuse Fresnel factor and assume no incident
diffuse loss. This allows us to directly specify the incident diffuse color.

Our subsurface parameter blends between the base diffuse shape and one inspired by the HanrahanKrueger subsurface BRDF [11]. This is useful for giving a subsurface appearance on distant objects
and on objects where the average scattering path length is small; it’s not, however, a substitute for
doing full subsurface transport as it won’t bleed light into the shadows or through the surface

#### Specular D detail
For our BRDF, we chose to have two fixed specular lobes, both using the GTR model. The primary
lobe uses γ = 2, and the secondary lobe uses γ = 1. The primary lobe represents the base material
and may be anisotropic and/or metallic. The secondary lobe represents a clearcoat layer over the base
material, and is thus always isotropic and non-metallic.
For roughness, we found that mapping α = roughness2
results in a more perceptually linear
change in the roughness. Without this remapping, very small and non-intuitive values were required
for matching shiny materials. Also, interpolating between a rough and smooth material would always
produce a rough result.
In place of an explicit index of refraction, or IOR, our specular parameter determines the incident
specular amount. The normalized range of this parameter is remapped linearly to the incident specular
range [0.0, 0.08]. This corresponds to IOR values in the range [1.0, 1.8], encompassing most common
materials. Notably, the middle of the parameter range corresponds to an IOR of 1.5, a very typical
value, and is also our default. The specular parameter may be pushed beyond one to reach higher IOR
values but should be done with caution. This mapping of the parameter has helped greatly in getting
15
artists to make plausible materials given that real-world incident reflectance values are so unintuitively
low.

#### specular G details
For our model, we took a hybrid approach. Given that the Smith shadowing factor is available for the
primary specular, we use the G derived for GGX by Walter but remap the roughness to reduce the
extreme gain for shiny surfaces. Specifically, we linearly scale the original roughness from the [0, 1]
range to a reduced range, [0.5, 1], for the purposes of computing G.

This remapping was based on comparisons with measured data as well as artist feedback that the
specular was just “too hot” for small roughness values. This gives us a G function that varies with
roughness, is at least partially physically based, and seems plausible


### Burley 2015
We introduced a new physically based shading model on Wreck-It Ralph [Bur12], and we used this
single, general-purpose BRDF on all materials (except hair). This model, which has come to be known
as the Disney BRDF, is able to reproduce a wide range of materials with only a few parameters.

We switched from ad hoc lighting and shading to path-traced
global illumination where refraction, subsurface scattering, and indirect illumination were all integrated
in a single physically based framework.
With path-traced global illumination, energy conservation
becomes critical as non-energy conserving materials may amplify light and prevent the image from
converging. The additive nature of ad hoc shading is generally not energy conserving given that the
various components are redundant representations of the refracted energy. In order to ensure energy
conservation, we extended our BRDF to a unified BSDF model where all such effects are accounted
for in a consistent, energy-conserving way

We would like to represent all forms of scattering within a unified model. The most general model perhaps would be a specular BSDF for the surface combined with a volumetric scattering
model for the interior as shown in Figure 3 to the right. Such a
model can reproduce all physical effects, though it is not always
practical to do so; e.g. in this model surface color is derived
solely from absorption during volume scattering though it would
be sufficient and obviously more efficient to use a diffuse BRDF
when the scatter distance is negligible.
Arguably, one might consider the BSSRDF as the most general unified representation of scattering since, mathematically at
least, it fully describes all scattering between any two points and

directions on a surface. However, the BSSRDF is an impractical
representation for non-diffuse scattering given that the entry and exit points and directions must be
known a priori, yet the exit point is often determined by tracing through the object after scattering at
the surface. The general form of the BSSRDF presents a chicken and egg problem (or more technically,
an example of the searchlight problem from radiative transport theory). The diffusion approximation
of the BSSRDF, however, sidesteps this problem.

Rather than trying to use a single general model, our unified model is a blend of models, combining
our existing BRDF with a specular BSDF and a subsurface scattering model. When the specular
BSDF is selected and combined with a volume shader for the interior, we have the general scattering
model of Figure 3, but we still have the more efficient approximations of the BRDF and subsurface
diffusion available.
We refer to our unified model as the Disney BSDF though this isn’t strictly correct given that our
model includes subsurface scattering. Because of this, we more formally refer to our unified model as
the Disney BSDF with integrated subsurface scattering.

The Disney BRDF is a blend of metallic and dielectric BRDF models based on a metallic shader parameter. For
our unified BSDF, we extend our dielectric BRDF with integrated subsurface scattering, and we blend in an additional
specular BSDF based on a new specTrans parameter.

maybe show figure 4?

We note that our Disney BRDF is already a blend of metallic and dielectric models. To support
refraction, we extend this by blending an additional specular BSDF model as shown in Figure


#### Disney BRDF recap
The Disney BRDF was largely based on observations of measured materials, primarily from the MERL
100 materials [MPBM03]. As such it was empirically inspired but follows an ad hoc construction where
existing physically derived models didn’t adequately reproduce measured material results.

Our BRDF includes two specular lobes: a microfacet reflection lobe with anisotropic roughness
and an optional clearcoat reflection lobe. The microfacet reflection follows standard models [CT81,
WMLT07].

#### Specular BSDF
Our specular BSDF directly extends the microfacet reflection lobe of our BRDF to refraction. 
We follow the derivation of Walter et al. [WMLT07].
Rays are refracted according to Snell’s law,
ηi sin θi = ηo sin θo where ηi and ηo are the indices of refraction, or IORs, of the two media at the
refractive surface. Note that refraction only depends on the ratio of indices, η = ηo/ηi
, commonly
referred to as the relative IOR. When coming from air (where ηi ≈ 1.0003 and typically assumed to be
1) the IOR and relative IOR are the same, but care must be taken to invert the value when refracting
out of an object.

####  Dielectric BRDF with integrated subsurface scattering
To extend our dielectric BRDF with subsurface scattering, we first refactor our diffuse lobe into directional microsurface effects and non-directional (i.e. Lambertian) subsurface effects, then we replace the
Lambertian portion of the diffuse lobe with either a diffusion model or a volumetric scattering model.
This preserves the microsurface effects, allowing our diffusion model to converge to the same result as
the diffuse BRDF when the scatter distance is small.

#### Path-traced subsurface scattering
We are now finding that path-traced subsurface scattering can be a practical alternative that is nearly
as efficient as diffusion while avoiding most of the artifacts

#### Index of Refraction
For our BRDF we avoided exposing IOR directly as a parameter because it was considered unintuitive
for artists. Instead, we inferred the IOR from the Fresnel equation at normal incidence, controlled by
the familiar specular parameter but rescaled to a plausible range

For our BSDF, the IOR controls the degree of bending for
refractive solids. To ensure energy conservation and plausibility,
we require the same IOR to be used for reflection and refraction,
and our artists now find it intuitive to control this with the IOR
directly when they understand the plausible range. IOR values
range from 1 for air to perhaps 2 for the vast majority of materials
with most common materials in the 1.3–1.6 range and only a few
extremely dense materials going beyond 2 (e.g. for diamond,
η ≈ 2.5), and [Pol08]. Artists merely need to understand that
the plausible range is [1, 2] with common materials close to 1.5.

As we noted in a 2014 addendum to our 2012 notes [Bur12] and restate here, based on Heitz’ analysis [Hei14] we eliminated our ad hoc remapping of the Smith G roughness for our primary specular
lobe, and we adopted Heitz’ anisotropic form.

### Conclusion
Originally, we had 10 parameters in our reflectance layers. With the BSDF, we now have 12 for
solids, adding specTrans and scatterDistance, or 13 for thin surfaces, adding specTrans, diffTrans, and
flatness as previously described

We have presented our new Disney BSDF, an extension of our BRDF with integrated refraction and
subsurface scattering, and we have described our novel diffusion profile which has efficiency, accuracy,
and controllability advantages. We have shown production examples and discussed implementation
issues and limitations.
In conclusion, we would like to emphasize that we believe the value of our model comes more from
our unified approach—having a single model with a intuitive, minimal set of parameters that can be
used for nearly all materials—and less from the specific constituent parts. We will continue to improve
the robustness or plausibility of the model and extend it as our needs dictate, but we remain committed
to the generality and simplicity of our approach.
