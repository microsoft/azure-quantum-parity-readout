---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: python
    language: python
    name: python
---

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
```

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

```python
gen_labels = True #set to False when you want to make the figure for the paper (so that tick/axis labels, etc, are not generated)
```

```python
def S11_fun_phi_tau(f,f0,ki,kc,phi,tau):
    k = ki + kc
    return np.exp(-1j*2*np.pi*f*tau)*(1 - 2*kc/k*np.exp(1j*phi) / (1 + 2*1j*(f-f0)/k))
```

```python
#define resonator params
f_ref = 1e6*np.linspace(670,730,201)
phi = 0*np.pi/2
f0  = 1e6*700
ki = 1e6*1
kcmag = 1e6*10
kc = kcmag*np.exp(1j*phi)
k = ki + np.real(kc)
tau = 0e-9
detuning = 0e6
delta_f0 = -k/8#10
delta_ki = k/5
```

```python
#plot params
av = 0.2
tickz = [-1, 0, 1]

#make the reference trace
S11_ref = S11_fun_phi_tau(f_ref,f0,ki,kcmag,phi,tau)

cmap = LinearSegmentedColormap.from_list('my_gradient', (
    (0.000, (0.000, 0.188, 0.953)),
    (0.392, (0.000, 0.098, 0.486)),
    (0.500, (0.000, 0.000, 0.000)),
    (0.631, (0.498, 0.008, 0.031)),
    (1.000, (0.941, 0.012, 0.059)))).reversed()

#this code allows us to plot the ref trace w cintinuously varying color
points = np.array([np.real(S11_ref),np.imag(S11_ref)]).transpose().reshape(-1,1,2)
segs = np.concatenate([points[:-1],points[1:]],axis=1)

# calculate other things to plot
S11_prime = S11_fun_phi_tau(f0+detuning,f0+delta_f0,ki+delta_ki,kcmag,phi,tau)
nearest_ind = np.argmin(abs(S11_prime-S11_ref))
nearest_point = S11_ref[nearest_ind]

dw = -k/20
dwind = np.argmin(abs(f_ref-(f0-dw)))
dw_IQ = S11_ref[dwind]

f0ind = np.argmin(abs(f_ref-f0))
f0_IQ = S11_ref[f0ind]

# plot resonator in IQ-space
fig,ax = plt.subplots(figsize=(3, 3), dpi=240)
sc = plt.scatter(points[:,0,0], points[:,0,1], c=(f_ref-f0)/k, s=3, cmap=cmap)

ax.set_xlim(-1, 1) # line collections don't auto-scale the plot
ax.set_ylim(-1,1)

ax.scatter(np.real(S11_prime),np.imag(S11_prime), color = 'tab:green', s=3)

# Distance Annotations Helpers
to_IQ_pair = lambda x: np.array([np.real(x), np.imag(x)])

arrowprops = dict(
    arrowstyle='|-|',
    fc="tab:orange", ec="tab:orange",
    mutation_scale=2,
    mutation_aspect=1,
    linewidth=1
)

# Delta S rad
pt1 = to_IQ_pair(nearest_point)
pt2 = to_IQ_pair(S11_prime)

transform = 1.3*(pt2-pt1)[::-1]
transform[0] = -transform[0]
pt1 += 0.1*transform
pt2 += 0.1*transform

pt1 *= 1.02
pt2 *= 0.97

ax.annotate(
    "", arrowprops=arrowprops,
    xy=pt1,
    xytext=pt2,
)
ax.text(*(pt1 + np.array([0.18, -0.02])), "$\delta S_{11}^\mathrm{rad}$")

# Delta S tan
scale_safeshift = 0.93
pt1 = to_IQ_pair(f0_IQ)
pt2 = to_IQ_pair(dw_IQ)

transform = (pt2-pt1)*0.2
pt2 += transform
pt1 -= transform

ax.annotate(
    "", arrowprops=arrowprops,
    xy=scale_safeshift*pt1,
    xytext=scale_safeshift*pt2,
)
ax.text(*(scale_safeshift*pt1 + np.array([0.08, 0.07])), "$\delta S_{11}^\mathrm{tan}$")

# Lines at y=x=0
# ymin=xmin=-0.003 so that the crosshairs are phase matched and constructively interfere
ax.axvline(0, c='k', lw=0.5, ls='--', alpha=av, ymin=-0.003)
ax.axhline(0, c='k', lw=0.5, ls='--', alpha=av, xmin=-0.003)

ax.set_aspect('equal', adjustable='box')
cbar = plt.colorbar(sc, shrink=0.81)
cbar.set_label("Detuning [$\kappa$]")
ax.set_xticks(tickz)
ax.set_yticks(tickz)

if gen_labels:
    ax.set_xlabel("$\mathrm{Re}\,S_{11}$")
    ax.set_ylabel("$\mathrm{Im}\,S_{11}$")
else:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.set_ticklabels([])
plt.savefig("cq_conversion.pdf", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.01)

plt.show()
```
