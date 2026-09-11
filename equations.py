# -*- coding: utf-8 -*-
"""
The numbered equations of the manuscript, in order.

Each entry is (number, LaTeX).  Numbers are referenced from the prose by the
same index, so the two cannot drift apart.
"""

EQUATIONS = [
 # --- transducer model ---------------------------------------------------
 (1,  r"n_p\sin\theta_{\mathrm{inc}}=\Re\left\{\sqrt{\frac{\varepsilon_m(\lambda)\,"
      r"\varepsilon_d(\lambda)}{\varepsilon_m(\lambda)+\varepsilon_d(\lambda)}}\right\},"
      r"\qquad \lambda=\lambda^{\mathrm{res}}_k"),

 (2,  r"\mathbf{M}_j=\begin{bmatrix} \cos\beta_j & -i\,q_j^{-1}\sin\beta_j \\ "
      r"-i\,q_j\sin\beta_j & \cos\beta_j \end{bmatrix},\quad "
      r"\beta_j=\frac{2\pi d_j}{\lambda}\kappa_j,\quad "
      r"q_j=\frac{\kappa_j}{\varepsilon_j},\quad "
      r"\kappa_j=\sqrt{\varepsilon_j-n_p^{2}\sin^{2}\theta_{\mathrm{inc}}}"),

 (3,  r"R(\lambda)=\left|\frac{(M_{11}+M_{12}q_N)q_1-(M_{21}+M_{22}q_N)}"
      r"{(M_{11}+M_{12}q_N)q_1+(M_{21}+M_{22}q_N)}\right|^{2},\quad "
      r"\mathbf{M}=\prod_{j=2}^{N-1}\mathbf{M}_j"),

 (4,  r"\ell_k=\frac{\lambda^{\mathrm{res}}_k}{2\pi}\,"
      r"\frac{\sqrt{\left|\Re\{\varepsilon_m\}+\varepsilon_c\right|}}{\varepsilon_c}"),

 (5,  r"\Delta n^{\mathrm{eff}}_k=(n_a-n_{c,k})\left[1-"
      r"\exp\left(-\frac{2 d_{a,k}}{\ell_k}\right)\right]"),

 # --- adsorption ---------------------------------------------------------
 (6,  r"\frac{d\theta_{k,j}(t)}{dt}=k^{a}_{k,j}\,c_j(t)^{\,n_j}"
      r"\left[1-\sum_{i=1}^{J}\theta_{k,i}(t)\right]-k^{d}_{k,j}\,\theta_{k,j}(t)"),

 (7,  r"\theta^{\,\mathrm{eq}}_{k,j}=\frac{\left(K_{k,j}c_j\right)^{n_j}}"
      r"{1+\sum_{i=1}^{J}\left(K_{k,i}c_i\right)^{n_i}},\qquad "
      r"K_{k,j}=k^{a}_{k,j}/k^{d}_{k,j}"),

 (8,  r"x_k(t)=S^{\mathrm{srf}}_k\,\Delta n^{\mathrm{eff}}_k\!\left(\Theta_k(t)\right)"
      r"+\sum_{m=1}^{M}\psi_{k,m}\,\xi_m(t)+\epsilon_k(t),\qquad "
      r"\Theta_k(t)=\sum_{j=1}^{J} w_j\,\theta_{k,j}(t)"),

 (9,  r"\psi_{k,m}=\left.\frac{\partial\lambda^{\mathrm{res}}_k}"
      r"{\partial\zeta_m}\right|_{\zeta_m=\zeta_m^{0}},\qquad "
      r"\zeta\in\{\,T_{\mathrm{stack}},\;\mathrm{age},\;n_{\mathrm{gas}},\;"
      r"\lambda\text{-scale}\,\}"),

 (10, r"\mathrm{rank}(\Psi)=M'\ll K,\qquad "
      r"\mathcal{S}=\mathrm{span}(\Psi),\qquad "
      r"\left\|\mathbf{P}_{\mathcal{S}^{\perp}}\mathbf{s}_c\right\|\big/"
      r"\left\|\mathbf{s}_c\right\|=\cos\vartheta_c>0"),

 # --- mechanism A ---------------------------------------------------------
 (11, r"\mathbf{Q}\mathbf{R}=\mathbf{U},\qquad "
      r"\mathbf{a}_t=\mathbf{Q}^{\top}\mathbf{x}_t\in\mathbb{R}^{r}"),

 (12, r"\mathbf{z}_t=\mathbf{x}_t-\mathbf{Q}\left(\mathbf{g}\odot\mathbf{a}_t\right),"
      r"\qquad \mathbf{g}=\sigma(\boldsymbol{\gamma})\in(0,1)^{r}"),

 # --- mechanism B ---------------------------------------------------------
 (13,  r"\mathbf{u}_t=\log\left(1+\exp\left(\mathbf{W}_{\!u}\mathbf{z}_t+\mathbf{b}_u\right)\right)\in\mathbb{R}_{+}^{J'},\qquad J'\ll K T"),

 (14,  r"k^{a}_{k,j}=K_{k,j}\,k^{d}_{k,j},\qquad k^{d}_{k,j}=\exp\left(\eta^{d}_{k,j}\right),\qquad K_{k,j}=\exp\left(\eta^{K}_{k,j}\right)"),

 (15,  r"\vartheta^{\ast}_{t,k,j}=\frac{k^{a}_{k,j}u_{t,j}\left(1-s_{t-1,k}\right)}{k^{a}_{k,j}u_{t,j}+k^{d}_{k,j}}"),

 (16,  r"\vartheta_{t,k,j}=\vartheta^{\ast}_{t,k,j}+\left(\vartheta_{t-1,k,j}-\vartheta^{\ast}_{t,k,j}\right)\exp\left[-\left(k^{a}_{k,j}u_{t,j}+k^{d}_{k,j}\right)\Delta t\right]"),

 (17,  r"s_{t,k}=\min\left(\sum_{j=1}^{J'}\omega_{k,j}\,\vartheta_{t,k,j},\;0.98\right),\qquad \omega_{k,j}=\sigma\!\left(\tilde\omega_{k,j}\right)/J'"),

 (18,  r"\hat{z}_{t,k}=\sum_{j=1}^{J'}v_{k,j}\,\vartheta_{t,k,j}+b^{v}_{k},\qquad \mathbf{e}_t=\mathbf{z}_t-\hat{\mathbf{z}}_t"),

 # --- mechanism C ---------------------------------------------------------
 (19,  r"\boldsymbol{\zeta}_t=\sigma\left(\mathbf{W}_{\!\zeta}\left[\mathbf{e}_t;\;\mathbf{z}_t-\mathbf{z}_{t-1}\right]+\mathbf{R}_{\zeta}\mathbf{h}_{t-1}+\Gamma\,\mathrm{vec}\left(\Delta\boldsymbol{\vartheta}_t\right)\right)"),

 (20, r"\boldsymbol{\rho}_t=\sigma\left(\mathbf{W}_{\!\rho}"
      r"\left[\mathbf{e}_t;\;\mathbf{z}_t-\mathbf{z}_{t-1}\right]+"
      r"\mathbf{R}_{\rho}\mathbf{h}_{t-1}\right),\qquad "
      r"\tilde{\mathbf{n}}_t=\tanh\left(\mathbf{W}_{\!n}"
      r"\left[\mathbf{e}_t;\;\mathbf{z}_t-\mathbf{z}_{t-1}\right]+"
      r"\boldsymbol{\rho}_t\odot\mathbf{R}_{n}\mathbf{h}_{t-1}\right)"),

 (21, r"\mathbf{h}_t=\left(\mathbf{1}-\boldsymbol{\zeta}_t\right)\odot"
      r"\tilde{\mathbf{n}}_t+\boldsymbol{\zeta}_t\odot\mathbf{h}_{t-1}"),

 # --- head ----------------------------------------------------------------
 (22,  r"\boldsymbol{\phi}=\mathrm{BN}\left[\mathrm{vec}\left(\boldsymbol{\vartheta}_T\right);\;\mathrm{vec}\left(\overline{\boldsymbol{\vartheta}}\right);\;\mathrm{vec}\left(\boldsymbol{\vartheta}^{\max}\right);\;\mathrm{vec}\left(\boldsymbol{\vartheta}^{\max}-\boldsymbol{\vartheta}_T\right);\;\mathrm{vec}\left(\left|\Delta\boldsymbol{\vartheta}\right|^{\max}\right);\;\mathbf{h}_T;\;\overline{\mathbf{h}}\right]"),

 (23,  r"\boldsymbol{\alpha}=\log\left(1+\exp\left(\mathbf{W}_{\!o}\boldsymbol{\phi}+\mathbf{b}_o\right)\right)+\mathbf{1},\qquad S=\sum_{c=1}^{C}\alpha_c,\qquad \hat{p}_c=\alpha_c/S"),

 (24, r"\upsilon=C/S,\qquad b_c=(\alpha_c-1)/S,\qquad "
      r"\upsilon+\sum_{c=1}^{C}b_c=1"),

 # --- objective -----------------------------------------------------------
 (25, r"\mathcal{L}_{\mathrm{cls}}=\frac{\sum_{i}\varpi_{y_i}\sum_{c=1}^{C}"
      r"y_{i,c}\left[\psi_0(S_i)-\psi_0(\alpha_{i,c})\right]}"
      r"{\sum_{i}\varpi_{y_i}},\qquad "
      r"\varpi_{c}\propto N/(C N_c)"),

 (26, r"\mathcal{L}_{\mathrm{kl}}=\mathrm{KL}\!\left[\mathrm{Dir}"
      r"\left(\tilde{\boldsymbol{\alpha}}_i\right)\,\|\,"
      r"\mathrm{Dir}\left(\mathbf{1}\right)\right],\qquad "
      r"\tilde{\boldsymbol{\alpha}}_i=\mathbf{y}_i+"
      r"\left(\mathbf{1}-\mathbf{y}_i\right)\odot\boldsymbol{\alpha}_i"),

 (27, r"\mathcal{L}_{\mathrm{kin}}=\frac{1}{TK}\sum_{t=1}^{T}"
      r"\left\|\hat{\mathbf{z}}_t-\mathrm{sg}\!\left[\mathbf{z}_t\right]"
      r"\right\|_2^{2}"),

 (28, r"\mathcal{L}_{\mathrm{drift}}=\left\|f_{\mathrm{co}}"
      r"\left(\mathbf{a}\right)-\boldsymbol{\nu}\right\|_2^{2}+"
      r"\left\|f_{\mathrm{adv}}\!\left(\mathcal{R}_{\lambda_{\mathrm{rev}}}"
      r"\left(\boldsymbol{\phi}\right)\right)-\boldsymbol{\nu}\right\|_2^{2},"
      r"\quad \frac{\partial\mathcal{R}_{\lambda}}{\partial\boldsymbol{\phi}}"
      r"=-\lambda\mathbf{I}"),

 (29, r"\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{evi}}"
      r"\min\!\left(1,\tfrac{\mathrm{epoch}}{E_w}\right)\mathcal{L}_{\mathrm{kl}}"
      r"+\lambda_{\mathrm{kin}}\mathcal{L}_{\mathrm{kin}}+"
      r"\lambda_{\mathrm{drift}}\mathcal{L}_{\mathrm{drift}}"),

 (30,  r"\mathcal{O}\!\left(T\left(Kr+KJ'+KJ'H+KH+H^{2}\right)+Kr^{2}\right)"),

 # --- evaluation ----------------------------------------------------------
 (31, r"\mathrm{ECE}=\sum_{b=1}^{B}\frac{|\mathcal{I}_b|}{N}"
      r"\left|\frac{1}{|\mathcal{I}_b|}\sum_{i\in\mathcal{I}_b}"
      r"\mathbb{1}\!\left[\hat{y}_i=y_i\right]-"
      r"\frac{1}{|\mathcal{I}_b|}\sum_{i\in\mathcal{I}_b}\hat{p}_{i,\hat{y}_i}"
      r"\right|"),

 (32, r"\mathrm{BS}=\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}"
      r"\left(\hat{p}_{i,c}-y_{i,c}\right)^{2},\qquad "
      r"\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log\hat{p}_{i,y_i}"),

 (33, r"\mathrm{BA}=\frac{1}{C}\sum_{c=1}^{C}\frac{\mathrm{TP}_c}{N_c},\qquad "
      r"\mathrm{MCC}=\frac{N\,\mathrm{tr}(\mathbf{G})-"
      r"\sum_c G_{\cdot c}G_{c\cdot}}"
      r"{\sqrt{\left(N^{2}-\sum_c G_{\cdot c}^{2}\right)"
      r"\left(N^{2}-\sum_c G_{c\cdot}^{2}\right)}}"),

 (34, r"\delta=\frac{\#\{a_i>b_j\}-\#\{a_i<b_j\}}{n_a n_b},\qquad "
      r"d_z=\frac{\overline{a-b}}{\mathrm{sd}(a-b)},\qquad "
      r"\mathrm{CD}=q_{\varsigma}\sqrt{\frac{L(L+1)}{6 D}}"),
]

N_EQ = len(EQUATIONS)
