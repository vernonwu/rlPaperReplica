### Introduction
In his paper “Learning to Predict by the Methods of Temporal Differences”, Sutton[1] proposed  $TD(\lambda)$, a class of incremental model-free prediction-learning procedures, synthesizing Monte-Carlo learning and previous works on Temporal Difference methods[2]. He further demonstrates their convergence and optimality with respect to the *Widrow-Hoff* approach.

Central to Sutton's discourse are experiments on a random walk prediction problem, showcasing the efficacy of $T D(\lambda)$ methods, particularly in Figures $3$, $4$, and $5$ . The replication of these experiments is crucial for reinforcing the foundational principles of TD learning and examining their applicability in contemporary contexts.

### Theory
observation-outcome sequence: $x_1, x_2, x_3, \ldots, x_m, z$

Prediction: $P\left(x_t, w\right)$

for each sequence, produce prediction $P_1, P_2, P_3, \ldots, P_m$

Update:

$$w \leftarrow w+\sum_{t=1}^m \Delta w_t$$

$\mathrm{TD(1)}$:

$z-P_t=\sum_{k=t}^m\left(P_{k+1}-P_k\right) \quad$ where $\quad P_{m+1} \stackrel{\text { def }}{=} z$

therefore

$$\begin{aligned} w \leftarrow w+\sum_{t=1}^m \alpha\left(z-P_t\right) \nabla_w P_t & =w+\sum_{t=1}^m \alpha \sum_{k=t}^m\left(P_{k+1}-P_k\right) \nabla_w P_t \\ & =w+\sum_{k=1}^m \alpha \sum_{t=1}^k\left(P_{k+1}-P_k\right) \nabla_w P_t \\ & =w+\sum_{t=1}^m \alpha\left(P_{t+1}-P_t\right) \sum_{k=1}^t \nabla_w P_k\end{aligned}$$

or:

$$\Delta w_t=\alpha\left(P_{t+1}-P_t\right) \sum_{k=1}^t \nabla_w P_k$$

$\mathrm{TD(\lambda)}$:

Consider an exponential decayed weighting with recency:

$$\Delta w_t=\alpha\left(P_{t+1}-P_t\right) \sum_{k=1}^t \lambda^{t-k} \nabla_w P_k$$

def: $\quad e_t$ = $\sum_{k=1}^t \lambda^{t-k} \nabla_w P_k$

then: $\quad e_{t+1} = \lambda e_t + \nabla_w P_{t+1}$

### Quick Start
pip install -r requirements.txt
run RL_replicate.py

### Results
See Sutton_1988_TD_lambda.ipynb