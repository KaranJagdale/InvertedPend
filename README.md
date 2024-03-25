## Controling inverted pendulum using RL

Here, I have tried to control the inverted pendulum at its unstable equilibrium point, i.e., upright position using various control techniques. As of now, I have implemented a PID and state feedback controller in another repository [controller-comparoson](https://github.com/KaranJagdale/controller_comparison/tree/master) In this repository I tried unconventinal method of reinforcement learning for controling the inverted pendulumn. 

[InvPendProd.ipynb](https://github.com/KaranJagdale/InvertedPend/blob/main/InvPendProd.ipynb) contains the required code. The pendulum model used is explained in [controller-comparison](https://github.com/KaranJagdale/controller_comparison). We apply Q-learning to obtain the Q-values of all the state-action pairs. 

The state of the inverted pendulum is given by $\theta$ and the control input is the torque $\tau$.
First we need to discretize the system as Q-learning cannot directly be appplied on a continuous system. We discretize $\theta$ into $100$ values between $[0, 2\pi)$, i.e., the discretized states are, $\{0, 0.628, 1.256, \dots, 5.652\}$. The discretized control, $\tau$ can take nine values, i.e., $\{-\tau_m, -\frac{3 \tau_m}{4}, \dots,  0, \dots, \frac{3 \tau_m}{4} \tau_m\}$. Where $\tau_m$ is four times the torque required for maintaining the pendulum in the horizontal position and is given by,
$$\tau_m = 2mgl.$$

Where, $m, l, g$ are mass of the pendulum, length of the pendulum and the acceleration due to gravity, respectively.

The reason for having less resolution in torque, i.e., actions is to keep the state-action pairs less and to get somwhat satisfactory results in less training. 

We perform episodic Q-learning to learn the Q-function. To genetate the data for Q-learning we apply random input action to the pendulum. An episode is marked complete if the pendulum reaches its target position or the number of samples become more than a threshold value. We perform training for $150000$ episodes with one episode, with each episode have maximum $800$ time instants. At each instance in the episide, we update the Q-function as,
$$Q_{k+1}(s,a) = (1 - \epsilon_k) Q_k(s,a) + \epsilon_k (r(s,a) + \gamma \max_{a'}(Q_k(s',a')), \text{ if } (s_k, a_k) = (s, a) \text{ and } s_{k+1} = s'$$
 
Otherwise,
$$Q_{k+1}(s,a) = Q_k(s,a), \text{ if } (s_k, a_k) \neq (s,a)$$

In the above equation, $\epsilon_k$ is the tuning parameter that determines the weight of the current Q-value and the weight of the information obtained by the experiimental data. Generally, $\epsilon_k$ is chosen close to zero so that the current Q-value has higher weight than the information obtained from the data in the update equation. $\gamma$ is called the discount factor and its value decides how the reward of future actions diminish over time in deciding the current optimal action. In the current setting, $\epsilon_k = 0.1$ and $\gamma = 0.9$ is used. Following video shows the result.
![](https://github.com/KaranJagdale/InvertedPend/blob/main/Invpend_QLearn.gif)


