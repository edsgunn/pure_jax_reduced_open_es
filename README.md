# pure_jax_reduced_open_es

This repo contains a pure JAX implementation of the Evolution Strategies with Probabilistic Communication algorithm developed for my Masters dissertation. 
This algorithm builds upon [OpenAI's evolution strategies algorithm](https://arxiv.org/abs/1703.03864) by introducing a new communication scheme that allows the communication between agents to be tunable to the bandwidth available. 
The problem setting is that of training an agent on a single agent RL task in a distributed manner. Many nodes can run episodes separately and communicate between each other to increase the rate of learning. However, the bandwith available for communication is limited. We therefore need to maximise the rate of learning while constrinaing communication to a particular rate. We can break down the amount of communication into two components. The first is the number of messages sent and the second is the size of those messages. 

In the OpenAI ES algorithm every node communicates every episode with a single scalar value. Since we cannot reduce the size of the messages here (beyond quantisation) our algorithm seeks to reduce the number of messages sent by agents. To do this we prioritise episodes that are particularly useful for learning. The problem with this is that agents do not know what rewards other agents recieved in this episode and therefore do not know how useful the reward they recieved was. Each agent can model their expected reward for an episode and if the reward recieved is sufficiently improbable then it is deemed worth communicating. More details on the precise algorithm can be found in [the report](https://github.com/edsgunn/pure_jax_reduced_open_es/blob/main/Report.pdf).

## Usage

Try the notebooks! The implementation of the reduced OpenES algorithm can be found in `reduced_open_es.py`.
