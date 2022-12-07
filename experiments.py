import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.initializers import RandomNormal, Zeros

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

import os
import sys
import argparse

# from learner import SimpleHoverEnv, ComplexHoverEnv, HoverEnv
from learner import HoverEnv


def make_model(in_size, out_size, hidden_size=128, hidden_layers=2, output_activation='linear', weight_init=RandomNormal(), bias_init=Zeros()):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + (in_size,)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_size, kernel_initializer=weight_init, bias_initializer=bias_init))
        model.add(Activation('relu'))
    model.add(Dense(out_size, kernel_initializer=weight_init, bias_initializer=bias_init))
    model.add(Activation(output_activation))

    return model


def checkpoint(epoch: int, expname: str, model, env):
    dqn.model.save(f'models/{expname}-ep{epoch}.h5')
    np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)


# TODO(escottrose01): implement actor-critic methods
# def make_critic_model(in_shape, out_size, hidden_size=128, hidden_layers=2):
#     action_input = Input(shape=(nb_actions,), name='action_input')
#     obs_input = Input(shape=(1,) + in_shape)
#     flat_obs = Flatten()(obs_input)
#     output = Concatenate()([action_input, flat_obs])
#     for _ in range(hidden_layers):
#         output = Dense(hidden_size)(output)
#         output = Activation('relu')(output)
#     output = Dense(1)(output)
#     output = Activation('linear')(output)
#     model = Model(inputs=[action_input, obs_input], outputs=output)
#     return model, action_input


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('-fr', '--fix-rocket', action='store_true',
                    help='if present, fixes rocket start position during training')
parser.add_argument('-ft', '--fix-target', action='store_true',
                    help='if present, fixes target position during training')
parser.add_argument('-a', '--agent', help='the type of agent to train', choices=['dqn', 'ddpg'])
parser.add_argument('-ahs', '--agent-hidden-size', type=int, default=128, help='the size of agent hidden layers')
parser.add_argument('-ahl', '--agent-hidden-layers', type=int, default=2, help='the number of agent hidden layers')
parser.add_argument('-chs', '--critic-hidden-size', type=int, default=128,
                    help='the size of critic hidden layers, if using actor-critic')
parser.add_argument('-chl', '--critic-hidden-layers', type=int, default=2,
                    help='the number of critic hidden layers, if using actor-critic')
parser.add_argument('-i', '--iter', type=int, default=100000, help='the number of iterations to train')
parser.add_argument('-c', '--control', help='the control method for the rocket',
                    choices=['simple', 'direct', 'indirect', 'continuous', 'assisted'])
parser.add_argument('-g', '--gamma', type=float, default=0.996, help='value decay parameter')
args = parser.parse_args()

# experiments = args.experiments or range(1, 3)
fix_rocket = args.fix_rocket
fix_target = args.fix_target
agent_type = args.agent
control_type = args.control
agent_hidden_size = args.agent_hidden_size
agent_hidden_layers = args.agent_hidden_layers
agent_hidden_size = args.critic_hidden_size
agent_hidden_layers = args.critic_hidden_layers
iters = args.iter
gamma = args.gamma

if agent_type == 'ddpg':
    raise NotImplementedError()


env = HoverEnv(fix_target=fix_target, fix_rocket=fix_rocket, control_type=control_type)
expname = f'hover-{control_type}'
if fix_target:
    expname += '-ft'
if fix_rocket:
    expname += '-fr'

nb_actions = env.action_space.n if control_type != 'continuous' else env.action_space[0]
memory = SequentialMemory(limit=500000, window_length=1)

if agent_type == 'dqn':
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 0.75, 0.0, 0.0, iters)
    model = make_model(env.observation_space.shape[0], nb_actions,
                       hidden_size=agent_hidden_size, hidden_layers=agent_hidden_layers)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, target_model_update=1e-4,
                   policy=policy, gamma=gamma, enable_double_dqn=True, enable_dueling_network=True)
    dqn.compile(Adam(learning_rate=2e-4), metrics=['mae'])

    checkpoint_callback = LambdaCallback(lambda epoch, logs: epoch %
                                         500 == 0 and checkpoint(epoch, expname, dqn.model, env))
    dqn.fit(env, nb_steps=iters, visualize=True, verbose=1, log_interval=20000,
            callbacks=[checkpoint_callback], action_repetition=10)

model.save(f'models/{expname}-final.h5')
np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)

# plot rewards
model.save(f'models/{expname}-final.h5')
fig = plt.figure()
avg_100 = 0.01*np.ones(100)
avg = np.convolve(env.reward_history, avg_100, 'same')
plt.plot(env.reward_history, marker='o', markersize=1, color='tab:purple', linestyle='None')
plt.plot(avg, color='tab:blue')
plt.xlabel('Training Episodes')
plt.ylabel('Reward')
plt.title(f'Average Reward for {control_type.capitalize()} Hover')
plt.savefig(f'results/{expname}-rewards.svg', format='svg', dpi=300)
plt.show()


##### Experiment 1: simple hover #####
# if 0 in experiments:
#     env = SimpleHoverEnv(fix_target=fix_target, fix_rocket=fix_rocket)
#     expname = 'hover-simple'
#     if fix_target:
#         expname += '-fixtarget'
#     if fix_rocket:
#         expname += '-fixrocket'

#     nb_actions = env.action_space.n
#     model = make_model(env.observation_space.shape, nb_actions)
#     # policy = EpsGreedyQPolicy(eps=0.001)
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 0.01, 0.001, 0.01, 20000)
#     memory = SequentialMemory(limit=500000, window_length=1)
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                    target_model_update=1e-4, policy=policy, gamma=0.995,
#                    enable_double_dqn=True, enable_dueling_network=True)
#     dqn.compile(Adam(lr=2e-4), metrics=['mae'])

#     checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch % 200 == 0 and dqn.model.save(
#         f'models/{expname}-ep{epoch}.h5') and np.savetxt(f'results/{expname}-rewards.csv', env.reward_history))

#     dqn.fit(env, nb_steps=500000, visualize=True, verbose=1,
#             log_interval=10000, callbacks=[checkpoint], action_repetition=10)

#     dqn.model.save(f'models/{expname}-final.h5')
#     np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)

#     # plot rewards
#     fig = plt.figure()
#     avg_100 = 0.01*np.ones(100)
#     avg = np.convolve(env.reward_history, avg_100, 'same')
#     plt.plot(env.reward_history, marker='o', markersize=2, color='tab:purple', linestyle='None')
#     plt.plot(avg, color='tab:blue')
#     plt.xlabel('Training Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average Reward for Simple Hovering Task')
#     plt.savefig(f'results/{expname}-rewards.svg', format='svg', dpi=300)
#     plt.show()

# ##### Experiment 2: complex hover, direct control #####
# if 1 in experiments:
#     env = ComplexHoverEnv(fix_target=fix_target, fix_rocket=fix_rocket, control_type='direct')
#     expname = 'hover-complex-direct'
#     if fix_target:
#         expname += '-fixtarget'
#     if fix_rocket:
#         expname += '-fixrocket'

#     nb_actions = env.action_space.n
#     model = make_model(env.observation_space.shape, nb_actions, hidden_size=256, hidden_layers=3)
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 1.0, 0.005, 0.01, 250000)
#     memory = SequentialMemory(limit=5000000, window_length=1)
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                    target_model_update=1e-4, policy=policy, gamma=0.99,
#                    enable_double_dqn=True, enable_dueling_network=True)
#     dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#     checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch % 500 == 0 and dqn.model.save(
#         f'models/{expname}-ep{epoch}.h5') and np.savetxt(f'results/{expname}-rewards.csv', env.reward_history))

#     dqn.fit(env, nb_steps=1000000, visualize=True, verbose=1,
#             log_interval=20000, callbacks=[checkpoint], action_repetition=10)
#     dqn.model.save('models/hover-complex-direct-final.h5')
#     np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)

#     # plot rewards
#     fig = plt.figure()
#     avg_100 = 0.01*np.ones(100)
#     avg = np.convolve(env.reward_history, avg_100, 'same')
#     plt.plot(env.reward_history, marker='o', markersize=2, color='tab:purple', linestyle='None')
#     plt.plot(avg, color='tab:blue')
#     plt.xlabel('Training Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average Reward for Complex Hovering Task, Direct Control')
#     plt.savefig(f'results/${expname}-rewards.svg', format='svg', dpi=300)
#     plt.show()

# ##### Experiment 3: complex hover, indirect control #####
# if 2 in experiments:
#     env = ComplexHoverEnv(fix_target=fix_target, fix_rocket=fix_rocket, control_type='indirect')
#     expname = 'hover-complex-indirect'
#     if fix_target:
#         expname += '-fixtarget'
#     if fix_rocket:
#         expname += '-fixrocket'

#     nb_actions = env.action_space.n
#     model = make_model(env.observation_space.shape, nb_actions, hidden_size=64, hidden_layers=5)
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 1.0, 0.005, 0.01, 250000)
#     memory = SequentialMemory(limit=5000000, window_length=1)
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                    target_model_update=1e-4, policy=policy, gamma=0.999,
#                    enable_double_dqn=True, enable_dueling_network=True)
#     dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#     checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch %
#                                 500 == 0 and dqn.model.save(f'models/{expname}-ep{epoch}.h5') and np.savetxt(f'results/{expname}-rewards.csv', env.reward_history))

#     dqn.fit(env, nb_steps=1000000, visualize=True, verbose=1,
#             log_interval=20000, callbacks=[checkpoint])
#     dqn.model.save(f'models/{expname}-final.h5')
#     np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)

#     # plot rewards
#     fig = plt.figure()
#     avg_100 = 0.01*np.ones(100)
#     avg = np.convolve(env.reward_history, avg_100, 'same')
#     plt.plot(env.reward_history, marker='o', markersize=2, color='tab:purple', linestyle='None')
#     plt.plot(avg, color='tab:blue')
#     plt.xlabel('Training Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average Reward for Complex Hovering Task, Indirect Control')
#     plt.savefig(f'results/{expname}-rewards.svg', format='svg', dpi=300)
#     plt.show()

# ##### Experiment 4: complex hover, assisted control #####
# if 3 in experiments:
#     env = ComplexHoverEnv(fix_target=fix_target, fix_rocket=fix_rocket, control_type='assisted')
#     expname = 'hover-complex-assist'
#     if fix_target:
#         expname += '-fixtarget'
#     if fix_rocket:
#         expname += '-fixrocket'

#     nb_actions = env.action_space.n
#     model = make_model(env.observation_space.shape, nb_actions, hidden_size=128, hidden_layers=5)
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 0.5, 0.001, 0.01, 2000000)
#     memory = SequentialMemory(limit=200000, window_length=1)
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                    target_model_update=1e-4, policy=policy, gamma=0.999,
#                    enable_double_dqn=True, enable_dueling_network=True)
#     dqn.compile(Adam(lr=2e-4), metrics=['mae'])

#     checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch %
#                                 500 == 0 and dqn.model.save(f'models/{expname}-ep{epoch}.h5') and np.savetxt(f'results/{expname}-rewards.csv', env.reward_history))

#     dqn.fit(env, nb_steps=5000000, visualize=True, verbose=1,
#             log_interval=20000, callbacks=[checkpoint], action_repetition=10)
#     dqn.model.save(f'models/{expname}-final.h5')
#     np.savetxt(f'results/{expname}-rewards.csv', env.reward_history)

#     # plot rewards
#     fig = plt.figure()
#     avg_100 = 0.01*np.ones(100)
#     avg = np.convolve(env.reward_history, avg_100, 'same')
#     plt.plot(env.reward_history, marker='o', markersize=2, color='tab:purple', linestyle='None')
#     plt.plot(avg, color='tab:blue')
#     plt.xlabel('Training Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average Reward for Complex Hovering Task, Assisted Control')
#     plt.savefig(f'results/{expname}-rewards.svg', format='svg', dpi=300)
#     plt.show()


# if 2 in experiments:
#     env = ComplexHoverEnv(fix_target=True, fix_rocket=False, discrete=False)

#     weight_init = RandomNormal(stddev=0.00001)

#     nb_actions = env.action_space.shape[0]
#     actor = make_model(env.observation_space.shape, nb_actions, hidden_size=64,
#                        hidden_layers=2, output_activation='linear', weight_init=weight_init)
#     critic, action_input = make_critic_model(env.observation_space.shape, nb_actions, hidden_size=64, hidden_layers=2)
#     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 1.0, 0.005, 0.01, 250000)
#     memory = SequentialMemory(limit=5000000, window_length=1)
#     agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                       memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
#                       gamma=0.99, target_model_update=1e-5)
#     #   target_model_update=1e-4, policy=policy, gamma=0.99,
#     #   enable_double_dqn=True, enable_dueling_network=True)
#     agent.compile(Adam(lr=1e-5), metrics=['mae'])

#     checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch %
#                                 500 == 0 and actor.save(f'models/hover-complex-indirect-ep{epoch}.h5'))

#     agent.fit(env, nb_steps=1000000, visualize=True, verbose=1, log_interval=5000, callbacks=[checkpoint])
#     actor.save('models/hover-complex-indirect-final.h5')
#     np.savetxt('results/hover-complex-indirect-rewards.csv', env.reward_history)

#     # plot rewards
#     fig = plt.figure()
#     avg_100 = 0.01*np.ones(100)
#     avg = np.convolve(env.reward_history, avg_100, 'same')
#     plt.plot(env.reward_history, marker='o', markersize=2, color='tab:purple', linestyle='None')
#     plt.plot(avg, color='tab:blue')
#     plt.xlabel('Training Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average Reward for Complex Hovering Task, Indirect Control')
#     plt.savefig("results/hover-complex-indirect-rewards.svg", format='svg', dpi=300)
#     plt.show()
