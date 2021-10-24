import tensorflow as tf
import tflearn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io

from scipy.integrate import odeint
from scipy.optimize import fsolve

from functionsForDDPG import ReplayBuffer
from simula_VdV import modelVdV
from simula_VdV import stepVdV

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # w_init = tflearn.initializations.zeros()

        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    '''
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor Networkself.
    '''

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic networks
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online networks
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau) \
            + tf.multiply(self.target_network_params[i],1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None,1]) #Q(s,a) is a scalar

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the actions
        # For each action in the minibatch (i.e., for each x in xs),
        # This will sum up the gradients of each critic output in the minibatch_size
        # w.r.t. that action. Each output is independent of all actions
        #except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None,self.a_dim])
        net = tflearn.fully_connected(inputs,400) #only the states are inputs of the first layer
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net,300)
        t2 = tflearn.fully_connected(action,300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # weights are init to uniform(-3e-3,3e-3)
        w_init = tflearn.initializations.uniform (minval=-0.003, maxval=0.003)
        # w_init = tflearn.initializations.zeros()
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs:inputs,
            self.action:action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
        self.inputs: inputs,
        self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
        self.target_inputs: inputs,
        self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class DecreasingNoise:
    def __init__(self, noise_bound=[0.5,0.05], N0=10000):
        self.noise_bound = noise_bound
        self.N0 = N0
        self.Iter = 0 # number of time the funciton is called

    def __call__(self):
        decreasing_factor = float(self.N0 / (self.N0 + self.Iter))
        xx = [2.*self.noise_bound[0]*np.random.rand() - self.noise_bound[0],\
            2*self.noise_bound[1]*np.random.rand() - self.noise_bound[1] ]
        x = [i * decreasing_factor for i in xx]
        self.Iter +=1
        return x



class OrnstreinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) *self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev =self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu=[]=={}, sigma={})'.format(self.mu, self.sigma)

# Tensorflow Summary summary_ops

def build_summaries():
    episode_reward = tf.Variable(0.) # create a variable with value equal to 0
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
#

def train(sess, actor, critic, actor_noise, replay_buffer, BUFFER_SIZE, MINIBATCH_SIZE,  \
MAX_TOT_STEPS, MAX_EP_STEPS, FBSP, deltaT):

    # List of sum of rewards initializations
    ReturnsList = []

    # Number of total episodes atualization
    NumTotEp = 0

    # Initial number of episodes
    i=0

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer()) # initialize the variables - mandatory
    writer = tf.summary.FileWriter('./results/tf_ddpg/', sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    ## Initialize replay memory
    #replay_buffer = ReplayBuffer(BUFFER_SIZE)


    # for i in range(MAX_EPISODES):
    while NumTotEp < MAX_TOT_STEPS:



        # Initial MV
        Q = 1000
        # Tk = 100
        # if random initiazation of the states
        Tk = 60. + np.random.rand() * 340
        Tin = 110
        CAin = 5.1
        s = fsolve(modelVdV,[5.1, 0, 100],args=(0,Q,Tk,Tin,CAin)) # fixed initial states

    #
    #     s = env.reset()''
    #
        ep_reward = 0
        ep_ave_max_q = 0
    #
        for j in range(MAX_EP_STEPS):
    #
    #         if args['render_env']:
    #             env.render()
    #
            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + np.array([1,0.1]) * actor_noise()
            a = a[0]
            # print(a)

            s2, r , terminal = stepVdV(deltaT,s,Q,Tk,Tin,CAin,a,FBSP)
            # print(s2, r)
            # s2, r, terminal, info = env.step(a[0])

            #atualização das MV
            Q = Q + a[0]
            Tk = Tk + a[1]

            # I add the new experience to the buffer
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,\
            np.reshape(s2, (actor.s_dim,)))
            #print(replay_buffer.count)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    y_i.append(r_batch[k] + critic.gamma * target_q[k]) # não tem passo terminal
                    # if t_batch[k]:
                    #     y_i.append(r_batch[k])
                    # else:
                    #     y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if j == MAX_EP_STEPS-1 or terminal ==1:

                # number of episodes actualization
                i += 1
                # total number of steps actualization
                NumTotEp += j

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.4f} | Episode: {:d} | Total steps: {:d} | Qmax: {:.4f}'.format(ep_reward, \
                        i, NumTotEp , (ep_ave_max_q / float(j))))

                ReturnsList.append(ep_reward)

                # Simulate_VdV(actor,NB_STEPS, deltaT)

                break

    t=np.linspace(0,i-1, i)
    plt.figure(10)
    plt.plot(t,ReturnsList)
    plt.xlabel('time steps')
    plt.ylabel('return')
    plt.show()

def train_online(sess, actor, critic, actor_noise, replay_buffer, BUFFER_SIZE, MINIBATCH_SIZE,  \
NB_STEPS, FBSP, deltaT):

    #sess.run(tf.global_variables_initializer()) # initialize the variables - mandatory

    ## Initialize target network weights
    #actor.update_target_network()
    #critic.update_target_network()

    # Needed to enab3le BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    # Initial MV
    Q = 1000
    Tk = 100
    Tin = 110
    CAin = 5.1
    s = fsolve(modelVdV,[5.1, 0, 100],args=(0,Q,Tk,Tin,CAin)) # fixed initial states

    # Memory initializations
    Qlist = [Q]
    Tklist = [Tk]
    CAlist = [s[0]]
    CBlist = [s[1]]
    FBlist = [s[1]*Q]
    Tlist = [s[2]]
    FBSPlist = [FBSP]
    CBMAXlist =[CBMAX]


    for j in range(NB_STEPS):

        if j == 5000:
            Tin = 70

        # Added exploration noise
        # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
        a = actor.predict(np.reshape(s, (1, actor.s_dim))) + np.array([.1,0.01]) * actor_noise()
        a = a[0]
        # print(a)

        s2, r , terminal = stepVdV(deltaT,s,Q,Tk,Tin,CAin,a,FBSP)
        # print(s2, r)
        # s2, r, terminal, info = env.step(a[0])

        #atualização das MV
        Q = Q + a[0]
        Tk = Tk + a[1]

        # I add the new experience to the buffer
        replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,\
        np.reshape(s2, (actor.s_dim,)))
        #print(replay_buffer.count)

        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        if replay_buffer.size() > MINIBATCH_SIZE:
            s_batch, a_batch, r_batch, s2_batch = \
                replay_buffer.sample_batch(MINIBATCH_SIZE)

            # Calculate targets
            target_q = critic.predict_target(
                s2_batch, actor.predict_target(s2_batch))

            y_i = []
            for k in range(MINIBATCH_SIZE):
                y_i.append(r_batch[k] + critic.gamma * target_q[k]) # não tem passo terminal
                # if t_batch[k]:
                #     y_i.append(r_batch[k])
                # else:
                #     y_i.append(r_batch[k] + critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(
                s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

            #ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(s_batch)
            grads = critic.action_gradients(s_batch, a_outs)
            actor.train(s_batch, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

        s = s2

        # memory atualization
        Qlist.append(Q)
        Tklist.append(Tk)
        CAlist.append(s[0])
        CBlist.append(s[1])
        FBlist.append(s[1]*Q)
        Tlist.append(s[2])
        FBSPlist.append(FBSP)
        CBMAXlist.append(CBMAX)

    # showing results
    t=np.linspace(0,NB_STEPS*deltaT, NB_STEPS + 1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t,FBlist,'b-',label=r'$F_{B}$')
    plt.plot(t,FBSPlist,'k--',label=r'$F_{B}^{sp}$')
    plt.xlabel('time(h)')
    plt.ylabel('$F_{B}(mol/h)$')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(t,CBlist,'c-',label=r'Reactor Concentration of B')
    plt.plot(t,CBMAXlist,'k--',label=r'$C_{B}^{max}$')
    plt.xlabel('time(h)')
    plt.ylabel('$C_{B}(mol/L)$')
    plt.legend(loc='best')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(t,Qlist,'r-')
    plt.xlabel('time(h)')
    plt.ylabel('$F(L/h)$')
    plt.legend(['Inlet Flow'],loc='best')
    plt.subplot(212)
    plt.plot(t,Tklist,'r-')
    plt.xlabel('time(h)')
    plt.ylabel('$T_{k}(°C)$')
    plt.legend(['Jacket Temperature'],loc='best')

    plt.figure(3)
    plt.subplot(211)
    plt.plot(t,CAlist,'g-')
    plt.xlabel('time(h)')
    plt.ylabel('$C_{A}(mol/L)$')
    plt.legend(['Reactor Concentration of A'],loc='best')
    plt.subplot(212)
    plt.plot(t,Tlist,'m-')
    plt.xlabel('time(h)')
    plt.ylabel('$T(°C)$')
    plt.legend(['Reactor Temperature'],loc='best')


    plt.show()





def Simulate_VdV(actor, NB_STEPS, deltaT):

    # Initial MV
    Q = 1000
    Tk = 100
    Tin = 110
    CAin = 5.1
    s = fsolve(modelVdV,[5.1, 0, 100],args=(0,Q,Tk,Tin,CAin)) # fixed initial states
    print(s)

    # Memory initializations
    Qlist = [Q]
    Tklist = [Tk]
    CAlist = [s[0]]
    CBlist = [s[1]]
    FBlist = [s[1]*Q]
    Tlist = [s[2]]
    FBSPlist = [FBSP]
    CBMAXlist =[CBMAX]

    for j in range(NB_STEPS):

        if j == 5000:
            Tin = 90

        a = actor.predict(np.reshape(s, (1, actor.s_dim)))
        a = a[0]

        s, r, terminal= stepVdV(deltaT,s,Q,Tk,Tin,CAin,a,FBSP)

        #atualização das MV
        Q = Q + a[0]
        Tk = Tk + a[1]

        # memory atualization
        Qlist.append(Q)
        Tklist.append(Tk)
        CAlist.append(s[0])
        CBlist.append(s[1])
        FBlist.append(s[1]*Q)
        Tlist.append(s[2])
        FBSPlist.append(FBSP)
        CBMAXlist.append(CBMAX)

    # showing results
    t=np.linspace(0,NB_STEPS*deltaT, NB_STEPS + 1)

    scipy.io.savemat('C:/Users/Bruno/Dropbox/ufrj/Reinforcement learning/Ellen_Jasmine/arr.mat', mdict={'t': t,'FBlist': FBlist, \
    'CBlist': CBlist, 'Qlist': Qlist, 'Tklist' : Tklist})

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t,FBlist,'b-',label=r'$F_{B}$')
    plt.plot(t,FBSPlist,'k--',label=r'$F_{B}^{sp}$')
    plt.xlabel('time(h)')
    plt.ylabel('$F_{B}(mol/h)$')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(t,CBlist,'c-',label=r'Reactor Concentration of B')
    plt.plot(t,CBMAXlist,'k--',label=r'$C_{B}^{max}$')
    plt.xlabel('time(h)')
    plt.ylabel('$C_{B}(mol/L)$')
    plt.legend(loc='best')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(t,Qlist,'r-')
    plt.xlabel('time(h)')
    plt.ylabel('$F(L/h)$')
    plt.legend(['Inlet Flow'],loc='best')
    plt.subplot(212)
    plt.plot(t,Tklist,'r-')
    plt.xlabel('time(h)')
    plt.ylabel('$T_{k}(°C)$')
    plt.legend(['Jacket Temperature'],loc='best')

    plt.figure(3)
    plt.subplot(211)
    plt.plot(t,CAlist,'g-')
    plt.xlabel('time(h)')
    plt.ylabel('$C_{A}(mol/L)$')
    plt.legend(['Reactor Concentration of A'],loc='best')
    plt.subplot(212)
    plt.plot(t,Tlist,'m-')
    plt.xlabel('time(h)')
    plt.ylabel('$T(°C)$')
    plt.legend(['Reactor Temperature'],loc='best')


    plt.show()

# main program

# Constants
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 1E6
MINIBATCH_SIZE = 64
MAX_EP_STEPS = 10000 # 10000
MAX_TOT_STEPS = 3000000 #1500000
GAMMA = 0.99
# RANDOM_SEED = 1234
NB_STEPS = 10000
deltaT = 0.01
FBSP = 2000
CBMAX = 1.13

# actor_noise = OrnstreinUhlenbeckActionNoise(mu=np.zeros(2))
# for i in range(200):
#     print(np.array([1,0.1]) * actor_noise())

with tf.Session() as sess: #open a session to evaluate graph nodes
    state_dim = 3
    action_dim = 2
    action_bound = [1,.1] #10 L/h e 1°C

    # # seeds definition
    # np.random.seed(RANDOM_SEED)
    # tf.set_random_seed(RANDOM_SEED)

    actor = ActorNetwork(sess,state_dim, action_dim, action_bound, \
    ACTOR_LEARNING_RATE, TAU, MINIBATCH_SIZE)

    #sess.run(tf.global_variables_initializer())


    critic = CriticNetwork(sess,state_dim, action_dim, \
    CRITIC_LEARNING_RATE, TAU, GAMMA, actor.get_num_trainable_vars())

    actor_noise = OrnstreinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    #actor_noise = DecreasingNoise()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    train(sess, actor, critic, actor_noise, replay_buffer, BUFFER_SIZE, MINIBATCH_SIZE, \
    MAX_TOT_STEPS, MAX_EP_STEPS, FBSP, deltaT)

    #tvars = actor.network_params
    #tvars_vals = actor.sess.run(tvars)

    #for var, val in zip(tvars, tvars_vals):
    #    print(var.name, val)  # Prints the name of the variable alongside its value

    #pickle_actorout=open('teste5.pickle','wb')
    #pickle.dump(tvars_vals,pickle_actorout)
    #pickle_actorout.close()

    #pickle_actorin=open('teste5.pickle','rb')
    #parameters_actor = pickle.load(pickle_actorin)
    #print(parameters_actor)

    #atualizar the actor network parameters
    #[actor.network_params[i].assign(parameters_actor[i]) for i in range(len(actor.network_params))]

    #print(parameters_actor)

    Simulate_VdV(actor,NB_STEPS, deltaT)

    #train_online(sess, actor, critic, actor_noise, replay_buffer, BUFFER_SIZE, MINIBATCH_SIZE,  \
    #NB_STEPS, FBSP, deltaT)
