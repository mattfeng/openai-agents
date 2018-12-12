from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
import numpy as np

class DDQN(object):

    def __init__(self, input_size, n_actions, layers, learning_rate):
        self.lr = learning_rate
        self.input = Input(shape=input_size)

        prev_layer = self.input
        for layer in layers:
            prev_layer = layer(prev_layer)
        
        self.qvals = Dense(n_actions, activation="linear")(prev_layer)
        self.model = Model(inputs=self.input, outputs=self.qvals)
        self.opt = SGD(lr=self.lr, momentum=0.5, decay=1e-6, clipnorm=2)
        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer=self.opt)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        result = self.model.fit(X, y, epochs=1, shuffle=True, verbose=0)

        return result

    def estimate_q(self, state):
        """
        Args:
            state: Matrix where each row is a different state.
        """
        estimate = self.model.predict(state)
        return estimate
    
    def get_model_weights(self):
        """
        """
        return self.model.get_weights()
    
    def set_model_weights(self, weights):
        """
        """
        self.model.set_weights(weights)
    
    def save_weights(self, fname):
        self.model.save_weights(fname, overwrite=True)

        
if __name__ == "__main__":
    import numpy as np

    ddqn = DDQN((4,), 2, 10, 1e-3)
    ddqn.estimate_q(np.array([[1, 2, 3, 4]]))

