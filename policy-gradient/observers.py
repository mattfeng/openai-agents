class StepObserver(object):
    def __init__(self, agent):
        """

        Args:
            agent: The RL agent to modify as events are observed.
        """
        self.agent = agent

    def __call__(self, event):
        if event == "finish_step":
            self.finish_step()
        else:
            pass
    
    def finish_step(self):
        print("Step finished.")
