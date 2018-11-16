import random

class Environment:
    def __init__(self):
        self._state_count = 1
        self._state = [random.random()*60+50, random.random()*40+20, 0]
        self._consumers = [45, 60, 30]

    def get_state(self):
        return self._state

    def next_state(self, action):
        water_pool = sum(self._state)
        max_required = sum(action)

        reward = [0, 0, 0, 0]
        
        if max_required > 0:
            reward[0] = min([water_pool*action[0]/max_required, self._consumers[0]])
            reward[1] = min([water_pool*action[1]/max_required, self._consumers[1]])
            reward[2] = min([water_pool*action[2]/max_required, self._consumers[2]])

            self._state[2] = water_pool*action[3]/max_required
            if self._state[2] > 50:
                self._state[2] = 50
                reward[3] = 0

        if self._state_count % 500 == 0:
            self._state[0] = random.random()*60+50
            self._state[1] = random.random()*40+20
        self._state_count += 1
        return reward