

class PID:
    # modified PI controller from the udacity Self-Driving car project 3
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp # proportion gain
        self.Ki = Ki # integral gain
        self.Kd = Kd # derivative gain
        self.set_point = 0. # desired value for the output
        self.integral = 0. # accumulated integral of the error
        self.old_measurement = 0. # holds the last measured error

    def set_desired(self, desired):
        # set the desired output value to controll around
        self.set_point = desired

    def clear_PID(self):
        # resets the PID controller
        self.set_point = 0.
        self.integral = 0.
        self.old_error = 0.

    def update(self, measurement):
        # perform the PID control function
        # proportional error
        error = self.set_point - measurement

        # integral error
        self.integral += error
        # derivative error
        delta_error = self.old_measurement - measurement
        self.old_measurement = measurement
        return self.Kp * error + self.Ki * self.integral + self.Kd * delta_error
