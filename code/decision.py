import numpy as np

def forward(Rover, speed, steer, can_move):
    # moves the rover forward
    if can_move:
        if Rover.vel < Rover.max_vel:
            # Set throttle value to controlled throttle setting
            Rover.throttle = np.clip(Rover.PID.update(speed), -10, 10)
        else:  # Else coast
            Rover.throttle = 0
        Rover.brake = 0
        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(steer, -15, 15)
    else:
        # objects in the way so turn around
        Rover.mode ='turn_around'
        Rover.PID.clear_PID()

def backward(Rover):
    # moves the rover backwards if the rover has a un-level pitch.
    # i.e the rover has driven up a wall or onto an object
    if not (Rover.pitch < Rover.pitch_cutoff or Rover.pitch > 360 - Rover.pitch_cutoff):
        Rover.PID.set_desired(-1 * Rover.throttle_set)
        Rover.throttle = np.clip(Rover.PID.update(Rover.vel), -10, 10)
        Rover.brake = 0
        # go straigt back with no angle
        Rover.steer = 0
    else:
        Rover.mode ='turn_around'
        Rover.PID.clear_PID()

def stop(Rover):
    # stop the movement of the rover
    # If we're in stop mode but still moving keep braking
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.steer = 0
    Rover.PID.clear_PID()

    # once stop set the next state
    if Rover.vel == 0:
        if Rover.can_go_forward:
            Rover.mode = 'forward'
        else:
            Rover.mode = 'turn_around'

def brake(Rover, brake_force, steer):
    # slow the rover down when near objects
    vel_gain = 100
    # determine the brake force needed to stop in the distance
    # force = velocity(m/s) / distance(pixel)
    # vel_gain to boost mm to pixel ratio, +1 to prevent div(zero)
    #brake_force = np.abs(Rover.vel) * vel_gain / (np.abs(distance) + 1)
    #if brake_force > 1:
    #    brake_force = Rover.brake_set
    #else:
    #    brake_force = brake_force * Rover.brake_set
    Rover.throttle = 0
    Rover.brake = 0.1
    Rover.steer = np.clip(steer, -15, 15)

def turn_around(Rover):
    # turns the rover around until the path is clear
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.PID.clear_PID()

    # check if the path is clear
    if Rover.can_go_forward:
        Rover.mode = 'forward'
        Rover.turn_dir = 'none'
        Rover.brake = 0
        Rover.PID.clear_PID()
    elif Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = 0.1
        Rover.steer = 0
    else:
        # check which direction the rover last rotated about and
        # continue in that direction.
        if Rover.turn_dir == 'none':
            # first movement will turn to the side of the best chance
            # of a clear path or if none will turn right
            if len(Rover.nav_angles) > 0:
                if np.mean(Rover.nav_angles * 180 / np.pi) > 0:
                    Rover.turn_dir = 'left'
                    Rover.steer = 15
                else:
                    Rover.turn_dir = 'right'
                    Rover.steer = -15
            else:
                Rover.turn_dir = 'right'
                Rover.steer = -15
        elif Rover.turn_dir == 'left':
            Rover.steer = 15
        elif Rover.turn_dir == 'right':
            Rover.steer = -15
        else:
            Rover.turn_dir = 'none'
            Rover.steer = -15

        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0

def sample_collect(Rover):
    # moves towards the found sample and picks it up when close
    Rover.mode = 'sample'

    # check that there is sample data available
    if len(Rover.sample_dists) > 0:
        distance = np.mean(Rover.sample_dists)
    else:
        distance = 30
    if len(Rover.sample_angles) > 0:
        steer = np.mean(Rover.sample_angles * 180 / np.pi)
    else:
        steer = 0

    # check if the sample can be picked up
    if Rover.near_sample > 0:
        if Rover.vel > 0.2:
            # travelling to fast so stop
            brake(Rover, 1, steer*0.8)
        elif Rover.vel <= 0.1:
            # the sample can be picked up so pick it up
            stop(Rover)
            Rover.send_pickup = True
            Rover.mode = 'turn_around'
    elif distance < 40:
        # slow the rover down if it is close to the sample
        if Rover.vel > 0.2:
            Rover.PID.clear_PID()
            brake(Rover, 0.1, steer*0.8)
        else:
            # the rover stop not close enough to pick up the sample
            Rover.PID.set_desired(0.2)
            forward(Rover, Rover.vel, 0.8*steer, True)
    elif Rover.sample_angles is not None:
        # sample is not close so move towards it
        Rover.PID.set_desired(Rover.throttle_set)
        forward(Rover, Rover.vel, steer*0.8, True)
    else:
        Rover.mode = 'turn_around'

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # check which state the rover is in and calles the functions to handle that state

    # only perform state updates if the rover is not picking up a sample rock
    if Rover.picking_up == 0 and Rover.send_pickup is False:

        if Rover.mode == 'sample': # state in sample found
            Rover.skip_next = True # do not skip image proccessing
            if Rover.sample_angles.any():
                sample_collect(Rover)
            else:
                Rover.mode = 'turn_around'
        #elif Rover.mode == 'stop':
        #    stop(Rover)
        elif Rover.mode == 'turn_around':
            turn_around(Rover)
        elif Rover.mode == 'forward' and Rover.can_go_forward:
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                if len(Rover.nav_angles) > 0:
                    steer = np.mean(Rover.nav_angles * 180 / np.pi)
                else:
                    steer = 0
                Rover.PID.set_desired(Rover.throttle_set)
                forward(Rover, Rover.vel, steer*0.9, Rover.can_go_forward)
            else:
                Rover.mode = 'turn_around'
                turn_around(Rover)
        else:
            Rover.mode = 'turn_around'
            turn_around(Rover)

    return Rover

