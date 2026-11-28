import random

# '''
# Each new active hall call is randomly assigned to an elevator.
# Each elevator maintains its own queue of assigned calls.
# Calls are serviced in the order they were assigned.
# Once a call is no longer active, its assignment is removed.
# If an elevator has no new call to service,  it stops where it is
# '''


class RandomAgent:
    def __init__(self, num_elevators, seed=None):
        '''
        Initialize the random agent with the following attributes:
        - num_elevators: number of elevators in the environment
        - seed: random seed for reproducibility during testing (optional)
        - assignments: tracks what calls have been assigned to elevators
        '''
        
        self.num_elevators = num_elevators
        self.random_num = random.Random(seed)   # Use a local random generator for reproducibility
        self.assignments = {}                   # dictionary to keep track of elevator assignments to calls, required for consistent dispatches

    def act(self, observation):
        '''
        Given an obervation:
        - identify active calls (i.e., floors with waiting passengers)
        - assign a random elevator to each new active call (remove old ones)
        - create and return the action list
            action list = list of integers length 2*num_elevators where each pair of numbers represents an elevator 
            [dispatch_target, elevator_direction]
            - dispatch_target: the floor to which the elevator is dispatched (between 1 and maxfloornumber=10)
            - elevator_direction: the direction in which the elevator should move (-1=down, 0=no direction, 1=up)
        '''
        
        # get active calls from the observation
        active_calls = get_active_calls(observation)
        
        # update assignments list to remove serviced calls and add new calls
        self.assignments = update_assignments(self.assignments, active_calls, self.num_elevators, self.random_num)

        # generate action
        action = generate_action(observation, self.assignments, self.num_elevators)
        
        # PRINT STATEMENTS FOR DEBUGGING 
        
        # default_action = [-1, 0]
        # print("-----------------------------------------------------------")
        # for elevator in range(self.num_elevators):
        #     curr_action = [action[2 * elevator], action[2 * elevator +1]]
        #     if curr_action == default_action:
        #         print(f"Elevator {elevator} Drop Off in Progress at {observation.ElevatorStates[elevator].ReservedTargetFloors}")
        #     else:
        #         print(f"Elevator {elevator} Pickup at {curr_action}")
            
        # print(f"Active Calls: {active_calls}")
        # print(f"Elevator Assignments: {self.assignments}")
        # print(f"Action List: {action}")
        
        # #print("Elevator Positions: ", observation.ElevatorStates)
                
        # for elevatorState in observation.ElevatorStates:
        #     #print(f"Elevator Positions: {elevatorstate.Floor}, {elevatorstate.Velocity}, {elevatorstate.Direction}, {elevatorstate.CurrentDispatchTarget}")
        #     print(
        #         f"Floor: {elevatorState.Floor}, "
        #         f"Direction: {elevatorState.Direction}, "
        #         f"Dispatch Target: {elevatorState.CurrentDispatchTarget}, "
        #         f"Dispatch Direction: {elevatorState.DispatchTargetDirection}, "
        #         f"Door State: {elevatorState.DoorState}, "
        #         f"Reserved Floors: {elevatorState.ReservedTargetFloors}"
        #         )
        return action
    
   
### HELPER FUNCTIONS 
def get_active_calls(observation):
    '''
    Given an observation, return a set of active calls: (floor, direction)
    '''
    
    # create an empty list to store active calls
    active_calls = []

    # add upward floors with direction = 1
    for floor in observation.RequiringUpwardFloors:
        call = (floor, 1)
        
        # check if it exists before adding for uniqueness
        if call not in active_calls:
            active_calls.append(call)

    # add downward floors with direction =-1
    for floor in observation.RequiringDownwardFloors:
        call = (floor, -1)
        
        # check if it exists before adding for uniqueness
        if call not in active_calls:
            active_calls.append(call)

    return active_calls


def update_assignments(assignments, active_calls, num_elevators, random_num):
    """
    update assignments dictionary to remove calls that have already been serviced and add new ones
    """
    
    # generate empty dictionary to store elevator assignments
    updated_assignments = {}

    # if the call is still active keep the assignment, otherwise remove assignment
    # for key, value pair in assignments dictionary
    for call, elevator in assignments.items():
    
        if call in active_calls:
            updated_assignments[call] = elevator
            
    # randomly assign new hall calls that are not already assigned
    for call in active_calls:
        if call not in updated_assignments:
            updated_assignments[call] = random_num.randrange(num_elevators)
        
    return updated_assignments


def generate_action(observation, assignments, numElevators):
    """
    Convert elevator assignments into a LiftSim action
    """
    # initiate default action list of 0's
    action = [-1, 0] * (numElevators)

    # for each elevator
    for elevator in range(numElevators):

        # get list of calls assigned to elevator
        assigned_calls = []
        
        # get state information for the elevator
        elevator_info = observation.ElevatorStates[elevator]


        # The problem could be the transition between these states:
        # 1. Elevator reaches pickup floor.
        # 2. Doors open.
        # 3. Passenger boards.
        # 4. Doors close.
        # 5. ReservedTargetFloors becomes populated.
        # 6. Elevator begins drop-off.
        # If ReservedTargetFloors is still empty during steps 2–4, agent can dispatch another pickup.
        
        # allow the elevator to finish serving onboard passengers before taking new assignment
        # allow the elevator to close/open doors before sending new assignment 
        # i.e. DO NOT send new action if there are still dropoffs to make OR the elevator doors are in transition state 
        if elevator_info.ReservedTargetFloors or elevator_info.DoorState > 0:
            continue
        
        # for each item in assignments dict
        for call, assigned_elevator in assignments.items():
            # if elevator matches add to list
            if assigned_elevator == elevator:
                assigned_calls.append(call)
        
        # if a call was assigned to elevator, create action corresponding to call on FIFO basis
        if assigned_calls:
            
            # get the call from the first list item
            floor, direction = assigned_calls[0]

            # update action list at correct index
            action[2 * elevator] = floor
            action[2 * elevator + 1] = direction

    return action
